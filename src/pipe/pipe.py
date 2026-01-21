import contextlib
import os
import pickle
import random
import re
import resource
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue, Value, freeze_support

# Change the start method if not already set elsewhere
if not mp.get_start_method(allow_none=True):
    mp.set_start_method("spawn", force=True)


def _check_picklable(obj, name: str, is_thread: bool = False) -> None:
    """Check if a worker object can be pickled and warn if not.

    This helps catch common issues where workers have unpicklable state
    initialized in __init__ instead of load().
    """
    if is_thread:
        return  # Threads don't need pickling

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        # Find problematic attributes
        problematic = []
        if hasattr(obj, "__dict__"):
            for attr_name, attr_val in obj.__dict__.items():
                try:
                    pickle.dumps(attr_val)
                except Exception:
                    problematic.append(f"  - self.{attr_name}: {type(attr_val).__name__}")

        msg = f"\n⚠️  WARNING: Worker '{name}' may not be picklable for multiprocessing.\n"
        msg += f"   Error: {e}\n"
        if problematic:
            msg += "   Problematic attributes:\n"
            msg += "\n".join(problematic) + "\n"
        msg += "   Consider initializing these in load() instead of __init__.\n"
        print(msg)
    except Exception as e:
        # Catch-all for other pickling issues
        print(f"\n⚠️  WARNING: Worker '{name}' pickle check failed: {e}\n")


def _serialize_tensors_recursive(obj):
    """Recursively serialize tensors to bytes in nested structures.

    This avoids PyTorch's file descriptor sharing mechanism which breaks
    when the sender process exits before the receiver deserializes.
    """
    import io
    try:
        if torch.is_tensor(obj):
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            return {"__tensor_bytes__": buffer.getvalue(), "__tensor_shape__": list(obj.shape)}
        if isinstance(obj, dict):
            return {k: _serialize_tensors_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize_tensors_recursive(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_serialize_tensors_recursive(item) for item in obj)
        return obj
    except Exception as e:
        print(f"Warning: Failed to serialize tensor: {e}")
        return obj


def _deserialize_tensors_recursive(obj):
    """Recursively deserialize tensors from bytes in nested structures."""
    import io
    try:
        if isinstance(obj, dict):
            if "__tensor_bytes__" in obj:
                buffer = io.BytesIO(obj["__tensor_bytes__"])
                return torch.load(buffer, weights_only=True)
            return {k: _deserialize_tensors_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deserialize_tensors_recursive(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_deserialize_tensors_recursive(item) for item in obj)
        return obj
    except Exception as e:
        print(f"Warning: Failed to deserialize tensor: {e}")
        return obj


def _get_fd_info():
    """Get file descriptor usage information"""
    try:
        pid = os.getpid()
        try:
            fd_count = len(os.listdir(f"/proc/{pid}/fd"))
        except Exception:
            fd_count = "unknown"
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        return {
            "pid": pid,
            "fd_count": fd_count,
            "soft_limit": soft_limit,
            "hard_limit": hard_limit,
        }
    except Exception as e:
        return {"error": str(e)}


def _increase_fd_limit(worker_id: str):
    """Increase FD limit to hard limit, critical for tensor passing"""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= hard:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Worker {worker_id}: Increased FD limit from {soft} to {hard}")
    except Exception as e:
        print(f"Worker {worker_id}: Warning - could not increase FD limit: {e}")


def _worker_run(
    worker,
    in_queue=None,
    out_queue=None,
    should_stop=None,
    working=None,
    gpu_id=None,
    timing_dict=None,
    worker_id=None,
    share_tensors=False,
    raise_errors=False,
    stage_end_counter=None,
    stage_worker_count=None,
    stage_idx=None,
    stage_name=None,
    is_final_stage=False,
    expected_consumers=1,
    upstream_done=None,
    stage_done=None,
    debug=False,
):
    """Worker process using Event-based completion signaling.

    Instead of passing "end" sentinels through queues (which causes race conditions),
    workers check if upstream_done Event is set AND the input queue is empty.
    """
    worker_desc = f"{stage_name} ({worker_id})" if stage_name else worker_id
    is_root = (in_queue is None)

    _increase_fd_limit(worker_desc)
    fd_info = _get_fd_info()
    print(f"Worker {worker_desc} starting up - FD info: {fd_info}")

    if gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
            print(f"Worker process set to GPU {gpu_id}")
        except Exception as e:
            print(f"Failed to set GPU {gpu_id}: {e}")

    def handle_signal(signum, frame):
        print(f"Worker {worker_desc} received signal {signum}, stopping gracefully")
        should_stop.value = 1

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if hasattr(worker, "load"):
        print(f"Worker {worker_desc} calling load()")
        worker.load()

    items_processed = 0
    total_process_time = 0.0
    total_audio_duration = 0.0
    worker_start_wall_time = time.time()
    last_fd_check = time.time()
    fd_check_interval = 30
    consecutive_empty = 0
    EMPTY_THRESHOLD = 10  # 1 second at 0.1s timeout

    def extract_audio_duration(obj):
        if obj is None:
            return 0.0
        if isinstance(obj, dict):
            dur = obj.get("duration", 0.0)
            if isinstance(dur, list):
                return sum(dur)
            if dur:
                return dur
            if "sample" in obj:
                return obj["sample"].get("duration", 0.0)
            if "segments" in obj:
                return sum(s.get("duration", 0.0) for s in obj["segments"])
            return 0.0
        if isinstance(obj, list):
            return sum(extract_audio_duration(x) for x in obj if x != "end")
        return 0.0

    while not should_stop.value:
        now = time.time()
        if now - last_fd_check > fd_check_interval:
            fd_info = _get_fd_info()
            if debug:
                print(f"Worker {worker_desc} FD check: {fd_info['fd_count']}/{fd_info['soft_limit']} (items: {items_processed})")
            last_fd_check = now

        try:
            if should_stop.value == 1:
                break

            if is_root:
                # Root worker: generate items
                start_time = time.time()
                result = worker()
                process_time = time.time() - start_time

                if result is None:
                    continue
                if not isinstance(result, list):
                    result = [result]

                has_end = "end" in result
                valid_items = [r for r in result if r is not None and r != "end"]

                if timing_dict is not None and worker_id is not None and valid_items:
                    items_processed += len(valid_items)
                    total_process_time += process_time
                    total_audio_duration += extract_audio_duration(valid_items)
                    timing_dict[worker_id] = {
                        "items": items_processed,
                        "total_time": total_process_time,
                        "avg_time": total_process_time / items_processed if items_processed else 0,
                        "audio_duration": total_audio_duration,
                        "start_wall_time": worker_start_wall_time,
                    }

                if out_queue:
                    for item in valid_items:
                        serialized = _serialize_tensors_recursive(item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue

                if has_end:
                    # Root generator finished - signal downstream via Event
                    break

            else:
                # Non-root worker: pull from input queue
                if out_queue is not None and out_queue.full():
                    time.sleep(0.001)
                    continue

                try:
                    item = in_queue.get(timeout=0.1)
                    consecutive_empty = 0
                except Empty:
                    consecutive_empty += 1
                    # Check if upstream is done AND queue is drained
                    if upstream_done is not None and upstream_done.is_set():
                        if consecutive_empty >= EMPTY_THRESHOLD:
                            break
                    continue
                except (OSError, EOFError, BrokenPipeError):
                    break

                # Handle autoscaler scale-down signal
                if item == "worker_stop":
                    print(f"Worker {worker_id} received stop signal, exiting gracefully")
                    break

                # Skip any legacy "end" sentinels (shouldn't happen with Event-based)
                if item == "end":
                    continue

                # Deserialize any tensors from bytes
                item = _deserialize_tensors_recursive(item)

                # Process the item
                start_time = time.time()
                result = worker(item)
                process_time = time.time() - start_time

                if result is None:
                    continue
                if not isinstance(result, list):
                    result = [result]

                valid_items = [r for r in result if r is not None and r != "end"]

                if timing_dict is not None and worker_id is not None:
                    items_processed += 1
                    total_process_time += process_time
                    total_audio_duration += extract_audio_duration(result) or extract_audio_duration(item)
                    if items_processed % 10 == 0:
                        timing_dict[worker_id] = {
                            "items": items_processed,
                            "total_time": total_process_time,
                            "avg_time": total_process_time / items_processed,
                            "audio_duration": total_audio_duration,
                            "start_wall_time": worker_start_wall_time,
                        }

                if out_queue:
                    for out_item in valid_items:
                        serialized = _serialize_tensors_recursive(out_item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue

        except (ConnectionError, FileNotFoundError):
            should_stop.value = 2
            break
        except Exception as e:
            print(f"Worker {worker_desc} error: {e}")
            traceback.print_exc()
            if raise_errors:
                raise e

    # Flush any buffered items by calling worker("end") if non-root
    if not is_root and out_queue:
        try:
            # Call worker("end") to trigger flush (e.g., for batchers)
            result = worker("end")
            if result is not None:
                if not isinstance(result, list):
                    result = [result]
                valid_items = [r for r in result if r is not None and r != "end"]
                if valid_items:
                    print(f"Worker {worker_desc} flushing {len(valid_items)} items")
                    for item in valid_items:
                        serialized = _serialize_tensors_recursive(item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue
        except Exception as e:
            print(f"Worker {worker_desc} flush error: {e}")

    # Also call flush() method if worker has it
    if hasattr(worker, "flush") and out_queue:
        try:
            flushed_items = list(worker.flush())
            if flushed_items:
                print(f"Worker {worker_desc} flushing {len(flushed_items)} items via flush()")
                for item in flushed_items:
                    if item is not None:
                        serialized = _serialize_tensors_recursive(item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue
        except Exception as e:
            print(f"Worker {worker_desc} flush() error: {e}")

    # Worker exiting - coordinate with siblings
    if stage_end_counter is not None and stage_worker_count is not None:
        with stage_end_counter.get_lock():
            stage_end_counter.value += 1
            finished_workers = stage_end_counter.value

        current_worker_count = stage_worker_count.value
        stage_desc = f"{stage_name}" if stage_name else f"stage_{stage_idx}"
        print(f"Worker {worker_id} finished ({finished_workers}/{current_worker_count} at {stage_desc})")

        # Last worker to finish signals downstream via Event
        if finished_workers >= current_worker_count:
            if stage_done is not None:
                stage_done.set()
                print(f"All workers finished at {stage_desc}, signaling downstream")


def _threaded_worker_run(
    worker,
    in_queue=None,
    out_queue=None,
    should_stop=None,
    working=None,
    gpu_id=None,
    timing_dict=None,
    worker_id=None,
    share_tensors=False,
    raise_errors=False,
    num_threads=1,
    stage_end_counter=None,
    stage_worker_count=None,
    stage_idx=None,
    stage_name=None,
    is_final_stage=False,
    expected_consumers=1,
    upstream_done=None,
    stage_done=None,
    debug=False,
):
    """Threaded worker using Event-based completion signaling."""
    worker_desc = f"{stage_name} ({worker_id})" if stage_name else worker_id
    is_root = (in_queue is None)

    _increase_fd_limit(worker_desc)
    fd_info = _get_fd_info()
    print(f"Threaded worker {worker_desc} starting with {num_threads} threads - FD info: {fd_info}")

    if gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
            print(f"Threaded worker {worker_desc} set to GPU {gpu_id}")
        except Exception as e:
            print(f"Failed to set GPU {gpu_id}: {e}")

    thread_stop = threading.Event()

    def handle_signal(signum, frame):
        print(f"Threaded worker {worker_desc} received signal {signum}, stopping")
        should_stop.value = 1
        thread_stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if hasattr(worker, "load"):
        print(f"Threaded worker {worker_desc} calling load()")
        worker.load()

    worker_start_wall_time = time.time()

    def extract_audio_duration(obj):
        if obj is None:
            return 0.0
        if isinstance(obj, dict):
            dur = obj.get("duration", 0.0)
            if isinstance(dur, list):
                return sum(dur)
            if dur:
                return dur
            if "sample" in obj:
                return obj["sample"].get("duration", 0.0)
            if "segments" in obj:
                return sum(s.get("duration", 0.0) for s in obj["segments"])
            return 0.0
        if isinstance(obj, list):
            return sum(extract_audio_duration(x) for x in obj if x != "end")
        return 0.0

    def thread_fn():
        local_items = 0
        local_time = 0.0
        local_audio = 0.0
        consecutive_empty = 0
        EMPTY_THRESHOLD = 10  # 1 second at 0.1s timeout

        while not should_stop.value and not thread_stop.is_set():
            try:
                if is_root:
                    # Root: generate items
                    start_time = time.time()
                    result = worker()
                    process_time = time.time() - start_time

                    if result is None:
                        continue
                    if not isinstance(result, list):
                        result = [result]

                    has_end = "end" in result
                    valid_items = [r for r in result if r is not None and r != "end"]

                    if timing_dict is not None and worker_id is not None and valid_items:
                        local_items += len(valid_items)
                        local_time += process_time
                        local_audio += extract_audio_duration(valid_items)

                    if out_queue:
                        for item in valid_items:
                            serialized = _serialize_tensors_recursive(item)
                            while not should_stop.value and not thread_stop.is_set():
                                try:
                                    out_queue.put(serialized, timeout=0.1)
                                    break
                                except Full:
                                    continue

                    if has_end:
                        thread_stop.set()
                        return
                else:
                    # Non-root: pull from queue
                    if out_queue is not None and out_queue.full():
                        time.sleep(0.001)
                        continue

                    try:
                        item = in_queue.get(timeout=0.1)
                        consecutive_empty = 0
                    except Empty:
                        consecutive_empty += 1
                        if upstream_done is not None and upstream_done.is_set():
                            if consecutive_empty >= EMPTY_THRESHOLD:
                                return
                        continue
                    except (OSError, EOFError, BrokenPipeError):
                        return

                    if should_stop.value:
                        with contextlib.suppress(Full):
                            in_queue.put(item)
                        return

                    if item == "worker_stop":
                        print(f"Thread {worker_id} received stop signal")
                        return

                    if item == "end":
                        continue

                    # Deserialize any tensors from bytes
                    item = _deserialize_tensors_recursive(item)

                    # Process item
                    start_time = time.time()
                    result = worker(item)
                    process_time = time.time() - start_time

                    if result is None:
                        continue
                    if not isinstance(result, list):
                        result = [result]

                    valid_items = [r for r in result if r is not None and r != "end"]

                    if timing_dict is not None and worker_id is not None:
                        local_items += 1
                        local_time += process_time
                        local_audio += extract_audio_duration(result) or extract_audio_duration(item)
                        if local_items % 10 == 0:
                            timing_dict[worker_id] = {
                                "items": local_items,
                                "total_time": local_time,
                                "avg_time": local_time / local_items,
                                "audio_duration": local_audio,
                                "start_wall_time": worker_start_wall_time,
                            }

                    if out_queue:
                        for out_item in valid_items:
                            serialized = _serialize_tensors_recursive(out_item)
                            while not should_stop.value:
                                try:
                                    out_queue.put(serialized, timeout=0.1)
                                    break
                                except Full:
                                    continue

            except (ConnectionError, FileNotFoundError):
                should_stop.value = 2
                return
            except Exception as e:
                print(f"Thread {worker_desc} error: {e}")
                traceback.print_exc()
                if raise_errors:
                    raise e

    executor = ThreadPoolExecutor(max_workers=num_threads)
    futures = [executor.submit(thread_fn) for _ in range(num_threads)]

    try:
        while not should_stop.value and not thread_stop.is_set():
            if all(f.done() for f in futures):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"Threaded worker {worker_desc} received KeyboardInterrupt")
        thread_stop.set()
    finally:
        thread_stop.set()
        executor.shutdown(wait=False)
        for f in futures:
            if not f.done():
                f.cancel()

        if hasattr(worker, "flush") and out_queue:
            try:
                for item in worker.flush():
                    serialized = _serialize_tensors_recursive(item)
                    out_queue.put(serialized, timeout=1)
            except Exception as e:
                print(f"Threaded worker {worker_desc} flush error: {e}")

        # Coordinate via Event
        if stage_end_counter is not None and stage_worker_count is not None:
            with stage_end_counter.get_lock():
                stage_end_counter.value += 1
                finished_workers = stage_end_counter.value

            current_worker_count = stage_worker_count.value
            print(f"Threaded worker {worker_desc} finished ({finished_workers}/{current_worker_count})")

            if finished_workers >= current_worker_count:
                if stage_done is not None:
                    stage_done.set()
                    print(f"All workers finished at {stage_name}, signaling downstream")

        print(f"Threaded worker {worker_desc} shutdown complete")


def _health_monitor_thread(
    pipe_instance,
    should_stop,
    health_check_interval,
    stop_event,
):
    """Monitor worker processes for crashes using process.is_alive() and exitcode

    Args:
        pipe_instance: The Pipe instance to monitor and restart workers
        stop_event: Threading event to signal shutdown
    """
    print("Health monitor starting up")

    # Track crash history per worker to detect repeated crashes
    crash_history = {}  # worker_id -> list of (timestamp, exitcode)
    CRASH_WINDOW = 300  # 5 minutes
    MAX_CRASHES_IN_WINDOW = 3

    while not should_stop.value and not stop_event.is_set():
        time.sleep(health_check_interval)

        if should_stop.value or stop_event.is_set():
            break

        crashed_workers = []

        # Check each worker process for crashes
        for idx, (proc, worker_id, stage_name) in enumerate(pipe_instance.worker_info):
            # Check if process is alive
            if not proc.is_alive():
                exitcode = proc.exitcode
                # Only report as crashed if exitcode indicates abnormal termination
                # exitcode == 0 means normal exit, None means still running
                if exitcode is not None and exitcode != 0:
                    crashed_workers.append((idx, worker_id, stage_name, exitcode))

        # Handle crashed workers
        if crashed_workers:
            print(f"\n⚠️  HEALTH CHECK: {len(crashed_workers)} worker(s) have crashed:")
            for idx, worker_id, stage_name, exitcode in crashed_workers:
                print(f"   - {worker_id} ({stage_name}): exitcode={exitcode}")

            # Check if we should do full restart instead of individual worker restart
            need_full_restart = False
            current_time = time.time()

            for idx, worker_id, stage_name, exitcode in crashed_workers:
                # Track crash history
                if worker_id not in crash_history:
                    crash_history[worker_id] = []
                crash_history[worker_id].append((current_time, exitcode))

                # Clean old entries outside the crash window
                crash_history[worker_id] = [
                    (t, e)
                    for t, e in crash_history[worker_id]
                    if current_time - t < CRASH_WINDOW
                ]

                # Check for conditions that require full restart:
                # 1. Segfault (exitcode -11) - queue corruption likely
                # 2. Bus error (exitcode -7) - memory/queue issues
                # 3. Repeated crashes (3+ in 5 minutes) - persistent problem
                if exitcode in (-11, -7):  # SIGSEGV, SIGBUS
                    print(f"   ⚠️  Serious error detected (exitcode {exitcode})")
                    need_full_restart = True
                    break
                if len(crash_history[worker_id]) >= MAX_CRASHES_IN_WINDOW:
                    print(
                        f"   ⚠️  Repeated crashes detected ({len(crash_history[worker_id])} in {CRASH_WINDOW}s)"
                    )
                    need_full_restart = True
                    break

            if need_full_restart and pipe_instance.allow_full_restart:
                print("   🔄 Triggering full pipeline restart to recreate queues...")
                try:
                    pipe_instance.restart(reason="Worker crash requiring queue refresh")
                    crash_history.clear()
                    print("   ✓ Full pipeline restart complete")
                except Exception as e:
                    print(f"   ✗ Failed to restart pipeline: {e}")
            elif need_full_restart:
                print(
                    "   ⚠️  Full restart needed but disabled - restarting workers individually"
                )
                for idx, worker_id, stage_name, exitcode in crashed_workers:
                    try:
                        pipe_instance._restart_worker(idx, worker_id)
                        print(f"   ✓ Restarted {worker_id}")
                    except Exception as e:
                        print(f"   ✗ Failed to restart {worker_id}: {e}")
            else:
                print("   Restarting crashed workers individually...")
                for idx, worker_id, stage_name, exitcode in crashed_workers:
                    try:
                        pipe_instance._restart_worker(idx, worker_id)
                        print(f"   ✓ Restarted {worker_id}")
                    except Exception as e:
                        print(f"   ✗ Failed to restart {worker_id}: {e}")

    print("Health monitor shutting down")


def _stats_monitor_thread(
    pipe_instance,
    stop_event,
    interval_seconds=30,
):
    """Background thread that periodically prints queue and timing stats."""
    while not stop_event.is_set():
        stop_event.wait(interval_seconds)
        if stop_event.is_set():
            break

        if not pipe_instance.timing_dict:
            continue

        # Calculate total items from final stage
        total_items = 0
        final_stage_idx = len(pipe_instance.jobs) - 1
        for worker_id, stats in dict(pipe_instance.timing_dict).items():
            match = re.match(r"stage_(\d+)", worker_id)
            stage_idx = int(match.group(1)) if match else -1
            if stage_idx == final_stage_idx:
                total_items += stats.get("items", 0)

        print(f"\n--- Timing & Queue Report (after {total_items} items) ---")

        # Print queue information
        print("  Queue Status:")
        for i, queue in enumerate(pipe_instance.queues):
            try:
                queue_size = queue.qsize()
                max_size = queue._maxsize if hasattr(queue, "_maxsize") else "unlimited"

                if i < len(pipe_instance.jobs):
                    source_func = pipe_instance.jobs[i]["func"]
                    if hasattr(source_func, "__name__"):
                        source_name = source_func.__name__
                    elif hasattr(source_func, "__class__"):
                        source_name = source_func.__class__.__name__
                    else:
                        source_name = str(type(source_func).__name__)

                    print(
                        f"    stage_{i} ({source_name}): {queue_size} items (max: {max_size})"
                    )
                else:
                    print(f"    stage_{i}: {queue_size} items (max: {max_size})")
            except Exception as e:
                print(f"    stage_{i}: unable to read size ({e})")

        # Group by stage index
        stages = {}
        for worker_id, stats in dict(pipe_instance.timing_dict).items():
            match = re.match(r"stage_(\d+)", worker_id)
            stage_idx = int(match.group(1)) if match else -1
            if stage_idx not in stages:
                stages[stage_idx] = []
            stages[stage_idx].append((worker_id, stats))

        print("\n  Worker Timing:")
        for stage_idx in sorted(stages.keys()):
            workers = stages[stage_idx]
            if 0 <= stage_idx < len(pipe_instance.jobs):
                func = pipe_instance.jobs[stage_idx]["func"]
                if hasattr(func, "__name__"):
                    stage_name = func.__name__
                elif hasattr(func, "__class__"):
                    stage_name = func.__class__.__name__
                else:
                    stage_name = type(func).__name__
            else:
                stage_name = f"stage_{stage_idx}"

            stage_total_items = 0
            stage_total_time = 0
            stage_total_audio = 0.0
            stage_earliest_start = None
            worker_rtfs = []
            now = time.time()

            for worker_id, stats in workers:
                items = stats.get("items", 0)
                total_time = stats.get("total_time", 0)
                audio_dur = stats.get("audio_duration", 0)
                start_wall = stats.get("start_wall_time")

                stage_total_items += items
                stage_total_time += total_time
                stage_total_audio += audio_dur

                # Per-worker RTF
                if start_wall is not None:
                    worker_elapsed = now - start_wall
                    worker_rtf = audio_dur / worker_elapsed if worker_elapsed > 0 else 0
                    worker_rtfs.append(worker_rtf)
                    if (
                        stage_earliest_start is None
                        or start_wall < stage_earliest_start
                    ):
                        stage_earliest_start = start_wall

            # Calculate stage RTF based on wall time
            wall_elapsed = now - stage_earliest_start if stage_earliest_start else 0
            stage_rtf = stage_total_audio / wall_elapsed if wall_elapsed > 0 else 0
            avg_worker_rtf = sum(worker_rtfs) / len(worker_rtfs) if worker_rtfs else 0
            print(
                f"    {stage_name}: {stage_total_items} items, {stage_total_time:.1f}s, {stage_rtf:.0f}x RTF ({avg_worker_rtf:.0f}x/worker)"
            )

        print("--- End Timing & Queue Report ---\n")

    print("Stats monitor shutting down")


def _get_cpu_usage():
    """Get current CPU usage as a ratio (0.0 to 1.0) by reading /proc/stat."""
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        # cpu user nice system idle iowait irq softirq steal guest guest_nice
        idle = int(parts[4])
        total = sum(int(x) for x in parts[1:])
        return idle, total
    except Exception:
        return None, None


def _autoscaler_thread(
    pipe_instance,
    should_stop,
    stop_event,
    check_interval=1.0,
    scale_up_threshold=0.8,
    scale_down_threshold=0.2,
    scale_up_samples=3,
    scale_down_samples=5,
    cooldown_seconds=3.0,
    cpu_limit=0.85,
):
    """Monitor input queue pressure and dynamically scale workers.

    Scale-up logic:
    - Scale up if input queue is consistently full (this stage is the bottleneck)
    - Don't scale up if CPU usage is above cpu_limit (machine is saturated)

    Scale-down logic:
    - Scale down if input queue is consistently nearly empty (excess capacity)
    - Don't scale below min_workers

    Note: Output queue state is irrelevant - full output just means we're keeping up.

    Args:
        pipe_instance: The Pipe instance to monitor and scale
        should_stop: Shared Value for stop signal
        stop_event: Threading event to signal shutdown
        check_interval: Seconds between checks (default: 1)
        scale_up_threshold: Input queue fill ratio that triggers scale up (default: 0.8)
        scale_down_threshold: Input queue fill ratio below which to scale down (default: 0.2)
        scale_up_samples: Consecutive samples before scaling up (default: 3)
        scale_down_samples: Consecutive samples before scaling down (default: 5)
        cooldown_seconds: Minimum seconds between scaling actions per stage (default: 3)
        cpu_limit: Don't scale up if CPU usage exceeds this ratio (default: 0.85)
    """
    print("Autoscaler started")

    high_pressure_counts = {}  # stage_idx -> consecutive high-pressure samples
    low_pressure_counts = {}   # stage_idx -> consecutive low-pressure samples
    last_scale_time = {}       # stage_idx -> timestamp of last scaling action

    # CPU tracking
    prev_idle, prev_total = _get_cpu_usage()
    cpu_saturated = False

    while not should_stop.value and not stop_event.is_set():
        time.sleep(check_interval)

        if should_stop.value or stop_event.is_set():
            break

        current_time = time.time()

        # Check CPU usage
        curr_idle, curr_total = _get_cpu_usage()
        if prev_idle is not None and curr_idle is not None:
            idle_delta = curr_idle - prev_idle
            total_delta = curr_total - prev_total
            if total_delta > 0:
                cpu_usage = 1.0 - (idle_delta / total_delta)
                cpu_saturated = cpu_usage > cpu_limit
            prev_idle, prev_total = curr_idle, curr_total

        for stage_idx, job in enumerate(pipe_instance.jobs):
            if not job.get("autoscale"):
                continue

            # Root stage has no input queue
            if stage_idx == 0:
                continue

            # Cooldown check
            if stage_idx in last_scale_time:
                if current_time - last_scale_time[stage_idx] < cooldown_seconds:
                    continue

            try:
                total_workers = pipe_instance.stage_worker_counts[stage_idx].value
                finished_workers = pipe_instance.stage_end_counters[stage_idx].value
                active_workers = total_workers - finished_workers
                max_workers = job.get("max_workers", active_workers * 4)
                min_workers = job.get("min_workers", 1)
                # Validate indices
                num_queues = len(pipe_instance.queues)
                num_workers = len(pipe_instance.stage_worker_counts)
                if stage_idx - 1 >= num_queues:
                    continue  # Queue not yet created
                if stage_idx >= num_workers:
                    continue  # Worker count not yet set
                if active_workers <= 0:
                    continue  # All workers have exited

                in_queue = pipe_instance.queues[stage_idx - 1]
                in_size = in_queue.qsize()
                in_max = in_queue._maxsize if hasattr(in_queue, "_maxsize") else 0
                if in_max <= 0:
                    continue

                in_fill = in_size / in_max

                # Check output queue - if full, workers would just block on put
                out_queue = pipe_instance.queues[stage_idx] if stage_idx < len(pipe_instance.queues) else None
                out_blocked = False
                if out_queue:
                    out_size = out_queue.qsize()
                    out_max = out_queue._maxsize if hasattr(out_queue, "_maxsize") else 0
                    if out_max > 0:
                        out_fill = out_size / out_max
                        out_blocked = out_fill >= 0.9  # Output queue near full

                # Scale UP: input queue full = this stage is the bottleneck
                # But don't scale if CPU is already saturated or output queue is blocked
                if active_workers < max_workers and in_fill >= scale_up_threshold:
                    if cpu_saturated:
                        # CPU is saturated, don't scale up - reset counter
                        high_pressure_counts[stage_idx] = 0
                        continue

                    if out_blocked:
                        # Output queue is full - adding workers won't help, they'll just block
                        high_pressure_counts[stage_idx] = 0
                        continue

                    high_pressure_counts[stage_idx] = high_pressure_counts.get(stage_idx, 0) + 1
                    low_pressure_counts[stage_idx] = 0

                    if high_pressure_counts[stage_idx] >= scale_up_samples:
                        high_pressure_counts[stage_idx] = 0
                        last_scale_time[stage_idx] = current_time
                        print(f"   Autoscale UP: stage {stage_idx} ({job.get('name', '?')}) in_fill={in_fill:.0%} -> {active_workers + 1} workers")
                        _spawn_additional_worker(pipe_instance, stage_idx, job)

                # Scale DOWN: input queue empty = excess capacity
                elif active_workers > min_workers and in_fill <= scale_down_threshold:
                    low_pressure_counts[stage_idx] = low_pressure_counts.get(stage_idx, 0) + 1
                    high_pressure_counts[stage_idx] = 0

                    if low_pressure_counts[stage_idx] >= scale_down_samples:
                        low_pressure_counts[stage_idx] = 0
                        last_scale_time[stage_idx] = current_time
                        print(f"   Autoscale DOWN: stage {stage_idx} ({job.get('name', '?')}) in_fill={in_fill:.0%} -> {active_workers - 1} workers")
                        _signal_worker_to_stop(pipe_instance, stage_idx)
                else:
                    # Reset counters if neither condition met
                    high_pressure_counts[stage_idx] = 0
                    low_pressure_counts[stage_idx] = 0

            except Exception as e:
                import traceback
                print(f"Autoscaler error at stage {stage_idx}: {e}\n{traceback.format_exc()}")

    print("Autoscaler stopped")


def _signal_worker_to_stop(pipe_instance, stage_idx):
    """Signal one worker at a stage to stop after completing current item."""
    # Put a special "worker_stop" sentinel in the input queue
    # Workers will check for this and exit gracefully
    in_queue = pipe_instance.queues[stage_idx - 1] if stage_idx > 0 else None
    if in_queue:
        try:
            in_queue.put("worker_stop", timeout=0.1)
            # NOTE: Do NOT decrement stage_worker_counts here!
            # The worker count must reflect the total workers that need to finish
            # before "all workers done" is triggered. Decrementing here causes a
            # race condition where the stopped worker triggers "all done" before
            # other workers finish processing.
            print(f"   Signaled worker at stage {stage_idx} to stop")
        except Exception as e:
            print(f"   Failed to signal worker stop: {e}")


def _spawn_additional_worker(pipe_instance, stage_idx, job):
    """Spawn an additional worker for the given stage."""
    from torch.multiprocessing import Process

    func = job["func"]
    is_threaded = job.get("thread", False)

    # Get stage name
    if hasattr(func, "__name__"):
        stage_name = func.__name__
    elif hasattr(func, "__class__"):
        stage_name = func.__class__.__name__
    else:
        stage_name = str(type(func).__name__)

    current_count = pipe_instance.stage_worker_counts[stage_idx].value
    new_worker_idx = current_count
    worker_id = f"stage_{stage_idx}_worker_{new_worker_idx}"

    print(f"🚀 Autoscaling: Adding worker {new_worker_idx + 1} to {stage_name} (stage {stage_idx})")

    # Get queues
    in_queue = pipe_instance.queues[stage_idx - 1] if stage_idx > 0 else None
    out_queue = pipe_instance.queues[stage_idx] if stage_idx < len(pipe_instance.queues) else None

    # Determine if this is the final stage
    is_final_stage = stage_idx == len(pipe_instance.jobs) - 1

    # Event-based completion signaling
    upstream_done = pipe_instance.stage_done_events[stage_idx - 1] if stage_idx > 0 else None
    stage_done = pipe_instance.stage_done_events[stage_idx]

    # Update worker count BEFORE spawning (for end signal coordination)
    pipe_instance.stage_worker_counts[stage_idx].value += 1
    new_worker_count = pipe_instance.stage_worker_counts[stage_idx].value

    if is_threaded:
        # For threaded workers, spawn a new threaded process
        args = (
            func,
            in_queue,
            out_queue,
            pipe_instance.should_stop,
            pipe_instance.working,
            None,  # gpu_id
            pipe_instance.timing_dict,
            worker_id,
            pipe_instance.share_tensors,
            pipe_instance.raise_errors,
            job.get("num_workers", 4),  # num_threads
            pipe_instance.stage_end_counters[stage_idx],
            pipe_instance.stage_worker_counts[stage_idx],  # Pass Value, not int
            stage_idx,
            stage_name,
            is_final_stage,
            pipe_instance.expected_consumers,
            upstream_done,  # upstream_done Event
            stage_done,  # stage_done Event
            pipe_instance.debug,
        )
        proc = Process(target=_threaded_worker_run, args=args, daemon=True)
    else:
        args = (
            func,
            in_queue,
            out_queue,
            pipe_instance.should_stop,
            pipe_instance.working,
            None,  # gpu_id
            pipe_instance.timing_dict,
            worker_id,
            pipe_instance.share_tensors,
            pipe_instance.raise_errors,
            pipe_instance.stage_end_counters[stage_idx],
            pipe_instance.stage_worker_counts[stage_idx],  # Pass Value, not int
            stage_idx,
            stage_name,
            is_final_stage,
            pipe_instance.expected_consumers,
            upstream_done,  # upstream_done Event
            stage_done,  # stage_done Event
            pipe_instance.debug,
        )
        proc = Process(target=_worker_run, args=args, daemon=True)

    proc.start()
    print(f"   Worker {worker_id} started with PID {proc.pid}")

    # Update tracking
    pipe_instance.processes.append(proc)
    pipe_instance.worker_info.append((proc, worker_id, stage_name))

    # Store config for potential restart
    pipe_instance.worker_configs[worker_id] = {
        "target": _threaded_worker_run if is_threaded else _worker_run,
        "args": args,
        "stage_name": stage_name,
    }


class PipeIterator:
    """Lightweight iterator that reads from a multiprocessing queue.

    Multiple PipeIterator instances can safely read from the same queue,
    with each getting different items automatically distributed by the queue.
    """

    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.queue.get(timeout=1.0)
                if item == "end":
                    raise StopIteration
                return _deserialize_tensors_recursive(item)
            except Empty:
                continue


class Pipe:
    def __init__(
        self,
        debug=False,
        share_tensors=False,
        raise_errors=None,
        health_check_interval=30,
        expected_consumers=1,
        stats_interval=30,
        allow_full_restart=True,
        autoscale=False,
        max_workers_per_stage=8,
    ):
        """
        Multiprocessing pipeline for data processing

        Args:
            debug: Enable debug output showing queue sizes
            share_tensors: Deprecated, no longer used. Tensor serialization is automatic.
            raise_errors: If True, raise errors instead of catching them. If None, defaults to True in debug mode
            health_check_interval: Seconds between health checks (default: 30)
            expected_consumers: Number of consumers (DDP workers) that will read from this pipe.
                               When > 1, end signals are multiplied for all consumers.
            stats_interval: Seconds between timing/queue stats prints (default: 30, 0 to disable)
            allow_full_restart: Allow health monitor to restart entire pipeline on repeated crashes (default: True)
            autoscale: If True, all stages start with 1 worker and scale up/down based on queue pressure
            max_workers_per_stage: Maximum workers per stage when global autoscale is enabled (default: 8)

        PyTorch Tensor Support:
        - Tensors are automatically serialized to bytes before passing through queues
        - Tensors are automatically deserialized when received by workers
        - Works with tensors in nested structures (dicts, lists, tuples)
        - No special handling required - just return tensors directly from workers
        """
        self.debug = debug
        self.share_tensors = share_tensors
        self.raise_errors = raise_errors if raise_errors is not None else debug
        self.health_check_interval = health_check_interval
        self.allow_full_restart = allow_full_restart
        self.autoscale = autoscale
        self.max_workers_per_stage = max_workers_per_stage
        self.jobs = []
        self.queues: list[Queue] = []
        self.processes = []
        self.worker_info = []  # List of (process, worker_id, stage_name) for health monitoring
        self.worker_configs = {}  # Dict mapping worker_id to worker configuration
        self.started = False
        self.working = Value("i", 0)  # Shared counter between processes
        self.should_stop = Value("i", 0)  # Shared flag for stopping processes
        self.restart_needed = Value("i", 0)  # Shared flag for signaling restart needed
        self.health_monitor_thread = None  # Thread for health monitoring
        self.health_monitor_stop_event = threading.Event()
        self.stats_monitor_thread = None  # Thread for stats printing
        self.stats_monitor_stop_event = threading.Event()
        self.stats_interval = stats_interval
        self.autoscaler_thread = None  # Thread for auto-scaling
        self.autoscaler_stop_event = threading.Event()
        self.gpus = self._get_gpu_count()
        self.expected_consumers = expected_consumers

        # Timing infrastructure for stats monitor
        if self.stats_interval > 0:
            import multiprocessing

            self.manager = multiprocessing.Manager()
            self.timing_dict = self.manager.dict()
        else:
            self.manager = None
            self.timing_dict = None

        # End signal coordination - shared counters and Events per stage
        self.stage_end_counters = []  # Will store Value('i', 0) for each stage
        self.stage_worker_counts = []  # Will store Value('i', N) for worker count per stage
        self.stage_done_events = []  # Will store Event() for each stage

    def _get_gpu_count(self):
        """Get the number of available GPUs"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            print("CUDA not available, pergpu flag will be ignored")
            return 0
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
            return 0

    def add(
        self,
        func: callable,
        workers=1,
        outqn=None,
        pergpu=False,
        thread=False,
        gpu_id=None,
        autoscale=None,
        min_workers=None,
        max_workers=None,
    ):
        """Add a worker function to the pipeline

        Args:
            func: Worker function or class
            outqn: Size of output queue (default: 1) 0 for unlimited out
            workers: Number of worker processes for this stage (default: 1)
            pergpu: If True, spawn workers on each available GPU (default: False)
            thread: If True, run multiple threads within a single process (default: False)
            gpu_id: If set, pin all workers to this specific GPU ID (overrides pergpu)
            autoscale: If True/False, explicitly set autoscaling for this stage. If None (default), uses global Pipe setting
            min_workers: Minimum workers when autoscaling (default: initial workers count)
            max_workers: Maximum workers when autoscaling (default: workers * 4, capped at GPU count for GPU stages)
        """
        # Get stage name for logging
        if hasattr(func, "__name__"):
            stage_name = func.__name__
        elif hasattr(func, "__class__"):
            stage_name = func.__class__.__name__
        else:
            stage_name = str(type(func).__name__)

        # Check if worker is picklable (warn if not)
        _check_picklable(func, stage_name, is_thread=thread)

        gpu_count = self.gpus
        is_gpu_stage = pergpu or gpu_id is not None

        if pergpu:
            if gpu_count > 0:
                actual_workers = workers * gpu_count
                print(f"Per-GPU mode: {workers} workers per GPU ({actual_workers} total for {gpu_count} GPUs)")
            else:
                actual_workers = workers
                pergpu = False
                is_gpu_stage = False
                print(f"No GPUs available, falling back to {workers} CPU workers")
        else:
            actual_workers = workers

        # Determine max_workers, capping at GPU count for GPU stages
        if max_workers is None:
            default_max = actual_workers * 4
        else:
            default_max = max_workers

        if is_gpu_stage and gpu_count > 0:
            effective_max = min(default_max, gpu_count)
            if effective_max < default_max:
                print(f"  {stage_name}: max_workers capped at {effective_max} (GPU count)")
        else:
            effective_max = default_max

        # Apply global autoscale if enabled (per-stage explicit setting takes precedence)
        # GPU stages never autoscale - they're limited by GPU count and sharing causes contention
        if is_gpu_stage:
            stage_autoscale = False
        elif autoscale is not None:
            stage_autoscale = autoscale  # Per-stage explicit setting
        else:
            stage_autoscale = self.autoscale  # Fall back to global setting

        # Use global max_workers_per_stage if no max_workers specified and global autoscale is on
        if max_workers is None and self.autoscale and not is_gpu_stage:
            effective_max = self.max_workers_per_stage

        self.jobs.append(
            {
                "func": func,
                "name": stage_name,
                "outqn": outqn,
                "num_workers": actual_workers,
                "pergpu": pergpu,
                "thread": thread,
                "gpu_id": gpu_id,
                "is_gpu_stage": is_gpu_stage,
                "autoscale": stage_autoscale,
                "max_workers": effective_max,
                "min_workers": min_workers if min_workers is not None else 1,
            }
        )

    def start(self):
        """Start the pipeline with configured workers"""
        print(f"Starting pipeline with {len(self.jobs)} jobs")
        self.started = True

        if not self.jobs:
            raise ValueError("No workers added to pipeline")

        # In debug mode, skip multiprocessing setup
        if self.debug:
            print("Debug mode - skipping multiprocessing setup")
            return self

        print("Setting up multiprocessing...")

        # First pass: create all queues, end counters, worker counts, and Events
        for i, job in enumerate(self.jobs):
            # Get configured queue sizes (0 = infinite)
            outq_size = job.get("outqn") or 0
            self.queues.append(Queue(maxsize=outq_size))

            # Create shared end counter for this stage
            self.stage_end_counters.append(Value("i", 0))

            # Create Event for signaling stage completion
            self.stage_done_events.append(Event())

            # Store total worker count for this stage as a shared Value
            # For threaded jobs without pergpu, there's only 1 process
            # For all other cases, num_workers processes are spawned
            if job.get("thread", False) and not job.get("pergpu", False):
                worker_count = 1
            else:
                worker_count = job["num_workers"]
            self.stage_worker_counts.append(Value("i", worker_count))

        # Second pass: spawn workers (now we have all Events created)
        for i, job in enumerate(self.jobs):
            in_queue = self.queues[i - 1] if i > 0 else None
            out_queue = self.queues[i] if i < len(self.queues) else None
            is_final_stage = i == len(self.jobs) - 1

            # Event-based completion signaling
            upstream_done = self.stage_done_events[i - 1] if i > 0 else None
            stage_done = self.stage_done_events[i]

            # Get stage name from function
            func = job["func"]
            if hasattr(func, "__name__"):
                stage_name = func.__name__
            elif hasattr(func, "__class__"):
                stage_name = func.__class__.__name__
            else:
                stage_name = str(type(func).__name__)

            # Check if this job uses threading
            if job.get("thread", False):
                if job.get("pergpu", False):
                    # For threading + pergpu: spawn one process per worker, each with one thread
                    for worker_idx in range(job["num_workers"]):
                        num_threads = 1  # One thread per process when using pergpu

                        # Determine GPU ID
                        gpu_id = worker_idx % self.gpus  # Cycle through available GPUs

                        # Create unique worker ID for timing
                        worker_id = (
                            f"stage_{i}_threaded_worker_{worker_idx}_gpu_{gpu_id}"
                        )

                        args = (
                            job["func"],
                            in_queue,
                            out_queue,
                            self.should_stop,
                            self.working,
                            gpu_id,
                            self.timing_dict,  # timing_dict
                            worker_id,  # worker_id
                            self.share_tensors,
                            self.raise_errors,
                            num_threads,  # num_threads
                            self.stage_end_counters[i],  # stage_end_counter
                            self.stage_worker_counts[i],  # stage_worker_count
                            i,  # stage_idx
                            stage_name,  # stage_name
                            is_final_stage,  # is_final_stage
                            self.expected_consumers,  # expected_consumers
                            upstream_done,  # upstream_done Event
                            stage_done,  # stage_done Event
                            self.debug,  # debug
                        )

                        # Store configuration for restart
                        self.worker_configs[worker_id] = {
                            "target": _threaded_worker_run,
                            "args": args,
                            "stage_name": stage_name,
                        }

                        p = mp.Process(target=_threaded_worker_run, args=args)
                        p.start()
                        self.processes.append(p)
                        self.worker_info.append((p, worker_id, stage_name))
                else:
                    # For threading without pergpu: one process with multiple threads
                    num_threads = job["num_workers"]
                    worker_id = f"stage_{i}_threaded_{num_threads}threads"

                    args = (
                        job["func"],
                        in_queue,
                        out_queue,
                        self.should_stop,
                        self.working,
                        None,  # gpu_id
                        self.timing_dict,  # timing_dict
                        worker_id,  # worker_id
                        self.share_tensors,
                        self.raise_errors,
                        num_threads,  # num_threads
                        self.stage_end_counters[i],  # stage_end_counter
                        self.stage_worker_counts[i],  # stage_worker_count
                        i,  # stage_idx
                        stage_name,  # stage_name
                        is_final_stage,  # is_final_stage
                        self.expected_consumers,  # expected_consumers
                        upstream_done,  # upstream_done Event
                        stage_done,  # stage_done Event
                        self.debug,  # debug
                    )

                    # Store configuration for restart
                    self.worker_configs[worker_id] = {
                        "target": _threaded_worker_run,
                        "args": args,
                        "stage_name": stage_name,
                    }

                    p = mp.Process(target=_threaded_worker_run, args=args)
                    p.start()
                    self.processes.append(p)
                    self.worker_info.append((p, worker_id, stage_name))
            else:
                # Create workers as separate processes (original behavior)
                for worker_idx in range(job["num_workers"]):
                    # Determine GPU ID if pergpu is enabled
                    gpu_id = None
                    if job.get("pergpu", False):
                        gpu_id = worker_idx % self.gpus  # Cycle through available GPUs

                    # Create unique worker ID for timing
                    worker_id = f"stage_{i}_worker_{worker_idx}"
                    if gpu_id is not None:
                        worker_id += f"_gpu_{gpu_id}"

                    print(f"Starting worker process: {stage_name} ({worker_id})")

                    args = (
                        job["func"],
                        in_queue,
                        out_queue,
                        self.should_stop,
                        self.working,
                        gpu_id,
                        self.timing_dict,  # timing_dict
                        worker_id,  # worker_id
                        self.share_tensors,
                        self.raise_errors,
                        self.stage_end_counters[i],  # stage_end_counter
                        self.stage_worker_counts[i],  # stage_worker_count
                        i,  # stage_idx
                        stage_name,  # stage_name
                        is_final_stage,  # is_final_stage
                        self.expected_consumers,  # expected_consumers
                        upstream_done,  # upstream_done Event
                        stage_done,  # stage_done Event
                        self.debug,  # debug
                    )

                    # Store configuration for restart
                    self.worker_configs[worker_id] = {
                        "target": _worker_run,
                        "args": args,
                        "stage_name": stage_name,
                    }

                    p = mp.Process(target=_worker_run, args=args)
                    p.start()
                    self.processes.append(p)
                    self.worker_info.append((p, worker_id, stage_name))
                    print(
                        f"Worker process {stage_name} ({worker_id}) started with PID {p.pid}"
                    )

        # Start health monitor if monitoring is enabled
        if self.health_check_interval > 0:
            print(
                f"Starting health monitor (check interval: {self.health_check_interval}s)"
            )

            self.health_monitor_stop_event.clear()
            self.health_monitor_thread = threading.Thread(
                target=_health_monitor_thread,
                args=(
                    self,  # Pass pipe instance
                    self.should_stop,
                    self.health_check_interval,
                    self.health_monitor_stop_event,
                ),
                daemon=True,
            )
            self.health_monitor_thread.start()
            print("Health monitor thread started")

        # Start stats monitor thread if enabled
        if self.stats_interval > 0:
            self.stats_monitor_thread = threading.Thread(
                target=_stats_monitor_thread,
                args=(self, self.stats_monitor_stop_event, self.stats_interval),
                daemon=True,
            )
            self.stats_monitor_thread.start()

        # Start autoscaler thread if any stage has autoscale enabled
        has_autoscale = any(job.get("autoscale") for job in self.jobs)
        if has_autoscale:
            print("Starting autoscaler (monitoring queue pressure)")
            self.autoscaler_stop_event.clear()
            self.autoscaler_thread = threading.Thread(
                target=_autoscaler_thread,
                args=(self, self.should_stop, self.autoscaler_stop_event),
                daemon=True,
            )
            self.autoscaler_thread.start()

        return self

    def _restart_worker(self, worker_idx, worker_id):
        """Restart a single crashed worker

        Args:
            worker_idx: Index in worker_info list
            worker_id: Worker ID string
        """
        if worker_id not in self.worker_configs:
            raise ValueError(f"No configuration found for worker {worker_id}")

        config = self.worker_configs[worker_id]

        # Clean up old process
        old_proc, _, _ = self.worker_info[worker_idx]
        with contextlib.suppress(Exception):
            old_proc.join(timeout=1)

        # Start new process with same configuration
        p = mp.Process(
            target=config["target"],
            args=config["args"],
        )
        p.start()

        # Update worker_info and processes list
        self.worker_info[worker_idx] = (p, worker_id, config["stage_name"])
        # Also update the processes list if needed
        for i, proc in enumerate(self.processes):
            if proc == old_proc:
                self.processes[i] = p
                break

        print(f"Worker {worker_id} restarted with PID {p.pid}")

    def _restart_stage(self, stage_idx):
        """Restart a single stage with fresh queues

        This recreates the input and output queues for the stage and restarts
        all workers in that stage. Adjacent stages are NOT restarted - they
        continue using their existing queues.

        Args:
            stage_idx: Index of the stage to restart
        """
        print(f"   Restarting stage {stage_idx}...")

        # Find all workers in this stage
        stage_workers = []
        for idx, (proc, worker_id, stage_name) in enumerate(self.worker_info):
            if worker_id.startswith(f"stage_{stage_idx}_"):
                stage_workers.append((idx, proc, worker_id, stage_name))

        if not stage_workers:
            raise ValueError(f"No workers found for stage {stage_idx}")

        # Stop all workers in this stage
        for idx, proc, worker_id, stage_name in stage_workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)

        # Get the job config for this stage
        job = self.jobs[stage_idx]

        # Recreate the output queue for this stage (with same maxsize)
        old_out_queue = self.queues[stage_idx]
        old_out_maxsize = (
            old_out_queue._maxsize if hasattr(old_out_queue, "_maxsize") else 0
        )

        # Drain any items from the old queue before closing
        drained_items = []
        try:
            while True:
                item = old_out_queue.get_nowait()
                if item != "end":
                    drained_items.append(item)
        except Empty:
            pass

        # Close old queue and create new one
        try:
            old_out_queue.cancel_join_thread()
            old_out_queue.close()
        except Exception:
            pass

        new_out_queue = Queue(maxsize=old_out_maxsize)
        self.queues[stage_idx] = new_out_queue

        # Put drained items back into new queue
        for item in drained_items:
            try:
                new_out_queue.put_nowait(item)
            except Full:
                break

        print(
            f"   Recreated output queue for stage {stage_idx} (recovered {len(drained_items)} items)"
        )

        # Reset the end counter for this stage
        self.stage_end_counters[stage_idx].value = 0

        # Get queue references
        in_queue = self.queues[stage_idx - 1] if stage_idx > 0 else None
        out_queue = new_out_queue

        # Get stage name
        func = job["func"]
        if hasattr(func, "__name__"):
            stage_name = func.__name__
        elif hasattr(func, "__class__"):
            stage_name = func.__class__.__name__
        else:
            stage_name = str(type(func).__name__)

        # Restart all workers in this stage with new queue
        is_final_stage = stage_idx == len(self.jobs) - 1
        next_stage_worker_count = None if is_final_stage else self.stage_worker_counts[stage_idx + 1]

        for idx, old_proc, worker_id, _ in stage_workers:
            # Determine if this is a threaded or process worker
            config = self.worker_configs.get(worker_id)
            if not config:
                print(f"   Warning: No config for {worker_id}, skipping")
                continue

            # Build new args with updated queue reference
            old_args = config["args"]

            if config["target"] == _threaded_worker_run:
                # Threaded worker args order:
                # (func, in_queue, out_queue, should_stop, working, gpu_id, timing_dict,
                #  worker_id, share_tensors, raise_errors, num_threads, stage_end_counter,
                #  stage_worker_count, stage_idx, stage_name, is_final_stage, expected_consumers,
                #  next_stage_worker_count)
                new_args = (
                    old_args[0],  # func
                    in_queue,  # in_queue (possibly same as before)
                    out_queue,  # out_queue (new)
                    self.should_stop,
                    self.working,
                    old_args[5],  # gpu_id
                    self.timing_dict,
                    worker_id,
                    self.share_tensors,
                    self.raise_errors,
                    old_args[10],  # num_threads
                    self.stage_end_counters[stage_idx],
                    self.stage_worker_counts[stage_idx],
                    stage_idx,
                    stage_name,
                    is_final_stage,
                    self.expected_consumers,
                    next_stage_worker_count,
                    self.debug,
                )
            else:
                # Process worker args order:
                # (func, in_queue, out_queue, should_stop, working, gpu_id, timing_dict,
                #  worker_id, share_tensors, raise_errors, stage_end_counter,
                #  stage_worker_count, stage_idx, stage_name, is_final_stage, expected_consumers,
                #  next_stage_worker_count)
                new_args = (
                    old_args[0],  # func
                    in_queue,
                    out_queue,
                    self.should_stop,
                    self.working,
                    old_args[5],  # gpu_id
                    self.timing_dict,
                    worker_id,
                    self.share_tensors,
                    self.raise_errors,
                    self.stage_end_counters[stage_idx],
                    self.stage_worker_counts[stage_idx],
                    stage_idx,
                    stage_name,
                    is_final_stage,
                    self.expected_consumers,
                    next_stage_worker_count,
                    self.debug,
                )

            # Update stored config
            config["args"] = new_args

            # Start new process
            p = mp.Process(target=config["target"], args=new_args)
            p.start()

            # Update tracking
            self.worker_info[idx] = (p, worker_id, stage_name)
            for i, proc in enumerate(self.processes):
                if proc == old_proc:
                    self.processes[i] = p
                    break

            print(f"   Restarted {worker_id} with PID {p.pid}")

        print(f"   Stage {stage_idx} restart complete")

    def restart(self, reason="ConnectionError"):
        """Restart the entire pipeline"""
        self._stop(force=True)

        # Recreate manager if timing is enabled
        if self.stats_interval > 0:
            import multiprocessing

            self.manager = multiprocessing.Manager()
            self.timing_dict = self.manager.dict()

        self.start()
        print(f"Pipeline restarted due to {reason}")

    def _stop(self, force=False):
        """Stop all worker processes cleanly or forcefully"""
        # Set the stop flag
        self.should_stop.value = 1  # Atomic write for simple int

        # Stop health monitor thread first
        if (
            self.health_monitor_thread is not None
            and self.health_monitor_thread.is_alive()
        ):
            print("Stopping health monitor...")
            try:
                self.health_monitor_stop_event.set()
                self.health_monitor_thread.join(timeout=2)
            except Exception as e:
                print(f"Error stopping health monitor: {e}")
            self.health_monitor_thread = None

        # Stop stats monitor thread
        if (
            self.stats_monitor_thread is not None
            and self.stats_monitor_thread.is_alive()
        ):
            self.stats_monitor_stop_event.set()
            self.stats_monitor_thread.join(timeout=2)
            self.stats_monitor_thread = None

        if force:
            print("Force stopping all processes...")

            # Aggressive shutdown - terminate immediately
            for p in self.processes:
                if p.is_alive():
                    p.terminate()

            # Give a brief moment for termination
            time.sleep(0.1)

            # Kill any remaining processes
            for p in self.processes:
                if p.is_alive():
                    p.kill()

            # Clean up shared CUDA tensor references
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
        else:
            # Signal all processes to stop
            for q in self.queues:
                with contextlib.suppress(Full):
                    q.put("end", timeout=1)

            # Wait for processes to finish
            for p in self.processes:
                try:
                    p.join(timeout=2)  # Give processes more time to clean up
                    if p.is_alive():
                        p.terminate()
                except KeyboardInterrupt:
                    if p.is_alive():
                        p.terminate()
                except Exception as e:
                    print(f"Error stopping process: {e}")

        # Join any remaining processes after termination
        for p in self.processes:
            try:
                if p.is_alive():
                    p.join(timeout=1)
            except Exception:
                pass

        # Shut down manager BEFORE closing queues to avoid semaphore leaks
        if self.manager is not None:
            try:
                self.manager.shutdown()
                self.manager = None
            except Exception as e:
                print(f"Error shutting down manager: {e}")

        # Clear all queues - cancel background threads first
        for q in self.queues:
            try:
                q.cancel_join_thread()  # Don't wait for queue to flush
                q.close()
            except Exception as e:
                print(f"Error closing queue: {e}")

        # Reset state
        self.processes = []
        self.queues = []
        self.worker_info = []
        self.worker_configs = {}
        self.working.value = 0
        self.should_stop.value = 0
        self.restart_needed.value = 0

        # Reset end counters and clear Events for next run
        for counter in self.stage_end_counters:
            counter.value = 0
        for event in self.stage_done_events:
            event.clear()

    def __iter__(self):
        """Iterate over results from the last worker using Event-based completion."""
        if not self.started:
            self.start()

        # Debug mode: run sequentially without processes/queues
        if self.debug:
            yield from self._debug_sequential_run()
            return

        final_done = self.stage_done_events[-1]
        consecutive_empty = 0
        EMPTY_THRESHOLD = 5

        while True:
            # Check if a worker signaled restart needed
            if self.should_stop.value == 2:
                print("\nWorker encountered connection error - restarting pipeline...")
                self.restart(reason="Worker connection error")
                consecutive_empty = 0
                continue

            try:
                item = self.queues[-1].get(timeout=0.1)
                consecutive_empty = 0

                # Skip any legacy "end" sentinels
                if item == "end":
                    continue

                yield _deserialize_tensors_recursive(item)
            except Empty:
                consecutive_empty += 1
                # Check if final stage is done AND queue is drained
                if final_done.is_set() and consecutive_empty >= EMPTY_THRESHOLD:
                    print("Iterator: pipeline complete")
                    self._stop()
                    return
            except (ConnectionError, FileNotFoundError):
                self.restart()
                consecutive_empty = 0
            except KeyboardInterrupt:
                print("Ctrl+C detected - force stopping pipeline")
                self._stop(force=True)
                return

    def __del__(self):
        """Cleanup when object is garbage collected"""
        try:
            if self.started and (self.processes or self.queues):
                self._stop(force=True)
        except Exception:
            pass  # Ignore errors during cleanup

    def _debug_sequential_run(self):
        """Run the pipeline sequentially in debug mode - no processes, no queues"""
        print("Running in debug mode - sequential execution")

        if not self.jobs:
            return

        # Initialize all workers
        workers = []
        for job in self.jobs:
            worker_func = job["func"]
            # If it's already an instance, use it directly
            if callable(worker_func) and not hasattr(worker_func, "__name__"):
                workers.append(worker_func)
            else:
                workers.append(worker_func)

        # Call load() on workers that have it (same as process mode)
        for worker in workers:
            if hasattr(worker, "load"):
                worker.load()

        # Keep a queue of items to process through the pipeline
        pending_items = []
        root_ended = False

        # Keep running until root ends and there are no items left in the pipeline
        while not root_ended or pending_items:
            # Get more items from root if it hasn't ended
            if not root_ended:
                root_items = workers[0]()
                if not isinstance(root_items, list):
                    root_items = [root_items]

                for root_item in root_items:
                    if root_item == "end":
                        root_ended = True
                        break
                    pending_items.append((1, root_item))  # Start at stage 1 (skip root)

            # Process pending items
            if pending_items:
                stage_idx, item = pending_items.pop(0)

                # If this is beyond the final stage, yield the result
                if stage_idx >= len(workers):
                    yield item
                    continue

                # Process through the current worker
                worker = workers[stage_idx]

                try:
                    result = worker(item)
                except Exception as e:
                    if self.raise_errors:
                        print(f"Error in worker at stage {stage_idx}: {e}")
                        raise e
                    else:
                        print(
                            f"Error in worker at stage {stage_idx}: {e}, continuing..."
                        )
                        continue

                # Handle different result types
                if result is None:
                    # Worker needs more items (like a batcher), don't iterate to next function
                    continue
                elif result == "end":
                    # End signal, stop processing
                    break
                elif isinstance(result, list):
                    # Worker returned multiple items
                    for res_item in result:
                        if res_item != "end":
                            pending_items.append((stage_idx + 1, res_item))
                else:
                    # Single item result
                    pending_items.append((stage_idx + 1, result))


class Batcher:
    def __init__(self, size, collate_fn=None):
        self.size = size
        self.collate = collate_fn
        self.current = size * [None]
        self.idx = 0

    def __call__(self, item):
        self.current[self.idx] = item
        self.idx += 1

        if self.idx == self.size:
            out = self.current
            self.idx = 0
            if self.collate is not None:
                out = self.collate(out)
            return out
        return None


class BufferAndShuffle:
    def __init__(self, size, batch_size):
        self.size = size
        self.buffer = []
        self.first = True
        self.batch_size = batch_size * 4

    def __call__(self, it):
        self.buffer.append(it)
        size = self.batch_size if self.first else self.size

        if len(self.buffer) > size:
            if self.first:
                self.first = False
                print("Buffered First")
            random.shuffle(self.buffer)
            out = self.buffer
            self.buffer = []
            return out
        return None


def download():
    """This is the root function"""
    print("root")
    return [*list(range(100)), "end"]


def work(item):
    time.sleep(0.6)
    return [item, item]


def upload(item):
    time.sleep(0.2)
    # print(f"Uploaded {item}")
    return item


def slow_worker(item):
    """Really slow worker for testing timing reports"""
    time.sleep(2.0)  # 2 second delay per item
    print(f"Slowly processed {item}")
    return item * 10


class StateFullWork:
    def __init__(self):
        self.create_some_state = []
        time.sleep(1)
        print("Initialized stateful worker")

    def __call__(self, item):
        time.sleep(random.random() * 0.5)
        # Check if we're on a GPU
        if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(
                f"Processing {item} on GPU {torch.cuda.current_device()}: {device_name}"
            )
        else:
            print(f"Processing {item} on CPU")

        if item % 2 == 0:
            return [item, item]
        return item


class RetrieveSQL:
    def __init__(self, query, chunk_size=20, skip_count=False) -> None:
        self.offset = 0
        self.chunk_size = chunk_size
        self.base_query = query
        self.query = query
        self.skip_count = skip_count
        self.count = 0 if skip_count else self.get_total_count()

    def __call__(self):
        from data.db import query

        # For queries with data-modifying CTEs, execute directly without offset/limit
        if self.skip_count:
            items = query(self.query)
            # End when no items are returned (no more work to claim)
            if len(items) == 0:
                return "end"
            return items

        # Check if query already has ORDER BY clause
        if "ORDER BY" in self.query.upper():
            # Query already has ordering, just add LIMIT and OFFSET
            items = query(self.query + f" LIMIT {self.chunk_size} OFFSET {self.offset}")
        else:
            # No ordering specified, use random order
            items = query(
                self.query
                + f" ORDER BY RANDOM() LIMIT {self.chunk_size} OFFSET {self.offset}"
            )

        self.offset += len(items)

        # Check if we should end - either no items returned or reached expected count
        if len(items) == 0 or self.offset >= self.count:
            # Recheck count in case database was modified during processing
            updated_count = self.get_total_count()
            if updated_count > self.count:
                # Count increased (new items added), reset offset and continue
                self.count = updated_count
                self.offset = 0
                return items if len(items) > 0 else []
            # Count unchanged or decreased (items processed) - we've reached the end
            return "end"

        return items

    def get_total_count(self):
        from data.db import query

        count_query = f"SELECT COUNT(*) as count FROM ({self.base_query}) as subquery"
        result = query(count_query)
        return result[0]["count"] if result else 0


class SQLConnection:
    """Reusable database connection class for pipeline workers"""

    def __init__(self, query=None):
        self.connection = None
        self.cursor = None
        self.query = query

    def _get_connection(self):
        import psycopg
        from data.constants import CONNECTION_STRING
        from psycopg.rows import dict_row

        connection_string = CONNECTION_STRING
        if "sslmode" not in connection_string:
            connection_string += "?sslmode=require"
        connection_string = connection_string.replace("+asyncpg", "")

        return psycopg.connect(
            connection_string, row_factory=dict_row, connect_timeout=10
        )

    def _ensure_connection(self):
        try:
            if self.connection is None or self.connection.closed:
                if self.cursor:
                    self.cursor.close()
                    self.cursor = None
                self.connection = self._get_connection()
                self.cursor = self.connection.cursor()
            elif self.cursor is None:
                self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Database connection error: {e}")
            self.connection = None
            self.cursor = None
            raise

    def execute(self, *params, query=None):
        """Execute predefined query or custom query with parameters and commit"""
        self._ensure_connection()
        try:
            query_to_run = query or self.query
            if not query_to_run:
                raise ValueError("No query provided and no predefined query set")

            self.cursor.execute(query_to_run, params)
            self.connection.commit()
        except Exception as e:
            print(f"Database update error: {e}")
            self.connection = None
            self.cursor = None
            raise

    def close(self):
        """Close the connection and cursor"""
        if self.cursor:
            self.cursor.close()
        if self.connection and not self.connection.closed:
            self.connection.close()

    def __del__(self):
        self.close()


class RTF:
    def __init__(self) -> None:
        self.start = None
        self.total = 0

    def load(self):
        if self.start is None:
            self.start = time.perf_counter()

    def __call__(self, item):
        if self.start is None:
            self.start = time.perf_counter()
        self.total += item["duration"]
        rtf = self.total / (time.perf_counter() - self.start)
        print(f"{rtf:.2f}x")
        return item


def io_worker(item):
    """Simulate IO-heavy work that benefits from threading"""
    time.sleep(0.1)  # Simulate IO delay
    print(f"Thread processed {item}")
    return item * 2


def long_download():
    """Root function that produces many items"""
    print("Starting long download...")
    items = []
    for i in range(40):
        if i % 20 == 0:
            print(f"Downloaded batch {i}")
        time.sleep(0.01)  # Small delay to make it interruptible
        items.append(i)
    items.append("end")
    return items


def debug_io_worker(item):
    """IO worker that shows thread activity"""
    thread_id = threading.current_thread().ident
    print(f"Thread {thread_id} processing {item}")
    time.sleep(0.2)  # Longer delay to make cancellation more likely
    return item * 2


def test_download():
    """Test root function that produces a known number of items"""
    print("Starting test download...")
    items = list(range(20))  # 20 items total
    items.append("end")
    return items


def test_worker(item):
    """Test worker that doubles items"""
    time.sleep(0.1)  # Small delay to make it realistic
    return item * 2


def test_final_worker(item):
    """Final worker that adds 1000"""
    time.sleep(0.05)
    return item + 1000


if __name__ == "__main__":
    freeze_support()
    print("Testing end signal handling and item preservation...")
    print("Expected: 20 items (0*2+1000=1000, 1*2+1000=1002, ..., 19*2+1000=1038)")

    # Test with multiple workers per stage to verify end signal coordination
    pipe = Pipe(debug=False, stats_interval=5)
    pipe.add(test_download, outqn=10)
    pipe.add(test_worker, workers=1, outqn=10)  # 1 worker first
    pipe.add(test_final_worker, workers=1, outqn=0)  # 1 worker - simplify

    results = []
    try:
        for count, result in enumerate(pipe):
            print(f"Result {count}: {result}")
            results.append(result)
    except KeyboardInterrupt:
        print("\n=== KEYBOARD INTERRUPT DETECTED ===")
        print("Force stopping pipeline...")
        pipe._stop(force=True)
        print("Pipeline force stop completed")

    print("\nTest Results:")
    print(f"Items received: {len(results)}")
    print("Expected: 20 items")

    if len(results) == 20:
        expected_results = [i * 2 + 1000 for i in range(20)]
        results_sorted = sorted(results)
        expected_sorted = sorted(expected_results)

        if results_sorted == expected_sorted:
            print("SUCCESS: All items processed correctly, no items lost!")
        else:
            print("FAILED: Items were modified incorrectly")
            print(f"Expected: {expected_sorted}")
            print(f"Got: {results_sorted}")
    else:
        print(f"FAILED: Wrong number of items. Expected 20, got {len(results)}")
        if len(results) < 20:
            print("Items were lost during processing!")
        else:
            print("Extra items were generated!")
