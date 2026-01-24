import contextlib
import os
import pickle
import resource
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Full

import torch

from ._shm import _item_from_shm, _item_to_shm


def _check_picklable(obj, name: str, is_thread: bool = False) -> None:
    if is_thread:
        return

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        problematic = []
        if hasattr(obj, "__dict__"):
            for attr_name, attr_val in obj.__dict__.items():
                try:
                    pickle.dumps(attr_val)
                except Exception:
                    problematic.append(f"  - self.{attr_name}: {type(attr_val).__name__}")

        msg = f"\nWARNING: Worker '{name}' may not be picklable for multiprocessing.\n"
        msg += f"   Error: {e}\n"
        if problematic:
            msg += "   Problematic attributes:\n"
            msg += "\n".join(problematic) + "\n"
        msg += "   Consider initializing these in load() instead of __init__.\n"
        print(msg)
    except Exception as e:
        print(f"\nWARNING: Worker '{name}' pickle check failed: {e}\n")


def _get_fd_info():
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
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= hard:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
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
    """Worker process using Event-based completion signaling."""
    worker_desc = f"{stage_name} ({worker_id})" if stage_name else worker_id
    is_root = (in_queue is None)

    # Limit CPU threads to avoid oversubscription across worker processes
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"

    _increase_fd_limit(worker_desc)

    if gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
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
    EMPTY_THRESHOLD = 10

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
                        serialized = _item_to_shm(item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue

                if has_end:
                    break

            else:
                from queue import Empty

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
                            break
                    continue
                except (OSError, EOFError, BrokenPipeError):
                    break

                if item == "worker_stop":
                    print(f"Worker {worker_id} received stop signal, exiting gracefully")
                    break

                if item == "end":
                    continue

                item = _item_from_shm(item)

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
                        serialized = _item_to_shm(out_item)
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

    # Flush buffered items
    if not is_root and out_queue:
        try:
            result = worker("end")
            if result is not None:
                if not isinstance(result, list):
                    result = [result]
                valid_items = [r for r in result if r is not None and r != "end"]
                if valid_items:
                    print(f"Worker {worker_desc} flushing {len(valid_items)} items")
                    for item in valid_items:
                        serialized = _item_to_shm(item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue
        except Exception as e:
            print(f"Worker {worker_desc} flush error: {e}")

    if hasattr(worker, "flush") and out_queue:
        try:
            flushed_items = list(worker.flush())
            if flushed_items:
                print(f"Worker {worker_desc} flushing {len(flushed_items)} items via flush()")
                for item in flushed_items:
                    if item is not None:
                        serialized = _item_to_shm(item)
                        while not should_stop.value:
                            try:
                                out_queue.put(serialized, timeout=0.1)
                                break
                            except Full:
                                continue
        except Exception as e:
            print(f"Worker {worker_desc} flush() error: {e}")

    # Coordinate with siblings
    if stage_end_counter is not None and stage_worker_count is not None:
        with stage_end_counter.get_lock():
            stage_end_counter.value += 1
            finished_workers = stage_end_counter.value

        current_worker_count = stage_worker_count.value
        stage_desc = f"{stage_name}" if stage_name else f"stage_{stage_idx}"
        print(f"Worker {worker_id} finished ({finished_workers}/{current_worker_count} at {stage_desc})")

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

    # Limit CPU threads to avoid oversubscription across worker processes
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"

    _increase_fd_limit(worker_desc)

    if gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
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
        from queue import Empty

        local_items = 0
        local_time = 0.0
        local_audio = 0.0
        consecutive_empty = 0
        EMPTY_THRESHOLD = 10

        while not should_stop.value and not thread_stop.is_set():
            try:
                if is_root:
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
                            serialized = _item_to_shm(item)
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

                    item = _item_from_shm(item)

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
                            serialized = _item_to_shm(out_item)
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
                    serialized = _item_to_shm(item)
                    out_queue.put(serialized, timeout=1)
            except Exception as e:
                print(f"Threaded worker {worker_desc} flush error: {e}")

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


def _signal_worker_to_stop(pipe_instance, stage_idx):
    """Signal one worker at a stage to stop after completing current item."""
    in_queue = pipe_instance.queues[stage_idx - 1] if stage_idx > 0 else None
    if in_queue:
        try:
            in_queue.put("worker_stop", timeout=0.1)
            print(f"   Signaled worker at stage {stage_idx} to stop")
        except Exception as e:
            print(f"   Failed to signal worker stop: {e}")


def _spawn_additional_worker(pipe_instance, stage_idx, job):
    """Spawn an additional worker for the given stage."""
    from torch.multiprocessing import Process

    func = job["func"]
    is_threaded = job.get("thread", False)

    if hasattr(func, "__name__"):
        stage_name = func.__name__
    elif hasattr(func, "__class__"):
        stage_name = func.__class__.__name__
    else:
        stage_name = str(type(func).__name__)

    current_count = pipe_instance.stage_worker_counts[stage_idx].value
    new_worker_idx = current_count
    worker_id = f"stage_{stage_idx}_worker_{new_worker_idx}"

    print(f"Autoscaling: Adding worker {new_worker_idx + 1} to {stage_name} (stage {stage_idx})")

    in_queue = pipe_instance.queues[stage_idx - 1] if stage_idx > 0 else None
    out_queue = pipe_instance.queues[stage_idx] if stage_idx < len(pipe_instance.queues) else None

    is_final_stage = stage_idx == len(pipe_instance.jobs) - 1

    upstream_done = pipe_instance.stage_done_events[stage_idx - 1] if stage_idx > 0 else None
    stage_done = pipe_instance.stage_done_events[stage_idx]

    pipe_instance.stage_worker_counts[stage_idx].value += 1

    if is_threaded:
        args = (
            func,
            in_queue,
            out_queue,
            pipe_instance.should_stop,
            pipe_instance.working,
            None,
            pipe_instance.timing_dict,
            worker_id,
            pipe_instance.share_tensors,
            pipe_instance.raise_errors,
            job.get("num_workers", 4),
            pipe_instance.stage_end_counters[stage_idx],
            pipe_instance.stage_worker_counts[stage_idx],
            stage_idx,
            stage_name,
            is_final_stage,
            pipe_instance.expected_consumers,
            upstream_done,
            stage_done,
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
            None,
            pipe_instance.timing_dict,
            worker_id,
            pipe_instance.share_tensors,
            pipe_instance.raise_errors,
            pipe_instance.stage_end_counters[stage_idx],
            pipe_instance.stage_worker_counts[stage_idx],
            stage_idx,
            stage_name,
            is_final_stage,
            pipe_instance.expected_consumers,
            upstream_done,
            stage_done,
            pipe_instance.debug,
        )
        proc = Process(target=_worker_run, args=args, daemon=True)

    proc.start()
    print(f"   Worker {worker_id} started with PID {proc.pid}")

    pipe_instance.processes.append(proc)
    pipe_instance.worker_info.append((proc, worker_id, stage_name))

    pipe_instance.worker_configs[worker_id] = {
        "target": _threaded_worker_run if is_threaded else _worker_run,
        "args": args,
        "stage_name": stage_name,
    }
