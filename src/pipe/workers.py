import contextlib
import inspect
import os
import pickle
import resource
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full

import torch

from .types import End, WorkerStop
from .queues import _InputChannel, _OutputChannel
from .shm import _item_from_shm, _item_to_shm


def _log(msg):
    if os.environ.get("PIPE_VERBOSE") == "1":
        print(msg)


def _cpu_chunk(cpu_pool, worker_idx, num_workers):
    """Split a stage's `cpus=` pool into per-worker contiguous slices.

    Worker `worker_idx` of `num_workers` gets its slice (sizes differ by <=1), so an
    N-core mel stage over M workers ends up with each worker owning ~N/M dedicated
    cores rather than all M workers fighting over the whole pool. If a stage has more
    workers than cores, workers round-robin a single (oversubscribed) core. Returns
    None when the stage has no cpus= pool."""
    if not cpu_pool:
        return None
    n = len(cpu_pool)
    if num_workers >= n:
        return [cpu_pool[worker_idx % n]]
    base, rem = divmod(n, num_workers)
    start = worker_idx * base + min(worker_idx, rem)
    size = base + (1 if worker_idx < rem else 0)
    return list(cpu_pool[start:start + size])


def _setup_cpu(cpu_affinity, cpu_threads, worker_desc):
    """Pin this worker to a CPU core set (if any) and size its BLAS/torch thread count.

    `cpu_affinity` is the list of core ids this worker may run on (its slice of the
    stage's `cpus=` pool, from `_cpu_chunk`), or None for no pinning. Thread count is
    resolved by precedence: an explicit `cpu_threads` wins, else the affinity slice
    size, else the default 2 that avoids oversubscription across unpinned workers.
    Pinning is best-effort and Linux-only (os.sched_setaffinity); the thread count is
    always applied. `cpu_threads` needs no pinning — it just lifts the flat 2-cap for a
    CPU-heavy stage (e.g. mel)."""
    if cpu_affinity:
        setaff = getattr(os, "sched_setaffinity", None)
        if setaff is not None:
            try:
                setaff(0, set(cpu_affinity))
                _log(f"Worker {worker_desc} pinned to CPUs {sorted(cpu_affinity)}")
            except OSError as e:
                print(f"Worker {worker_desc}: could not pin to CPUs {cpu_affinity}: {e}")
        else:
            print(f"Worker {worker_desc}: CPU affinity unsupported on this platform; ignoring cpus=")

    if cpu_threads:
        n = cpu_threads
    elif cpu_affinity:
        n = len(cpu_affinity)
    else:
        n = 2
    torch.set_num_threads(n)
    s = str(n)
    os.environ["OMP_NUM_THREADS"] = s
    os.environ["MKL_NUM_THREADS"] = s
    os.environ["OPENBLAS_NUM_THREADS"] = s


def _skip_shm_for_output(is_final_stage):
    """Check if shm should be skipped for output queue."""
    return bool(is_final_stage and os.environ.get("PIPE_NO_SHM_OUTPUT"))


def _make_push(all_queues, stage_names, should_stop):
    """Build the worker.push(stage, item) primitive.

    Drops `item` onto the INPUT queue of the target stage (= the OUTPUT queue of
    the stage before it), so a downstream worker can send an item BACK to an
    earlier stage for reprocessing (e.g. re-render on a failed WER check). `stage`
    may be an int stage index, a stage name (str), or the worker CLASS / a worker
    INSTANCE (resolved by its `__name__`). Names/classes resolve to the FIRST
    stage with that name. Pushing to stage 0 (the root, no input queue) is invalid.

    This is forward-flow's escape hatch and is BEST-EFFORT: pipe's completion is
    signalled forward-only (End sentinels + upstream-done events), so an item
    pushed back after the target stage has already drained at end-of-run may be
    dropped. Cap retries on the item so the cycle always terminates."""
    name_to_idx = {}
    if stage_names:
        for i, nm in enumerate(stage_names):
            name_to_idx.setdefault(nm, i)  # first match wins

    def _push(stage, item, block=True):
        if item is None:
            return
        if isinstance(stage, bool):
            raise TypeError("push: stage must be an index, name, or class")
        if isinstance(stage, int):
            idx = stage
        else:
            # str name, a worker class, or a worker instance -> resolve by name
            name = (
                stage if isinstance(stage, str)
                else (stage if isinstance(stage, type) else type(stage)).__name__
            )
            if name not in name_to_idx:
                raise ValueError(f"push: unknown stage {stage!r}")
            idx = name_to_idx[name]
        if idx <= 0 or idx >= len(all_queues) + 1:
            raise ValueError(
                f"push: stage {stage!r} (idx {idx}) has no input queue to push to"
            )
        target = all_queues[idx - 1]  # input queue of stage idx == output of idx-1
        serialized = _item_to_shm(item, skip=False)
        if not block:
            with contextlib.suppress(Full):
                target.put_nowait(serialized)
            return
        while not (should_stop is not None and should_stop.value):
            try:
                target.put(serialized, timeout=0.1)
                return
            except Full:
                time.sleep(0.01)

    return _push


def _is_end(item):
    return item is End


def _is_worker_stop(item):
    return item is WorkerStop


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
    raise_errors=False,
    stage_end_counter=None,
    stage_worker_count=None,
    stage_idx=None,
    stage_name=None,
    is_final_stage=False,
    expected_consumers=1,
    upstream_done=None,
    stage_done=None,
    sequential=False,
    batch_size=0,
    drain_event=None,
    drain=True,
    all_queues=None,
    stage_names=None,
    cpu_affinity=None,
    cpu_threads=None,
    out_chunk=0,
    out_chunk_ms=10.0,
):
    """Worker process using Event-based completion signaling."""
    worker_desc = f"{stage_name} ({worker_id})" if stage_name else worker_id
    is_root = (in_queue is None)

    # Pin to this worker's CPU slice (if cpus= was set) and size threads: explicit
    # cpu_threads > slice size > the default 2-thread cap that avoids oversubscription.
    _setup_cpu(cpu_affinity, cpu_threads, worker_desc)

    _increase_fd_limit(worker_desc)

    if gpu_id is not None:
        try:
            # Isolate this process to its GPU *before* any CUDA context is created.
            # set_device() alone initializes CUDA, and some model/library loads then
            # create a second bare ~500 MB primary context on cuda:0 regardless of the
            # current device. Pinning visibility to the one GPU (env var, honoured
            # only while no context exists yet — which is true here) makes that
            # impossible. Physical id stays discoverable via CUDA_VISIBLE_DEVICES.
            # Safe because this is one process per GPU; the *threaded* GPU worker
            # keeps set_device() (CUDA_VISIBLE_DEVICES is process-global, so it can't
            # isolate co-resident GPU threads).
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)  # logical 0 == physical gpu_id
        except Exception as e:
            print(f"Failed to set GPU {gpu_id}: {e}")

    def handle_signal(signum, frame):
        _log(f"Worker {worker_desc} received signal {signum}, stopping gracefully")
        should_stop.value = 1

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _has_custom_run = hasattr(worker, "run") and not is_root

    # All reads go through a chunk-aware channel: transparently unpacks Chunk
    # messages (many payloads per queue op) while passing bare items and
    # sentinels straight through.
    in_ch = _InputChannel(in_queue) if in_queue is not None else None

    # All writes go through an output channel. With out_chunk > 0 it bundles N
    # serialized payloads into ONE queue message (size-OR-timeout flush); with
    # out_chunk=0 it is a plain blocking put — identical to the old inline loops.
    out_ch = (
        _OutputChannel(out_queue, chunk_size=out_chunk, chunk_ms=out_chunk_ms)
        if out_queue is not None else None
    )

    def _emit(obj):
        out_ch.send(_item_to_shm(obj, skip=_skip_shm_for_output(is_final_stage)))

    # Inject pull/put for workers with run()
    if _has_custom_run and in_queue is not None:
        def _pull(n=batch_size or 1):
            items = []
            while len(items) < n and not should_stop.value:
                try:
                    raw = in_ch.get_nowait()
                except Empty:
                    if items:
                        break  # got partial batch, return it
                    if upstream_done is not None and upstream_done.is_set():
                        return []  # truly done
                    time.sleep(0.01)
                    continue
                if _is_worker_stop(raw):
                    in_ch.put(WorkerStop)
                    break
                raw = _item_from_shm(raw)
                if _is_end(raw):
                    continue
                items.append(raw)
            # No fresh input: a partial output chunk older than chunk_ms must
            # not wait for more sends to flush it.
            if not items and out_ch is not None:
                out_ch.maybe_flush()
            return items
        worker.pull = _pull

    if _has_custom_run and out_queue is not None:
        def _put(item):
            if item is None:
                return
            _emit(item)
        worker.put = _put

    # push(stage, item): send an item BACK to an earlier stage's input queue
    # (e.g. re-render on a failed WER check). Available to every non-root worker.
    if all_queues is not None and not is_root:
        worker.push = _make_push(all_queues, stage_names, should_stop)

    if hasattr(worker, "load"):
        _log(f"Worker {worker_desc} calling load()")
        worker.load()

    if _has_custom_run:
        print(f"Worker {worker_desc} using custom run()")
        if hasattr(worker, "pull"):
            worker.run()
        else:
            worker.run(
                in_queue=in_queue,
                out_queue=out_queue,
                should_stop=should_stop,
                upstream_done=upstream_done,
                serialize=lambda item: _item_to_shm(item, skip=_skip_shm_for_output(is_final_stage)),
                deserialize=lambda raw: None if _is_end(v := _item_from_shm(raw)) else v,
                timing_dict=timing_dict,
                worker_id=worker_id,
                worker_start_wall_time=time.time(),
            )

    items_processed = 0
    total_process_time = 0.0
    total_audio_duration = 0.0
    worker_start_wall_time = time.time()
    last_fd_check = time.time()
    fd_check_interval = 30
    consecutive_empty = 0
    EMPTY_THRESHOLD = 30

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
            return sum(extract_audio_duration(x) for x in obj)
        return 0.0

    while not should_stop.value and not _has_custom_run:
        now = time.time()
        if now - last_fd_check > fd_check_interval:
            fd_info = _get_fd_info()
            if sequential:
                print(f"Worker {worker_desc} FD check: {fd_info['fd_count']}/{fd_info['soft_limit']} (items: {items_processed})")
            last_fd_check = now

        if out_ch is not None:
            out_ch.maybe_flush()

        try:
            if should_stop.value == 1:
                break

            if (not drain) and drain_event is not None and drain_event.is_set():
                break

            if is_root:
                start_time = time.time()
                result = worker()
                process_time = time.time() - start_time

                if result is None:
                    continue

                # Handle generator results - iterate and emit items one by one
                if inspect.isgenerator(result):
                    for gen_item in result:
                        if should_stop.value or (drain_event is not None and drain_event.is_set()):
                            break
                        if gen_item is None or _is_end(gen_item):
                            continue
                        if timing_dict is not None and worker_id is not None:
                            items_processed += 1
                            total_audio_duration += extract_audio_duration(gen_item)
                        if out_queue:
                            _emit(gen_item)
                    # Generator exhausted = done
                    if timing_dict is not None and worker_id is not None:
                        total_process_time += time.time() - start_time
                        timing_dict[worker_id] = {
                            "items": items_processed,
                            "total_time": total_process_time,
                            "avg_time": total_process_time / items_processed if items_processed else 0,
                            "audio_duration": total_audio_duration,
                            "start_wall_time": worker_start_wall_time,
                        }
                    break

                # Handle list/single item results (original behavior)
                if _is_end(result):
                    break
                if not isinstance(result, list):
                    result = [result]

                has_end = any(_is_end(r) for r in result)
                valid_items = [r for r in result if r is not None and not _is_end(r)]

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
                        _emit(item)

                if has_end:
                    break

            else:
                try:
                    item = in_ch.get(timeout=0.1)
                    consecutive_empty = 0
                except Empty:
                    consecutive_empty += 1
                    if upstream_done is not None and upstream_done.is_set():
                        if consecutive_empty >= EMPTY_THRESHOLD:
                            break
                    continue
                except (OSError, EOFError, BrokenPipeError):
                    break

                # Graceful early-exit primitive: a worker leaves the pool when it
                # pulls a WorkerStop sentinel. Currently only driven by the
                # planned autoscaler (src/pipe/_planned/autoscale.py); left inert
                # here so scale-down works the moment that feature is re-enabled.
                if _is_worker_stop(item):
                    _log(f"Worker {worker_id} received stop signal, exiting gracefully")
                    # Items already unpacked from a chunk belong back on the shared
                    # queue — siblings must process them, we're leaving the pool.
                    in_ch.requeue_buffered(should_stop)
                    # And a partial output chunk must reach downstream.
                    if out_ch is not None:
                        out_ch.flush()
                    # Skip coordination - we were removed from the pool, not finishing normally
                    # (count is decremented by whatever signalled the stop)
                    while should_stop is not None and not should_stop.value:
                        time.sleep(0.1)
                    return

                item = _item_from_shm(item)

                # Check for End sentinel from upstream - skip it and continue draining
                # The upstream_done event + empty threshold will signal exit
                if _is_end(item):
                    continue

                if batch_size > 0:
                    batch = [item]
                    deadline = time.monotonic() + 0.05
                    while len(batch) < batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            raw = in_ch.get(timeout=min(remaining, 0.01))
                        except Empty:
                            if time.monotonic() >= deadline:
                                break
                            continue
                        if _is_worker_stop(raw):
                            in_ch.put(WorkerStop)
                            break
                        raw = _item_from_shm(raw)
                        if _is_end(raw):
                            continue
                        batch.append(raw)
                    start_time = time.time()
                    result = worker(batch)
                    process_time = time.time() - start_time
                    n_items = len(batch)
                    input_audio = sum(extract_audio_duration(b) for b in batch)
                else:
                    start_time = time.time()
                    result = worker(item)
                    process_time = time.time() - start_time
                    n_items = 1
                    input_audio = extract_audio_duration(item)

                if result is None:
                    continue

                # Handle generator results from middle workers
                if inspect.isgenerator(result):
                    gen_count = 0
                    for gen_item in result:
                        if should_stop.value:
                            break
                        if gen_item is None or _is_end(gen_item):
                            continue
                        gen_count += 1
                        if out_queue:
                            _emit(gen_item)
                    if timing_dict is not None and worker_id is not None:
                        items_processed += n_items
                        total_process_time += time.time() - start_time
                        total_audio_duration += input_audio
                        if items_processed % 10 == 0:
                            timing_dict[worker_id] = {
                                "items": items_processed,
                                "total_time": total_process_time,
                                "avg_time": total_process_time / items_processed,
                                "audio_duration": total_audio_duration,
                                "start_wall_time": worker_start_wall_time,
                            }
                    continue

                # Handle list/single item results
                if not isinstance(result, list):
                    result = [result]

                valid_items = [r for r in result if r is not None and not _is_end(r)]

                if timing_dict is not None and worker_id is not None:
                    items_processed += n_items
                    total_process_time += process_time
                    total_audio_duration += extract_audio_duration(result) or input_audio
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
                        _emit(out_item)

        except (ConnectionError, FileNotFoundError):
            should_stop.value = 2
            break
        except Exception as e:
            print(f"Worker {worker_desc} error: {e}")
            traceback.print_exc()
            if raise_errors:
                raise e

    if not _has_custom_run and hasattr(worker, "flush") and out_queue:
        try:
            flushed_items = list(worker.flush())
            if flushed_items:
                _log(f"Worker {worker_desc} flushing {len(flushed_items)} items via flush()")
                for item in flushed_items:
                    if item is not None:
                        _emit(item)
        except Exception as e:
            print(f"Worker {worker_desc} flush() error: {e}")

    # Emit any partial output chunk BEFORE sibling coordination: the last
    # finisher puts the End sentinel, which must never overtake pending items.
    if out_ch is not None:
        out_ch.flush()

    # Coordinate with siblings
    if stage_end_counter is not None and stage_worker_count is not None:
        with stage_end_counter.get_lock():
            stage_end_counter.value += 1
            finished_workers = stage_end_counter.value

        current_worker_count = stage_worker_count.value
        stage_desc = f"{stage_name}" if stage_name else f"stage_{stage_idx}"
        _log(f"Worker {worker_id} finished ({finished_workers}/{current_worker_count} at {stage_desc})")

        if finished_workers >= current_worker_count:
            # Put End sentinel on output queue so downstream knows we're done
            if out_queue is not None:
                try:
                    serialized = _item_to_shm(End, skip=_skip_shm_for_output(is_final_stage))
                    out_queue.put(serialized, timeout=1.0)
                    _log(f"Put End sentinel on queue for {stage_desc}")
                except Full:
                    print(f"Warning: Could not put End sentinel on queue for {stage_desc}")
            if stage_done is not None:
                stage_done.set()
                _log(f"All workers finished at {stage_desc}, signaling downstream")

    # Stay alive until pipeline stops - keeps tensor file descriptors valid
    # Without this, tensors in queues become invalid when worker exits
    while should_stop is not None and not should_stop.value:
        time.sleep(0.1)


def _threaded_worker_run(
    worker,
    in_queue=None,
    out_queue=None,
    should_stop=None,
    working=None,
    gpu_id=None,
    timing_dict=None,
    worker_id=None,
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
    sequential=False,
    batch_size=0,
    drain_event=None,
    drain=True,
    all_queues=None,
    stage_names=None,
    cpu_affinity=None,
    cpu_threads=None,
    out_chunk=0,
    out_chunk_ms=10.0,
):
    """Threaded worker using Event-based completion signaling."""
    worker_desc = f"{stage_name} ({worker_id})" if stage_name else worker_id
    is_root = (in_queue is None)

    # Pin to this worker's CPU slice (if cpus= was set) and size threads: explicit
    # cpu_threads > slice size > the default 2-thread cap that avoids oversubscription.
    _setup_cpu(cpu_affinity, cpu_threads, worker_desc)

    _increase_fd_limit(worker_desc)

    if gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"Failed to set GPU {gpu_id}: {e}")

    thread_stop = threading.Event()

    def handle_signal(signum, frame):
        _log(f"Threaded worker {worker_desc} received signal {signum}, stopping")
        should_stop.value = 1
        thread_stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _has_custom_run = hasattr(worker, "run") and not is_root

    # Inject pull/put for workers with run(). run() is invoked once (not per
    # thread), so a single chunk-aware channel is safe here.
    if _has_custom_run and in_queue is not None:
        _pull_ch = _InputChannel(in_queue)

        def _pull(n=batch_size or 1):
            items = []
            while len(items) < n and not should_stop.value and not thread_stop.is_set():
                try:
                    raw = _pull_ch.get_nowait()
                except Empty:
                    if items:
                        break  # got partial batch, return it
                    if upstream_done is not None and upstream_done.is_set():
                        return []  # truly done
                    time.sleep(0.01)
                    continue
                if _is_worker_stop(raw):
                    _pull_ch.put(WorkerStop)
                    break
                raw = _item_from_shm(raw)
                if _is_end(raw):
                    continue
                items.append(raw)
            return items
        worker.pull = _pull

    _put_ch = None
    if _has_custom_run and out_queue is not None:
        # run() is invoked once (single-threaded), so one output channel is safe.
        _put_ch = _OutputChannel(out_queue, chunk_size=out_chunk, chunk_ms=out_chunk_ms)

        def _put(item):
            if item is None:
                return
            _put_ch.send(_item_to_shm(item, skip=_skip_shm_for_output(is_final_stage)))
        worker.put = _put

    # push(stage, item): send an item BACK to an earlier stage (see _make_push).
    if all_queues is not None and not is_root:
        worker.push = _make_push(all_queues, stage_names, should_stop)

    if hasattr(worker, "load"):
        _log(f"Threaded worker {worker_desc} calling load()")
        worker.load()

    # Delegate to custom run() if worker defines it
    if _has_custom_run:
        print(f"Threaded worker {worker_desc} using custom run()")
        if hasattr(worker, "pull"):
            worker.run()
        else:
            worker.run(
                in_queue=in_queue,
                out_queue=out_queue,
                should_stop=should_stop,
                upstream_done=upstream_done,
                serialize=lambda item: _item_to_shm(item, skip=_skip_shm_for_output(is_final_stage)),
                deserialize=lambda raw: None if _is_end(v := _item_from_shm(raw)) else v,
                timing_dict=timing_dict,
                worker_id=worker_id,
                worker_start_wall_time=time.time(),
            )
        # Partial output chunk must land before the End sentinel below.
        if _put_ch is not None:
            _put_ch.flush()
        if stage_end_counter is not None and stage_worker_count is not None:
            with stage_end_counter.get_lock():
                stage_end_counter.value += 1
                finished_workers = stage_end_counter.value
            current_worker_count = stage_worker_count.value
            print(f"Threaded worker {worker_desc} finished ({finished_workers}/{current_worker_count})")
            if finished_workers >= current_worker_count:
                if out_queue is not None:
                    with contextlib.suppress(Full):
                        serialized = _item_to_shm(End, skip=_skip_shm_for_output(is_final_stage))
                        out_queue.put(serialized, timeout=1.0)
                        print(f"Put End sentinel on queue for {stage_name}")
                if stage_done is not None:
                    stage_done.set()
                    print(f"All workers finished at {stage_name}, signaling downstream")
        while should_stop is not None and not should_stop.value:
            time.sleep(0.1)
        return

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
            return sum(extract_audio_duration(x) for x in obj)
        return 0.0

    def thread_fn():
        local_items = 0
        local_time = 0.0
        local_audio = 0.0
        consecutive_empty = 0
        EMPTY_THRESHOLD = 30
        # Per-thread channels: neither the input chunk buffer nor the output
        # pending list is thread-safe, so each thread owns its own. A chunk
        # grabbed by one thread is processed wholly by that thread.
        in_ch = _InputChannel(in_queue) if in_queue is not None else None
        out_ch = (
            _OutputChannel(out_queue, chunk_size=out_chunk, chunk_ms=out_chunk_ms)
            if out_queue is not None else None
        )

        def _emit(obj):
            out_ch.send(_item_to_shm(obj, skip=_skip_shm_for_output(is_final_stage)))

        try:
            _thread_loop(in_ch, out_ch, _emit, local_items, local_time, local_audio,
                         consecutive_empty, EMPTY_THRESHOLD)
        finally:
            # Every exit path (done, worker_stop, force stop, error) must emit
            # the partial chunk; _blocking_put gives up if should_stop is set.
            if out_ch is not None:
                out_ch.flush()

    def _thread_loop(in_ch, out_ch, _emit, local_items, local_time, local_audio,
                     consecutive_empty, EMPTY_THRESHOLD):
        while not should_stop.value and not thread_stop.is_set():
            if out_ch is not None:
                out_ch.maybe_flush()
            try:
                if (not drain) and drain_event is not None and drain_event.is_set():
                    return
                if is_root:
                    start_time = time.time()
                    result = worker()
                    process_time = time.time() - start_time

                    if result is None:
                        continue

                    # Handle generator results
                    if inspect.isgenerator(result):
                        for gen_item in result:
                            if should_stop.value or thread_stop.is_set() or (drain_event is not None and drain_event.is_set()):
                                break
                            if gen_item is None or _is_end(gen_item):
                                continue
                            if timing_dict is not None and worker_id is not None:
                                local_items += 1
                                local_audio += extract_audio_duration(gen_item)
                            if out_queue:
                                _emit(gen_item)
                        # Generator exhausted = done
                        if timing_dict is not None and worker_id is not None:
                            local_time += time.time() - start_time
                        thread_stop.set()
                        return

                    # Handle list/single item results (original behavior)
                    if _is_end(result):
                        thread_stop.set()
                        return
                    if not isinstance(result, list):
                        result = [result]

                    has_end = any(_is_end(r) for r in result)
                    valid_items = [r for r in result if r is not None and not _is_end(r)]

                    if timing_dict is not None and worker_id is not None and valid_items:
                        local_items += len(valid_items)
                        local_time += process_time
                        local_audio += extract_audio_duration(valid_items)

                    if out_queue:
                        for item in valid_items:
                            _emit(item)

                    if has_end:
                        thread_stop.set()
                        return
                else:
                    try:
                        item = in_ch.get(timeout=0.1)
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
                        in_ch.requeue_buffered(should_stop)
                        return

                    # Inert graceful early-exit primitive (see _worker_run above);
                    # only the planned autoscaler emits WorkerStop today.
                    if _is_worker_stop(item):
                        _log(f"Thread {worker_id} received stop signal")
                        # Return chunk-buffered items to the shared queue for siblings.
                        in_ch.requeue_buffered(should_stop)
                        # Just exit - coordination is skipped for stopped workers
                        # (count is decremented by whatever signalled the stop)
                        return

                    item = _item_from_shm(item)

                    # Check for End sentinel from upstream - skip it and continue draining
                    # The upstream_done event + empty threshold will signal exit
                    if _is_end(item):
                        continue

                    if batch_size > 0:
                        batch = [item]
                        deadline = time.monotonic() + 0.05
                        while len(batch) < batch_size:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            try:
                                raw = in_ch.get(timeout=min(remaining, 0.01))
                            except Empty:
                                if time.monotonic() >= deadline:
                                    break
                                continue
                            if _is_worker_stop(raw):
                                in_ch.put(WorkerStop)
                                break
                            raw = _item_from_shm(raw)
                            if _is_end(raw):
                                continue
                            batch.append(raw)
                        start_time = time.time()
                        result = worker(batch)
                        process_time = time.time() - start_time
                        n_items = len(batch)
                        input_audio = sum(extract_audio_duration(b) for b in batch)
                    else:
                        start_time = time.time()
                        result = worker(item)
                        process_time = time.time() - start_time
                        n_items = 1
                        input_audio = extract_audio_duration(item)

                    if result is None:
                        continue

                    # Handle generator results from middle workers
                    if inspect.isgenerator(result):
                        for gen_item in result:
                            if should_stop.value or thread_stop.is_set():
                                break
                            if gen_item is None or _is_end(gen_item):
                                continue
                            if out_queue:
                                _emit(gen_item)
                        if timing_dict is not None and worker_id is not None:
                            local_items += n_items
                            local_time += time.time() - start_time
                            local_audio += input_audio
                            if local_items % 10 == 0:
                                timing_dict[worker_id] = {
                                    "items": local_items,
                                    "total_time": local_time,
                                    "avg_time": local_time / local_items,
                                    "audio_duration": local_audio,
                                    "start_wall_time": worker_start_wall_time,
                                }
                        continue

                    # Handle list/single item results
                    if not isinstance(result, list):
                        result = [result]

                    valid_items = [r for r in result if r is not None and not _is_end(r)]

                    if timing_dict is not None and worker_id is not None:
                        local_items += n_items
                        local_time += process_time
                        local_audio += extract_audio_duration(result) or input_audio
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
                            _emit(out_item)

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
        _log(f"Threaded worker {worker_desc} received KeyboardInterrupt")
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
                    serialized = _item_to_shm(item, skip=_skip_shm_for_output(is_final_stage))
                    out_queue.put(serialized, timeout=1)
            except Exception as e:
                print(f"Threaded worker {worker_desc} flush error: {e}")

        if stage_end_counter is not None and stage_worker_count is not None:
            with stage_end_counter.get_lock():
                stage_end_counter.value += 1
                finished_workers = stage_end_counter.value

            current_worker_count = stage_worker_count.value
            _log(f"Threaded worker {worker_desc} finished ({finished_workers}/{current_worker_count})")

            if finished_workers >= current_worker_count:
                # Put End sentinel on output queue so downstream knows we're done
                if out_queue is not None:
                    try:
                        serialized = _item_to_shm(End, skip=_skip_shm_for_output(is_final_stage))
                        out_queue.put(serialized, timeout=1.0)
                        _log(f"Put End sentinel on queue for {stage_name}")
                    except Full:
                        print(f"Warning: Could not put End sentinel on queue for {stage_name}")
                if stage_done is not None:
                    stage_done.set()
                    _log(f"All workers finished at {stage_name}, signaling downstream")

        _log(f"Threaded worker {worker_desc} shutdown complete")

    # Stay alive until pipeline stops - keeps tensor file descriptors valid
    while should_stop is not None and not should_stop.value:
        time.sleep(0.1)

