import contextlib
import inspect
import os
from queue import Empty, Full
import threading
import time

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue, Value

from .monitors import _autoscaler_thread, _collect_stats, _create_progress, _health_monitor_thread, _stats_monitor_thread, _stats_monitor_thread_text
from .shm import _cleanup_stale_shm, _item_from_shm


def _log(msg):
    if os.environ.get("PIPE_QUIET") != "1":
        print(msg)
from .workers import (
    _check_picklable,
    _is_end,
    _threaded_worker_run,
    _worker_run,
)

if not mp.get_start_method(allow_none=True):
    mp.set_start_method("spawn", force=True)


class PipeIterator:
    """Lightweight iterator that reads from a multiprocessing queue."""

    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.queue.get(timeout=1.0)
                item = _item_from_shm(item)
                if _is_end(item):
                    raise StopIteration
                return item
            except Empty:
                continue


class Pipe:
    def __init__(
        self,
        debug=False,
        raise_errors=None,
        health_check_interval=30,
        expected_consumers=1,
        stats_interval=0.2,
        stats_mode="rich",
        allow_full_restart=True,
        autoscale=False,
        max_workers_per_stage=8,
        use_shm=False,
        output_shm=False,
    ):
        # Set env vars for shm control (inherited by spawned workers)
        if not use_shm:
            os.environ["PIPE_NO_SHM"] = "1"
        elif "PIPE_NO_SHM" in os.environ:
            del os.environ["PIPE_NO_SHM"]
        if not output_shm:
            os.environ["PIPE_NO_SHM_OUTPUT"] = "1"
        elif "PIPE_NO_SHM_OUTPUT" in os.environ:
            del os.environ["PIPE_NO_SHM_OUTPUT"]
        _cleanup_stale_shm()
        self.debug = debug
        self.use_shm = use_shm
        self.raise_errors = raise_errors if raise_errors is not None else debug
        self.health_check_interval = health_check_interval
        self.allow_full_restart = allow_full_restart
        self.autoscale = autoscale
        self.max_workers_per_stage = max_workers_per_stage
        self.jobs = []
        self.queues: list[Queue] = []
        self.processes = []
        self.worker_info = []
        self.worker_configs = {}
        self.started = False
        self.working = Value("i", 0)
        self.should_stop = Value("i", 0)
        self.restart_needed = Value("i", 0)
        self.health_monitor_thread = None
        self.health_monitor_stop_event = threading.Event()
        self.stats_monitor_thread = None
        self.stats_monitor_stop_event = threading.Event()
        self.stats_interval = stats_interval
        self.stats_mode = stats_mode
        self.autoscaler_thread = None
        self.autoscaler_stop_event = threading.Event()
        self.gpus = self._get_gpu_count()
        self.expected_consumers = expected_consumers

        import multiprocessing
        self.manager = multiprocessing.Manager()
        self.timing_dict = self.manager.dict()

        self.stage_end_counters = []
        self.stage_worker_counts = []
        self.stage_done_events = []

    def _get_gpu_count(self):
        try:
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            _log("CUDA not available, pergpu flag will be ignored")
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
        if hasattr(func, "__name__"):
            stage_name = func.__name__
        elif hasattr(func, "__class__"):
            stage_name = func.__class__.__name__
        else:
            stage_name = str(type(func).__name__)

        _check_picklable(func, stage_name, is_thread=thread)

        gpu_count = self.gpus
        is_gpu_stage = pergpu or gpu_id is not None

        if pergpu:
            if gpu_count > 0:
                actual_workers = workers * gpu_count
                _log(
                    f"Per-GPU mode: {workers} workers per GPU ({actual_workers} total for {gpu_count} GPUs)"
                )
            else:
                actual_workers = workers
                pergpu = False
                is_gpu_stage = False
                _log(f"No GPUs available, falling back to {workers} CPU workers")
        else:
            actual_workers = workers

        if max_workers is None:
            default_max = actual_workers * 4
        else:
            default_max = max_workers

        if is_gpu_stage and gpu_count > 0:
            effective_max = min(default_max, gpu_count)
            if effective_max < default_max:
                _log(f"  {stage_name}: max_workers capped at {effective_max} (GPU count)")
        else:
            effective_max = default_max

        if is_gpu_stage:
            stage_autoscale = False
        elif autoscale is not None:
            stage_autoscale = autoscale
        else:
            stage_autoscale = self.autoscale

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
        _log(f"Starting pipeline with {len(self.jobs)} jobs")
        self.started = True

        if not self.jobs:
            raise ValueError("No workers added to pipeline")

        if self.debug:
            _log("Debug mode - skipping multiprocessing setup")
            return self

        _log("Setting up multiprocessing...")

        for i, job in enumerate(self.jobs):
            outq_size = job.get("outqn") or 0
            self.queues.append(Queue(maxsize=outq_size))
            self.stage_end_counters.append(Value("i", 0))
            self.stage_done_events.append(Event())

            if job.get("thread", False) and not job.get("pergpu", False):
                worker_count = 1
            else:
                worker_count = job["num_workers"]
            self.stage_worker_counts.append(Value("i", worker_count))

        for i, job in enumerate(self.jobs):
            in_queue = self.queues[i - 1] if i > 0 else None
            out_queue = self.queues[i] if i < len(self.queues) else None
            is_final_stage = i == len(self.jobs) - 1

            upstream_done = self.stage_done_events[i - 1] if i > 0 else None
            stage_done = self.stage_done_events[i]

            func = job["func"]
            if hasattr(func, "__name__"):
                stage_name = func.__name__
            elif hasattr(func, "__class__"):
                stage_name = func.__class__.__name__
            else:
                stage_name = str(type(func).__name__)

            if job.get("thread", False):
                if job.get("pergpu", False):
                    for worker_idx in range(job["num_workers"]):
                        num_threads = 1
                        gpu_id = worker_idx % self.gpus

                        worker_id = f"stage_{i}_threaded_worker_{worker_idx}_gpu_{gpu_id}"

                        args = (
                            job["func"],
                            in_queue,
                            out_queue,
                            self.should_stop,
                            self.working,
                            gpu_id,
                            self.timing_dict,
                            worker_id,
                            self.raise_errors,
                            num_threads,
                            self.stage_end_counters[i],
                            self.stage_worker_counts[i],
                            i,
                            stage_name,
                            is_final_stage,
                            self.expected_consumers,
                            upstream_done,
                            stage_done,
                            self.debug,
                        )

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
                    num_threads = job["num_workers"]
                    worker_id = f"stage_{i}_threaded_{num_threads}threads"

                    args = (
                        job["func"],
                        in_queue,
                        out_queue,
                        self.should_stop,
                        self.working,
                        None,
                        self.timing_dict,
                        worker_id,
                        self.raise_errors,
                        num_threads,
                        self.stage_end_counters[i],
                        self.stage_worker_counts[i],
                        i,
                        stage_name,
                        is_final_stage,
                        self.expected_consumers,
                        upstream_done,
                        stage_done,
                        self.debug,
                    )

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
                for worker_idx in range(job["num_workers"]):
                    gpu_id = None
                    if job.get("pergpu", False):
                        gpu_id = worker_idx % self.gpus

                    worker_id = f"stage_{i}_worker_{worker_idx}"
                    if gpu_id is not None:
                        worker_id += f"_gpu_{gpu_id}"

                    _log(f"Starting worker process: {stage_name} ({worker_id})")

                    args = (
                        job["func"],
                        in_queue,
                        out_queue,
                        self.should_stop,
                        self.working,
                        gpu_id,
                        self.timing_dict,
                        worker_id,
                        self.raise_errors,
                        self.stage_end_counters[i],
                        self.stage_worker_counts[i],
                        i,
                        stage_name,
                        is_final_stage,
                        self.expected_consumers,
                        upstream_done,
                        stage_done,
                        self.debug,
                    )

                    self.worker_configs[worker_id] = {
                        "target": _worker_run,
                        "args": args,
                        "stage_name": stage_name,
                    }

                    p = mp.Process(target=_worker_run, args=args)
                    p.start()
                    self.processes.append(p)
                    self.worker_info.append((p, worker_id, stage_name))
                    _log(f"Worker process {stage_name} ({worker_id}) started with PID {p.pid}")

        if self.health_check_interval > 0:
            _log(f"Starting health monitor (check interval: {self.health_check_interval}s)")
            self.health_monitor_stop_event.clear()
            self.health_monitor_thread = threading.Thread(
                target=_health_monitor_thread,
                args=(
                    self,
                    self.should_stop,
                    self.health_check_interval,
                    self.health_monitor_stop_event,
                ),
                daemon=True,
            )
            self.health_monitor_thread.start()
            _log("Health monitor thread started")

        if self.stats_interval > 0 and self.stats_mode != "external":
            if self.stats_mode == "text":
                self.progress = None
                monitor_fn = _stats_monitor_thread_text
            else:
                from rich.console import Console
                console = Console() if self.stats_mode == "rich" else None
                self.progress = _create_progress(console)
                self._stage_task_ids = {}
                for idx, job in enumerate(self.jobs):
                    task_id = self.progress.add_task(job["name"], total=None, info="")
                    self._stage_task_ids[idx] = task_id
                if self.stats_mode == "rich":
                    self.progress.start()
                monitor_fn = _stats_monitor_thread
            self.stats_monitor_thread = threading.Thread(
                target=monitor_fn,
                args=(self, self.stats_monitor_stop_event, self.stats_interval),
                daemon=True,
            )
            self.stats_monitor_thread.start()

        has_autoscale = any(job.get("autoscale") for job in self.jobs)
        if has_autoscale:
            _log("Starting autoscaler (monitoring queue pressure)")
            self.autoscaler_stop_event.clear()
            self.autoscaler_thread = threading.Thread(
                target=_autoscaler_thread,
                args=(self, self.should_stop, self.autoscaler_stop_event),
                daemon=True,
            )
            self.autoscaler_thread.start()

        return self

    def get_stats(self):
        return _collect_stats(self)

    def _restart_worker(self, worker_idx, worker_id):
        if worker_id not in self.worker_configs:
            raise ValueError(f"No configuration found for worker {worker_id}")

        config = self.worker_configs[worker_id]

        old_proc, _, _ = self.worker_info[worker_idx]
        with contextlib.suppress(Exception):
            old_proc.join(timeout=1)

        p = mp.Process(
            target=config["target"],
            args=config["args"],
        )
        p.start()

        self.worker_info[worker_idx] = (p, worker_id, config["stage_name"])
        for i, proc in enumerate(self.processes):
            if proc == old_proc:
                self.processes[i] = p
                break

        print(f"Worker {worker_id} restarted with PID {p.pid}")

    def _restart_stage(self, stage_idx):
        print(f"   Restarting stage {stage_idx}...")

        stage_workers = []
        for idx, (proc, worker_id, stage_name) in enumerate(self.worker_info):
            if worker_id.startswith(f"stage_{stage_idx}_"):
                stage_workers.append((idx, proc, worker_id, stage_name))

        if not stage_workers:
            raise ValueError(f"No workers found for stage {stage_idx}")

        for idx, proc, worker_id, stage_name in stage_workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)

        job = self.jobs[stage_idx]

        old_out_queue = self.queues[stage_idx]
        old_out_maxsize = old_out_queue._maxsize if hasattr(old_out_queue, "_maxsize") else 0

        drained_items = []
        try:
            while True:
                item = old_out_queue.get_nowait()
                if item != "end":
                    drained_items.append(item)
        except Empty:
            pass

        try:
            old_out_queue.cancel_join_thread()
            old_out_queue.close()
        except Exception:
            pass

        new_out_queue = Queue(maxsize=old_out_maxsize)
        self.queues[stage_idx] = new_out_queue

        for item in drained_items:
            try:
                new_out_queue.put_nowait(item)
            except Full:
                break

        print(
            f"   Recreated output queue for stage {stage_idx} (recovered {len(drained_items)} items)"
        )

        self.stage_end_counters[stage_idx].value = 0

        in_queue = self.queues[stage_idx - 1] if stage_idx > 0 else None
        out_queue = new_out_queue

        func = job["func"]
        if hasattr(func, "__name__"):
            stage_name = func.__name__
        elif hasattr(func, "__class__"):
            stage_name = func.__class__.__name__
        else:
            stage_name = str(type(func).__name__)

        is_final_stage = stage_idx == len(self.jobs) - 1
        next_stage_worker_count = (
            None if is_final_stage else self.stage_worker_counts[stage_idx + 1]
        )

        for idx, old_proc, worker_id, _ in stage_workers:
            config = self.worker_configs.get(worker_id)
            if not config:
                print(f"   Warning: No config for {worker_id}, skipping")
                continue

            old_args = config["args"]

            if config["target"] == _threaded_worker_run:
                new_args = (
                    old_args[0],
                    in_queue,
                    out_queue,
                    self.should_stop,
                    self.working,
                    old_args[5],
                    self.timing_dict,
                    worker_id,
                    self.raise_errors,
                    old_args[9],
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
                new_args = (
                    old_args[0],
                    in_queue,
                    out_queue,
                    self.should_stop,
                    self.working,
                    old_args[5],
                    self.timing_dict,
                    worker_id,
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

            config["args"] = new_args

            p = mp.Process(target=config["target"], args=new_args)
            p.start()

            self.worker_info[idx] = (p, worker_id, stage_name)
            for i, proc in enumerate(self.processes):
                if proc == old_proc:
                    self.processes[i] = p
                    break

            print(f"   Restarted {worker_id} with PID {p.pid}")

        print(f"   Stage {stage_idx} restart complete")

    def restart(self, reason="ConnectionError"):
        self._stop(force=True)

        if self.stats_interval > 0:
            import multiprocessing

            self.manager = multiprocessing.Manager()
            self.timing_dict = self.manager.dict()

        self.start()
        print(f"Pipeline restarted due to {reason}")

    def stop(self, force=False):
        self._stop(force=force)

    def _stop(self, force=False):
        self.should_stop.value = 1

        if self.health_monitor_thread is not None and self.health_monitor_thread.is_alive():
            _log("Stopping health monitor...")
            try:
                self.health_monitor_stop_event.set()
                self.health_monitor_thread.join(timeout=2)
            except Exception as e:
                print(f"Error stopping health monitor: {e}")
            self.health_monitor_thread = None

        if self.stats_monitor_thread is not None and self.stats_monitor_thread.is_alive():
            self.stats_monitor_stop_event.set()
            self.stats_monitor_thread.join(timeout=2)
            self.stats_monitor_thread = None
        if hasattr(self, "progress") and self.progress is not None:
            if getattr(self, "stats_mode", "rich") == "rich":
                self.progress.stop()
            self.progress = None

        if force:
            _log("Force stopping all processes...")

            for p in self.processes:
                if p.is_alive():
                    p.terminate()

            time.sleep(0.1)

            for p in self.processes:
                if p.is_alive():
                    p.kill()

            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
        else:
            for q in self.queues:
                with contextlib.suppress(Full):
                    q.put("end", timeout=1)

            for p in self.processes:
                try:
                    p.join(timeout=2)
                    if p.is_alive():
                        p.terminate()
                except KeyboardInterrupt:
                    if p.is_alive():
                        p.terminate()
                except Exception as e:
                    print(f"Error stopping process: {e}")

        for p in self.processes:
            try:
                if p.is_alive():
                    p.join(timeout=1)
            except Exception:
                pass

        if self.manager is not None:
            try:
                self.manager.shutdown()
                self.manager = None
            except Exception as e:
                print(f"Error shutting down manager: {e}")

        for q in self.queues:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception as e:
                print(f"Error closing queue: {e}")

        self.processes = []
        self.queues = []
        self.worker_info = []
        self.worker_configs = {}
        self.working.value = 0
        self.should_stop.value = 0
        self.restart_needed.value = 0

        for counter in self.stage_end_counters:
            counter.value = 0
        for event in self.stage_done_events:
            event.clear()

    def __iter__(self):
        if not self.started:
            self.start()

        if self.debug:
            yield from self._debug_sequential_run()
            return

        final_done = self.stage_done_events[-1]
        consecutive_empty = 0
        EMPTY_THRESHOLD = 5

        while True:
            if self.should_stop.value == 2:
                print("\nWorker encountered connection error - restarting pipeline...")
                self.restart(reason="Worker connection error")
                consecutive_empty = 0
                continue

            try:
                item = self.queues[-1].get(timeout=0.1)
                consecutive_empty = 0
                item = _item_from_shm(item)

                if _is_end(item):
                    continue

                yield item
            except Empty:
                consecutive_empty += 1
                if final_done.is_set() and consecutive_empty >= EMPTY_THRESHOLD:
                    _log("Iterator: pipeline complete")
                    self._stop()
                    return
            except (ConnectionError, FileNotFoundError):
                self.restart()
                consecutive_empty = 0
            except KeyboardInterrupt:
                print("Ctrl+C detected - force stopping pipeline")
                self._stop(force=True)
                return

    def run(self):
        count = 0
        for _ in self:
            count += 1
        return count

    def __del__(self):
        try:
            if self.started and (self.processes or self.queues):
                self._stop(force=True)
        except Exception:
            pass

    def _debug_sequential_run(self):
        _log("Running in debug mode - sequential execution")

        if not self.jobs:
            return

        workers = []
        for job in self.jobs:
            worker_func = job["func"]
            if callable(worker_func) and not hasattr(worker_func, "__name__"):
                workers.append(worker_func)
            else:
                workers.append(worker_func)

        for worker in workers:
            if hasattr(worker, "load"):
                worker.load()

        pending_items = []
        root_ended = False

        while not root_ended or pending_items:
            if not root_ended:
                root_result = workers[0]()

                # Handle generator from root worker
                if inspect.isgenerator(root_result):
                    for root_item in root_result:
                        if root_item is None or _is_end(root_item):
                            continue
                        pending_items.append((1, root_item))
                    root_ended = True
                else:
                    if _is_end(root_result):
                        root_ended = True
                    elif not isinstance(root_result, list):
                        root_result = [root_result]
                        for root_item in root_result:
                            if _is_end(root_item):
                                root_ended = True
                                break
                            if root_item is not None:
                                pending_items.append((1, root_item))
                    else:
                        for root_item in root_result:
                            if _is_end(root_item):
                                root_ended = True
                                break
                            if root_item is not None:
                                pending_items.append((1, root_item))

            if pending_items:
                stage_idx, item = pending_items.pop(0)

                if stage_idx >= len(workers):
                    yield item
                    continue

                worker = workers[stage_idx]

                try:
                    result = worker(item)
                except Exception as e:
                    if self.raise_errors:
                        print(f"Error in worker at stage {stage_idx}: {e}")
                        raise e
                    else:
                        print(f"Error in worker at stage {stage_idx}: {e}, continuing...")
                        continue

                if result is None:
                    continue
                # Handle generator from middle worker
                elif inspect.isgenerator(result):
                    for res_item in result:
                        if res_item is None or _is_end(res_item):
                            continue
                        pending_items.append((stage_idx + 1, res_item))
                elif _is_end(result):
                    break
                elif isinstance(result, list):
                    for res_item in result:
                        if res_item is not None and not _is_end(res_item):
                            pending_items.append((stage_idx + 1, res_item))
                else:
                    pending_items.append((stage_idx + 1, result))

        # Flush all workers that have flush() method and process remaining items
        for stage_idx, worker in enumerate(workers[1:], 1):  # Skip root worker
            if hasattr(worker, "flush"):
                for flushed_item in worker.flush():
                    if flushed_item is not None:
                        pending_items.append((stage_idx + 1, flushed_item))

        # Process any remaining items from flush
        while pending_items:
            stage_idx, item = pending_items.pop(0)

            if stage_idx >= len(workers):
                yield item
                continue

            worker = workers[stage_idx]
            try:
                result = worker(item)
            except Exception as e:
                if self.raise_errors:
                    raise e
                continue

            if result is None:
                continue
            elif inspect.isgenerator(result):
                for res_item in result:
                    if res_item is not None and not _is_end(res_item):
                        pending_items.append((stage_idx + 1, res_item))
            elif _is_end(result):
                continue
            elif isinstance(result, list):
                for res_item in result:
                    if res_item is not None and not _is_end(res_item):
                        pending_items.append((stage_idx + 1, res_item))
            else:
                pending_items.append((stage_idx + 1, result))
