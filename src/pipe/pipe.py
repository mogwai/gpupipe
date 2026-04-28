import contextlib
import inspect
import os
from queue import Empty, Full
import signal
import threading
import time

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue, Value



class InstrumentedQueue:
    def __init__(self, queue):
        self._queue = queue
        self.items_put = Value('l', 0)
        self.items_got = Value('l', 0)
        self.total_transit = Value('d', 0.0)

    def put(self, item, **kwargs):
        self._queue.put((item, time.time()), **kwargs)
        with self.items_put.get_lock():
            self.items_put.value += 1

    def get(self, **kwargs):
        item, put_time = self._queue.get(**kwargs)
        transit = time.time() - put_time
        with self.total_transit.get_lock():
            self.total_transit.value += transit
        with self.items_got.get_lock():
            self.items_got.value += 1
        return item

    def get_nowait(self):
        item, put_time = self._queue.get_nowait()
        transit = time.time() - put_time
        with self.total_transit.get_lock():
            self.total_transit.value += transit
        with self.items_got.get_lock():
            self.items_got.value += 1
        return item

    def put_nowait(self, item):
        self._queue.put_nowait((item, time.time()))
        with self.items_put.get_lock():
            self.items_put.value += 1

    def qsize(self):
        return self._queue.qsize()

    def full(self):
        return self._queue.full()

    def empty(self):
        return self._queue.empty()

    def close(self):
        return self._queue.close()

    def cancel_join_thread(self):
        return self._queue.cancel_join_thread()

    @property
    def _maxsize(self):
        return self._queue._maxsize


def _log(msg):
    if os.environ.get("PIPE_VERBOSE") == "1":
        print(msg)

from .monitors import _autoscaler_thread, _collect_stats, _health_monitor_thread, _stats_monitor_thread, _stats_monitor_thread_text
from .profiling import _profile_dir, _profiled_worker, print_profile_summary
from .shm import _cleanup_stale_shm, _item_from_shm
from .types import End
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
        sequential=False,
        debug=False,
        raise_errors=None,
        health_check_interval=30,
        expected_consumers=1,
        stats_interval=0.2,
        stats_mode="text",
        allow_full_restart=True,
        autoscale=False,
        max_workers_per_stage=8,
        use_shm=False,
        output_shm=False,
        profile=False,
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
        self.sequential = sequential
        self.debug = debug
        self.use_shm = use_shm
        self.raise_errors = raise_errors if raise_errors is not None else sequential
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
        self.drain_event = Event()
        self.health_monitor_thread = None
        self.health_monitor_stop_event = threading.Event()
        self.stats_monitor_thread = None
        self.stats_monitor_stop_event = threading.Event()
        self.stats_interval = stats_interval
        self.stats_mode = stats_mode
        self.autoscaler_thread = None
        self.autoscaler_stop_event = threading.Event()
        self.profile = profile
        self.profile_dir = None
        self.profile_rss = {}
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
        batch=0,
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
                "batch": batch,
            }
        )

    def _spawn(self, target, args, worker_id):
        if self.profile:
            from multiprocessing import Value as MpValue
            rss_val = MpValue("l", 0)
            self.profile_rss[worker_id] = rss_val
            p = mp.Process(
                target=_profiled_worker,
                args=(target, args, self.profile_dir, worker_id, rss_val),
            )
        else:
            p = mp.Process(target=target, args=args)
        p.start()
        return p

    def start(self):
        _log(f"Starting pipeline with {len(self.jobs)} jobs")
        self.started = True

        if not self.jobs:
            raise ValueError("No workers added to pipeline")

        if self.sequential:
            _log("Sequential mode - skipping multiprocessing setup")
            return self

        if self.profile:
            self.profile_dir = _profile_dir()
            print(f"Profiling enabled, saving to {self.profile_dir}")

        _log("Setting up multiprocessing...")

        for i, job in enumerate(self.jobs):
            outq_size = job.get("outqn") or 0
            q = Queue(maxsize=outq_size)
            self.queues.append(InstrumentedQueue(q) if self.debug else q)
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
                            self.sequential,
                            job.get("batch", 0),
                            self.drain_event,
                        )

                        self.worker_configs[worker_id] = {
                            "target": _threaded_worker_run,
                            "args": args,
                            "stage_name": stage_name,
                        }

                        p = self._spawn(_threaded_worker_run, args, worker_id)
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
                        self.sequential,
                        job.get("batch", 0),
                        self.drain_event,
                    )

                    self.worker_configs[worker_id] = {
                        "target": _threaded_worker_run,
                        "args": args,
                        "stage_name": stage_name,
                    }

                    p = self._spawn(_threaded_worker_run, args, worker_id)
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
                        self.sequential,
                        job.get("batch", 0),
                        self.drain_event,
                    )

                    self.worker_configs[worker_id] = {
                        "target": _worker_run,
                        "args": args,
                        "stage_name": stage_name,
                    }

                    p = self._spawn(_worker_run, args, worker_id)
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
                self._rich_console = Console()
                self.progress = None
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

        _log(f"Worker {worker_id} restarted with PID {p.pid}")

    def _restart_stage(self, stage_idx):
        _log(f"   Restarting stage {stage_idx}...")

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
                if not _is_end(item):
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

        _log(
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
        upstream_done = self.stage_done_events[stage_idx - 1] if stage_idx > 0 else None
        stage_done = self.stage_done_events[stage_idx]

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
                    upstream_done,
                    stage_done,
                    self.sequential,
                    job.get("batch", 0),
                    self.drain_event,
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
                    upstream_done,
                    stage_done,
                    self.sequential,
                    job.get("batch", 0),
                    self.drain_event,
                )

            config["args"] = new_args

            p = mp.Process(target=config["target"], args=new_args)
            p.start()

            self.worker_info[idx] = (p, worker_id, stage_name)
            for i, proc in enumerate(self.processes):
                if proc == old_proc:
                    self.processes[i] = p
                    break

            _log(f"   Restarted {worker_id} with PID {p.pid}")

        _log(f"   Stage {stage_idx} restart complete")

    def restart(self, reason="ConnectionError"):
        self._stop(force=True)

        if self.stats_interval > 0:
            import multiprocessing

            self.manager = multiprocessing.Manager()
            self.timing_dict = self.manager.dict()

        self.start()
        _log(f"Pipeline restarted due to {reason}")

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
                    q.put(End, timeout=1)

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

        if self.profile and self.profile_dir and self.worker_info:
            rss_resolved = {}
            for wid, val in self.profile_rss.items():
                rss_resolved[wid] = val.value
            print_profile_summary(self.profile_dir, rss_resolved, self.worker_info)

        self.processes = []
        self.queues = []
        self.worker_info = []
        self.worker_configs = {}
        self.profile_rss = {}
        self.working.value = 0
        self.should_stop.value = 0
        self.restart_needed.value = 0
        self.drain_event.clear()

        for counter in self.stage_end_counters:
            counter.value = 0
        for event in self.stage_done_events:
            event.clear()

    def __iter__(self):
        if not self.started:
            self.start()

        if self.sequential:
            yield from self._sequential_run()
            return

        # Catch the first Ctrl+C at the OS level so it drains regardless of
        # whether Python is currently in pipe code or in the user's loop body.
        # The default handler is restored after one hit, so a second Ctrl+C
        # raises KeyboardInterrupt as usual and the iterator force-stops below.
        prev_handler = None
        is_main_thread = threading.current_thread() is threading.main_thread()
        if is_main_thread:
            def _drain_on_sigint(signum, frame):
                signal.signal(signal.SIGINT, prev_handler if prev_handler is not None else signal.SIG_DFL)
                self.drain_event.set()
                print("Starting to drain... (Ctrl+C again to force stop)", flush=True)
            prev_handler = signal.signal(signal.SIGINT, _drain_on_sigint)

        try:
            final_done = self.stage_done_events[-1]
            consecutive_empty = 0
            EMPTY_THRESHOLD = 5

            while True:
                if self.should_stop.value == 2:
                    _log("Worker encountered connection error - restarting pipeline...")
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
                    # Second Ctrl+C: default handler restored, KeyboardInterrupt fired.
                    _log("Force stopping pipeline")
                    self._stop(force=True)
                    return
        finally:
            if is_main_thread and prev_handler is not None:
                with contextlib.suppress(Exception):
                    signal.signal(signal.SIGINT, prev_handler)

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

    def _sequential_run(self):
        _log("Running in sequential mode")

        if not self.jobs:
            return

        workers = [job["func"] for job in self.jobs]

        run_stages = set()
        for i, (w, job) in enumerate(zip(workers, self.jobs)):
            if hasattr(w, "run") and i > 0:
                run_stages.add(i)

        for worker in workers:
            if hasattr(worker, "load"):
                worker.load()

        pending_items = []
        root_ended = False

        def _add_result(result, next_stage):
            if result is None:
                return
            if inspect.isgenerator(result):
                for res_item in result:
                    if res_item is not None and not _is_end(res_item):
                        pending_items.append((next_stage, res_item))
            elif _is_end(result):
                return
            elif isinstance(result, list):
                for res_item in result:
                    if res_item is not None and not _is_end(res_item):
                        pending_items.append((next_stage, res_item))
            else:
                pending_items.append((next_stage, result))

        def _process_callable(worker, stage_idx, item):
            job = self.jobs[stage_idx]
            batch_sz = job.get("batch", 0)

            if batch_sz > 0:
                batch = [item]
                while len(batch) < batch_sz and pending_items and pending_items[0][0] == stage_idx:
                    _, next_item = pending_items.pop(0)
                    batch.append(next_item)
                result = worker(batch)
            else:
                result = worker(item)
            _add_result(result, stage_idx + 1)

        while not root_ended or pending_items:
            if not root_ended:
                root_result = workers[0]()

                if inspect.isgenerator(root_result):
                    for root_item in root_result:
                        if root_item is not None and not _is_end(root_item):
                            pending_items.append((1, root_item))
                    root_ended = True
                elif root_result is None:
                    pass
                elif _is_end(root_result):
                    root_ended = True
                elif isinstance(root_result, list):
                    for root_item in root_result:
                        if _is_end(root_item):
                            root_ended = True
                            break
                        if root_item is not None:
                            pending_items.append((1, root_item))
                else:
                    pending_items.append((1, root_result))

            if pending_items:
                stage_idx, item = pending_items.pop(0)

                if stage_idx >= len(workers):
                    yield item
                    continue

                worker = workers[stage_idx]

                if stage_idx in run_stages:
                    # Collect all currently pending items for this stage
                    buffer = [item]
                    remaining = []
                    for si, it in pending_items:
                        if si == stage_idx:
                            buffer.append(it)
                        else:
                            remaining.append((si, it))
                    pending_items[:] = remaining

                    batch_sz = self.jobs[stage_idx].get("batch", 1) or 1

                    def _make_pull(buf, bs):
                        def _pull(n=bs):
                            items = buf[:n]
                            del buf[:n]
                            return items
                        return _pull

                    def _make_put(pi, next_stage):
                        def _put(item):
                            if item is not None:
                                pi.append((next_stage, item))
                        return _put

                    worker.pull = _make_pull(buffer, batch_sz)
                    worker.put = _make_put(pending_items, stage_idx + 1)
                    try:
                        worker.run()
                    except Exception as e:
                        if self.raise_errors:
                            raise
                        print(f"Error in run() worker at stage {stage_idx}: {e}")
                    continue

                try:
                    _process_callable(worker, stage_idx, item)
                except Exception as e:
                    if self.raise_errors:
                        raise
                    print(f"Error in worker at stage {stage_idx}: {e}, continuing...")

        # Flush callable workers (run() workers manage their own flushing)
        for stage_idx, worker in enumerate(workers[1:], 1):
            if stage_idx in run_stages:
                continue
            if hasattr(worker, "flush"):
                for flushed_item in worker.flush():
                    if flushed_item is not None:
                        pending_items.append((stage_idx + 1, flushed_item))

        # Process remaining items from flush
        while pending_items:
            stage_idx, item = pending_items.pop(0)

            if stage_idx >= len(workers):
                yield item
                continue

            worker = workers[stage_idx]
            try:
                _process_callable(worker, stage_idx, item)
            except Exception as e:
                if self.raise_errors:
                    raise
                continue
