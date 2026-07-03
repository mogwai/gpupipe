"""Process lifecycle for a pipeline: spawning stage workers, health/stats/autoscale
monitor threads, per-worker and per-stage restart, and graceful/forced shutdown.

Mixed into `Pipe` (methods keep operating on `self`); split out to keep pipe.py
focused on the public API (init/add/iterate)."""
import contextlib
import os
import threading
import time
from queue import Empty, Full

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue, Value

from .monitors import (
    _autoscaler_thread,
    _health_monitor_thread,
    _stats_monitor_thread,
    _stats_monitor_thread_text,
)
from .profiling import _profile_dir, _profiled_worker, print_profile_summary
from .queues import InstrumentedQueue
from .types import End
from .workers import _cpu_chunk, _is_end, _threaded_worker_run, _worker_run


def _log(msg):
    if os.environ.get("PIPE_VERBOSE") == "1":
        print(msg)


class LifecycleMixin:
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

        # Stage 0 has no input queue, so it always stops on drain.
        self.jobs[0]["drain"] = False
        # drain=False stages must form a contiguous prefix (e.g. 0,1,2 — not 0,3).
        seen_drain = False
        for i, job in enumerate(self.jobs):
            if job["drain"]:
                seen_drain = True
            elif seen_drain:
                raise ValueError(
                    f"Stage {i} ({job['name']}) has drain=False but an earlier stage has drain=True. "
                    "drain=False stages must form a contiguous prefix from stage 0."
                )

        if self.sequential:
            _log("Sequential mode - skipping multiprocessing setup")
            return self

        if self.profile:
            self.profile_dir = _profile_dir()
            print(f"Profiling enabled, saving to {self.profile_dir}")

        _log("Setting up multiprocessing...")

        # Monotonic per-stage id allocator for autoscaled workers. Never reuses an
        # index (unlike the live stage_worker_counts, which is decremented on
        # scale-down) so spawned workers can't collide with a still-live worker id.
        self.stage_spawn_counts = [job["num_workers"] for job in self.jobs]

        for i, job in enumerate(self.jobs):
            # Resolve output-edge chunking. Explicit chunk= wins (0 = force off);
            # otherwise auto-adopt the DOWNSTREAM stage's batch size, so an edge
            # feeding a batch=B stage moves one B-item Chunk per queue op instead
            # of B separate messages. Stored on the job so autoscale/restart
            # respawns keep the same transport config.
            if job.get("chunk") is not None:
                job["chunk_eff"] = max(0, int(job["chunk"]))
            elif i + 1 < len(self.jobs) and self.jobs[i + 1].get("batch", 0) > 1:
                job["chunk_eff"] = self.jobs[i + 1]["batch"]
            else:
                job["chunk_eff"] = 0
            if job["chunk_eff"] > 0:
                _log(
                    f"Stage {i} ({job['name']}): output chunked x{job['chunk_eff']} "
                    f"(flush {job.get('chunk_ms', 10.0)}ms)"
                )

            # outqn is specified in ITEMS. On a chunked edge each queue slot holds
            # up to chunk_eff items, so scale maxsize down to keep the same item
            # capacity (and the same backpressure point). 0 stays unbounded.
            outq_size = job.get("outqn") or 0
            if outq_size and job["chunk_eff"] > 1:
                outq_size = max(1, outq_size // job["chunk_eff"])
            q = Queue(maxsize=outq_size)
            self.queues.append(InstrumentedQueue(q) if self.debug else q)
            self.stage_end_counters.append(Value("i", 0))
            self.stage_done_events.append(Event())

            if job.get("thread", False) and not job.get("pergpu", False):
                worker_count = 1
            else:
                worker_count = job["num_workers"]
            self.stage_worker_counts.append(Value("i", worker_count))

        # Names of every stage, in order, so workers can resolve a push(stage,...)
        # target by name as well as by index.
        stage_names_all = [job["name"] for job in self.jobs]

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
                if job.get("gpus"):
                    gpu_pool = job["gpus"]
                    for worker_idx in range(job["num_workers"]):
                        num_threads = 1
                        gpu_id = gpu_pool[worker_idx % len(gpu_pool)]

                        worker_id = f"stage_{i}_threaded_worker_{worker_idx}_gpu_{gpu_id}"
                        cpu_affinity = _cpu_chunk(job.get("cpus"), worker_idx, job["num_workers"])

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
                            job.get("drain", True),
                            self.queues,
                            stage_names_all,
                            cpu_affinity,
                            job.get("cpu_threads"),
                            job.get("chunk_eff", 0),
                            job.get("chunk_ms", 10.0),
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
                    # One process runs all threads, so it owns the whole pool; _setup_cpu
                    # sizes its thread count to len(cpus).
                    cpu_affinity = job.get("cpus")

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
                        job.get("drain", True),
                        self.queues,
                        stage_names_all,
                        cpu_affinity,
                        job.get("cpu_threads"),
                        job.get("chunk_eff", 0),
                        job.get("chunk_ms", 10.0),
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
                gpu_pool = job.get("gpus")
                for worker_idx in range(job["num_workers"]):
                    gpu_id = gpu_pool[worker_idx % len(gpu_pool)] if gpu_pool else None

                    worker_id = f"stage_{i}_worker_{worker_idx}"
                    if gpu_id is not None:
                        worker_id += f"_gpu_{gpu_id}"
                    cpu_affinity = _cpu_chunk(job.get("cpus"), worker_idx, job["num_workers"])

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
                        job.get("drain", True),
                        self.queues,
                        stage_names_all,
                        cpu_affinity,
                        job.get("cpu_threads"),
                        job.get("chunk_eff", 0),
                        job.get("chunk_ms", 10.0),
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
                    job.get("drain", True),
                    self.queues,
                    [j["name"] for j in self.jobs],
                    old_args[24],  # cpu_affinity slice, unchanged across restart
                    old_args[25],  # cpu_threads, unchanged across restart
                    job.get("chunk_eff", 0),
                    job.get("chunk_ms", 10.0),
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
                    job.get("drain", True),
                    self.queues,
                    [j["name"] for j in self.jobs],
                    old_args[23],  # cpu_affinity slice, unchanged across restart
                    old_args[24],  # cpu_threads, unchanged across restart
                    job.get("chunk_eff", 0),
                    job.get("chunk_ms", 10.0),
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
