"""Process lifecycle for a pipeline: spawning stage workers, health/stats
monitor threads, per-worker and per-stage restart, and graceful/forced shutdown.

Mixed into `Pipe` (methods keep operating on `self`); split out to keep pipe.py
focused on the public API (init/add/iterate)."""
import contextlib
import os
import threading
import time
from queue import Full

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue, Value

from .monitors import (
    _health_monitor_thread,
    _stats_monitor_thread,
    _stats_monitor_thread_text,
)
from .profiling import _profile_dir, _profiled_worker, print_profile_summary
from .queues import InstrumentedQueue
from .types import End
from .utils import _log
from .workers import _cpu_chunk, _threaded_worker_run, _worker_run


class LifecycleMixin:
    def _spawn(self, target, kwargs, worker_id):
        if self.profile:
            from multiprocessing import Value as MpValue
            rss_val = MpValue("l", 0)
            self.profile_rss[worker_id] = rss_val
            p = mp.Process(
                target=_profiled_worker,
                args=(target, kwargs, self.profile_dir, worker_id, rss_val),
            )
        else:
            p = mp.Process(target=target, kwargs=kwargs)
        p.start()
        return p

    def _spawn_and_register(self, target, kwargs):
        worker_id = kwargs["worker_id"]
        stage_name = kwargs["stage_name"]
        _log(f"Starting worker: {stage_name} ({worker_id})")
        self.worker_configs[worker_id] = {
            "target": target,
            "kwargs": kwargs,
            "stage_name": stage_name,
        }
        p = self._spawn(target, kwargs, worker_id)
        self.processes.append(p)
        self.worker_info.append((p, worker_id, stage_name))
        _log(f"Worker {stage_name} ({worker_id}) started with PID {p.pid}")
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

        if self.manager is not None and self.stats_interval > 0:
            pass  # reusing existing manager (restart path)
        elif self.stats_interval > 0:
            import multiprocessing
            self.manager = multiprocessing.Manager()
            self.timing_dict = self.manager.dict()

        if self.profile:
            self.profile_dir = _profile_dir()
            print(f"Profiling enabled, saving to {self.profile_dir}")

        _log("Setting up multiprocessing...")

        for i, job in enumerate(self.jobs):
            # Resolve output-edge chunking. Explicit chunk= wins (0 = force off);
            # otherwise auto-adopt the DOWNSTREAM stage's batch size, so an edge
            # feeding a batch=B stage moves one B-item Chunk per queue op instead
            # of B separate messages. Stored on the job so restart respawns keep
            # the same transport config.
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

            # Threaded stage without a GPU pool = ONE process running N threads;
            # every other shape spawns num_workers processes. Must mirror the
            # spawn branches below (keyed on "gpus", the canonical pool).
            if job.get("thread", False) and not job.get("gpus"):
                worker_count = 1
            else:
                worker_count = job["num_workers"]
            self.stage_worker_counts.append(Value("i", worker_count))

        # Names of every stage, in order, so workers can resolve a push(stage,...)
        # target by name as well as by index.
        stage_names_all = [job["name"] for job in self.jobs]

        # Scavenge stages: probe GPU occupancy ONCE before spawning so workers
        # whose GPU is already claimed start held (they wait before load() and
        # touch no CUDA state). Rebuilt from scratch on every start()/restart.
        self.scavenge_slots = []
        any_scavenge = any(job.get("scavenge") for job in self.jobs)
        busy_at_start = set()
        if any_scavenge:
            from .scavenge import busy_physical_gpus, physical_gpu

            busy_at_start = busy_physical_gpus({os.getpid()})
            if busy_at_start:
                _log(f"Scavenge: GPUs {sorted(busy_at_start)} busy at start")

        for i, job in enumerate(self.jobs):
            # Everything the worker run-loops take, by PARAMETER NAME. Spawning
            # with kwargs (not positional tuples) means adding a parameter can't
            # silently misalign the call — a mismatch raises TypeError at spawn.
            common = dict(
                worker=job["func"],
                in_queue=self.queues[i - 1] if i > 0 else None,
                out_queue=self.queues[i] if i < len(self.queues) else None,
                should_stop=self.should_stop,
                timing_dict=self.timing_dict,
                raise_errors=self.raise_errors,
                stage_end_counter=self.stage_end_counters[i],
                stage_worker_count=self.stage_worker_counts[i],
                stage_idx=i,
                stage_name=job["name"],
                is_final_stage=i == len(self.jobs) - 1,
                expected_consumers=self.expected_consumers,
                upstream_done=self.stage_done_events[i - 1] if i > 0 else None,
                stage_done=self.stage_done_events[i],
                sequential=self.sequential,
                batch_size=job.get("batch", 0),
                drain_event=self.drain_event,
                drain=job.get("drain", True),
                all_queues=self.queues,
                stage_names=stage_names_all,
                cpu_threads=job.get("cpu_threads"),
                out_chunk=job.get("chunk_eff", 0),
                out_chunk_ms=job.get("chunk_ms", 10.0),
            )

            gpu_pool = job.get("gpus")
            if job.get("thread", False):
                if gpu_pool:
                    # One single-threaded process per GPU worker.
                    for worker_idx in range(job["num_workers"]):
                        gpu_id = gpu_pool[worker_idx % len(gpu_pool)]
                        self._spawn_and_register(_threaded_worker_run, dict(
                            common,
                            gpu_id=gpu_id,
                            num_threads=1,
                            worker_id=f"stage_{i}_threaded_worker_{worker_idx}_gpu_{gpu_id}",
                            cpu_affinity=_cpu_chunk(job.get("cpus"), worker_idx, job["num_workers"]),
                        ))
                else:
                    # One process runs all threads, so it owns the whole cpus=
                    # pool; _setup_cpu sizes its thread count to len(cpus).
                    num_threads = job["num_workers"]
                    self._spawn_and_register(_threaded_worker_run, dict(
                        common,
                        gpu_id=None,
                        num_threads=num_threads,
                        worker_id=f"stage_{i}_threaded_{num_threads}threads",
                        cpu_affinity=job.get("cpus"),
                    ))
            else:
                for worker_idx in range(job["num_workers"]):
                    gpu_id = gpu_pool[worker_idx % len(gpu_pool)] if gpu_pool else None
                    worker_id = f"stage_{i}_worker_{worker_idx}"
                    if gpu_id is not None:
                        worker_id += f"_gpu_{gpu_id}"
                    scavenge_hold = scavenge_park = None
                    if job.get("scavenge") and gpu_id is not None:
                        phys = physical_gpu(gpu_id)
                        scavenge_hold = Value("i", 1 if phys in busy_at_start else 0)
                        if scavenge_hold.value:
                            _log(
                                f"Scavenge: {worker_id} starting held "
                                f"(GPU {phys} busy)"
                            )
                        # A stage that can release and rebuild its own GPU
                        # state gives the GPU back in seconds instead of
                        # paying a full VRAM->host checkpoint copy.
                        supports_park = hasattr(job["func"], "on_park")
                        scavenge_park = Value("i", 0) if supports_park else None
                        self.scavenge_slots.append({
                            "stage_name": job["name"],
                            "worker_id": worker_id,
                            "gpu": gpu_id,
                            "physical": phys,
                            "hold": scavenge_hold,
                            "park": scavenge_park,
                            "supports_park": supports_park,
                            "parked": False,
                            "frozen": False,
                            "frozen_mib": 0,
                        })
                    self._spawn_and_register(_worker_run, dict(
                        common,
                        gpu_id=gpu_id,
                        worker_id=worker_id,
                        cpu_affinity=_cpu_chunk(job.get("cpus"), worker_idx, job["num_workers"]),
                        scavenge_hold=scavenge_hold,
                        scavenge_park=scavenge_park,
                    ))

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

        if self.scavenge_slots:
            from .scavenge import _scavenger_thread

            self.scavenger_stop_event.clear()
            self.scavenger_thread = threading.Thread(
                target=_scavenger_thread,
                args=(
                    self,
                    self.scavenger_stop_event,
                    self.scavenge_poll,
                    self.scavenge_free_secs,
                ),
                daemon=True,
            )
            self.scavenger_thread.start()
            _log(
                f"Scavenger thread started ({len(self.scavenge_slots)} slot(s), "
                f"poll {self.scavenge_poll}s)"
            )

        if self.stats_interval > 0 and self.stats_mode != "external":
            if self.stats_mode == "text":
                monitor_fn = _stats_monitor_thread_text
            else:
                from rich.console import Console
                self._rich_console = Console()
                monitor_fn = _stats_monitor_thread
            self.stats_monitor_thread = threading.Thread(
                target=monitor_fn,
                args=(self, self.stats_monitor_stop_event, self.stats_interval),
                daemon=True,
            )
            self.stats_monitor_thread.start()

        return self

    def _restart_worker(self, worker_idx, worker_id):
        if worker_id not in self.worker_configs:
            raise ValueError(f"No configuration found for worker {worker_id}")

        config = self.worker_configs[worker_id]

        old_proc, _, _ = self.worker_info[worker_idx]
        with contextlib.suppress(Exception):
            old_proc.join(timeout=1)

        p = self._spawn(config["target"], config["kwargs"], worker_id)

        self.worker_info[worker_idx] = (p, worker_id, config["stage_name"])
        for i, proc in enumerate(self.processes):
            if proc == old_proc:
                self.processes[i] = p
                break

        _log(f"Worker {worker_id} restarted with PID {p.pid}")

    def restart(self, reason="ConnectionError"):
        self._stop(force=True)

        self.start()  # recreates the manager lazily if stats are enabled
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

        if self.scavenger_thread is not None and self.scavenger_thread.is_alive():
            self.scavenger_stop_event.set()
            self.scavenger_thread.join(timeout=2)
            self.scavenger_thread = None

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
                self.timing_dict = None
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
        self.should_stop.value = 0
        self.drain_event.clear()

        for counter in self.stage_end_counters:
            counter.value = 0
        for event in self.stage_done_events:
            event.clear()
