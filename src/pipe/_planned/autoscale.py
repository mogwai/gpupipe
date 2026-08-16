"""Queue-pressure autoscaling — PLANNED, not wired into the live pipeline.

This module is preserved verbatim from when autoscaling was part of the core
code path. It is NOT imported by the package and nothing in `pipe` calls it;
it is kept here so the feature can be re-integrated later without archaeology.

What it does
------------
A background thread watches each autoscale-enabled stage's input-queue fill
ratio and spawns/stops worker processes to track load:

- Scale UP:   input fill >= 80% for 3 consecutive samples -> spawn a worker,
              unless CPU is saturated (>85%) or the output queue is >=90% full.
- Scale DOWN: input fill <= 20% for 5 consecutive samples -> signal one worker
              to stop after its current item (never below min_workers).
- Cooldown:   3s minimum between scaling actions per stage.
- GPU / CPU-pinned / threaded stages never autoscale (a static pin can't track
  a live worker count; GPU stages are capped by device count).

Re-integration checklist (what was removed from the live code to land this)
---------------------------------------------------------------------------
1. `Pipe.__init__` (pipe.py): re-add params `autoscale=False,
   max_workers_per_stage=8`; store `self.autoscale`, `self.max_workers_per_stage`,
   `self.autoscaler_thread = None`, `self.autoscaler_stop_event = threading.Event()`.
2. `Pipe.add` (pipe.py): re-add params `autoscale=None, min_workers=None,
   max_workers=None`; recompute `effective_max` (default `actual_workers * 4`,
   clamped to GPU-pool size for GPU stages, or `max_workers_per_stage` when
   autoscaling); resolve `stage_autoscale` (off for GPU/threaded/`cpus=` stages,
   else per-stage override, else global); store `autoscale`, `max_workers`,
   `min_workers` back onto the job dict.
3. `LifecycleMixin.start` (lifecycle.py): re-add the monotonic id allocator
   `self.stage_spawn_counts = [job["num_workers"] for job in self.jobs]` and,
   after the stats thread, start `_autoscaler_thread` when
   `any(job.get("autoscale") for job in self.jobs)`.
4. Worker run-loops (workers.py) already handle the `WorkerStop` sentinel
   (`pipe.types.WorkerStop`) that `_signal_worker_to_stop` puts on the queue —
   that inert primitive was left in place, so scale-down works as soon as this
   thread drives it.
5. Tests live in `tests/planned/test_autoscale.py` (currently skipped/ignored).
6. Live spawning moved to kwargs (`_spawn_and_register` in lifecycle.py):
   convert `_spawn_additional_worker`'s positional args tuples to kwargs dicts
   keyed by the run-loop parameter names, and register configs under "kwargs"
   (not "args") so `_restart_worker` can respawn autoscaled workers.

Drift warning: `_spawn_additional_worker` imports `_worker_run` /
`_threaded_worker_run` lazily (inside the function) and hand-builds their full
positional args tuples — importing this module does NOT validate either. The
live smoke test `tests/test_planned_preserved.py` checks the import and the
tuple arity against the live signatures; if it fails, this module has drifted.
"""
import os
import time

from ..types import WorkerStop


def _log(msg):
    if os.environ.get("PIPE_VERBOSE") == "1":
        print(msg)


def _get_cpu_usage():
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
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
    _log("Autoscaler started")

    high_pressure_counts = {}
    low_pressure_counts = {}
    last_scale_time = {}

    prev_idle, prev_total = _get_cpu_usage()
    cpu_saturated = False

    while not should_stop.value and not stop_event.is_set():
        time.sleep(check_interval)

        if should_stop.value or stop_event.is_set():
            break

        current_time = time.time()

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

            if stage_idx == 0:
                continue

            if stage_idx in last_scale_time:
                if current_time - last_scale_time[stage_idx] < cooldown_seconds:
                    continue

            try:
                total_workers = pipe_instance.stage_worker_counts[stage_idx].value
                finished_workers = pipe_instance.stage_end_counters[stage_idx].value
                active_workers = total_workers - finished_workers
                max_workers = job.get("max_workers", active_workers * 4)
                min_workers = job.get("min_workers", 1)

                num_queues = len(pipe_instance.queues)
                num_workers = len(pipe_instance.stage_worker_counts)
                if stage_idx - 1 >= num_queues:
                    continue
                if stage_idx >= num_workers:
                    continue
                if active_workers <= 0:
                    continue

                in_queue = pipe_instance.queues[stage_idx - 1]
                in_size = in_queue.qsize()
                in_max = in_queue._maxsize if hasattr(in_queue, "_maxsize") else 0
                if in_max <= 0:
                    continue

                in_fill = in_size / in_max

                out_queue = pipe_instance.queues[stage_idx] if stage_idx < len(pipe_instance.queues) else None
                out_blocked = False
                if out_queue:
                    out_size = out_queue.qsize()
                    out_max = out_queue._maxsize if hasattr(out_queue, "_maxsize") else 0
                    if out_max > 0:
                        out_fill = out_size / out_max
                        out_blocked = out_fill >= 0.9

                if active_workers < max_workers and in_fill >= scale_up_threshold:
                    if cpu_saturated:
                        high_pressure_counts[stage_idx] = 0
                        continue

                    if out_blocked:
                        high_pressure_counts[stage_idx] = 0
                        continue

                    high_pressure_counts[stage_idx] = high_pressure_counts.get(stage_idx, 0) + 1
                    low_pressure_counts[stage_idx] = 0

                    if high_pressure_counts[stage_idx] >= scale_up_samples:
                        high_pressure_counts[stage_idx] = 0
                        last_scale_time[stage_idx] = current_time
                        _log(f"   Autoscale UP: stage {stage_idx} ({job.get('name', '?')}) "
                             f"in_fill={in_fill:.0%} -> {active_workers + 1} workers")
                        _spawn_additional_worker(pipe_instance, stage_idx, job)

                elif active_workers > min_workers and in_fill <= scale_down_threshold:
                    low_pressure_counts[stage_idx] = low_pressure_counts.get(stage_idx, 0) + 1
                    high_pressure_counts[stage_idx] = 0

                    if low_pressure_counts[stage_idx] >= scale_down_samples:
                        low_pressure_counts[stage_idx] = 0
                        last_scale_time[stage_idx] = current_time
                        _log(f"   Autoscale DOWN: stage {stage_idx} ({job.get('name', '?')}) "
                             f"in_fill={in_fill:.0%} -> {active_workers - 1} workers")
                        _signal_worker_to_stop(pipe_instance, stage_idx)
                else:
                    high_pressure_counts[stage_idx] = 0
                    low_pressure_counts[stage_idx] = 0

            except Exception as e:
                import traceback
                print(f"Autoscaler error at stage {stage_idx}: {e}\n{traceback.format_exc()}")

    _log("Autoscaler stopped")


def _signal_worker_to_stop(pipe_instance, stage_idx):
    """Signal one worker at a stage to stop after completing current item."""
    in_queue = pipe_instance.queues[stage_idx - 1] if stage_idx > 0 else None
    if in_queue:
        try:
            in_queue.put(WorkerStop, timeout=0.1)
            # Decrement worker count so stage completion check works correctly
            with pipe_instance.stage_worker_counts[stage_idx].get_lock():
                pipe_instance.stage_worker_counts[stage_idx].value -= 1
            _log(f"   Signaled worker at stage {stage_idx} to stop")
        except Exception as e:
            _log(f"   Failed to signal worker stop: {e}")


def _spawn_additional_worker(pipe_instance, stage_idx, job):
    """Spawn an additional worker for the given stage."""
    from torch.multiprocessing import Process

    from ..workers import _cpu_chunk, _threaded_worker_run, _worker_run

    func = job["func"]
    is_threaded = job.get("thread", False)

    if hasattr(func, "__name__"):
        stage_name = func.__name__
    elif hasattr(func, "__class__"):
        stage_name = func.__class__.__name__
    else:
        stage_name = str(type(func).__name__)

    new_worker_idx = pipe_instance.stage_spawn_counts[stage_idx]
    pipe_instance.stage_spawn_counts[stage_idx] += 1
    worker_id = f"stage_{stage_idx}_worker_{new_worker_idx}"

    # cpus= stages force autoscale off (a static pin can't track a live worker count),
    # so this is normally None; computed defensively to keep the args arity consistent.
    cpu_affinity = _cpu_chunk(job.get("cpus"), new_worker_idx, job.get("num_workers", 1))

    _log(f"Autoscaling: Adding worker {new_worker_idx + 1} to {stage_name} (stage {stage_idx})")

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
            None,
            pipe_instance.timing_dict,
            worker_id,
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
            pipe_instance.sequential,
            job.get("batch", 0),
            pipe_instance.drain_event,
            job.get("drain", True),
            pipe_instance.queues,
            [j["name"] for j in pipe_instance.jobs],
            cpu_affinity,
            job.get("cpu_threads"),
            job.get("chunk_eff", 0),
            job.get("chunk_ms", 10.0),
        )
        proc = Process(target=_threaded_worker_run, args=args, daemon=True)
    else:
        args = (
            func,
            in_queue,
            out_queue,
            pipe_instance.should_stop,
            None,
            pipe_instance.timing_dict,
            worker_id,
            pipe_instance.raise_errors,
            pipe_instance.stage_end_counters[stage_idx],
            pipe_instance.stage_worker_counts[stage_idx],
            stage_idx,
            stage_name,
            is_final_stage,
            pipe_instance.expected_consumers,
            upstream_done,
            stage_done,
            pipe_instance.sequential,
            job.get("batch", 0),
            pipe_instance.drain_event,
            job.get("drain", True),
            pipe_instance.queues,
            [j["name"] for j in pipe_instance.jobs],
            cpu_affinity,
            job.get("cpu_threads"),
            job.get("chunk_eff", 0),
            job.get("chunk_ms", 10.0),
            None,  # scavenge_hold: GPU stages never autoscale
            None,  # scavenge_park: ditto
            None,  # scavenge_ipc: ditto
        )
        proc = Process(target=_worker_run, args=args, daemon=True)

    proc.start()
    _log(f"   Worker {worker_id} started with PID {proc.pid}")

    pipe_instance.processes.append(proc)
    pipe_instance.worker_info.append((proc, worker_id, stage_name))

    pipe_instance.worker_configs[worker_id] = {
        "target": _threaded_worker_run if is_threaded else _worker_run,
        "args": args,
        "stage_name": stage_name,
    }
