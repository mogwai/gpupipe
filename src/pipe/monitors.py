import os
import re
import time

from .workers import _signal_worker_to_stop, _spawn_additional_worker


def _log(msg):
    if os.environ.get("PIPE_VERBOSE") == "1":
        print(msg)


def _health_monitor_thread(
    pipe_instance,
    should_stop,
    health_check_interval,
    stop_event,
):
    _log("Health monitor starting up")

    crash_history = {}
    CRASH_WINDOW = 300
    MAX_CRASHES_IN_WINDOW = 3

    while not should_stop.value and not stop_event.is_set():
        time.sleep(health_check_interval)

        if should_stop.value or stop_event.is_set():
            break

        crashed_workers = []

        for idx, (proc, worker_id, stage_name) in enumerate(pipe_instance.worker_info):
            if not proc.is_alive():
                exitcode = proc.exitcode
                if exitcode is not None and exitcode != 0:
                    crashed_workers.append((idx, worker_id, stage_name, exitcode))

        if crashed_workers:
            print(f"\nHEALTH CHECK: {len(crashed_workers)} worker(s) have crashed:")
            for idx, worker_id, stage_name, exitcode in crashed_workers:
                print(f"   - {worker_id} ({stage_name}): exitcode={exitcode}")

            need_full_restart = False
            current_time = time.time()

            for idx, worker_id, stage_name, exitcode in crashed_workers:
                if worker_id not in crash_history:
                    crash_history[worker_id] = []
                crash_history[worker_id].append((current_time, exitcode))

                crash_history[worker_id] = [
                    (t, e)
                    for t, e in crash_history[worker_id]
                    if current_time - t < CRASH_WINDOW
                ]

                if exitcode in (-11, -7):  # SIGSEGV, SIGBUS
                    print(f"   Serious error detected (exitcode {exitcode})")
                    need_full_restart = True
                    break
                if len(crash_history[worker_id]) >= MAX_CRASHES_IN_WINDOW:
                    print(
                        f"   Repeated crashes detected ({len(crash_history[worker_id])} in {CRASH_WINDOW}s)"
                    )
                    need_full_restart = True
                    break

            if need_full_restart and pipe_instance.allow_full_restart:
                print("   Triggering full pipeline restart to recreate queues...")
                try:
                    pipe_instance.restart(reason="Worker crash requiring queue refresh")
                    crash_history.clear()
                    print("   Full pipeline restart complete")
                except Exception as e:
                    print(f"   Failed to restart pipeline: {e}")
            elif need_full_restart:
                print(
                    "   Full restart needed but disabled - restarting workers individually"
                )
                for idx, worker_id, stage_name, exitcode in crashed_workers:
                    try:
                        pipe_instance._restart_worker(idx, worker_id)
                        print(f"   Restarted {worker_id}")
                    except Exception as e:
                        print(f"   Failed to restart {worker_id}: {e}")
            else:
                print("   Restarting crashed workers individually...")
                for idx, worker_id, stage_name, exitcode in crashed_workers:
                    try:
                        pipe_instance._restart_worker(idx, worker_id)
                        print(f"   Restarted {worker_id}")
                    except Exception as e:
                        print(f"   Failed to restart {worker_id}: {e}")

    _log("Health monitor shutting down")


def _queue_bar(qsize, qmax, width=10):
    if not qmax:
        return ""
    fill = qsize / qmax if qmax > 0 else 0
    filled = int(fill * width)
    bar = "|" * filled + " " * (width - filled)
    if fill >= 0.8:
        color = "red"
    elif fill >= 0.5:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]\\[{bar}][/{color}] {qsize}/{qmax}"


def _collect_stats(pipe_instance):
    stages = {}
    for worker_id, stats in dict(pipe_instance.timing_dict).items():
        match = re.match(r"stage_(\d+)", worker_id)
        stage_idx = int(match.group(1)) if match else -1
        if stage_idx not in stages:
            stages[stage_idx] = []
        stages[stage_idx].append((worker_id, stats))

    now = time.time()
    result = []

    for stage_idx in range(len(pipe_instance.jobs)):
        done = pipe_instance.stage_done_events[stage_idx].is_set()

        q = pipe_instance.queues[stage_idx]
        try:
            qsize = q.qsize()
            qmax = q._maxsize if hasattr(q, "_maxsize") else 0
            if qmax > 1000000:
                qmax = 0
        except Exception:
            qsize = 0
            qmax = 0

        # On a chunked edge each queue message holds up to chunk_eff items, so
        # scale the display back to ITEMS (upper-bound estimate). The fill
        # ratio — what the autoscaler thresholds use — is unchanged by this.
        chunk_eff = pipe_instance.jobs[stage_idx].get("chunk_eff", 0)
        if chunk_eff > 1:
            qsize *= chunk_eff
            qmax *= chunk_eff

        stage_items = 0
        stage_rtf = 0.0
        avg_worker_rtf = 0.0
        has_audio = False

        if stage_idx in stages:
            stage_earliest_start = None
            worker_rtfs = []
            stage_total_audio = 0.0

            for _, stats in stages[stage_idx]:
                stage_items += stats.get("items", 0)
                audio_dur = stats.get("audio_duration", 0)
                start_wall = stats.get("start_wall_time")
                stage_total_audio += audio_dur

                if start_wall is not None:
                    worker_elapsed = now - start_wall
                    if worker_elapsed > 0:
                        worker_rtfs.append(audio_dur / worker_elapsed)
                    if stage_earliest_start is None or start_wall < stage_earliest_start:
                        stage_earliest_start = start_wall

            wall_elapsed = now - stage_earliest_start if stage_earliest_start else 0
            stage_rtf = stage_total_audio / wall_elapsed if wall_elapsed > 0 else 0
            avg_worker_rtf = sum(worker_rtfs) / len(worker_rtfs) if worker_rtfs else 0
            has_audio = stage_total_audio > 0

        total_workers = pipe_instance.stage_worker_counts[stage_idx].value
        finished = pipe_instance.stage_end_counters[stage_idx].value
        active = total_workers - finished

        result.append({
            "stage_idx": stage_idx,
            "done": done,
            "qsize": qsize,
            "qmax": qmax,
            "items": stage_items,
            "active": active,
            "total_workers": total_workers,
            "stage_rtf": stage_rtf,
            "avg_worker_rtf": avg_worker_rtf,
            "has_audio": has_audio,
        })

    return result


def _stats_monitor_thread(pipe_instance, stop_event, interval_seconds=30):
    from rich.live import Live
    from rich.table import Table

    console = pipe_instance._rich_console
    prev_snapshot = {}
    rates = {}
    EMA_ALPHA = 0.3

    def build_table():
        all_stats = _collect_stats(pipe_instance)
        now = time.time()
        any_audio = any(s["has_audio"] for s in all_stats)

        table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        table.add_column("Stage", style="cyan", min_width=14)
        table.add_column("Items", justify="right", min_width=8)
        table.add_column("Rate", justify="right", min_width=8)
        table.add_column("Queue", min_width=18)
        table.add_column("Wkrs", justify="right", min_width=6)
        if any_audio:
            table.add_column("RTF", justify="right", min_width=10)

        for s in all_stats:
            if s["done"]:
                continue
            idx = s["stage_idx"]
            items = s["items"]

            prev_time, prev_items = prev_snapshot.get(idx, (now, items))
            dt = now - prev_time
            if dt > 0 and prev_time != now:
                instant_rate = (items - prev_items) / dt
                old_rate = rates.get(idx, instant_rate)
                rates[idx] = EMA_ALPHA * instant_rate + (1 - EMA_ALPHA) * old_rate
            prev_snapshot[idx] = (now, items)

            rate = rates.get(idx, 0)
            name = pipe_instance.jobs[idx]["name"]
            items_str = f"{items:,}"
            rate_str = f"{rate:,.0f}/s" if rate > 0 else ""
            queue_str = _queue_bar(s["qsize"], s["qmax"])
            wkrs_str = f"{s['active']}/{s['total_workers']}"

            row = [name, items_str, rate_str, queue_str, wkrs_str]
            if any_audio:
                if s["has_audio"]:
                    row.append(f"{s['stage_rtf']:.0f}/{s['avg_worker_rtf']:.0f}x")
                else:
                    row.append("")
            table.add_row(*row)

        return table

    with Live(build_table(), console=console, refresh_per_second=4) as live:
        while not stop_event.is_set():
            stop_event.wait(interval_seconds)
            if stop_event.is_set():
                break
            if not pipe_instance.timing_dict:
                continue
            live.update(build_table())


def _stats_monitor_thread_text(pipe_instance, stop_event, interval_seconds=30):
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"

    def queue_color(qsize, qmax):
        if not qmax:
            return DIM
        fill = qsize / qmax
        if fill > 0.5:
            return GREEN
        elif fill > 0.2:
            return YELLOW
        return RED

    while not stop_event.is_set():
        stop_event.wait(interval_seconds)
        if stop_event.is_set():
            break

        if not pipe_instance.timing_dict:
            continue

        all_stats = _collect_stats(pipe_instance)

        total_items = 0
        final_idx = len(pipe_instance.jobs) - 1
        for s in all_stats:
            if s["stage_idx"] == final_idx:
                total_items = s["items"]

        parts = []
        for s in all_stats:
            if s["done"]:
                continue
            name = pipe_instance.jobs[s["stage_idx"]]["name"][:4]
            qc = queue_color(s["qsize"], s["qmax"])
            q_str = f"{s['qsize']}/{s['qmax']}" if s["qmax"] else str(s["qsize"])
            parts.append(f"{CYAN}{name}{RESET}|{qc}{q_str}{RESET}|{s['stage_rtf']:.0f}/{s['avg_worker_rtf']:.0f}x")

        print(f"[{total_items}] " + "\u25b8".join(parts))

    _log("Stats monitor shutting down")


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
                        _log(f"   Autoscale UP: stage {stage_idx} ({job.get('name', '?')}) in_fill={in_fill:.0%} -> {active_workers + 1} workers")
                        _spawn_additional_worker(pipe_instance, stage_idx, job)

                elif active_workers > min_workers and in_fill <= scale_down_threshold:
                    low_pressure_counts[stage_idx] = low_pressure_counts.get(stage_idx, 0) + 1
                    high_pressure_counts[stage_idx] = 0

                    if low_pressure_counts[stage_idx] >= scale_down_samples:
                        low_pressure_counts[stage_idx] = 0
                        last_scale_time[stage_idx] = current_time
                        _log(f"   Autoscale DOWN: stage {stage_idx} ({job.get('name', '?')}) in_fill={in_fill:.0%} -> {active_workers - 1} workers")
                        _signal_worker_to_stop(pipe_instance, stage_idx)
                else:
                    high_pressure_counts[stage_idx] = 0
                    low_pressure_counts[stage_idx] = 0

            except Exception as e:
                import traceback
                print(f"Autoscaler error at stage {stage_idx}: {e}\n{traceback.format_exc()}")

    _log("Autoscaler stopped")
