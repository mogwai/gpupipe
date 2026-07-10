import re
import time

from .utils import _is_tty, _log, _pin_stats_line, _unpin_stats_line


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
            else:
                if need_full_restart:
                    print("   Full restart needed but disabled - restarting workers individually")
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
    fill = qsize / qmax
    filled = int(fill * width)
    bar = "|" * filled + " " * (width - filled)
    if fill >= 0.8:
        color = "red"
    elif fill >= 0.5:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]\\[{bar}][/{color}] {qsize}/{qmax}"


_UNBOUNDED_QMAX = 1_000_000


def _collect_stats(pipe_instance):
    stages = {}
    td = pipe_instance.timing_dict
    for worker_id, stats in (dict(td) if td is not None else {}).items():
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
            if qmax > _UNBOUNDED_QMAX:
                qmax = 0  # torch mp queues report a huge maxsize when unbounded
        except Exception:
            qsize = 0
            qmax = 0

        # On a chunked edge each queue message holds up to chunk_eff items, so
        # scale the display back to ITEMS (upper-bound estimate). The fill
        # ratio is unchanged by this.
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
            if dt > 0:
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
        if fill >= 0.8:
            return RED
        elif fill >= 0.5:
            return YELLOW
        return GREEN

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

        line = f"[{total_items}] " + "\u25b8".join(parts)
        if _is_tty():
            # Repaint one pinned line in place; print_above()/Pipe.print() write
            # messages on their own line above it.
            _pin_stats_line(line)
        else:
            print(line)

    _unpin_stats_line()
    _log("Stats monitor shutting down")
