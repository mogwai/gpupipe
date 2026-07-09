"""Profiling support for Pipe workers.

When profile=True, each worker process runs under cProfile and tracks peak RSS.
Results are saved to a temp directory and summarized at pipeline stop.
"""
import cProfile
import os
import pstats
import resource
import tempfile
from io import StringIO


def _profile_dir():
    d = os.path.join(tempfile.gettempdir(), f"pipe_profile_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    return d


def _profiled_worker(target, kwargs, profile_dir, worker_id, peak_rss_kb):
    prof = cProfile.Profile()
    prof.enable()
    try:
        target(**kwargs)
    finally:
        prof.disable()
        prof_path = os.path.join(profile_dir, f"{worker_id}.prof")
        prof.dump_stats(prof_path)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with peak_rss_kb.get_lock():
            peak_rss_kb.value = rss


def print_profile_summary(profile_dir, profile_rss, worker_info):
    print("\n=== Profile Summary ===\n")

    rows = []
    for _, worker_id, stage_name in worker_info:
        prof_path = os.path.join(profile_dir, f"{worker_id}.prof")
        rss_kb = profile_rss.get(worker_id)
        rss_mb = rss_kb / 1024 if rss_kb else 0

        if os.path.exists(prof_path):
            stats = pstats.Stats(prof_path, stream=StringIO())
            stats.sort_stats("cumulative")
            total_time = stats.total_tt  # total CPU time across all functions

            sio = StringIO()
            ps = pstats.Stats(prof_path, stream=sio)
            ps.sort_stats("cumulative")
            ps.print_stats(5)
            top_funcs = sio.getvalue()

            rows.append((worker_id, stage_name, total_time, rss_mb, top_funcs))
        else:
            rows.append((worker_id, stage_name, 0, rss_mb, ""))

    print(f"{'worker':<40} {'stage':<20} {'cpu_time':>10} {'peak_rss':>10}")
    print("-" * 84)
    for worker_id, stage_name, cpu_time, rss_mb, _ in rows:
        print(f"{worker_id:<40} {stage_name:<20} {cpu_time:>9.2f}s {rss_mb:>8.0f}MB")

    seen_stages = set()
    print("\n=== Top Functions by Stage ===\n")
    for worker_id, stage_name, _, _, top_funcs in rows:
        if stage_name in seen_stages or not top_funcs:
            continue
        seen_stages.add(stage_name)
        print(f"--- {stage_name} ({worker_id}) ---")
        lines = top_funcs.strip().split("\n")
        for line in lines:
            if line.strip():
                print(f"  {line}")
        print()

    print(f"Profile data saved to: {profile_dir}")
    print(f"Inspect with: python -c \"import pstats; p = pstats.Stats('{profile_dir}/<worker>.prof'); p.sort_stats('cumulative').print_stats(20)\"")
    print()
