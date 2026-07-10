"""Tests for observability & recovery surfaces: get_stats, profile=True,
health-monitor crash restart, and bounded push(timeout=).

Run with: pytest test_observability.py -q
"""
import glob
import os
import time

from conftest import Collector, FastWorker, Generator, SlowWorker

from pipe import Pipe

# === get_stats ===

def test_get_stats_live_fields():
    """get_stats() must report queue/worker fields per stage while running,
    and per-worker item counts when stats are enabled (akro polls this)."""
    pipe = Pipe(stats_interval=0.1, stats_mode="external", health_check_interval=0)
    pipe.add(Generator(60, delay=0.01), outqn=50)
    pipe.add(SlowWorker(0.01), workers=2, outqn=50)
    pipe.add(Collector(), workers=1, outqn=0)

    snapshots = []
    for _ in pipe:
        snapshots.append(pipe.get_stats())

    assert snapshots, "no stats collected"
    last = snapshots[-1]
    assert len(last) == 3
    for s in last:
        for key in ("stage_idx", "done", "qsize", "qmax", "items", "active", "total_workers"):
            assert key in s, f"missing {key}"
    # with stats enabled, per-worker timing flows into items counts eventually
    assert any(s["items"] > 0 for snap in snapshots for s in snap), \
        "no per-worker item counts despite stats_interval > 0"


def test_get_stats_without_stats_interval():
    """stats_interval=0: no Manager/timing_dict exists — get_stats must still
    return queue/worker fields (items are 0) instead of raising."""
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Generator(5), outqn=10)
    pipe.add(Collector(), workers=1, outqn=0)
    pipe.start()
    try:
        stats = pipe.get_stats()
        assert len(stats) == 2
        assert all(s["items"] == 0 for s in stats)
    finally:
        list(pipe)  # drain to completion


# === profile=True ===

def test_profile_writes_prof_files_and_summary(capsys):
    pipe = Pipe(stats_interval=0, health_check_interval=0, profile=True)
    pipe.add(Generator(20), outqn=50)
    pipe.add(FastWorker(), workers=2, outqn=0)
    n = sum(1 for _ in pipe)
    assert n == 20
    out = capsys.readouterr().out
    assert "Profile Summary" in out
    profile_dir = pipe.profile_dir
    assert profile_dir and glob.glob(os.path.join(profile_dir, "*.prof")), \
        "no .prof files written"


# === health monitor: crash detection -> automatic worker restart ===

class CrashOnce:
    """Hard-exits the whole worker process on the first item, exactly once
    (coordinated via a flag file so the restarted worker proceeds normally)."""

    def __init__(self, flag_path):
        self.flag_path = flag_path

    def __call__(self, item):
        if not os.path.exists(self.flag_path):
            open(self.flag_path, "w").write("crashed")
            os._exit(1)  # simulate segfault-style death (nonzero exitcode)
        item["survived"] = True
        return item


def test_health_monitor_restarts_crashed_worker(tmp_path):
    flag = str(tmp_path / "crashed.flag")
    pipe = Pipe(stats_interval=0, health_check_interval=1, raise_errors=False)
    pipe.add(Generator(30, delay=0.05), outqn=5)   # slow source keeps the run alive
    pipe.add(CrashOnce(flag), workers=1, outqn=50)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    assert os.path.exists(flag), "worker never crashed — test is vacuous"
    # the crashed worker took one item down with it; the restarted worker
    # must process the rest (>= proves detection + restart happened)
    assert len(results) >= 25, f"only {len(results)} items — worker not restarted?"
    assert all(r.get("survived") for r in results)


# === push(timeout=): bounded, returns False instead of deadlocking ===

def test_push_timeout_returns_false_on_full_queue():
    """A push into a permanently-full queue must give up after `timeout`
    seconds and report False — the unbounded variant deadlocks the pipe
    (see the push() docstring and the drained-target case)."""
    from queue import Queue

    from pipe.workers import _make_push

    target = Queue(maxsize=1)
    target.put("occupied")  # nobody will ever consume this
    push = _make_push([target, Queue()], ["A", "B"], should_stop=None)

    t0 = time.monotonic()
    landed = push(1, {"x": 1}, timeout=0.5)
    elapsed = time.monotonic() - t0
    assert landed is False
    assert 0.4 < elapsed < 3.0, f"timeout not honoured ({elapsed:.2f}s)"

    # non-blocking variant reports the drop immediately
    assert push(1, {"x": 2}, block=False) is False
    # and a push that fits reports True
    target.get()
    assert push(1, {"x": 3}, timeout=0.5) is True
