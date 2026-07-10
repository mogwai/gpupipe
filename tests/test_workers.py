"""Worker type tests - threaded, process, filter, expand workers.

Run with: pytest test_workers.py -n auto
"""
import pytest
from conftest import (
    Batcher,
    BatchGenerator,
    Collector,
    ExpandWorker,
    FastWorker,
    FilterWorker,
    Generator,
    SlowWorker,
    run_pipeline,
)

from pipe import Pipe

# === THREADED WORKER TESTS ===

@pytest.mark.parametrize("run", range(5))
def test_threaded_workers(run):
    """Threaded workers - run multiple times to catch race conditions."""
    n_items = 100
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (SlowWorker(0.02), {"workers": 4, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_process_workers(run):
    """Process workers - run multiple times to catch race conditions."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 100}),
        (SlowWorker(0.02), {"workers": 4, "outqn": 100, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_process_workers_with_batcher(run):
    """Process workers with Batcher stage - tests end signal with batching."""
    n_items = 100
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 200}),
        (Batcher(10), {"workers": 1, "outqn": 200}),
        (SlowWorker(0.02), {"workers": 4, "outqn": 200, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing: {expected - got}"
    assert got == expected


def test_concurrent_end_threaded():
    """Many threads competing for end signal."""
    n_items = 200
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 100}),
        (SlowWorker(0.01), {"workers": 8, "outqn": 100, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_concurrent_end_process():
    """Many processes competing for end signal."""
    n_items = 200
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 100}),
        (SlowWorker(0.01), {"workers": 8, "outqn": 100, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_concurrent_end_multiprocess():
    """Many processes competing for end signal."""
    n_items = 200
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 100}),
        (SlowWorker(0.01), {"workers": 4, "outqn": 100, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_mixed_threading_modes():
    """Mix of threaded and process workers."""
    n_items = 50
    stages = [
        (Generator(n_items), {"outqn": 30}),
        (FastWorker(), {"workers": 2, "outqn": 30, "thread": True}),
        (FastWorker(), {"workers": 2, "outqn": 30, "thread": False}),
        (FastWorker(), {"workers": 2, "outqn": 30, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


# === FILTER AND EXPAND WORKER TESTS ===

def test_filter_worker():
    """Filter drops half the items, end should still propagate."""
    n_items = 100
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (FilterWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = {i for i in range(n_items) if i % 2 == 0}
    assert len(results) == len(expected)
    assert {r["id"] for r in results} == expected


def test_filter_worker_process():
    """Filter drops half the items with process workers."""
    n_items = 100
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (FilterWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = {i for i in range(n_items) if i % 2 == 0}
    assert len(results) == len(expected)
    assert {r["id"] for r in results} == expected


def test_expand_worker():
    """Expander creates 3x items."""
    n_items = 20
    stages = [
        (Generator(n_items), {"outqn": 100}),
        (ExpandWorker(3), {"workers": 2, "outqn": 100, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items * 3
    assert {r["original_id"] for r in results} == set(range(n_items))


def test_expand_worker_process():
    """Expander creates 3x items with process workers."""
    n_items = 20
    stages = [
        (Generator(n_items), {"outqn": 100}),
        (ExpandWorker(3), {"workers": 2, "outqn": 100, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items * 3
    assert {r["original_id"] for r in results} == set(range(n_items))


# === ROOT STAGE WORKER COUNT (root must not run in parallel and duplicate) ===

class GenRoot:
    """Generator-based root yielding ids 0..n-1."""
    def __init__(self, n_items):
        self.n_items = n_items

    def __call__(self):
        for i in range(self.n_items):
            yield {"id": i}


def test_root_workers_clamped_no_duplication():
    """A root requested with workers>1 used to emit the stream N times; now once."""
    n = 20
    for src in (Generator(n), GenRoot(n)):
        results = run_pipeline([
            (src, {"workers": 3, "outqn": 100}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ])
        ids = sorted(r["id"] for r in results)
        assert ids == list(range(n)), f"{type(src).__name__}: {ids}"


def test_root_clamped_to_single_worker():
    pipe = Pipe(stats_interval=0)
    pipe.add(Generator(5), workers=4)
    pipe.add(Collector(), workers=1, outqn=0)
    assert pipe.jobs[0]["num_workers"] == 1   # root clamped
    assert pipe.jobs[1]["num_workers"] == 1   # non-root untouched


# === EXPECTED CONSUMERS (DDP shared-queue mode) ===

def test_expected_consumers_multi_reader():
    """expected_consumers=N must put N End sentinels on the final queue so each
    DDP-style PipeIterator reader terminates (shared mode in fluac/vui/akro:
    rank 0 runs the pipe, every rank reads pipe.queues[-1] via PipeIterator)."""
    import threading

    from pipe import PipeIterator

    n = 30
    pipe = Pipe(stats_interval=0, health_check_interval=0, expected_consumers=2)
    pipe.add(Generator(n), outqn=100)
    pipe.add(FastWorker(), workers=2, outqn=0)
    pipe.start()

    got = [[], []]

    def reader(i):
        for item in PipeIterator(pipe.queues[-1]):
            got[i].append(item["id"])

    threads = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    alive = [t.is_alive() for t in threads]
    pipe.stop()
    assert not any(alive), f"reader(s) never got an End sentinel: {alive}"
    assert sorted(got[0] + got[1]) == list(range(n))
