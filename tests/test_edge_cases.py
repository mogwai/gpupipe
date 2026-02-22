"""Edge case tests for pipe framework - item counts, queue pressure, timing.

Run with: pytest test_edge_cases.py -n auto
"""
import time

from pipe import Pipe, End


class Generator:
    """Simple generator returning items one at a time."""
    def __init__(self, n_items: int, delay: float = 0):
        self.n_items = n_items
        self.delay = delay
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        if self.delay:
            time.sleep(self.delay)
        item = {"id": self._idx}
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class BatchGenerator:
    """Generator that returns items in batches."""
    def __init__(self, n_items: int, batch_size: int = 10):
        self.n_items = n_items
        self.batch_size = batch_size
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        items = []
        for _ in range(min(self.batch_size, self.n_items - self._idx)):
            items.append({"id": self._idx})
            self._idx += 1
        if self._idx >= self.n_items:
            items.append(End)
        return items


class SlowWorker:
    """Worker with configurable delay."""
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(self.delay)
        item["processed"] = True
        return item


class FastWorker:
    """Worker with no delay."""
    def load(self):
        pass

    def __call__(self, item):
        item["processed"] = True
        return item


class Collector:
    """Simple pass-through collector."""
    def load(self):
        pass

    def __call__(self, item):
        return item


def run_pipeline(stages: list[tuple], timeout: float = 60, health_check: int = 0) -> list[dict]:
    """Run pipeline and collect results."""
    pipe = Pipe(debug=False, raise_errors=True, stats_interval=0, health_check_interval=health_check)
    for stage, kwargs in stages:
        pipe.add(stage, **kwargs)
    results = []
    start = time.time()
    for item in pipe:
        results.append(item)
        if time.time() - start > timeout:
            raise TimeoutError(f"Pipeline timeout after {timeout}s")
    return results


# === EDGE CASES: ITEM COUNTS ===

def test_zero_items():
    """Empty pipeline."""
    stages = [
        (Generator(0), {"outqn": 10}),
        (FastWorker(), {"workers": 2, "outqn": 10, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 0


def test_zero_items_process():
    """Empty pipeline with process workers."""
    stages = [
        (Generator(0), {"outqn": 10}),
        (FastWorker(), {"workers": 2, "outqn": 10, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 0


def test_one_item():
    """Single item pipeline."""
    stages = [
        (Generator(1), {"outqn": 10}),
        (FastWorker(), {"workers": 2, "outqn": 10, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 1
    assert results[0]["id"] == 0


def test_one_item_process():
    """Single item pipeline with process workers."""
    stages = [
        (Generator(1), {"outqn": 10}),
        (FastWorker(), {"workers": 2, "outqn": 10, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 1
    assert results[0]["id"] == 0


def test_more_workers_than_items():
    """10 workers, 3 items - some workers never get work."""
    stages = [
        (Generator(3), {"outqn": 10}),
        (FastWorker(), {"workers": 10, "outqn": 10, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 3
    assert {r["id"] for r in results} == {0, 1, 2}


def test_more_workers_than_items_process():
    """4 process workers, 3 items - some workers never get work."""
    stages = [
        (Generator(3), {"outqn": 10}),
        (FastWorker(), {"workers": 4, "outqn": 10, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 3
    assert {r["id"] for r in results} == {0, 1, 2}


def test_many_more_workers_than_items_process():
    """10 process workers, 3 items - exposes end signal propagation bug."""
    stages = [
        (Generator(3), {"outqn": 10}),
        (FastWorker(), {"workers": 10, "outqn": 10, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == 3
    assert {r["id"] for r in results} == {0, 1, 2}


# === EDGE CASES: QUEUE PRESSURE ===

def test_fast_upstream_slow_downstream():
    """Upstream faster than downstream - tests backpressure."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 10}),
        (SlowWorker(0.02), {"workers": 2, "outqn": 10, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_fast_upstream_slow_downstream_process():
    """Upstream faster than downstream - tests backpressure with process workers."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 10}),
        (SlowWorker(0.02), {"workers": 2, "outqn": 10, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_slow_upstream_fast_downstream():
    """Upstream slower than downstream - queue goes empty repeatedly."""
    n_items = 30
    stages = [
        (Generator(n_items, delay=0.02), {"outqn": 50}),
        (FastWorker(), {"workers": 4, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_slow_upstream_fast_downstream_process():
    """Upstream slower than downstream - queue goes empty repeatedly with process workers."""
    n_items = 30
    stages = [
        (Generator(n_items, delay=0.02), {"outqn": 50}),
        (FastWorker(), {"workers": 4, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


# === EDGE CASES: GENERATOR TIMING ===

def test_generator_done_before_workers_start_threaded():
    """Generator completes before workers even start processing - extreme case."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 100), {"outqn": 200}),
        (SlowWorker(0.02), {"workers": 8, "outqn": 200, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


def test_generator_done_before_workers_start_process():
    """Generator completes before process workers even start processing - extreme case."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 100), {"outqn": 200}),
        (SlowWorker(0.02), {"workers": 8, "outqn": 200, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected
