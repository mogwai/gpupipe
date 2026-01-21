"""Stress tests for pipe framework.

Run with: pytest tests/test_stress.py -n auto
"""
import time
import pytest

from pipe import Pipe


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
            return "end"
        if self.delay:
            time.sleep(self.delay)
        item = {"id": self._idx}
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, "end"]
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
            return "end"
        items = []
        for _ in range(min(self.batch_size, self.n_items - self._idx)):
            items.append({"id": self._idx})
            self._idx += 1
        if self._idx >= self.n_items:
            items.append("end")
        return items


class SlowWorker:
    """Worker with configurable delay."""
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            return item
        time.sleep(self.delay)
        item["processed"] = True
        return item


class FastWorker:
    """Worker with no delay."""
    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            return item
        item["processed"] = True
        return item


class Batcher:
    """Batches items, with flush support."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            result = self.buffer.copy() if self.buffer else []
            self.buffer = []
            if result:
                result.append("end")
                return result
            return "end"
        self.buffer.append(item)
        if len(self.buffer) >= self.size:
            result = self.buffer.copy()
            self.buffer = []
            return result
        return None


class GPUBatcher:
    """Collects items into batches for GPU processing."""
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            if self.buffer:
                batch = self.buffer.copy()
                self.buffer = []
                return [{"batch": batch, "batch_size": len(batch)}, "end"]
            return "end"

        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            batch = self.buffer.copy()
            self.buffer = []
            return {"batch": batch, "batch_size": len(batch)}
        return None


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


@pytest.mark.parametrize("run", range(5))
def test_fast_generator_slow_workers_threaded(run):
    """Generator finishes immediately, slow workers must process all items."""
    n_items = 50
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 100}),
        (SlowWorker(0.05), {"workers": 4, "outqn": 100, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_fast_generator_slow_workers_process(run):
    """Generator finishes immediately, slow process workers must process all items."""
    n_items = 50
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 100}),
        (SlowWorker(0.05), {"workers": 4, "outqn": 100, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_instant_dump_very_slow_workers_threaded(run):
    """All items dumped instantly, workers take 100ms each - tests end signal timing."""
    n_items = 20
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 50}),
        (SlowWorker(0.1), {"workers": 4, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=30)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_instant_dump_very_slow_workers_process(run):
    """All items dumped instantly, process workers take 100ms each - tests end signal timing."""
    n_items = 20
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 50}),
        (SlowWorker(0.1), {"workers": 4, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=30)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


def test_stress_slow_workers():
    """Slow workers to maximize race condition window."""
    n_items = 50
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 100}),
        (SlowWorker(0.1), {"workers": 2, "outqn": 100}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items


def test_stress_many_items():
    """Stress test with many items."""
    n_items = 500
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 100}),
        (Batcher(32), {"workers": 1, "outqn": 100}),
        (SlowWorker(0.05), {"workers": 2, "outqn": 100}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_stress_large_threaded():
    """Stress test with many items and threaded workers."""
    n_items = 1000
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 200}),
        (SlowWorker(0.005), {"workers": 8, "outqn": 200, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=120)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_stress_large_process():
    """Stress test with many items and process workers."""
    n_items = 1000
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 200}),
        (SlowWorker(0.005), {"workers": 8, "outqn": 200, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=120)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_many_pipelines_sequential():
    """Create 20 pipelines sequentially - test cleanup."""
    for _ in range(20):
        stages = [
            (Generator(10), {"outqn": 10}),
            (FastWorker(), {"workers": 2, "outqn": 10, "thread": True}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages)
        assert len(results) == 10


def test_many_pipelines_sequential_process():
    """Create 20 pipelines sequentially with process workers - test cleanup."""
    for _ in range(20):
        stages = [
            (Generator(10), {"outqn": 10}),
            (FastWorker(), {"workers": 2, "outqn": 10, "thread": False}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages)
        assert len(results) == 10
