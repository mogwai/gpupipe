"""Worker type tests - threaded, process, filter, expand workers.

Run with: pytest test_workers.py -n auto
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


class FilterWorker:
    """Drops items where id is odd."""
    def load(self):
        pass

    def __call__(self, item):
        if item["id"] % 2 == 0:
            return item
        return None


class ExpandWorker:
    """Expands each item into multiple."""
    def __init__(self, factor: int = 3):
        self.factor = factor

    def load(self):
        pass

    def __call__(self, item):
        return [{"id": item["id"], "sub": i, "original_id": item["id"]} for i in range(self.factor)]


class Batcher:
    """Batches items, with flush support."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.size:
            result = self.buffer.copy()
            self.buffer = []
            return result
        return None

    def flush(self):
        if self.buffer:
            result = self.buffer.copy()
            self.buffer = []
            for item in result:
                yield item


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
