"""Batching tests for pipe framework.

Run with: pytest test_batching.py -n auto
"""
import time

from pipe import Pipe, End


class Generator:
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
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(self.delay)
        item["processed"] = True
        return item


class FastWorker:
    def load(self):
        pass

    def __call__(self, item):
        item["processed"] = True
        return item


class ExpandWorker:
    def __init__(self, factor: int = 3):
        self.factor = factor

    def load(self):
        pass

    def __call__(self, item):
        return [{"id": item["id"], "sub": i, "original_id": item["id"]} for i in range(self.factor)]


class Batcher:
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


class FlushingBatcher:
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
    def load(self):
        pass

    def __call__(self, item):
        return item


def run_pipeline(stages: list[tuple], timeout: float = 60, health_check: int = 0) -> list[dict]:
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


def test_partial_batch_flush():
    """25 items, batch size 10 - last 5 need flush."""
    n_items = 25
    stages = [
        (BatchGenerator(n_items, 5), {"outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_partial_batch_flush_process():
    """25 items, batch size 10 - last 5 need flush with process workers."""
    n_items = 25
    stages = [
        (BatchGenerator(n_items, 5), {"outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_flushing_batcher():
    """Worker that uses flush() mechanism."""
    n_items = 55
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (FlushingBatcher(10), {"workers": 1, "outqn": 50}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_large_batch_small_data():
    """Batch size larger than total items."""
    n_items = 5
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (Batcher(100), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_large_batch_small_data_process():
    """Batch size larger than total items with process workers."""
    n_items = 5
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (Batcher(100), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_consecutive_batchers():
    """Batcher -> Batcher with different sizes."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 10), {"outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_consecutive_batchers_process():
    """Batcher -> Batcher with different sizes with process workers."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 10), {"outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_queue_size_one():
    """Maximum backpressure - queue size 1."""
    n_items = 20
    stages = [
        (Generator(n_items), {"outqn": 1}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 1, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_queue_size_one_process():
    """Maximum backpressure - queue size 1 with process workers."""
    n_items = 20
    stages = [
        (Generator(n_items), {"outqn": 1}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 1, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_expand_then_batch():
    """Expander -> Batcher."""
    n_items = 10
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (ExpandWorker(3), {"workers": 1, "outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items * 3  # 10 * 3 = 30


def test_expand_then_batch_process():
    """Expander -> Batcher with process workers."""
    n_items = 10
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (ExpandWorker(3), {"workers": 1, "outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items * 3  # 10 * 3 = 30
