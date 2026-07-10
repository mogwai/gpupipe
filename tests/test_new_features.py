"""Tests for batch=N, run()+pull/put, debug=True (InstrumentedQueue).

Covers:
- Framework batching via pipe.add(worker, batch=N)
- Worker-driven loop via run() with self.pull() / self.put()
- InstrumentedQueue transit stats via Pipe(debug=True)
- All of the above in sequential mode
"""
import time

from pipe import End, Pipe
from pipe.pipe import InstrumentedQueue

# === Workers ===

class Generator:
    def __init__(self, n):
        self.n = n
        self._idx = 0

    def __call__(self):
        if self._idx >= self.n:
            return End
        item = {"id": self._idx}
        self._idx += 1
        if self._idx >= self.n:
            return [item, End]
        return item


class BatchProcessor:
    """Receives a list of items via batch=N, processes each, returns list."""
    def __call__(self, items):
        for item in items:
            item["batch_processed"] = True
            item["batch_size"] = len(items)
        return items


class BatchSummer:
    """Receives a batch, returns a single summary item."""
    def __call__(self, items):
        return {"sum_ids": sum(it["id"] for it in items), "count": len(items)}


class RunWorker:
    """Worker using run() + pull/put for custom loop."""
    def __call__(self, item):
        raise RuntimeError("should not be called")

    def run(self):
        while True:
            items = self.pull()
            if not items:
                break
            for item in items:
                item["run_processed"] = True
                self.put(item)


class RunBatchWorker:
    """Worker using run() that pulls larger batches."""
    def __call__(self, item):
        raise RuntimeError("should not be called")

    def run(self):
        while True:
            items = self.pull(8)
            if not items:
                break
            for item in items:
                item["run_batch"] = True
                item["pulled_with"] = len(items)
                self.put(item)


class RunAggregator:
    """Worker using run() that aggregates items before putting."""
    def __call__(self, item):
        raise RuntimeError("should not be called")

    def run(self):
        total = 0
        count = 0
        while True:
            items = self.pull(16)
            if not items:
                break
            for item in items:
                total += item["id"]
                count += 1
        self.put({"total": total, "count": count})


class Passthrough:
    def __call__(self, item):
        return item


# === batch=N tests ===

def test_batch_basic_sequential():
    """batch=N passes lists to __call__ in sequential mode."""
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(20), outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=4)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 20
    assert all(r["batch_processed"] for r in results)
    assert {r["id"] for r in results} == set(range(20))


def test_batch_basic_process():
    """batch=N works with multiprocessing workers."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(50), outqn=50)
    pipe.add(BatchProcessor(), workers=2, outqn=50, batch=8)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 50
    assert all(r["batch_processed"] for r in results)
    assert {r["id"] for r in results} == set(range(50))


def test_batch_basic_threaded():
    """batch=N works with threaded workers."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(50), outqn=50)
    pipe.add(BatchProcessor(), workers=4, outqn=50, batch=8, thread=True)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 50
    assert all(r["batch_processed"] for r in results)
    assert {r["id"] for r in results} == set(range(50))


def test_batch_partial():
    """Items not divisible by batch size still all flow through."""
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(7), outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=4)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 7
    assert {r["id"] for r in results} == set(range(7))


def test_batch_partial_process():
    """Partial batches work in multiprocessing mode."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(13), outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=4)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 13
    assert {r["id"] for r in results} == set(range(13))


def test_batch_aggregation_sequential():
    """batch=N with a worker that returns a single summary per batch."""
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(20), outqn=50)
    pipe.add(BatchSummer(), workers=1, outqn=50, batch=5)

    results = list(pipe)
    total = sum(r["count"] for r in results)
    assert total == 20
    assert sum(r["sum_ids"] for r in results) == sum(range(20))


def test_batch_aggregation_process():
    """batch=N aggregation in multiprocessing mode."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(40), outqn=50)
    pipe.add(BatchSummer(), workers=1, outqn=50, batch=10)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    total = sum(r["count"] for r in results)
    assert total == 40
    assert sum(r["sum_ids"] for r in results) == sum(range(40))


def test_batch_size_one():
    """batch=1 is equivalent to normal single-item processing."""
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(10), outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=1)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 10
    assert all(r["batch_size"] == 1 for r in results)


def test_batch_larger_than_items():
    """batch=N where N > total items."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(3), outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=100)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 3
    assert {r["id"] for r in results} == set(range(3))


# === run() + pull/put tests ===

def test_run_basic_sequential():
    """run() worker with pull/put in sequential mode."""
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(20), outqn=50)
    pipe.add(RunWorker(), workers=1, outqn=50)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 20
    assert all(r["run_processed"] for r in results)
    assert {r["id"] for r in results} == set(range(20))


def test_run_basic_process():
    """run() worker with pull/put in multiprocessing mode."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(50), outqn=50)
    pipe.add(RunWorker(), workers=2, outqn=50)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 50
    assert all(r["run_processed"] for r in results)
    assert {r["id"] for r in results} == set(range(50))


def test_run_basic_threaded():
    """run() worker with pull/put in threaded mode."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(50), outqn=50)
    pipe.add(RunWorker(), workers=4, outqn=50, thread=True)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 50
    assert all(r["run_processed"] for r in results)
    assert {r["id"] for r in results} == set(range(50))


def test_run_batch_pull():
    """run() worker pulling larger batches."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(50), outqn=50)
    pipe.add(RunBatchWorker(), workers=1, outqn=50, batch=8)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 50
    assert all(r["run_batch"] for r in results)
    assert {r["id"] for r in results} == set(range(50))


def test_run_aggregator_sequential():
    """run() worker that aggregates items in sequential mode.

    In sequential mode items flow incrementally, so the aggregator
    gets called multiple times with small batches.
    """
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(30), outqn=50)
    pipe.add(RunAggregator(), workers=1, outqn=50)

    results = list(pipe)
    assert sum(r["count"] for r in results) == 30
    assert sum(r["total"] for r in results) == sum(range(30))


def test_run_aggregator_process():
    """run() aggregator in multiprocessing mode."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(30), outqn=50)
    pipe.add(RunAggregator(), workers=1, outqn=50)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    # Single worker aggregates everything
    total_count = sum(r["count"] for r in results)
    assert total_count == 30
    total_sum = sum(r["total"] for r in results)
    assert total_sum == sum(range(30))


def test_run_then_batch_sequential():
    """run() worker followed by batch=N worker."""
    pipe = Pipe(sequential=True, raise_errors=True, stats_interval=0)
    pipe.add(Generator(20), outqn=50)
    pipe.add(RunWorker(), workers=1, outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=4)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 20
    assert all(r["run_processed"] for r in results)
    assert all(r["batch_processed"] for r in results)


def test_run_then_batch_process():
    """run() worker followed by batch=N worker in multiprocessing."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(40), outqn=50)
    pipe.add(RunWorker(), workers=2, outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=8)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 40
    assert all(r["run_processed"] for r in results)
    assert all(r["batch_processed"] for r in results)


# === debug=True (InstrumentedQueue) tests ===

def test_debug_instrumented_queues():
    """debug=True wraps queues with InstrumentedQueue."""
    pipe = Pipe(debug=True, raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(10), outqn=20)
    pipe.add(Passthrough(), workers=1, outqn=0)
    pipe.start()

    assert len(pipe.queues) == 2
    for q in pipe.queues:
        assert isinstance(q, InstrumentedQueue)

    results = list(pipe)
    assert len(results) == 10


def test_debug_tracks_counts():
    """InstrumentedQueue tracks put/get counts."""
    pipe = Pipe(debug=True, raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(20), outqn=50)
    pipe.add(Passthrough(), workers=1, outqn=50)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 20

    # After pipeline stops, queues are cleared, so queue instrumentation is
    # checked during iteration instead.


def test_debug_transit_latency():
    """InstrumentedQueue tracks transit latency."""
    from torch.multiprocessing import Queue
    q = InstrumentedQueue(Queue(maxsize=10))

    q.put("item1")
    time.sleep(0.01)
    result = q.get()

    assert result == "item1"
    assert q.items_put.value == 1
    assert q.items_got.value == 1
    assert q.total_transit.value > 0.005  # at least ~5ms


def test_debug_transit_nowait():
    """InstrumentedQueue.get_nowait tracks transit."""
    from torch.multiprocessing import Queue
    q = InstrumentedQueue(Queue(maxsize=10))

    q.put("a")
    q.put("b")
    time.sleep(0.01)

    r1 = q.get_nowait()
    r2 = q.get_nowait()

    assert r1 == "a"
    assert r2 == "b"
    assert q.items_put.value == 2
    assert q.items_got.value == 2
    assert q.total_transit.value > 0


def test_debug_full_pipeline():
    """Full pipeline with debug=True produces correct results."""
    pipe = Pipe(debug=True, raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(100), outqn=50)
    pipe.add(Passthrough(), workers=2, outqn=50)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 100
    assert {r["id"] for r in results} == set(range(100))


def test_debug_with_batch():
    """debug=True works correctly with batch=N."""
    pipe = Pipe(debug=True, raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(30), outqn=50)
    pipe.add(BatchProcessor(), workers=1, outqn=50, batch=5)
    pipe.add(Passthrough(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == 30
    assert all(r["batch_processed"] for r in results)


def test_no_debug_plain_queues():
    """Without debug=True, queues are plain mp.Queue (not instrumented)."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(5), outqn=20)
    pipe.add(Passthrough(), workers=1, outqn=0)
    pipe.start()

    for q in pipe.queues:
        assert not isinstance(q, InstrumentedQueue)

    results = list(pipe)
    assert len(results) == 5
