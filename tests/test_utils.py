"""Tests for the shipped helper workers in pipe.utils.

Run with: pytest test_utils.py -q
"""
import time

from pipe import End, Pipe
from pipe.utils import AsyncPoolWorker, Batcher, BufferAndShuffle

# === Batcher ===

def test_batcher_emits_distinct_objects():
    """Each emitted batch must be its own list (no aliasing/overwrite)."""
    b = Batcher(2)
    out = [r for r in (b(i) for i in range(4)) if r is not None]
    assert len(out) == 2
    assert out[0] is not out[1]
    assert out[0] == [0, 1]   # not overwritten by the second batch
    assert out[1] == [2, 3]


def test_batcher_flush_emits_trailing():
    """Items in a partial final batch must come out via flush()."""
    b = Batcher(3)
    emitted = [r for r in (b(i) for i in range(7)) if r is not None]
    assert emitted == [[0, 1, 2], [3, 4, 5]]
    assert list(b.flush()) == [6]          # trailing item, emitted individually
    assert list(b.flush()) == []           # empty buffer -> nothing


def test_batcher_collate():
    b = Batcher(2, collate_fn=lambda items: {"batch": items})
    assert b(0) is None
    assert b(1) == {"batch": [0, 1]}
    b(2)
    assert list(b.flush()) == [{"batch": [2]}]


# === BufferAndShuffle ===

def test_buffer_and_shuffle_flush():
    bs = BufferAndShuffle(size=4, batch_size=1)
    for i in range(13):
        bs(i)
    remaining = sorted(bs.flush())
    assert remaining == [10, 11, 12]
    assert bs.buffer == []
    assert list(bs.flush()) == []


# === end-to-end: helpers must not drop the partial tail in a real pipeline ===

class Source:
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


class Pass:
    def __call__(self, item):
        item["seen"] = True
        return item


class Collector:
    def __call__(self, item):
        return item


def _run(stages, timeout=60):
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    for stage, kwargs in stages:
        pipe.add(stage, **kwargs)
    results, start = [], time.time()
    for item in pipe:
        results.append(item)
        if time.time() - start > timeout:
            raise TimeoutError("pipeline timeout")
    return results


def test_batcher_pipeline_no_loss():
    """Every item survives a real Batcher stage, including the partial tail (25 % 10).

    Mirrors the layout in test_batching (thread stage in the middle, process final
    stage) so the result is deterministic rather than racing on shutdown.
    """
    n = 25
    for thread in (False, True):
        results = _run([
            (Source(n), {"outqn": 50}),
            (Batcher(10), {"workers": 1, "outqn": 50}),
            (Pass(), {"workers": 2, "thread": thread, "outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ])
        assert {r["id"] for r in results} == set(range(n)), f"thread={thread}"
        assert len(results) == n, f"thread={thread} got {len(results)}"


# === AsyncPoolWorker (streaming async pool, run()/pull()/put()) ===

class VariableLatencySource:
    def __init__(self, n):
        self.n = n
        self._i = 0

    def __call__(self):
        if self._i >= self.n:
            return End
        item = {"id": self._i, "delay": ((self.n - self._i) % 7) * 0.02}
        self._i += 1
        if self._i >= self.n:
            return [item, End]
        return item


class SlowFetch(AsyncPoolWorker):
    """Simulated download whose latency varies per item; id 13 fails."""

    async def process(self, item):
        import asyncio
        await asyncio.sleep(item["delay"])
        if item["id"] == 13:
            raise ValueError("permanent failure")
        item["fetched"] = True
        return item


class ExpandingFetch(AsyncPoolWorker):
    async def process(self, item):
        return [{"id": item["id"], "part": p} for p in range(2)]


def test_async_pool_worker_multiprocess():
    """All items arrive despite out-of-order completion; the failing one drops."""
    n = 20
    results = _run([
        (VariableLatencySource(n), {"outqn": 50}),
        (SlowFetch(max_concurrent=8), {"workers": 1, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ])
    ids = sorted(r["id"] for r in results)
    assert ids == [i for i in range(n) if i != 13]
    assert all(r["fetched"] for r in results)


def test_async_pool_worker_sequential():
    n = 10
    pipe = Pipe(sequential=True, stats_interval=0)
    pipe.add(VariableLatencySource(n))
    pipe.add(SlowFetch(max_concurrent=4), workers=1)
    ids = sorted(item["id"] for item in pipe)
    assert ids == [i for i in range(n) if i != 13]


def test_async_pool_worker_list_expansion():
    n = 6
    results = _run([
        (VariableLatencySource(n), {"outqn": 20}),
        (ExpandingFetch(max_concurrent=4), {"workers": 1, "outqn": 20}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ])
    assert len(results) == n * 2
    assert {(r["id"], r["part"]) for r in results} == {(i, p) for i in range(n) for p in range(2)}
