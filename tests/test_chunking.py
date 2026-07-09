"""Unit tests for chunked transport primitives (Chunk, _OutputChannel, _InputChannel)
plus integration tests for chunked edges feeding batch= stages."""
import pickle
import queue as thread_queue
import time

import pytest

from pipe.queues import Chunk, _InputChannel, _OutputChannel
from pipe.types import WorkerStop


class FakeStop:
    def __init__(self, value=0):
        self.value = value


# ---------------------------------------------------------------------------
# _OutputChannel
# ---------------------------------------------------------------------------

def test_output_channel_passthrough_when_disabled():
    q = thread_queue.Queue()
    ch = _OutputChannel(q, chunk_size=0)
    for i in range(5):
        ch.send(i)
    got = [q.get_nowait() for _ in range(5)]
    assert got == [0, 1, 2, 3, 4]
    assert q.empty()


def test_output_channel_flush_on_size():
    q = thread_queue.Queue()
    ch = _OutputChannel(q, chunk_size=3, chunk_ms=10_000)
    ch.send("a")
    ch.send("b")
    assert q.empty()  # below chunk_size, long timeout -> nothing emitted
    ch.send("c")
    msg = q.get_nowait()
    assert type(msg) is Chunk
    assert msg.items == ["a", "b", "c"]
    assert q.empty()


def test_output_channel_flush_on_timeout():
    q = thread_queue.Queue()
    ch = _OutputChannel(q, chunk_size=100, chunk_ms=20)
    ch.send("only")
    assert q.empty()
    time.sleep(0.03)
    ch.maybe_flush()
    msg = q.get_nowait()
    assert type(msg) is Chunk
    assert msg.items == ["only"]


def test_output_channel_timeout_checked_on_send():
    """A send after the deadline flushes even without maybe_flush()."""
    q = thread_queue.Queue()
    ch = _OutputChannel(q, chunk_size=100, chunk_ms=20)
    ch.send("first")
    time.sleep(0.03)
    ch.send("second")  # deadline passed -> flush both
    msg = q.get_nowait()
    assert msg.items == ["first", "second"]


def test_output_channel_explicit_flush_partial():
    q = thread_queue.Queue()
    ch = _OutputChannel(q, chunk_size=100, chunk_ms=10_000)
    ch.send("x")
    ch.send("y")
    ch.flush()
    msg = q.get_nowait()
    assert msg.items == ["x", "y"]
    ch.flush()  # empty flush is a no-op
    assert q.empty()


def test_output_channel_multiple_chunks():
    q = thread_queue.Queue()
    ch = _OutputChannel(q, chunk_size=2, chunk_ms=10_000)
    for i in range(5):
        ch.send(i)
    ch.flush()
    chunks = []
    while not q.empty():
        chunks.append(q.get_nowait().items)
    assert chunks == [[0, 1], [2, 3], [4]]


# ---------------------------------------------------------------------------
# _InputChannel
# ---------------------------------------------------------------------------

def test_input_channel_unpacks_chunks_in_order():
    q = thread_queue.Queue()
    q.put(Chunk([1, 2, 3]))
    q.put(Chunk([4]))
    ch = _InputChannel(q)
    assert [ch.get(timeout=0.1) for _ in range(4)] == [1, 2, 3, 4]
    with pytest.raises(thread_queue.Empty):
        ch.get_nowait()


def test_input_channel_bare_items_pass_through():
    """push() back-edge items are bare; sentinels are bare (never chunked)."""
    q = thread_queue.Queue()
    q.put({"id": 1})
    q.put(Chunk([{"id": 2}, {"id": 3}]))
    q.put(WorkerStop)
    ch = _InputChannel(q)
    assert ch.get(timeout=0.1) == {"id": 1}
    assert ch.get(timeout=0.1) == {"id": 2}
    assert ch.get(timeout=0.1) == {"id": 3}
    assert ch.get(timeout=0.1) is WorkerStop


def test_input_channel_list_payload_not_confused_with_chunk():
    """A worker item that IS a list must not be unpacked."""
    q = thread_queue.Queue()
    q.put([1, 2, 3])  # bare list payload
    ch = _InputChannel(q)
    assert ch.get(timeout=0.1) == [1, 2, 3]


def test_input_channel_buffered_count():
    q = thread_queue.Queue()
    q.put(Chunk(["a", "b", "c"]))
    ch = _InputChannel(q)
    assert ch.buffered() == 0
    assert ch.get(timeout=0.1) == "a"
    assert ch.buffered() == 2


def test_chunk_pickle_roundtrip():
    c = Chunk([{"id": 1}, "raw", [1, 2]])
    c2 = pickle.loads(pickle.dumps(c))
    assert type(c2) is Chunk
    assert c2.items == c.items


# ---------------------------------------------------------------------------
# End-to-end pipeline tests (multiprocessing)
# ---------------------------------------------------------------------------

from pipe import End, Pipe  # noqa: E402


class _Gen:
    def __init__(self, n):
        self.n = n
        self._i = 0

    def __call__(self):
        if self._i >= self.n:
            return End
        item = {"id": self._i}
        self._i += 1
        return item


class _BatchDoubler:
    """batch=N stage: receives a list, doubles each id."""

    def __call__(self, batch):
        assert isinstance(batch, list)
        return [{"id": b["id"], "doubled": b["id"] * 2, "bsz": len(batch)} for b in batch]


class _PassThrough:
    def __call__(self, item):
        return item


class _Splitter:
    def __call__(self, item):
        for j in range(3):
            yield {"id": item["id"] * 3 + j}


def test_auto_chunk_all_items_arrive():
    """Edge feeding a batch= stage auto-chunks; every item must arrive exactly once."""
    n = 200
    pipe = Pipe(stats_interval=0)
    pipe.add(_Gen(n), outqn=256)
    pipe.add(_BatchDoubler(), workers=2, batch=16, outqn=256)
    # chunk_eff auto-derived: stage 0 -> 16
    results = list(pipe)
    assert pipe.jobs[0]["chunk_eff"] == 16
    ids = sorted(r["id"] for r in results)
    assert ids == list(range(n)), f"got {len(ids)} items"
    assert all(r["doubled"] == r["id"] * 2 for r in results)


def test_auto_chunk_produces_real_batches():
    """With a fast producer, the batch stage should mostly see multi-item batches."""
    n = 400
    pipe = Pipe(stats_interval=0)
    pipe.add(_Gen(n), outqn=512)
    pipe.add(_BatchDoubler(), workers=1, batch=32, outqn=512)
    results = list(pipe)
    assert len(results) == n
    # At least some batches should be full-sized (chunked transport delivers
    # 32 items per queue message, so the collector can fill instantly).
    assert max(r["bsz"] for r in results) >= 16


def test_short_tail_flushes_on_timeout():
    """n smaller than the chunk size: the timeout flush must deliver everything."""
    n = 5
    pipe = Pipe(stats_interval=0)
    pipe.add(_Gen(n), outqn=64)
    pipe.add(_BatchDoubler(), workers=1, batch=64, outqn=64)
    results = list(pipe)
    assert sorted(r["id"] for r in results) == list(range(n))


def test_explicit_chunk_on_final_edge():
    """chunk= on the last stage: the consumer iterator unpacks transparently."""
    n = 150
    pipe = Pipe(stats_interval=0)
    pipe.add(_Gen(n), outqn=256)
    pipe.add(_PassThrough(), workers=2, outqn=256, chunk=25)
    results = list(pipe)
    assert pipe.jobs[1]["chunk_eff"] == 25
    assert sorted(r["id"] for r in results) == list(range(n))


def test_chunk_zero_forces_off():
    """chunk=0 overrides the auto-derive from downstream batch=."""
    n = 60
    pipe = Pipe(stats_interval=0)
    pipe.add(_Gen(n), outqn=64, chunk=0)
    pipe.add(_BatchDoubler(), workers=1, batch=16, outqn=64)
    results = list(pipe)
    assert pipe.jobs[0]["chunk_eff"] == 0
    assert sorted(r["id"] for r in results) == list(range(n))


def test_chunked_edge_with_generator_expansion():
    """A yielding (1->N) stage feeding a chunked edge into a batch stage."""
    n = 40
    pipe = Pipe(stats_interval=0)
    pipe.add(_Gen(n), outqn=128)
    pipe.add(_Splitter(), workers=2, outqn=256)
    pipe.add(_BatchDoubler(), workers=2, batch=16, outqn=256)
    results = list(pipe)
    assert sorted(r["id"] for r in results) == list(range(n * 3))
