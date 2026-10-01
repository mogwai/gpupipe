"""The shared payload store (pipe.shmstore): large arrays cross edges once and
are forwarded by handle; numpy arrives read-only, tensors writable-shared;
blocks are refcounted across processes and freed for reuse."""
import os
import time

import numpy as np
import pytest
import torch

pytest.importorskip("pipe._rustq")

from pipe._rustq import Store  # noqa: E402

from pipe import Pipe  # noqa: E402
from pipe.shmqueue import ShmQueue  # noqa: E402
from pipe.shmstore import STORE_MIN_BYTES, PayloadStore, _ndarray_block  # noqa: E402

BIG = STORE_MIN_BYTES // 4 * 4  # float32 elements' worth of bytes, rounded


@pytest.fixture
def store():
    s = PayloadStore(limit_bytes=256 << 20, seg_bytes=16 << 20)
    yield s
    s.close()


def hop(q, obj):
    q.put(obj)
    return q.get(timeout=5)


def test_large_numpy_arrives_read_only_view_of_the_store(store):
    q = ShmQueue(8, store=store)
    a = np.arange(BIG, dtype=np.float32)
    b = hop(q, {"audio": a})["audio"]
    assert np.array_equal(a, b) and b.dtype == a.dtype
    assert _ndarray_block(b) is not None
    assert not b.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        b[0] = 1.0
    c = b.copy()  # the documented way to get a private writable array
    c[0] = 1.0
    assert a[0] == 0.0 and b[0] == 0.0


def test_small_numpy_is_pickled_normally(store):
    q = ShmQueue(8, store=store)
    b = hop(q, np.arange(8, dtype=np.float32))
    assert _ndarray_block(b) is None and b.flags.writeable


def test_forwarding_a_view_reuses_the_block(store):
    q1, q2 = ShmQueue(8, store=store), ShmQueue(8, store=store)
    b = hop(q1, np.arange(4 * BIG, dtype=np.float32).reshape(4, BIG))
    used = store.core.used
    c = hop(q2, b)  # unchanged: forwarded, not copied
    s = hop(q2, b[1:3, ::2])  # a strided slice of it: still the same block
    assert _ndarray_block(c).addr == _ndarray_block(b).addr == _ndarray_block(s).addr
    assert np.array_equal(c, b) and np.array_equal(s, b[1:3, ::2])
    assert store.core.used == used


def test_blocks_are_freed_and_reused(store):
    q = ShmQueue(4, store=store)
    for i in range(300):
        b = hop(q, np.full(1 << 18, i, dtype=np.float32))  # 1 MiB each
        assert b[0] == i
        del b
    assert store.core.used <= 16 << 20  # one segment, recycled throughout


def test_cpu_tensors_go_through_the_store_writable_and_shared(store):
    q = ShmQueue(8, store=store)
    t = torch.arange(BIG, dtype=torch.float32)
    r = hop(q, t)
    assert torch.equal(r, t) and not t.is_shared()
    used = store.core.used
    fwd = hop(q, r[10:])  # forwarded view: same memory, no new block
    assert store.core.used == used
    fwd[0] = -1.0  # writable, and shared with the view it came from
    assert r[10] == -1.0


def test_requires_grad_survives_the_store(store):
    q = ShmQueue(8, store=store)
    t = torch.ones(BIG, requires_grad=True)
    r = hop(q, t)
    assert r.requires_grad and torch.equal(r.detach(), t.detach())


def test_store_full_falls_back_to_copying(capsys):
    small = PayloadStore(limit_bytes=1 << 20, seg_bytes=1 << 20)
    try:
        q = ShmQueue(8, ring_bytes=16 << 20, store=small)
        b = hop(q, np.ones(1 << 19, dtype=np.float32))  # 2 MiB > the whole store
        assert np.array_equal(b, np.ones(1 << 19, dtype=np.float32))
        assert _ndarray_block(b) is None and b.flags.writeable  # plain pickled copy
        assert "payload store can't fit" in capsys.readouterr().out
    finally:
        small.close()


def test_oversized_block_gets_its_own_segment_and_is_released(store):
    q = ShmQueue(2, store=store)
    a = np.random.rand(5 << 20)  # 40 MiB > seg_bytes/4: dedicated segment
    b = hop(q, a)
    assert np.array_equal(a, b)
    assert store.core.used >= 40 << 20
    del b
    assert store.core.used < 40 << 20


def test_close_unlinks_everything(store):
    q = ShmQueue(2, store=store)
    keep = hop(q, np.ones(BIG, dtype=np.float32))
    name = store.name
    store.close()
    with pytest.raises(OSError):
        Store.attach(name)
    assert keep.sum() == BIG  # views handed out stay valid after unlink


class _Waves:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        for i in range(self.n):
            yield {"id": i, "wave": np.full(1 << 16, i, dtype=np.float32)}


class _Tag:
    """Adds metadata and passes the waveform on unchanged."""

    def __call__(self, item):
        assert not item["wave"].flags.writeable
        item["tag"] = item["id"] * 2
        return item


class _Check:
    def __call__(self, item):
        w = item["wave"]
        item["ok"] = bool((w == item["id"]).all()) and not w.flags.writeable
        return item


def test_pipeline_passes_arrays_through_the_store(monkeypatch):
    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    monkeypatch.delenv("PIPE_STORE", raising=False)
    pipe = Pipe(stats_interval=0)
    pipe.add(_Waves(200), outqn=16)
    pipe.add(_Tag(), workers=2, outqn=16)
    pipe.add(_Check(), workers=2, outqn=16)
    pipe.start()
    name = pipe.store.name
    got = list(pipe)
    assert sorted(x["id"] for x in got) == list(range(200))
    assert all(x["ok"] and x["tag"] == 2 * x["id"] for x in got)
    assert all(not x["wave"].flags.writeable for x in got)
    pipe.stop()
    with pytest.raises(OSError):
        Store.attach(name)


def test_pipeline_store_can_be_disabled(monkeypatch):
    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    monkeypatch.setenv("PIPE_STORE", "0")
    pipe = Pipe(stats_interval=0)
    pipe.add(_Waves(5), outqn=4)
    pipe.start()
    assert pipe.store is None
    assert all(x["wave"].flags.writeable for x in pipe)


def test_store_is_attached_by_name_across_processes(store):
    import torch.multiprocessing as mp

    q = ShmQueue(4, store=store)
    p = mp.Process(target=_put_wave, args=(q,))
    p.start()
    b = q.get(timeout=30)
    p.join()
    assert np.array_equal(b, np.full(BIG, 3, dtype=np.float32)) and _ndarray_block(b) is not None


def _put_wave(q):
    q.put(np.full(BIG, 3, dtype=np.float32))
    os._exit(0)  # no teardown: the block must outlive its creator


def test_oversized_messages_travel_in_recycled_store_blocks(store):
    q = ShmQueue(4, ring_bytes=1 << 20, store=store)  # ring fast path <= 256 KiB
    blob = os.urandom(3 << 20)  # bytes aren't arrays: the whole message is big
    for i in range(40):
        r = hop(q, {"i": i, "blob": blob})
        assert r["i"] == i and r["blob"] == blob
    assert store.core.used <= 16 << 20  # one segment's blocks, reused each time


# === reference counting under load ===

def test_fanned_out_array_is_freed_only_after_every_copy(store):
    q = ShmQueue(8, store=store)
    b = hop(q, np.arange(BIG, dtype=np.float32))
    addr = _ndarray_block(b).addr
    copies = [hop(q, {"k": k, "a": b})["a"] for k in range(3)]  # three items share it
    assert all(_ndarray_block(c).addr == addr for c in copies)
    del b
    other = hop(q, np.ones(BIG, dtype=np.float32))  # copies still hold the block
    assert _ndarray_block(other).addr != addr
    del other
    del copies  # last reference gone: the block is free again (LIFO reuse)
    assert _ndarray_block(hop(q, np.ones(BIG, dtype=np.float32))).addr == addr


def _stress_producer(q, src, n, nbytes):
    for i in range(n):
        q.put({"src": src, "i": i, "a": np.full(nbytes // 8, src * 100_000 + i, dtype=np.int64)})


def _stress_forwarder(qin, qout, n):
    for _ in range(n):
        it = qin.get(timeout=120)
        if not (it["a"] == it["src"] * 100_000 + it["i"]).all():
            raise AssertionError(f"corrupt payload {it['src']}/{it['i']}")
        if it["i"] % 2 == 0:
            qout.put(it)  # forwarded by handle; odd ones are dropped (freed) here
    qout.put(None)


def test_refcounts_hold_up_across_many_processes(store):
    import torch.multiprocessing as mp

    qa, qb = ShmQueue(8, store=store), ShmQueue(8, store=store)
    producers, forwarders, n, nbytes = 4, 4, 150, 512 << 10
    procs = [mp.Process(target=_stress_producer, args=(qa, s, n, nbytes)) for s in range(producers)]
    procs += [mp.Process(target=_stress_forwarder, args=(qa, qb, producers * n // forwarders))
              for _ in range(forwarders)]
    for p in procs:
        p.start()
    got, done = [], 0
    while done < forwarders:
        it = qb.get(timeout=120)
        if it is None:
            done += 1
            continue
        assert (it["a"] == it["src"] * 100_000 + it["i"]).all() and not it["a"].flags.writeable
        got.append((it["src"], it["i"]))
        del it
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    assert sorted(got) == sorted((s, i) for s in range(producers) for i in range(0, n, 2))
    # Live blocks are bounded by the queues and workers, so a recycling store
    # stays at a few segments; leaked references would need ~375 MiB.
    assert store.core.used <= 64 << 20


def test_threads_share_a_store_safely(store):
    import threading

    errors = []

    def worker(k):
        q = ShmQueue(4, store=store)
        try:
            for i in range(150):
                b = hop(q, {"a": np.full(1 << 16, k * 1000 + i, dtype=np.float32)})["a"]
                if not (b == k * 1000 + i).all():
                    errors.append((k, i))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    assert errors == []
    assert store.core.used <= 32 << 20


def _put_mixed(q, src, n):
    for i in range(n):
        size = (64, 300 << 10, 1 << 20)[i % 3]  # ring fast path / spilled message / store array
        q.put({"src": src, "i": i, "blob": bytes([i % 251]) * size if i % 3 < 2 else None,
               "a": np.full(size // 4, i, dtype=np.float32) if i % 3 == 2 else None})


def test_mixed_sizes_cross_process(store):
    import torch.multiprocessing as mp

    q = ShmQueue(6, ring_bytes=256 << 10, store=store)  # payloads > 64 KiB leave the ring
    procs = [mp.Process(target=_put_mixed, args=(q, s, 60)) for s in range(2)]
    for p in procs:
        p.start()
    seen = set()
    for _ in range(120):
        it = q.get(timeout=120)
        i = it["i"]
        if i % 3 < 2:
            assert it["blob"] == bytes([i % 251]) * (64, 300 << 10)[i % 3]
        else:
            assert (it["a"] == i).all()
        seen.add((it["src"], i))
    for p in procs:
        p.join(timeout=30)
    assert len(seen) == 120


# === pipeline shapes ===

class _Arrays:
    def __init__(self, n, size=1 << 16, delay=0.0):
        self.n, self.size, self.delay = n, size, delay

    def __call__(self):
        for i in range(self.n):
            if self.delay:
                time.sleep(self.delay)
            yield {"id": i, "a": np.full(self.size, i, dtype=np.float32)}


class _Pass:
    def __call__(self, item):
        return item


class _CatBatch:
    """batch= stage, fed over an auto-chunked edge: concatenates its items' arrays."""

    def __call__(self, batch):
        assert all(not it["a"].flags.writeable for it in batch)
        cat = np.concatenate([it["a"] for it in batch])
        return {"ids": [it["id"] for it in batch], "sum": float(cat.sum(dtype=np.float64))}


class _Double:
    def __call__(self, item):
        item["b"] = item["a"] * 2  # a new array: enters the store on the way out
        return item


class _RetryEveryThird:
    def __call__(self, item):
        if item["id"] % 3 == 0 and not item.get("tries"):
            item["tries"] = 1
            if self.push(1, item, timeout=30):
                return None
        return item


class _CrashOnItem:
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, item):
        if item["id"] == 5 and not os.path.exists(self.flag):
            open(self.flag, "w").write("x")
            os._exit(1)  # dies holding store references
        return item


@pytest.fixture
def on_store(monkeypatch):
    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    monkeypatch.delenv("PIPE_STORE", raising=False)
    monkeypatch.delenv("PIPE_STORE_MB", raising=False)


def test_batch_stage_over_chunked_edge(on_store):
    n, size = 96, 1 << 15
    pipe = Pipe(stats_interval=0)
    pipe.add(_Arrays(n, size), outqn=32)
    pipe.add(_CatBatch(), workers=2, batch=8, outqn=16)
    pipe.start()
    assert pipe.jobs[0]["chunk_eff"] == 8  # the edge really is chunked
    out = list(pipe)
    assert sorted(i for b in out for i in b["ids"]) == list(range(n))
    assert all(b["sum"] == size * sum(b["ids"]) for b in out)


def test_threaded_stage_with_store_arrays(on_store):
    pipe = Pipe(stats_interval=0)
    pipe.add(_Arrays(120), outqn=16)
    pipe.add(_Double(), workers=6, thread=True, outqn=16)
    got = list(pipe)
    assert sorted(x["id"] for x in got) == list(range(120))
    assert all((x["a"] == x["id"]).all() and (x["b"] == 2 * x["id"]).all() for x in got)


def test_push_back_edge_carries_store_arrays(on_store):
    pipe = Pipe(stats_interval=0)
    pipe.add(_Arrays(60), outqn=64)
    pipe.add(_Pass(), workers=2, outqn=64)
    pipe.add(_RetryEveryThird(), workers=2, outqn=64)
    got = list(pipe)
    assert sorted(x["id"] for x in got) == list(range(60))
    assert all((x["a"] == x["id"]).all() for x in got)
    assert all(x.get("tries") == 1 for x in got if x["id"] % 3 == 0)


def test_worker_crash_while_holding_store_arrays(on_store, tmp_path):
    flag = str(tmp_path / "crashed")
    pipe = Pipe(stats_interval=0, health_check_interval=1, raise_errors=False)
    pipe.add(_Arrays(40, delay=0.05), outqn=4)
    pipe.add(_CrashOnItem(flag), workers=1, outqn=8)
    got = list(pipe)
    assert os.path.exists(flag), "worker never crashed: test is vacuous"
    assert len(got) >= 35  # the crash takes its in-hand item(s) with it
    assert all((x["a"] == x["id"]).all() for x in got)


def test_restart_gets_a_fresh_store(on_store):
    pipe = Pipe(stats_interval=0)
    pipe.add(_Arrays(5), outqn=4)
    pipe.start()
    old = pipe.store.name
    pipe.restart()
    assert pipe.store.name != old
    with pytest.raises(OSError):
        Store.attach(old)
    assert sorted(x["id"] for x in pipe) == list(range(5))


def test_full_store_in_a_pipeline_copies_instead(on_store, monkeypatch):
    monkeypatch.setenv("PIPE_STORE_MB", "1")
    pipe = Pipe(stats_interval=0)
    pipe.add(_Arrays(6, size=1 << 20), outqn=4)  # 4 MiB arrays never fit
    pipe.add(_Pass(), outqn=4)
    got = list(pipe)
    assert sorted(x["id"] for x in got) == list(range(6))
    assert all((x["a"] == x["id"]).all() and x["a"].flags.writeable for x in got)


class _BigArrays:
    """Arrays on both store paths: size-class blocks (2 MiB) and dedicated
    segments (24 MiB > seg_bytes/4)."""

    def __call__(self):
        for i in range(10_000):
            n = (24 << 20) if i % 4 == 0 else (2 << 20)
            yield {"id": i, "a": np.full(n // 4, i, dtype=np.float32)}


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm to list shm objects")
def test_force_stop_mid_traffic_leaves_nothing_behind(on_store):
    pipe = Pipe(stats_interval=0)
    pipe.add(_BigArrays(), outqn=16)
    pipe.add(_Pass(), workers=3, outqn=16)
    it = iter(pipe)
    for _ in range(20):
        next(it)
    names = [pipe.store.name[1:]] + [q._core.name[1:] for q in pipe.queues]
    workers = [p.pid for p in pipe.processes]

    def ours():
        spills = tuple(f"gpqs{p:x}_" for p in [os.getpid(), *workers])
        return [n for n in os.listdir("/dev/shm") if n.startswith(tuple(names)) or n.startswith(spills)]

    assert ours(), "nothing in shared memory: test is vacuous"
    pipe.stop(force=True)  # workers die mid-put / mid-alloc
    assert ours() == []


# === container-sized /dev/shm (Docker's default is 64 MiB) ===

def test_small_store_still_carries_arrays():
    small = PayloadStore(limit_bytes=24 << 20)  # segments sized down to fit (3 MiB)
    try:
        q = ShmQueue(4, store=small)
        for n in (512 << 10, 4 << 20):  # a size-class block, then a dedicated segment
            b = hop(q, np.ones(n // 4, dtype=np.float32))
            assert _ndarray_block(b) is not None and b.sum() == n // 4
            del b
        assert small.core.used <= small.core.limit
    finally:
        small.close()


def test_rings_shrink_to_fit_small_shm(monkeypatch):
    from pipe import shmqueue, shmstore

    monkeypatch.delenv("PIPE_RING_MB", raising=False)
    monkeypatch.setattr(shmstore, "shm_free_bytes", lambda: 64 << 20)
    assert shmqueue.edge_ring_bytes(256, 4) == 4 << 20  # a quarter of free, split 4 ways
    assert shmqueue.edge_ring_bytes(256, 64) == 256 << 10
    assert shmqueue.edge_ring_bytes(256, 10_000) == 64 << 10  # floor
    monkeypatch.setattr(shmstore, "shm_free_bytes", lambda: 64 << 30)
    assert shmqueue.edge_ring_bytes(256, 4) == shmqueue._ring_bytes(256)  # roomy: uncapped


def test_pipeline_in_docker_sized_shm(on_store, monkeypatch, capsys):
    from pipe import shmstore

    monkeypatch.delenv("PIPE_RING_MB", raising=False)
    monkeypatch.setattr(shmstore, "shm_free_bytes", lambda: 64 << 20)
    monkeypatch.setattr(shmstore, "_small_shm_warned", False)
    pipe = Pipe(stats_interval=0)
    pipe.add(_Arrays(40, size=1 << 18), outqn=64)  # 1 MiB arrays
    pipe.add(_Double(), workers=2, outqn=64)
    pipe.start()
    assert pipe.store.core.limit == 32 << 20
    assert all(q._core.ring_bytes <= 8 << 20 for q in pipe.queues)
    got = list(pipe)
    assert sorted(x["id"] for x in got) == list(range(40))
    assert all((x["b"] == 2 * x["id"]).all() for x in got)
    assert "--shm-size" in capsys.readouterr().out
