"""The Rust shared-memory queue (pipe.shmqueue.ShmQueue) under pipe's edges:
mp.Queue semantics, capacity, spill, cross-process/thread delivery, tensor
transport, crash recovery, and the make_queue fallback."""
import mmap
import os
import pickle
import signal
import struct
import sys
import threading
import time
from collections import Counter
from queue import Empty, Full

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("pipe._rustq")

from pipe import Pipe  # noqa: E402
from pipe.queues import make_queue  # noqa: E402
from pipe.shmqueue import ShmQueue  # noqa: E402


def _producer(q, start, n, size):
    for i in range(start, start + n):
        q.put({"id": i, "pad": b"x" * size})


def _consumer(q, n, out):
    for _ in range(n):
        out.put(q.get(timeout=30)["id"])


def _put_after(q, ready, delay, item):
    ready.set()
    time.sleep(delay)
    q.put(item)


def _send_tensors(q, read):
    q.put({"small": torch.arange(10, dtype=torch.float32), "big": torch.full((1 << 20,), 3.0)})
    read.wait()  # torch's fd strategy: the consumer fetches the fd from us at get()


def test_fifo_and_mp_queue_surface():
    q = ShmQueue(maxsize=3)
    assert q.empty() and not q.full() and q.qsize() == 0
    for i in range(3):
        q.put(i)
    assert q.full() and q.qsize() == 3
    with pytest.raises(Full):
        q.put_nowait(9)
    t0 = time.monotonic()
    with pytest.raises(Full):
        q.put(9, timeout=0.1)
    assert time.monotonic() - t0 >= 0.09
    assert [q.get(), q.get_nowait(), q.get(timeout=1)] == [0, 1, 2]
    with pytest.raises(Empty):
        q.get_nowait()
    with pytest.raises(Empty):
        q.get(timeout=0.05)
    q.close()
    q.cancel_join_thread()
    q.join_thread()


def test_capacity_is_maxsize_items_even_when_ring_bytes_run_out():
    # 10 x 40 KB items cannot fit a 64 KB ring: the overflow spills, and the
    # queue still holds exactly maxsize items, in order.
    q = ShmQueue(maxsize=10, ring_bytes=64 << 10)
    for i in range(10):
        q.put_nowait({"id": i, "pad": os.urandom(40 << 10)})
    with pytest.raises(Full):
        q.put_nowait({"id": 10})
    assert [q.get()["id"] for _ in range(10)] == list(range(10))


def test_unbounded_never_blocks_on_ring_bytes():
    q = ShmQueue(maxsize=0, ring_bytes=64 << 10)
    blob = os.urandom(30 << 10)
    for i in range(200):
        q.put_nowait((i, blob))
    assert q.qsize() == 200
    got = [q.get_nowait() for _ in range(200)]
    assert [i for i, _ in got] == list(range(200)) and all(b == blob for _, b in got)


def test_payload_larger_than_ring_spills():
    q = ShmQueue(maxsize=2, ring_bytes=64 << 10)
    blob = os.urandom(5 << 20)
    q.put(blob)
    q.put(b"small")
    assert q.get() == blob and q.get() == b"small"


def test_get_many_takes_up_to_n_in_order():
    q = ShmQueue()
    for i in range(5):
        q.put(i)
    assert q.get_many(3) == [0, 1, 2]
    assert q.get_many(10) == [3, 4]
    assert q.get_many(10, timeout=0.05) == []


@pytest.mark.parametrize("producers,consumers,size", [(1, 1, 16), (4, 4, 16), (3, 2, 20_000)])
def test_cross_process_no_loss_no_duplicates(producers, consumers, size):
    n = 2000 if size < 1000 else 300
    q = ShmQueue(maxsize=32, ring_bytes=256 << 10)
    out = mp.Queue()
    per_c = n * producers // consumers
    procs = [mp.Process(target=_producer, args=(q, p * n, n, size)) for p in range(producers)]
    procs += [mp.Process(target=_consumer, args=(q, per_c, out)) for _ in range(consumers)]
    for p in procs:
        p.start()
    got = [out.get(timeout=60) for _ in range(per_c * consumers)]
    for p in procs:
        p.join(timeout=30)
    assert Counter(got) == Counter(range(n * producers))
    assert q.qsize() == 0


def test_threads_share_one_queue():
    q = ShmQueue(maxsize=8)
    got = []
    lock = threading.Lock()

    def consume():
        while True:
            v = q.get(timeout=10)
            if v is None:
                return
            with lock:
                got.append(v)

    ts = [threading.Thread(target=consume) for _ in range(4)]
    for t in ts:
        t.start()
    for i in range(2000):
        q.put(i)
    for _ in ts:
        q.put(None)
    for t in ts:
        t.join(timeout=10)
    assert sorted(got) == list(range(2000))


def test_blocked_get_wakes_promptly_on_cross_process_put():
    q = ShmQueue()
    ready = mp.Event()
    p = mp.Process(target=_put_after, args=(q, ready, 0.3, "hi"))
    p.start()
    assert ready.wait(60)  # spawn time is not what this measures
    t0 = time.monotonic()
    assert q.get(timeout=10) == "hi"
    dt = time.monotonic() - t0
    p.join()
    assert 0.2 < dt < 2.0


def test_blocked_get_runs_python_signal_handlers():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals need the main thread")
    q = ShmQueue()

    class Interrupted(Exception):
        pass

    def handler(signum, frame):
        raise Interrupted

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        t0 = time.monotonic()
        with pytest.raises(Interrupted):
            q.get()  # no timeout: only the signal can end this
        assert time.monotonic() - t0 < 2.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def test_attach_by_pickle_and_owner_unlink():
    q = ShmQueue()
    q2 = pickle.loads(pickle.dumps(q))
    q.put("a")
    assert q2.get() == "a"
    q.close()
    q.put("still mapped")  # unlink only removes the name
    assert q2.get() == "still mapped"
    with pytest.raises(OSError):
        pickle.loads(pickle.dumps(q))


@pytest.mark.parametrize("t", [
    torch.arange(12, dtype=torch.float32).reshape(3, 4),
    torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),  # non-contiguous
    torch.randn(5).to(torch.bfloat16),
    torch.tensor([True, False, True]),
    torch.tensor(7, dtype=torch.int64),  # 0-dim
    torch.empty(0, 3),
])
def test_small_cpu_tensors_round_trip_inline(t):
    q = ShmQueue()
    q.put({"t": t})
    r = q.get()["t"]
    assert r.dtype == t.dtype and r.shape == t.shape and torch.equal(r, t)
    assert not t.is_shared()  # inline copy: the producer's tensor is untouched


def test_tensors_inline_only_up_to_the_ring_fast_path():
    q = ShmQueue(maxsize=4, ring_bytes=1 << 20)  # payloads > 256 KiB spill
    small, big = torch.zeros(32 << 10, dtype=torch.uint8), torch.zeros(512 << 10, dtype=torch.uint8)
    q.put(small)
    q.put(big)
    assert torch.equal(q.get(), small) and torch.equal(q.get(), big)
    assert not small.is_shared()  # copied inline
    assert big.is_shared()  # torch's reducer moved it to shared memory


def test_requires_grad_leaf_preserved_and_non_leaf_refused():
    q = ShmQueue()
    leaf = torch.ones(3, requires_grad=True)
    q.put(leaf)
    r = q.get()
    assert r.requires_grad and torch.equal(r.detach(), leaf.detach())
    with pytest.raises(RuntimeError):
        q.put(leaf * 2)  # same refusal torch's reducer gives


def test_large_and_small_tensors_cross_process():
    q = ShmQueue()
    read = mp.Event()
    p = mp.Process(target=_send_tensors, args=(q, read))
    p.start()
    item = q.get(timeout=60)
    read.set()
    p.join()
    assert torch.equal(item["small"], torch.arange(10, dtype=torch.float32))
    assert item["big"].shape == (1 << 20,) and bool((item["big"] == 3.0).all())


def test_lock_held_by_dead_process_is_recovered():
    import _posixshmem

    q = ShmQueue()
    p = mp.Process(target=time.sleep, args=(0,))
    p.start()
    p.join()
    dead = p.pid
    # Forge "held by a process that was SIGKILLed mid-operation": the lock
    # word (offset 64 of the control block) holds the dead pid.
    fd = _posixshmem.shm_open(q._core.name, os.O_RDWR, 0o600)
    try:
        with mmap.mmap(fd, 4096) as mm:
            struct.pack_into("<I", mm, 64, dead)
    finally:
        os.close(fd)
    t0 = time.monotonic()
    q.put("after", timeout=5)
    assert q.get(timeout=5) == "after"
    assert time.monotonic() - t0 < 2.0


def test_make_queue_default_and_override(monkeypatch):
    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    assert isinstance(make_queue(4), ShmQueue)
    monkeypatch.setenv("PIPE_QUEUE", "mp")
    assert not isinstance(make_queue(4), ShmQueue)


def test_make_queue_falls_back_when_ring_cannot_be_created(monkeypatch, capsys):
    import pipe.queues
    import pipe.shmqueue

    def boom(*a, **k):
        raise OSError("no space left")

    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    monkeypatch.setattr(pipe.shmqueue, "ShmQueue", boom)
    monkeypatch.setattr(pipe.queues, "_fallback_warned", False)
    q = make_queue(4)
    assert not isinstance(q, ShmQueue)
    assert "shared-memory queue unavailable" in capsys.readouterr().out
    monkeypatch.setenv("PIPE_QUEUE", "rust")
    with pytest.raises(OSError):
        make_queue(4)


class _Numbers:
    def __call__(self):
        for i in range(3000):
            yield {"id": i, "t": torch.full((4,), float(i))}


class _Double:
    def __call__(self, item):
        item["t"] = item["t"] * 2
        return item


def test_pipeline_runs_on_shm_queues(monkeypatch):
    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    pipe = Pipe(stats_interval=0)
    pipe.add(_Numbers(), outqn=64)
    pipe.add(_Double(), workers=3, outqn=64)
    pipe.add(_Double(), workers=2, outqn=64)
    pipe.start()
    assert all(isinstance(q, ShmQueue) for q in pipe.queues)
    got = {item["id"]: item["t"] for item in pipe}
    assert sorted(got) == list(range(3000))
    assert all(torch.equal(t, torch.full((4,), 4.0 * i)) for i, t in got.items())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="shm reservation (fallocate) is Linux-only")
def test_large_reservations_survive_signals():
    """A signal landing mid-reservation (EINTR) must be retried, not reported as
    a failed put — seen in the wild with multi-MB spills under a busy pipe."""
    got = []
    signal.signal(signal.SIGUSR1, lambda *a: got.append(1))
    main = threading.main_thread().ident
    stop = threading.Event()

    def pester():
        while not stop.is_set():
            signal.pthread_kill(main, signal.SIGUSR1)
            time.sleep(0.0002)

    t = threading.Thread(target=pester)
    t.start()
    try:
        q = ShmQueue(2, ring_bytes=64 << 10)  # everything over 16 KiB spills to a fresh reservation
        blob = os.urandom(32 << 20)
        for _ in range(20):
            q.put(blob)
            assert q.get() == blob
    finally:
        stop.set()
        t.join()
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    assert got, "no signals delivered: test is vacuous"


def _spills_of(pids):
    prefixes = tuple(f"gpqs{p:x}_" for p in pids)
    return sorted(n for n in os.listdir("/dev/shm") if n.startswith(prefixes))


def test_close_removes_unread_spilled_messages():
    q = ShmQueue(8, ring_bytes=64 << 10)  # no store: messages over 16 KiB spill to own objects
    for _ in range(5):
        q.put(os.urandom(100 << 10))
    q.put("small")
    before = _spills_of([os.getpid()]) if os.path.isdir("/dev/shm") else None
    assert q._core.unlink() == 5  # owner teardown reclaims what nobody will read
    if before is not None:
        assert len(before) >= 5 and not set(before) & set(_spills_of([os.getpid()]))


class _BigBlobs:
    def __call__(self):
        for i in range(1000):
            yield {"i": i, "blob": os.urandom(300 << 10)}


class _Forward:
    def __call__(self, item):
        return item


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm to list shm objects")
def test_force_stop_leaves_no_spilled_messages(monkeypatch):
    monkeypatch.delenv("PIPE_QUEUE", raising=False)
    monkeypatch.setenv("PIPE_STORE", "0")  # big messages take the ring's own spill path
    pipe = Pipe(stats_interval=0)
    pipe.add(_BigBlobs(), outqn=8)
    pipe.add(_Forward(), workers=2, outqn=8)
    it = iter(pipe)
    next(it)  # running, queues full of spilled messages
    time.sleep(1.0)
    pids = [p.pid for p in pipe.processes]
    assert _spills_of(pids), "nothing spilled: test is vacuous"
    pipe.stop(force=True)
    assert _spills_of(pids) == []


def _put_big(q, ready):
    ready.set()
    q.put(b"x" * (100 << 10))  # spills, then waits: the queue is full


def test_producer_killed_while_waiting_does_not_leak_its_spill():
    q = ShmQueue(1, ring_bytes=64 << 10)
    q.put("fill")  # maxsize 1: the next put has to wait
    ready = mp.Event()
    p = mp.Process(target=_put_big, args=(q, ready))
    p.start()
    assert ready.wait(60)  # spawn time varies with load; don't guess it
    deadline = time.monotonic() + 30
    while os.path.isdir("/dev/shm") and not _spills_of([p.pid]) and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)  # spilled; now blocked waiting for room
    assert p.is_alive()
    p.kill()
    p.join()
    if os.path.isdir("/dev/shm"):
        assert len(_spills_of([p.pid])) == 1
    assert q._core.unlink() == 1  # the dead producer's spill, found via the pending table
    if os.path.isdir("/dev/shm"):
        assert _spills_of([p.pid]) == []


# === /dev/shm full: oversized messages go to disk ===

def _spill_dir(q):
    import tempfile

    return os.path.join(tempfile.gettempdir(), q._core.name.lstrip("/") + ".spill")


def _put_via_disk(q, n, ready=None):
    q._core._force_disk_spill(True)  # as if /dev/shm were full in this process
    if ready is not None:
        ready.set()
    for i in range(n):
        q.put((i, bytes([i]) * (200 << 10)))


def test_full_shm_spills_to_disk_and_cleans_up():
    q = ShmQueue(8, ring_bytes=64 << 10)  # messages over 16 KiB spill
    q._core._force_disk_spill(True)
    blobs = [os.urandom(200 << 10) for _ in range(5)]
    for b in blobs[:3]:
        q.put(b)
    assert len(os.listdir(_spill_dir(q))) == 3
    assert [q.get() for _ in range(3)] == blobs[:3]
    assert os.listdir(_spill_dir(q)) == []  # each read removes its file
    for b in blobs[3:]:
        q.put(b)
    assert q._core.unlink() == 2  # unread disk spills counted and removed...
    assert not os.path.exists(_spill_dir(q))  # ...with the whole spill dir


def test_disk_spills_cross_process():
    q = ShmQueue(4, ring_bytes=64 << 10)
    p = mp.Process(target=_put_via_disk, args=(q, 12))
    p.start()
    got = [q.get(timeout=60) for _ in range(12)]
    p.join()
    assert [i for i, _ in got] == list(range(12))
    assert all(b == bytes([i]) * (200 << 10) for i, b in got)
    q.close()
    assert not os.path.exists(_spill_dir(q))


def test_writer_killed_holding_a_disk_spill_leaves_nothing():
    q = ShmQueue(1, ring_bytes=64 << 10)
    q.put("fill")  # the writer's spilled message has to wait for room
    ready = mp.Event()
    p = mp.Process(target=_put_via_disk, args=(q, 1, ready))
    p.start()
    assert ready.wait(60)
    deadline = time.monotonic() + 30
    while not (os.path.isdir(_spill_dir(q)) and os.listdir(_spill_dir(q))) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert os.listdir(_spill_dir(q)), "writer never spilled: test is vacuous"
    p.kill()
    p.join()
    q._core.unlink()
    assert not os.path.exists(_spill_dir(q))
