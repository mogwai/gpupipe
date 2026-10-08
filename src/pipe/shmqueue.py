"""Inter-stage queue over the Rust shared-memory ring (`pipe._rustq`).

The queue on every pipeline edge (multiprocessing.Queue's surface). The Rust core only
moves bytes: a put is one memcpy into a ring every attached process maps, a
get is one memcpy out, and nobody enters the kernel unless they have to sleep
— no feeder thread, no pipe, no semaphores. That is 2-8x the item throughput
of mp.Queue on small items (bench/bench_shmqueue.py).

Serialization stays here, on multiprocessing's ForkingPickler, so every
reducer torch registers still applies (CUDA tensors travel by IPC handle,
large CPU tensors by fd/file_system sharing). CPU tensors up to
PIPE_INLINE_TENSOR_BYTES (and small enough for the ring's fast path) are the
exception: their bytes ride inline in the ring. That beats torch's per-tensor
shm object + fd handoff up to several MB on Linux (~250us fixed per tensor
under the file_descriptor strategy; bench/bench_shmqueue.py tensors), and the
consumer gets its own copy rather than memory shared with the producer, so
the producer need not stay alive for the consumer to read it.

Larger numpy arrays and CPU tensors go further when the queue has a payload
store (shmstore.py, one per pipeline): written to shared memory once, then
passed hop to hop by handle with zero further copies.
"""
import io
import os
import pickle
import threading
import weakref
from multiprocessing import resource_tracker
from multiprocessing.reduction import ForkingPickler
from queue import Empty, Full

import numpy as np

from . import _torch, shmstore
from ._rustq import RingQueue

__all__ = ["ShmQueue"]

_INLINE_TENSOR_BYTES = int(os.environ.get("PIPE_INLINE_TENSOR_BYTES", 1 << 20))


def _ring_bytes(maxsize):
    """Ring size for a queue of `maxsize` items. Only a fast path: items that
    don't fit spill to their own shm object, so this never changes capacity.
    Linux reserves it in /dev/shm up front (size it down in small containers
    with PIPE_RING_MB)."""
    mb = os.environ.get("PIPE_RING_MB")
    if mb:
        return int(float(mb) * (1 << 20))
    if maxsize <= 0:
        return 8 << 20
    return min(max(maxsize * (64 << 10), 1 << 20), 16 << 20)


def edge_ring_bytes(maxsize, n_edges):
    """Ring size for one of a pipeline's `n_edges` edges: _ring_bytes, capped
    so all the rings together take at most a quarter of what /dev/shm has free
    (rings are reserved up front; the payload store and torch need the rest).
    A small ring only means more messages ride in store blocks or spills."""
    want = _ring_bytes(maxsize)
    free = shmstore.shm_free_bytes()
    if free is None or os.environ.get("PIPE_RING_MB"):
        return want
    return max(64 << 10, min(want, free // 4 // max(1, n_edges)))


def _rebuild_inline(buf, dtype, shape, requires_grad):
    import torch

    t = torch.frombuffer(buf, dtype=dtype) if len(buf) else torch.empty(0, dtype=dtype)
    t = t.reshape(shape)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _tensor_reducer(torch_reduce, torch):
    def reduce(t):
        st = _tls.store
        if st is not None:
            r = shmstore.reduce_tensor(t, st)
            if r is not None:
                return r
        if (
            t.device.type == "cpu"
            and t.layout is torch.strided
            and t.dtype in shmstore.tensor_itemsizes()
            and t.numel() * t.element_size() <= _tls.inline_max
            and (t.is_leaf or not t.requires_grad)  # torch refuses non-leaf grads; so do we
        ):
            raw = t.detach().contiguous().view(-1).view(torch.uint8).numpy()
            return _rebuild_inline, (pickle.PickleBuffer(raw), t.dtype, tuple(t.shape), t.requires_grad)
        return torch_reduce(t)

    return reduce


def _reduce_ndarray(a):
    st = _tls.store
    if st is not None:
        r = shmstore.reduce_ndarray(a, st)
        if r is not None:
            return r
    return a.__reduce_ex__(pickle.HIGHEST_PROTOCOL)


_tls = threading.local()


def _dumps(obj, inline_max, store):
    # ForkingPickler() copies its whole dispatch table on construction; keep
    # one per thread, rebuilt only if a reducer was registered since.
    _tls.inline_max = inline_max  # read by the reducers during dump()
    _tls.store = store
    st = getattr(_tls, "p", None)
    n = len(ForkingPickler._extra_reducers)
    if st is None or st[2] != n:
        buf = io.BytesIO()
        p = ForkingPickler(buf, pickle.HIGHEST_PROTOCOL)
        # Tensor fast paths only once a stage has imported torch (which
        # registers its reducers, changing n and so rebuilding this pickler).
        torch = _torch.loaded()
        if torch is not None and torch.Tensor in p.dispatch_table:
            p.dispatch_table[torch.Tensor] = _tensor_reducer(p.dispatch_table[torch.Tensor], torch)
        p.dispatch_table[np.ndarray] = _reduce_ndarray
        st = _tls.p = (buf, p, n)
    buf, p, _ = st
    buf.seek(0)
    buf.truncate()
    p.clear_memo()
    p.dump(obj)
    return buf.getvalue()


_loads = pickle.loads
# Marks a message whose pickle lives in a store block (see ShmQueue.put).
# Never the first byte of a real pickle: protocol 2+ pickles start with PROTO.
_SPILLED = b"\x00"


def _load(b):
    if b[:1] == _SPILLED:
        load, args = _loads(b[1:])
        return load(*args)
    return _loads(b)


def _release(core):
    core.unlink()
    resource_tracker.unregister(core.name, "shared_memory")


class ShmQueue:
    """multiprocessing.Queue surface (put/get/qsize/close/...) over RingQueue.

    The creating process owns the shm name: it is registered with the
    resource tracker (so a crashed parent still cleans it up) and unlinked on
    close() or garbage collection. Copies that reach other processes by
    pickling attach to the same ring and never unlink it.

    With a `store` (shmstore.PayloadStore), large arrays put on this queue go
    through the shared payload store instead of being copied."""

    def __init__(self, maxsize=0, ring_bytes=None, store=None, *, _core=None):
        self._finalizer = None
        if _core is None:
            _core = RingQueue(maxsize, ring_bytes or _ring_bytes(maxsize))
            resource_tracker.register(_core.name, "shared_memory")
            self._finalizer = weakref.finalize(self, _release, _core)
        self._core = _core
        self._maxsize = _core.maxsize
        # Inline only what the ring carries directly; a spilled tensor would
        # pay a shm object per item anyway, so leave those to torch.
        self._inline_max = min(_INLINE_TENSOR_BYTES, _core.spill_at)
        self._store = store.core if isinstance(store, shmstore.PayloadStore) else store
        self._spill_at = _core.spill_at

    def __reduce__(self):
        return (_rebuild, (self._core, self._store.name if self._store is not None else None))

    def __repr__(self):
        return f"<ShmQueue {self._core.name} maxsize={self._maxsize} ring={self._core.ring_bytes >> 10}KiB>"

    def put(self, obj, block=True, timeout=None):
        self.put_encoded(self.encode(obj), block, timeout)

    def encode(self, obj):
        """`obj` as this queue sends it, for `put_encoded`: made once however
        often a full queue makes the put retry. Each encoding shares the
        message's arrays anew (a store block reference, torch's fd for a big
        CPU tensor) and an attempt that didn't go never gave them back: a
        stage blocked on a full queue leaked them ~10 times a second (akro:
        its 16 GiB store full in a minute, then ~100k fds)."""
        data = _dumps(obj, self._inline_max, self._store)
        disk = False
        if len(data) > self._spill_at and self._store is not None:
            # Too big for the ring's fast path (e.g. a large bytes field): park
            # it in a recycled store block rather than a fresh shm object per
            # message, which costs reserve + page faults every time on Linux.
            blk = shmstore.alloc_bytes(self._store, data)
            if blk is not None:
                data = pickle.dumps((shmstore.load_spilled, (self._store.name, *blk.share())), 5)
                data = _SPILLED + data
            else:
                # Store full: spill to disk rather than growing shared memory
                # (RAM) past the store's cap with a shm object per message.
                disk = True
        return data, disk

    def put_encoded(self, encoded, block=True, timeout=None):
        """Put what `encode` made; raises Full as `put` does (try it again)."""
        data, disk = encoded
        if not self._core.put(data, block, timeout, disk):
            raise Full

    def put_nowait(self, obj):
        self.put(obj, False)

    def get(self, block=True, timeout=None):
        b = self._core.get(block, timeout)
        if b is None:
            raise Empty
        return _load(b)

    def get_nowait(self):
        return self.get(False)

    def get_many(self, max_items, block=True, timeout=None):
        """Up to max_items objects for one lock round-trip; [] if none came."""
        return [_load(b) for b in self._core.get_many(max_items, block, timeout)]

    def qsize(self):
        return self._core.qsize()

    def empty(self):
        return self._core.qsize() == 0

    def full(self):
        return 0 < self._maxsize <= self._core.qsize()

    def close(self):
        """Owner: unlink the name. Items already in the ring stay readable by
        processes that have it mapped; new attaches fail."""
        if self._finalizer is not None:
            self._finalizer()

    # Puts are synchronous (no feeder thread), so there is nothing to join.
    def join_thread(self):
        pass

    def cancel_join_thread(self):
        pass


def _rebuild(core, store_name):
    return ShmQueue(store=shmstore.store(store_name) if store_name else None, _core=core)
