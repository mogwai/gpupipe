"""Shared payload store: large arrays cross pipeline edges by handle.

Most of an item's bytes usually pass through a pipeline unchanged — the
array a loader produced rides through several stages
that each add a few metadata fields. Pickled normally, every hop copies it
several times. With the store, a numpy array or CPU tensor of at least
PIPE_STORE_MIN_BYTES is written into shared memory ONCE, the first time it is
put on a queue; the message carries a ~100-byte handle, the next stage gets a
zero-copy view, and putting that same view (or a slice of it) onward re-sends
the handle. Copies after the first hop: zero.

Semantics:
- numpy arrays arrive READ-ONLY. An in-place write raises
  ("assignment destination is read-only") instead of silently changing data
  another item or stage may share; call .copy() for a private writable array.
- CPU tensors arrive writable but shared (torch has no read-only tensors),
  just as torch.multiprocessing shares them today.
- Blocks are refcounted across processes and reused once every view is gone.
  If the store is full, items fall back to being copied; nothing blocks.
"""
import os
import pickle
import threading
import weakref
from multiprocessing import resource_tracker

import numpy as np
import torch

from ._rustq import Block, Store

STORE_MIN_BYTES = int(os.environ.get("PIPE_STORE_MIN_BYTES", 64 << 10))

_TENSOR_DTYPES = {
    torch.float32, torch.float64, torch.float16, torch.bfloat16,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
}
_ITEMSIZE = {dt: torch.empty((), dtype=dt).element_size() for dt in _TENSOR_DTYPES}

_attached = {}  # store name -> Store (this process)
_attach_lock = threading.Lock()
_tensor_blocks = weakref.WeakValueDictionary()  # block addr -> Block a tensor was built on
_full_warned = False


SMALL_SHM = 1 << 30  # below this much free /dev/shm, warn and size everything down
_small_shm_warned = False


def shm_free_bytes():
    """Free bytes in /dev/shm, or None where POSIX shm isn't a sized tmpfs
    (macOS: it is plain RAM)."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return None
    return st.f_bavail * st.f_frsize


def warn_if_small_shm():
    """One warning per process when /dev/shm is container-sized (Docker
    defaults to 64 MiB): everything still works, sized down to fit."""
    global _small_shm_warned
    free = shm_free_bytes()
    if free is not None and free < SMALL_SHM and not _small_shm_warned:
        _small_shm_warned = True
        print(f"WARNING: /dev/shm has only {free >> 20} MiB free (Docker's default is 64 MiB): "
              f"pipe's queues and payload store are sized down to fit, and large arrays may be "
              f"copied instead of shared. Run the container with --shm-size=8g (or more).")


def _default_limit():
    mb = os.environ.get("PIPE_STORE_MB")
    if mb:
        return int(float(mb) * (1 << 20))
    free = shm_free_bytes()
    if free is None:
        return 2 << 30
    return min(free // 2, 16 << 30)  # half of what's free, at most 16 GiB


def store(name):
    """This process's handle on the store called `name`, attached on first use."""
    s = _attached.get(name)
    if s is None:
        with _attach_lock:
            s = _attached.get(name)
            if s is None:
                s = _attached[name] = Store.attach(name)
    return s


class PayloadStore:
    """The owner's handle (the pipeline parent): creates the store, registers
    its names with the resource tracker so a crashed parent can't leak them,
    and unlinks everything on close(). Views already handed out stay valid."""

    def __init__(self, limit_bytes=None, seg_bytes=None):
        limit = limit_bytes or _default_limit()
        # 64 MiB segments, smaller for a small store so one still fits (a
        # 32 MiB store in a Docker-sized /dev/shm gets 4 MiB segments).
        self.core = Store(limit, seg_bytes or min(64 << 20, max(1 << 20, limit // 8)))
        self.name = self.core.name
        _attached[self.name] = self.core
        resource_tracker.register(self.name, "shared_memory")
        self._finalizer = weakref.finalize(self, _close, self.core)

    def close(self):
        self._finalizer()


def _close(core):
    names = core.segment_names()
    core.unlink()
    for n in [core.name, *names]:
        resource_tracker.unregister(n, "shared_memory")


def _alloc(st, nbytes):
    global _full_warned
    got = st.alloc(nbytes)
    if got is None:
        if not _full_warned:
            _full_warned = True
            free = shm_free_bytes()
            if st.used + nbytes <= st.limit and free is not None:
                why = f"/dev/shm is full ({free >> 20} MiB free; run Docker with a bigger --shm-size)"
            else:
                why = (f"the payload store can't fit a {nbytes >> 20} MiB array ({st.used >> 20} of "
                       f"{st.limit >> 20} MiB in use; raise PIPE_STORE_MB, or --shm-size in Docker)")
            print(f"WARNING: {why}: copying large arrays until shared memory frees up")
        return None
    blk, new_segment = got
    if new_segment is not None:
        resource_tracker.register(new_segment, "shared_memory")
    return blk


def alloc_bytes(st, data):
    """A store block holding a copy of `data` (a too-big queue message), or
    None if the store is full."""
    blk = _alloc(st, len(data))
    if blk is not None:
        memoryview(blk)[:] = data
    return blk


def load_spilled(name, seg, boff, dedicated):
    """Unpickle a queue message parked in a store block; the block is freed
    as soon as this returns (pickle copies everything out)."""
    return pickle.loads(store(name).adopt(seg, boff, dedicated))


def _ndarray_block(a):
    b = a.base
    while isinstance(b, np.ndarray):
        b = b.base
    return b if type(b) is Block else None


def reduce_ndarray(a, st):
    """Store reduction for a numpy array, or None to pickle it normally."""
    if a.nbytes < STORE_MIN_BYTES or a.dtype.hasobject:
        return None
    blk = _ndarray_block(a)
    if blk is not None and blk.store == st.name:
        off = a.__array_interface__["data"][0] - blk.addr
        try:  # an arbitrary view of the block: forward it as-is if it maps back
            np.ndarray(a.shape, a.dtype, buffer=blk, offset=off, strides=a.strides)
        except (TypeError, ValueError):
            pass
        else:
            return _rebuild_ndarray, (st.name, *blk.share(), a.dtype, a.shape, a.strides, off)
    blk = _alloc(st, a.nbytes)
    if blk is None:
        return None
    dst = np.ndarray(a.shape, a.dtype, buffer=blk)
    np.copyto(dst, a)
    return _rebuild_ndarray, (st.name, *blk.share(), a.dtype, a.shape, dst.strides, 0)


def _rebuild_ndarray(name, seg, boff, dedicated, dtype, shape, strides, off):
    a = np.ndarray(shape, dtype, buffer=store(name).adopt(seg, boff, dedicated), offset=off, strides=strides)
    a.flags.writeable = False
    return a


def reduce_tensor(t, st):
    """Store reduction for a CPU tensor, or None to fall back."""
    if not (
        t.device.type == "cpu"
        and t.layout is torch.strided
        and t.dtype in _TENSOR_DTYPES
        and t.numel() * t.element_size() >= STORE_MIN_BYTES
        and (t.is_leaf or not t.requires_grad)  # torch refuses non-leaf grads; so do we
    ):
        return None
    blk = _tensor_blocks.get(t.untyped_storage().data_ptr())
    if blk is not None and blk.store == st.name:
        off = t.data_ptr() - blk.addr
        return _rebuild_tensor, (st.name, *blk.share(), t.dtype, tuple(t.shape), t.stride(),
                                 off // t.element_size(), t.requires_grad)
    c = t.detach().contiguous()
    nbytes = c.numel() * c.element_size()
    blk = _alloc(st, nbytes)
    if blk is None:
        return None
    torch.frombuffer(blk, dtype=torch.uint8).copy_(c.view(-1).view(torch.uint8))
    return _rebuild_tensor, (st.name, *blk.share(), t.dtype, tuple(t.shape), c.stride(), 0, t.requires_grad)


def _rebuild_tensor(name, seg, boff, dedicated, dtype, shape, stride, storage_offset, requires_grad):
    blk = store(name).adopt(seg, boff, dedicated)
    base = torch.frombuffer(blk, dtype=dtype, count=len(blk) // _ITEMSIZE[dtype])
    _tensor_blocks[blk.addr] = blk
    t = base.as_strided(shape, stride, storage_offset)
    if requires_grad:
        t.requires_grad_(True)
    return t
