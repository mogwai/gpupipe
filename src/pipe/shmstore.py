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
  just as torch's own process sharing does.
- Blocks are refcounted across processes and reused once every view is gone.
  If the store is full, items fall back to being copied; nothing blocks.
"""
import functools
import os
import pickle
import re
import shutil
import tempfile
import threading
import weakref
from multiprocessing import resource_tracker

import numpy as np

from ._rustq import Block, Store

STORE_MIN_BYTES = int(os.environ.get("PIPE_STORE_MIN_BYTES", 64 << 10))

@functools.lru_cache(maxsize=1)
def tensor_itemsizes():
    """{dtype: bytes per element} for the tensor dtypes moved as raw bytes
    (inline or via the store). Built on first use: torch is only imported in
    processes whose stages use it."""
    import torch

    dts = (torch.float32, torch.float64, torch.float16, torch.bfloat16,
           torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
    return {dt: torch.empty((), dtype=dt).element_size() for dt in dts}

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


# Linux shm/spill names start "gpq"/"gps" + "<pid-ns inode>.<owner pid>_"
# (sys.rs owner_tag); spill dirs are "<ring name>.spill" in the temp dir.
_OWNED = re.compile(r"^gp[qs]([0-9a-f]+)\.([0-9a-f]+)_")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_dead_owners():
    """Remove queues, payload-store segments and spills left in /dev/shm (and
    spill dirs in the temp dir) by pipelines whose owner process died without
    cleaning up — a crash, OOM kill, or SIGKILL. A normal stop removes them,
    and the resource tracker covers what the owner registered, but segments a
    worker created on demand, and anything left when the tracker died too,
    would otherwise sit in RAM until reboot. Linux only (macOS can't list shm
    objects). Only names from this pid namespace are judged, so containers
    sharing /dev/shm (--ipc=host) never sweep each other's. Returns the count."""
    if not os.path.isdir("/dev/shm"):
        return 0
    try:
        ns = os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        return 0
    removed = 0
    for base, is_dir in (("/dev/shm", False), (tempfile.gettempdir(), True)):
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for name in names:
            m = _OWNED.match(name)
            if not m or int(m.group(1), 16) != ns or (is_dir and not name.endswith(".spill")):
                continue
            if _alive(int(m.group(2), 16)):
                continue
            path = os.path.join(base, name)
            try:
                if is_dir:
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
                removed += 1
            except OSError:
                pass
    return removed


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
        # Register every name the store can ever create, here in the owner:
        # segments are created by whichever worker needs room first, and a
        # worker registering its own raced the owner's unregister at stop (a
        # worker killed in between left the tracker a KeyError). One process
        # registering and unregistering one list can't race. Regular segments
        # are capped by limit // seg_bytes; dedicated ones are unlinked by
        # whoever frees them and swept by close().
        names = [self.name, *self.core.segment_names(self.core.limit // self.core.seg_bytes)]
        for n in names:
            resource_tracker.register(n, "shared_memory")
        self._finalizer = weakref.finalize(self, _close, self.core, names)

    def close(self):
        self._finalizer()


def _close(core, registered):
    core.unlink()
    for n in registered:
        resource_tracker.unregister(n, "shared_memory")


def _alloc(st, nbytes):
    global _full_warned
    got = st.alloc(nbytes)
    if got is None:
        if not _full_warned:
            _full_warned = True
            # Which limit stopped it: a new size-class block needs a whole new
            # segment, a big one its own dedicated segment.
            dedicated = nbytes + 64 > min(64 << 20, st.seg_bytes // 4)
            need = nbytes if dedicated else st.seg_bytes
            free = shm_free_bytes()
            if free is not None and free < need + (1 << 20):
                why = f"/dev/shm is full ({free >> 20} MiB free; give Docker a bigger --shm-size)"
            else:
                why = (f"the payload store is at its cap ({st.used >> 20} of {st.limit >> 20} MiB "
                       f"reserved; raise PIPE_STORE_MB)")
            print(f"WARNING: {why}: copying large arrays, and spilling big messages to disk, "
                  f"until shared memory frees up")
        return None
    return got


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
    import torch

    if not (
        t.device.type == "cpu"
        and t.layout is torch.strided
        and t.dtype in tensor_itemsizes()
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
    import torch

    blk = store(name).adopt(seg, boff, dedicated)
    base = torch.frombuffer(blk, dtype=dtype, count=len(blk) // tensor_itemsizes()[dtype])
    _tensor_blocks[blk.addr] = blk
    t = base.as_strided(shape, stride, storage_offset)
    if requires_grad:
        t.requires_grad_(True)
    return t
