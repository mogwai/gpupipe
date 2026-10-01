//! Payload store: one per pipeline, shared by every stage process. Large
//! array payloads are written into a block here once; queue messages carry a
//! (segment, offset) handle, receivers map the block zero-copy, and passing
//! the same buffer onward re-sends the handle, never the bytes.
//!
//! Layout: a control segment (mutex, segment counters, size-class free
//! lists) plus data segments created on demand — `seg_bytes` "regular"
//! segments carved into size-class blocks, and a "dedicated" segment per
//! block too big for that (unlinked again when the block is freed). Every
//! segment is reserved when created (sys::shm_map), and `limit` caps the
//! total: past it alloc returns None and the caller falls back to copying.
//!
//! Each block starts with a 64-byte header holding a cross-process refcount.
//! A Block object (any process) owns one reference; Block.share() takes one
//! more for a handle about to be pickled, which the receiver's Block adopts.
//! The last release puts the block back on its free list. References held
//! by a process that dies leak until the store is unlinked at pipeline stop.

use crate::sys::{lock, shm_map, shm_unlink, unique_name, unlock, Line, Map};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

const MAGIC: u64 = 0x7069_7065_7374_0001;
const CTL_BYTES: usize = 4096;
const BLK_HDR: u64 = 64;
const MIN_BLOCK: u64 = 4096;
const NCLASS: usize = 64;
// Largest size-class block (header included); bigger ones get a dedicated
// segment. Also capped at seg_bytes/4 so a regular segment holds several.
const MAX_CLASS_BLOCK: u64 = 1 << 26;
const DEDICATED_ALIGN: u64 = 1 << 16;

#[repr(C)]
struct Ctl {
    magic: AtomicU64,
    limit: u64,
    seg_bytes: u64,
    lock: Line<AtomicU32>,
    nreg: AtomicU64, // regular segments created (ids 0..nreg)
    nded: AtomicU64, // dedicated segments created (ids 0..nded)
    cur: AtomicU64,  // regular segment being carved; u64::MAX = none
    bump: AtomicU64, // next free offset in `cur`
    used: AtomicU64, // bytes of live segments, against `limit`
    // Free-list heads per size class: packed (seg, off) + 1; 0 = empty.
    free: [AtomicU64; NCLASS],
}

#[repr(C)]
struct Hdr {
    rc: AtomicU32,
    class: AtomicU32,
    len: AtomicU64,  // payload bytes (what the buffer exposes)
    next: AtomicU64, // free-list link while free
}

fn pack(seg: u64, off: u64) -> u64 {
    (seg << 40) | off
}

fn unpack(p: u64) -> (u64, u64) {
    (p >> 40, p & ((1 << 40) - 1))
}

/// Size class for a block of `total` bytes (header included): 4 classes per
/// power of two above 4 KiB, so rounding wastes at most 25%.
fn class_of(total: u64) -> (usize, u64) {
    if total <= MIN_BLOCK {
        return (0, MIN_BLOCK);
    }
    let p = 63 - (total - 1).leading_zeros() as u64; // 2^p < total <= 2^(p+1)
    let step = 1u64 << (p - 2);
    let k = (total - 1 - (1u64 << p)) / step;
    ((((p - 12) * 4) + k + 1) as usize, (1u64 << p) + (k + 1) * step)
}

fn seg_name(base: &str, id: u64, dedicated: bool) -> String {
    if dedicated {
        format!("{base}d{id:x}")
    } else {
        format!("{base}r{id:x}")
    }
}

enum SegRef {
    Regular(Arc<Map>),    // reused for the store's lifetime: keep mapped
    Dedicated(Weak<Map>), // one block: unmap once this process drops it
}

struct Inner {
    name: String,
    ctl: Map,
    pid: u32,
    segs: Mutex<HashMap<(u64, bool), SegRef>>,
}

impl Inner {
    fn ctl(&self) -> &Ctl {
        unsafe { &*(self.ctl.base as *const Ctl) }
    }

    fn max_class_block(&self) -> u64 {
        MAX_CLASS_BLOCK.min(self.ctl().seg_bytes / 4)
    }

    /// This process's mapping of a segment, attaching on first use.
    fn segment(&self, id: u64, dedicated: bool) -> PyResult<Arc<Map>> {
        let mut segs = self.segs.lock().unwrap();
        match segs.get(&(id, dedicated)) {
            Some(SegRef::Regular(m)) => return Ok(m.clone()),
            Some(SegRef::Dedicated(w)) => {
                if let Some(m) = w.upgrade() {
                    return Ok(m);
                }
            }
            None => {}
        }
        let m = Arc::new(shm_map(&seg_name(&self.name, id, dedicated), None)?);
        let r = if dedicated { SegRef::Dedicated(Arc::downgrade(&m)) } else { SegRef::Regular(m.clone()) };
        segs.insert((id, dedicated), r);
        Ok(m)
    }
}

fn hdr(m: &Map, off: u64) -> &Hdr {
    unsafe { &*(m.base.add(off as usize) as *const Hdr) }
}

/// One reference to a block: exposes its payload through the buffer
/// protocol (numpy/torch views keep the Block, hence the reference, alive).
#[pyclass(frozen, weakref, module = "pipe._rustq")]
pub struct Block {
    inner: Arc<Inner>,
    seg: Arc<Map>,
    seg_id: u64,
    off: u64,
    dedicated: bool,
    len: usize,
}

impl Block {
    fn hdr(&self) -> &Hdr {
        hdr(&self.seg, self.off)
    }

    fn data(&self) -> *mut u8 {
        unsafe { self.seg.base.add((self.off + BLK_HDR) as usize) }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if self.hdr().rc.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let c = self.inner.ctl();
        if self.dedicated {
            shm_unlink(&seg_name(&self.inner.name, self.seg_id, true));
            c.used.fetch_sub(self.seg.len as u64, Ordering::Relaxed);
            return;
        }
        // Blocks are only dropped from Python object teardown, which runs
        // attached to the interpreter.
        let py = unsafe { Python::assume_attached() };
        let h = self.hdr();
        let cls = h.class.load(Ordering::Relaxed) as usize;
        lock(py, &c.lock.0, self.inner.pid);
        h.next.store(c.free[cls].load(Ordering::Relaxed), Ordering::Relaxed);
        c.free[cls].store(pack(self.seg_id, self.off) + 1, Ordering::Relaxed);
        unlock(&c.lock.0);
    }
}

#[pymethods]
impl Block {
    /// Take one more reference for a handle about to be pickled; returns
    /// (segment, offset, dedicated) for Store.adopt() on the other side.
    fn share(&self) -> (u64, u64, bool) {
        self.hdr().rc.fetch_add(1, Ordering::Relaxed);
        (self.seg_id, self.off, self.dedicated)
    }

    /// Address of the payload in this process (to recognise views of it).
    #[getter]
    fn addr(&self) -> usize {
        self.data() as usize
    }

    #[getter]
    fn store(&self) -> &str {
        &self.inner.name
    }

    fn __len__(&self) -> usize {
        self.len
    }

    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        let b = slf.get();
        // Writable: numpy arrays are flipped read-only on the Python side;
        // torch has no read-only tensors and shares writes, as torch does.
        let rc =
            unsafe { ffi::PyBuffer_FillInfo(view, slf.as_ptr(), b.data().cast(), b.len as ffi::Py_ssize_t, 0, flags) };
        if rc != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
}

#[pyclass(frozen, module = "pipe._rustq")]
pub struct Store {
    inner: Arc<Inner>,
}

impl Store {
    fn block(&self, m: Arc<Map>, seg_id: u64, off: u64, dedicated: bool) -> Block {
        let len = hdr(&m, off).len.load(Ordering::Relaxed) as usize;
        Block { inner: self.inner.clone(), seg: m, seg_id, off, dedicated, len }
    }

    fn alloc_dedicated(&self, nbytes: u64) -> PyResult<Option<Block>> {
        let c = self.inner.ctl();
        let size = (BLK_HDR + nbytes).div_ceil(DEDICATED_ALIGN) * DEDICATED_ALIGN;
        if c.used.fetch_add(size, Ordering::Relaxed) + size > c.limit {
            c.used.fetch_sub(size, Ordering::Relaxed);
            return Ok(None);
        }
        let id = c.nded.fetch_add(1, Ordering::Relaxed);
        let m = match shm_map(&seg_name(&self.inner.name, id, true), Some(size as usize)) {
            Ok(m) => Arc::new(m),
            Err(_) => {
                // e.g. /dev/shm full: treat like a full store, caller copies.
                c.used.fetch_sub(size, Ordering::Relaxed);
                return Ok(None);
            }
        };
        let h = hdr(&m, 0);
        h.rc.store(1, Ordering::Relaxed);
        h.class.store(u32::MAX, Ordering::Relaxed);
        h.len.store(nbytes, Ordering::Relaxed);
        self.inner.segs.lock().unwrap().insert((id, true), SegRef::Dedicated(Arc::downgrade(&m)));
        Ok(Some(self.block(m, id, 0, true)))
    }
}

#[pymethods]
impl Store {
    /// Create a store capped at `limit_bytes` of shared memory, grown in
    /// `seg_bytes` segments.
    #[new]
    #[pyo3(signature = (limit_bytes, seg_bytes = 64 << 20))]
    fn new(limit_bytes: u64, seg_bytes: u64) -> PyResult<Self> {
        assert!(std::mem::size_of::<Ctl>() <= CTL_BYTES);
        let seg_bytes = seg_bytes.max(MIN_BLOCK * 16).next_multiple_of(DEDICATED_ALIGN);
        let name = unique_name("gps");
        let ctl = shm_map(&name, Some(CTL_BYTES))?;
        unsafe {
            let c = ctl.base as *mut Ctl;
            (*c).limit = limit_bytes;
            (*c).seg_bytes = seg_bytes;
        }
        let inner = Inner { name, ctl, pid: std::process::id(), segs: Mutex::new(HashMap::new()) };
        inner.ctl().cur.store(u64::MAX, Ordering::Relaxed);
        inner.ctl().magic.store(MAGIC, Ordering::Release);
        Ok(Store { inner: Arc::new(inner) })
    }

    #[staticmethod]
    fn attach(name: &str) -> PyResult<Self> {
        let ctl = shm_map(name, None)?;
        let inner = Inner { name: name.to_string(), ctl, pid: std::process::id(), segs: Mutex::new(HashMap::new()) };
        if inner.ctl.len < CTL_BYTES || inner.ctl().magic.load(Ordering::Acquire) != MAGIC {
            return Err(PyValueError::new_err(format!("{name} is not a pipe payload store")));
        }
        Ok(Store { inner: Arc::new(inner) })
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let attach = py.get_type::<Store>().getattr("attach")?;
        Ok((attach, (self.inner.name.clone(),)))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn limit(&self) -> u64 {
        self.inner.ctl().limit
    }

    /// Bytes of shared memory the store's segments currently hold.
    #[getter]
    fn used(&self) -> u64 {
        self.inner.ctl().used.load(Ordering::Relaxed)
    }

    /// A fresh block with room for `nbytes` (one reference, owned by the
    /// returned Block), plus the name of a regular segment this call had to
    /// create (for the resource tracker); None if the store is full.
    fn alloc(&self, py: Python<'_>, nbytes: u64) -> PyResult<Option<(Block, Option<String>)>> {
        let inner = &self.inner;
        let c = inner.ctl();
        let total = BLK_HDR + nbytes;
        if total > inner.max_class_block() {
            return Ok(self.alloc_dedicated(nbytes)?.map(|b| (b, None)));
        }
        let (cls, size) = class_of(total);
        lock(py, &c.lock.0, inner.pid);
        let got = (|| -> PyResult<Option<(u64, u64, Option<String>)>> {
            let head = c.free[cls].load(Ordering::Relaxed);
            if head != 0 {
                let (seg, off) = unpack(head - 1);
                let m = inner.segment(seg, false)?;
                c.free[cls].store(hdr(&m, off).next.load(Ordering::Relaxed), Ordering::Relaxed);
                return Ok(Some((seg, off, None)));
            }
            let cur = c.cur.load(Ordering::Relaxed);
            let bump = c.bump.load(Ordering::Relaxed);
            if cur != u64::MAX && bump + size <= c.seg_bytes {
                c.bump.store(bump + size, Ordering::Relaxed);
                return Ok(Some((cur, bump, None)));
            }
            if c.used.load(Ordering::Relaxed) + c.seg_bytes > c.limit {
                return Ok(None);
            }
            // Claim the id before creating the segment: a process killed while
            // creating it (reserving 64 MiB takes a while) must not leave a
            // segment past `nreg`, where unlink()'s sweep would never look.
            let id = c.nreg.load(Ordering::Relaxed);
            c.nreg.store(id + 1, Ordering::Relaxed);
            let name = seg_name(&inner.name, id, false);
            let Ok(m) = shm_map(&name, Some(c.seg_bytes as usize)) else {
                c.nreg.store(id, Ordering::Relaxed); // still under the lock: give it back
                return Ok(None); // /dev/shm full: same as a full store
            };
            inner.segs.lock().unwrap().insert((id, false), SegRef::Regular(Arc::new(m)));
            c.used.fetch_add(c.seg_bytes, Ordering::Relaxed);
            c.cur.store(id, Ordering::Relaxed);
            c.bump.store(size, Ordering::Relaxed);
            Ok(Some((id, 0, Some(name))))
        })();
        unlock(&c.lock.0);
        let Some((seg, off, created)) = got? else {
            return Ok(None);
        };
        let m = inner.segment(seg, false)?;
        let h = hdr(&m, off);
        h.rc.store(1, Ordering::Relaxed);
        h.class.store(cls as u32, Ordering::Relaxed);
        h.len.store(nbytes, Ordering::Relaxed);
        Ok(Some((self.block(m, seg, off, false), created)))
    }

    /// Wrap a reference that arrived in a handle (from Block.share()).
    fn adopt(&self, seg: u64, off: u64, dedicated: bool) -> PyResult<Block> {
        let m = self.inner.segment(seg, dedicated)?;
        if off + BLK_HDR > m.len as u64 {
            return Err(PyValueError::new_err("block handle outside its segment"));
        }
        Ok(self.block(m, seg, off, dedicated))
    }

    /// Names of the regular segments created so far (resource tracker).
    fn segment_names(&self) -> Vec<String> {
        (0..self.inner.ctl().nreg.load(Ordering::Relaxed)).map(|i| seg_name(&self.inner.name, i, false)).collect()
    }

    /// Remove every name the store created. Mapped blocks stay valid; no new
    /// process can attach. Called by the owner at pipeline stop.
    fn unlink(&self) {
        let c = self.inner.ctl();
        shm_unlink(&self.inner.name);
        for i in 0..c.nreg.load(Ordering::Relaxed) {
            shm_unlink(&seg_name(&self.inner.name, i, false));
        }
        for i in 0..c.nded.load(Ordering::Relaxed) {
            shm_unlink(&seg_name(&self.inner.name, i, true));
        }
    }
}
