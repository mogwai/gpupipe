//! Bounded multi-producer/multi-consumer byte queue in POSIX shared memory —
//! the transport under pipe's inter-stage queues (see src/pipe/shmqueue.py).
//!
//! One shm segment per queue: a 4 KiB control block (mutex, ring offsets,
//! wakeup sequence words) followed by a byte ring of variable-length records.
//! Any process that attaches by name exchanges payloads with one memcpy each
//! way and no kernel call on the uncontended path; a futex (Linux) / ulock
//! (macOS) is only touched when someone has to sleep. A payload the ring
//! can't take right now goes through its own shm object ("spill"), so
//! capacity is exactly `maxsize` items, as on multiprocessing.Queue.
//!
//! Crash tolerance: the mutex word holds the owner's pid, so a waiter can
//! take over a lock whose holder died (a SIGKILLed worker). Records are
//! written before the offset that publishes them, and consumers judge
//! emptiness by offsets, never by the item count, so a death mid-operation
//! loses at most the in-flight item and leaves the ring readable.

use crate::sys::{futex, is_no_space, lock, shm_map, shm_map_io, shm_unlink, unique_name, unlock, Line, Map, SLICE};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const MAGIC: u64 = 0x7069_7065_7271_0002;
const HDR: usize = 4096;
const REC_HDR: u64 = 8;
const KIND_DATA: u32 = 1;
const KIND_PAD: u32 = 2;
const KIND_SPILL: u32 = 3; // payload too big for the ring: in its own shm object
const KIND_FILE: u32 = 4; // ...and /dev/shm was full: in a file (see put)
// Spill objects of producers waiting for room: no record points at them yet,
// so this table (in the control block, under the lock) is how owner teardown
// finds them if the producer is killed mid-wait (force stop, hang_timeout).
const PENDING_OFF: usize = 2048;
const PENDING_SLOTS: usize = 48;
const PENDING_NAME: usize = 32;

#[repr(C)]
struct Shared {
    magic: AtomicU64,
    cap: u64,
    maxsize: u64,
    spill_at: u64,
    lock: Line<AtomicU32>, // 0 free, else holder pid | WAITERS
    head: AtomicU64,       // offsets and count are only written under `lock`
    tail: AtomicU64,
    count: AtomicU64,
    items_seq: Line<AtomicU32>, // bumped per put; consumers sleep on it
    space_seq: Line<AtomicU32>, // bumped per get; producers sleep on it
    waiting_consumers: AtomicU32,
    waiting_producers: AtomicU32,
}

fn align8(n: u64) -> u64 {
    (n + 7) & !7
}

/// Write `data` to its own shm object; returns the record payload
/// (u64 length + name). The consumer unlinks it after reading.
/// Errors keep their OS code: "no space" (is_no_space) is backpressure for
/// the caller, anything else is a real failure.
fn spill_write(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let name = unique_name("gpqs");
    let m = shm_map_io(&name, Some(data.len().max(1)))?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), m.base, data.len());
    }
    let mut rec = (data.len() as u64).to_le_bytes().to_vec();
    rec.extend_from_slice(name.as_bytes());
    Ok(rec)
}

fn spill_name(rec: &[u8]) -> &str {
    std::str::from_utf8(&rec[8..]).unwrap_or("")
}

fn spill_read<'py>(py: Python<'py>, rec: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let n = u64::from_le_bytes(rec[..8].try_into().unwrap()) as usize;
    let name = spill_name(rec);
    let m = shm_map(name, None)?;
    shm_unlink(name);
    Ok(PyBytes::new(py, unsafe { std::slice::from_raw_parts(m.base, n) }))
}

#[pyclass(frozen, module = "pipe._rustq")]
pub struct RingQueue {
    name: String,
    map: Map,
    pid: u32,
    force_disk: AtomicBool, // test hook: spill to disk as if /dev/shm were full
}

/// None = non-blocking; Some(None) = wait forever; Some(Some(t)) = until t.
fn deadline_of(block: bool, timeout: Option<f64>) -> PyResult<Option<Option<Instant>>> {
    if !block {
        return Ok(None);
    }
    match timeout {
        None => Ok(Some(None)),
        Some(t) if t < 0.0 => Err(PyValueError::new_err("'timeout' must be a non-negative number")),
        Some(t) => Ok(Some(Some(Instant::now() + Duration::from_secs_f64(t)))),
    }
}

fn expired(deadline: Option<Option<Instant>>) -> bool {
    match deadline {
        None => true,
        Some(None) => false,
        Some(Some(d)) => Instant::now() >= d,
    }
}

impl RingQueue {
    fn sh(&self) -> &Shared {
        unsafe { &*(self.map.base as *const Shared) }
    }

    fn data(&self) -> *mut u8 {
        unsafe { self.map.base.add(HDR) }
    }

    fn lock(&self, py: Python<'_>) {
        lock(py, &self.sh().lock.0, self.pid);
    }

    fn unlock(&self) {
        unlock(&self.sh().lock.0);
    }

    /// Sleep until `seq` moves off `seen` or `deadline` passes, in bounded
    /// slices so Python signal handlers run. Caller re-checks the ring.
    fn sleep_on(&self, py: Python<'_>, seq: &AtomicU32, seen: u32, deadline: Option<Instant>) -> PyResult<()> {
        let dur = match deadline {
            None => SLICE,
            Some(d) => d.saturating_duration_since(Instant::now()).min(SLICE),
        };
        if !dur.is_zero() {
            py.detach(|| futex::wait(seq, seen, dur));
        }
        py.check_signals()
    }

    fn pending_slot(&self, i: usize) -> *mut u8 {
        unsafe { self.map.base.add(PENDING_OFF + i * PENDING_NAME) }
    }

    /// Note a waiting producer's spill object (lock held). None if the table
    /// is full: the object then leaks only if that producer dies waiting.
    fn note_pending(&self, name: &str) -> Option<usize> {
        let b = name.as_bytes();
        if b.len() >= PENDING_NAME {
            return None;
        }
        (0..PENDING_SLOTS).find(|&i| unsafe {
            let p = self.pending_slot(i);
            if *p != 0 {
                return false;
            }
            std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len());
            *p.add(b.len()) = 0;
            true
        })
    }

    fn clear_pending(&self, slot: Option<usize>) {
        if let Some(i) = slot {
            unsafe { *self.pending_slot(i) = 0 };
        }
    }

    /// Empty the pending table, returning the names it held (lock held).
    fn take_pending(&self) -> Vec<String> {
        (0..PENDING_SLOTS)
            .filter_map(|i| unsafe {
                let p = self.pending_slot(i);
                if *p == 0 {
                    return None;
                }
                let name = std::ffi::CStr::from_ptr(p.cast()).to_string_lossy().into_owned();
                *p = 0;
                Some(name)
            })
            .collect()
    }

    /// This ring's directory for disk spills: one per ring, so owner teardown
    /// can remove whatever is left in it, a killed writer's files included.
    fn spill_dir(&self) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}.spill", self.name.trim_start_matches('/')))
    }

    /// Spill to a file when /dev/shm is full; returns the record payload
    /// (u64 length + file name in spill_dir). Page cache backed, so not slow.
    fn file_spill_write(&self, data: &[u8]) -> std::io::Result<Vec<u8>> {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = self.spill_dir();
        std::fs::create_dir_all(&dir)?;
        let name = format!("{:x}_{:x}", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
        std::fs::write(dir.join(&name), data)?;
        let mut rec = (data.len() as u64).to_le_bytes().to_vec();
        rec.extend_from_slice(name.as_bytes());
        Ok(rec)
    }

    fn file_spill_read<'py>(&self, py: Python<'py>, rec: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let n = u64::from_le_bytes(rec[..8].try_into().unwrap()) as usize;
        let path = self.spill_dir().join(spill_name(rec));
        let out = PyBytes::new_with(py, n, |buf| {
            std::fs::File::open(&path)
                .and_then(|mut f| f.read_exact(buf))
                .map_err(|e| PyOSError::new_err(format!("reading spilled message {}: {e}", path.display())))
        });
        let _ = std::fs::remove_file(&path);
        out
    }

    /// Payload bytes of one popped record, whatever carried it.
    fn materialize<'py>(&self, py: Python<'py>, kind: u32, b: Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyBytes>> {
        match kind {
            KIND_SPILL => spill_read(py, b.as_bytes()),
            KIND_FILE => self.file_spill_read(py, b.as_bytes()),
            _ => Ok(b),
        }
    }

    fn write_record(&self, pos: u64, kind: u32, payload: &[u8]) {
        unsafe {
            let p = self.data().add(pos as usize);
            std::ptr::write_unaligned(p as *mut u32, payload.len() as u32);
            std::ptr::write_unaligned(p.add(4) as *mut u32, kind);
            std::ptr::copy_nonoverlapping(payload.as_ptr(), p.add(REC_HDR as usize), payload.len());
        }
    }

    fn read_header(&self, pos: u64) -> (u64, u32) {
        unsafe {
            let p = self.data().add(pos as usize);
            (
                std::ptr::read_unaligned(p as *const u32) as u64,
                std::ptr::read_unaligned(p.add(4) as *const u32),
            )
        }
    }

    /// Pop one record if the ring has one (lock held).
    fn pop_locked<'py>(&self, py: Python<'py>) -> Option<(u32, Bound<'py, PyBytes>)> {
        let s = self.sh();
        let mut head = s.head.load(Ordering::Relaxed);
        if head == s.tail.load(Ordering::Relaxed) {
            // Empty by offsets; a producer that died between its two commit
            // stores can leave `count` one off, so resync it here.
            s.count.store(0, Ordering::Relaxed);
            return None;
        }
        let mut pos = head % s.cap;
        let (mut len, mut kind) = self.read_header(pos);
        if kind == KIND_PAD {
            head += s.cap - pos;
            pos = 0;
            (len, kind) = self.read_header(pos);
        }
        let b = PyBytes::new(py, unsafe {
            std::slice::from_raw_parts(self.data().add((pos + REC_HDR) as usize), len as usize)
        });
        s.head.store(head + REC_HDR + align8(len), Ordering::Relaxed);
        let c = s.count.load(Ordering::Relaxed);
        s.count.store(c.saturating_sub(1), Ordering::Relaxed);
        Some((kind, b))
    }

    /// Commit `n` pops: release the lock and wake producers waiting for room.
    fn finish_pops(&self, n: u32) {
        let s = self.sh();
        s.space_seq.0.fetch_add(n, Ordering::Release);
        let wake = s.waiting_producers.load(Ordering::Relaxed) > 0;
        self.unlock();
        if wake {
            futex::wake(&s.space_seq.0, n.min(i32::MAX as u32) as i32);
        }
    }

    /// Block (per deadline) until a record is available; lock held on Ok(true).
    fn wait_for_items(&self, py: Python<'_>, deadline: Option<Option<Instant>>) -> PyResult<bool> {
        let s = self.sh();
        loop {
            self.lock(py);
            if s.head.load(Ordering::Relaxed) != s.tail.load(Ordering::Relaxed) {
                return Ok(true);
            }
            if expired(deadline) {
                self.unlock();
                return Ok(false);
            }
            let seen = s.items_seq.0.load(Ordering::Relaxed);
            s.waiting_consumers.fetch_add(1, Ordering::Relaxed);
            self.unlock();
            let r = self.sleep_on(py, &s.items_seq.0, seen, deadline.flatten());
            s.waiting_consumers.fetch_sub(1, Ordering::Relaxed);
            r?;
        }
    }
}

#[pymethods]
impl RingQueue {
    /// Create a queue. `maxsize` bounds items (0 = unbounded); `ring_bytes`
    /// sizes the shared ring; payloads above `spill_at` (default and cap:
    /// ring/4) always go through their own shm object.
    #[new]
    #[pyo3(signature = (maxsize=0, ring_bytes=8 << 20, spill_at=None))]
    fn new(maxsize: u64, ring_bytes: u64, spill_at: Option<u64>) -> PyResult<Self> {
        let cap = align8(ring_bytes.max(64 << 10));
        let spill_at = spill_at.unwrap_or(cap / 4).min(cap / 4);
        let name = unique_name("gpq");
        let map = shm_map(&name, Some(HDR + cap as usize))?;
        assert!(std::mem::size_of::<Shared>() <= HDR);
        assert!(std::mem::size_of::<Shared>() <= PENDING_OFF && PENDING_OFF + PENDING_SLOTS * PENDING_NAME <= HDR);
        let q = RingQueue { name, map, pid: std::process::id(), force_disk: AtomicBool::new(false) };
        unsafe {
            let s = q.map.base as *mut Shared;
            (*s).cap = cap;
            (*s).maxsize = maxsize;
            (*s).spill_at = spill_at;
        }
        q.sh().magic.store(MAGIC, Ordering::Release);
        Ok(q)
    }

    /// Attach to an existing queue by shm name (what unpickling calls).
    #[staticmethod]
    fn attach(name: &str) -> PyResult<Self> {
        let map = shm_map(name, None)?;
        let q = RingQueue { name: name.to_string(), map, pid: std::process::id(), force_disk: AtomicBool::new(false) };
        if q.map.len < HDR || q.sh().magic.load(Ordering::Acquire) != MAGIC {
            return Err(PyValueError::new_err(format!("{name} is not a pipe ring queue")));
        }
        Ok(q)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let attach = py.get_type::<RingQueue>().getattr("attach")?;
        Ok((attach, (self.name.clone(),)))
    }

    /// Owner teardown: remove the name so nothing new can attach, and drop
    /// any messages nobody will read. A spilled message lives in its own shm
    /// object that only its reader unlinks, so one still queued at stop
    /// (force stop, unread End-time leftovers) would otherwise leak for good,
    /// as would one a producer was still waiting to queue when it was killed.
    /// Disk spills go with the ring's spill directory. Returns how many
    /// unread spilled messages were removed.
    fn unlink(&self, py: Python<'_>) -> usize {
        shm_unlink(&self.name);
        let s = self.sh();
        let mut names = Vec::new();
        let mut files = 0;
        self.lock(py);
        let tail = s.tail.load(Ordering::Relaxed);
        let mut head = s.head.load(Ordering::Relaxed);
        while head < tail {
            let pos = head % s.cap;
            let (len, kind) = self.read_header(pos);
            if kind == KIND_PAD {
                head += s.cap - pos;
                continue;
            }
            if kind == KIND_SPILL {
                let rec = unsafe {
                    std::slice::from_raw_parts(self.data().add((pos + REC_HDR) as usize), len as usize)
                };
                names.push(spill_name(rec).to_string());
            } else if kind == KIND_FILE {
                files += 1;
            }
            head += REC_HDR + align8(len);
        }
        s.head.store(tail, Ordering::Relaxed);
        s.count.store(0, Ordering::Relaxed);
        names.extend(self.take_pending());
        self.unlock();
        for n in &names {
            shm_unlink(n);
        }
        let _ = std::fs::remove_dir_all(self.spill_dir()); // disk spills, unread or orphaned
        names.len() + files
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn maxsize(&self) -> u64 {
        self.sh().maxsize
    }

    #[getter]
    fn ring_bytes(&self) -> u64 {
        self.sh().cap
    }

    /// Largest payload that travels in the ring itself; bigger ones spill.
    #[getter]
    fn spill_at(&self) -> u64 {
        self.sh().spill_at
    }

    /// Test hook: spill oversized messages to disk as if /dev/shm were full.
    fn _force_disk_spill(&self, on: bool) {
        self.force_disk.store(on, Ordering::Relaxed);
    }

    fn qsize(&self) -> u64 {
        self.sh().count.load(Ordering::Relaxed)
    }

    /// Put one payload. Returns False if it could not be placed (non-blocking
    /// and full, or the timeout passed); the caller raises queue.Full.
    #[pyo3(signature = (data, block=true, timeout=None))]
    fn put(&self, py: Python<'_>, data: &[u8], block: bool, timeout: Option<f64>) -> PyResult<bool> {
        let deadline = deadline_of(block, timeout)?;
        let s = self.sh();
        let mut spilled: Option<(u32, Vec<u8>)> = None;
        let mut must_spill = data.len() as u64 > s.spill_at;
        let discard = |sp: &Option<(u32, Vec<u8>)>| match sp {
            Some((KIND_SPILL, rec)) => shm_unlink(spill_name(rec)),
            Some((_, rec)) => {
                let _ = std::fs::remove_file(self.spill_dir().join(spill_name(rec)));
            }
            None => {}
        };
        let mut pending: Option<usize> = None;
        loop {
            if must_spill && spilled.is_none() {
                let in_shm = if self.force_disk.load(Ordering::Relaxed) {
                    Err(std::io::Error::from_raw_os_error(libc::ENOSPC))
                } else {
                    spill_write(data)
                };
                match in_shm {
                    Ok(rec) => spilled = Some((KIND_SPILL, rec)),
                    // /dev/shm is full: put it on disk. Waiting for shm to
                    // free up can deadlock — every worker may hold the shm the
                    // others wait for (seen in Docker's 64 MiB /dev/shm).
                    Err(e) if is_no_space(&e) => match self.file_spill_write(data) {
                        Ok(rec) => spilled = Some((KIND_FILE, rec)),
                        // Disk full too: wait as a last resort, Full at the
                        // deadline (raising would drop the item).
                        Err(fe) if is_no_space(&fe) => {
                            if expired(deadline) {
                                return Ok(false);
                            }
                            let seen = s.space_seq.0.load(Ordering::Relaxed);
                            self.sleep_on(py, &s.space_seq.0, seen, deadline.flatten())?;
                            continue;
                        }
                        Err(fe) => {
                            return Err(PyOSError::new_err(format!(
                                "spilling {} bytes to {}: {fe}",
                                data.len(),
                                self.spill_dir().display()
                            )))
                        }
                    },
                    Err(e) => {
                        return Err(PyOSError::new_err(format!(
                            "reserving {} bytes of shared memory: {e}",
                            data.len()
                        )))
                    }
                }
            }
            let (kind, payload): (u32, &[u8]) = match &spilled {
                Some((k, rec)) => (*k, rec),
                None => (KIND_DATA, data),
            };
            let rec = REC_HDR + align8(payload.len() as u64);
            self.lock(py);
            let head = s.head.load(Ordering::Relaxed);
            let tail = s.tail.load(Ordering::Relaxed);
            let pos = tail % s.cap;
            let pad = if s.cap - pos < rec { s.cap - pos } else { 0 };
            let has_slot = s.maxsize == 0 || s.count.load(Ordering::Relaxed) < s.maxsize;
            if has_slot && (tail - head) + pad + rec <= s.cap {
                let mut at = pos;
                if pad > 0 {
                    self.write_record(pos, KIND_PAD, &[]);
                    at = 0;
                }
                self.write_record(at, kind, payload);
                // Publish: the record is complete before the offset covers it.
                s.tail.store(tail + pad + rec, Ordering::Relaxed);
                s.count.fetch_add(1, Ordering::Relaxed);
                s.items_seq.0.fetch_add(1, Ordering::Release);
                self.clear_pending(pending); // a record owns the spill now
                let wake = s.waiting_consumers.load(Ordering::Relaxed) > 0;
                self.unlock();
                if wake {
                    futex::wake(&s.items_seq.0, 1);
                }
                return Ok(true);
            }
            if has_slot && kind == KIND_DATA {
                // Out of ring bytes but not out of slots: capacity is
                // `maxsize` items, so this one travels by spill instead.
                self.unlock();
                must_spill = true;
                continue;
            }
            if expired(deadline) {
                self.clear_pending(pending);
                self.unlock();
                discard(&spilled);
                return Ok(false);
            }
            if pending.is_none() {
                // Disk spills need no entry: the spill dir goes at teardown.
                if let Some((KIND_SPILL, rec)) = &spilled {
                    pending = self.note_pending(spill_name(rec));
                }
            }
            let seen = s.space_seq.0.load(Ordering::Relaxed);
            s.waiting_producers.fetch_add(1, Ordering::Relaxed);
            self.unlock();
            let r = self.sleep_on(py, &s.space_seq.0, seen, deadline.flatten());
            s.waiting_producers.fetch_sub(1, Ordering::Relaxed);
            if let Err(e) = r {
                self.lock(py);
                self.clear_pending(pending);
                self.unlock();
                discard(&spilled);
                return Err(e);
            }
        }
    }

    /// Pop one payload, or None if nothing arrived (caller raises queue.Empty).
    #[pyo3(signature = (block=true, timeout=None))]
    fn get<'py>(&self, py: Python<'py>, block: bool, timeout: Option<f64>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let deadline = deadline_of(block, timeout)?;
        loop {
            if !self.wait_for_items(py, deadline)? {
                return Ok(None);
            }
            let Some((kind, b)) = self.pop_locked(py) else {
                self.unlock();
                continue;
            };
            self.finish_pops(1);
            return self.materialize(py, kind, b).map(Some);
        }
    }

    /// Pop up to `max_items` payloads under one lock acquisition, waiting
    /// (per block/timeout) only for the first. Empty list = nothing arrived.
    #[pyo3(signature = (max_items=1, block=true, timeout=None))]
    fn get_many<'py>(
        &self,
        py: Python<'py>,
        max_items: usize,
        block: bool,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let deadline = deadline_of(block, timeout)?;
        let out = PyList::empty(py);
        if !self.wait_for_items(py, deadline)? {
            return Ok(out);
        }
        let mut got = Vec::new();
        while got.len() < max_items.max(1) {
            match self.pop_locked(py) {
                Some(rec) => got.push(rec),
                None => break,
            }
        }
        self.finish_pops(got.len() as u32);
        for (kind, b) in got {
            out.append(self.materialize(py, kind, b)?)?;
        }
        Ok(out)
    }
}
