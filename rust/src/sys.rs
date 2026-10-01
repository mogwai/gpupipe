//! OS plumbing shared by the ring queue and the payload store: process-shared
//! futex/ulock waits, a pid-owned mutex that survives a SIGKILLed holder, and
//! POSIX shared-memory mappings.

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use std::ffi::CString;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Longest uninterrupted sleep: blocked calls wake this often to run Python
// signal handlers and to check whether a lock holder died.
pub const SLICE: Duration = Duration::from_millis(50);
const SPIN: u32 = 64;
const WAITERS: u32 = 1 << 31;

#[repr(C, align(64))]
pub struct Line<T>(pub T);

#[cfg(target_os = "linux")]
pub mod futex {
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    // No FUTEX_PRIVATE_FLAG: the word lives in memory shared across processes.
    pub fn wait(a: &AtomicU32, expected: u32, timeout: Duration) {
        let ts = libc::timespec {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_nsec: timeout.subsec_nanos() as libc::c_long,
        };
        unsafe {
            libc::syscall(libc::SYS_futex, a.as_ptr(), libc::FUTEX_WAIT, expected, &ts as *const libc::timespec);
        }
    }

    pub fn wake(a: &AtomicU32, n: i32) {
        unsafe {
            libc::syscall(libc::SYS_futex, a.as_ptr(), libc::FUTEX_WAKE, n);
        }
    }
}

#[cfg(target_os = "macos")]
pub mod futex {
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    extern "C" {
        fn __ulock_wait(op: u32, addr: *mut libc::c_void, value: u64, timeout_us: u32) -> libc::c_int;
        fn __ulock_wake(op: u32, addr: *mut libc::c_void, wake_value: u64) -> libc::c_int;
    }
    const UL_COMPARE_AND_WAIT_SHARED: u32 = 3;
    const ULF_WAKE_ALL: u32 = 0x100;
    const ULF_NO_ERRNO: u32 = 0x0100_0000;

    pub fn wait(a: &AtomicU32, expected: u32, timeout: Duration) {
        let us = timeout.as_micros().clamp(1, u32::MAX as u128) as u32;
        unsafe {
            __ulock_wait(UL_COMPARE_AND_WAIT_SHARED | ULF_NO_ERRNO, a.as_ptr().cast(), expected as u64, us);
        }
    }

    pub fn wake(a: &AtomicU32, n: i32) {
        let all = if n > 1 { ULF_WAKE_ALL } else { 0 };
        unsafe {
            __ulock_wake(UL_COMPARE_AND_WAIT_SHARED | ULF_NO_ERRNO | all, a.as_ptr().cast(), 0);
        }
    }
}

fn pid_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 || *errno() != libc::ESRCH }
}

#[cfg(target_os = "linux")]
unsafe fn errno() -> *mut libc::c_int {
    libc::__errno_location()
}

#[cfg(target_os = "macos")]
unsafe fn errno() -> *mut libc::c_int {
    libc::__error()
}

/// Acquire a process-shared mutex word: 0 free, else holder pid | WAITERS.
///
/// Waiters sleep without the GIL, and anyone who finds the same holder for a
/// whole SLICE checks that it is still alive: a holder that was SIGKILLed
/// mid-operation (hang_timeout, OOM killer) has its lock taken over instead
/// of deadlocking every sibling. Callers keep shared state consistent at
/// every instant a holder could die (write data before publishing offsets).
pub fn lock(py: Python<'_>, l: &AtomicU32, me: u32) {
    if l.compare_exchange(0, me, Ordering::Acquire, Ordering::Relaxed).is_ok() {
        return;
    }
    for _ in 0..SPIN {
        std::hint::spin_loop();
        if l.load(Ordering::Relaxed) == 0 && l.compare_exchange(0, me, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            return;
        }
    }
    py.detach(|| {
        let mut since = Instant::now();
        loop {
            let v = l.load(Ordering::Relaxed);
            if v == 0 {
                // Taken with WAITERS set: others may still be asleep.
                if l.compare_exchange(0, me | WAITERS, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    return;
                }
                continue;
            }
            if v & WAITERS == 0 && l.compare_exchange(v, v | WAITERS, Ordering::Relaxed, Ordering::Relaxed).is_err() {
                continue;
            }
            futex::wait(l, v | WAITERS, SLICE);
            if l.load(Ordering::Relaxed) != v | WAITERS {
                since = Instant::now();
                continue;
            }
            let holder = v & !WAITERS;
            if since.elapsed() >= SLICE
                && holder != me
                && !pid_alive(holder)
                && l.compare_exchange(v | WAITERS, me | WAITERS, Ordering::Acquire, Ordering::Relaxed).is_ok()
            {
                return;
            }
        }
    });
}

pub fn unlock(l: &AtomicU32) {
    if l.swap(0, Ordering::Release) & WAITERS != 0 {
        futex::wake(l, 1);
    }
}

pub struct Map {
    pub base: *mut u8,
    pub len: usize,
}

// The mapping is process-shared memory coordinated through atomics and the
// in-segment mutexes, never through Rust aliasing rules.
unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Drop for Map {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.cast(), self.len);
        }
    }
}

/// Map a shm object: create it with `create_len` bytes (reserved up front on
/// Linux so a full /dev/shm fails here with ENOSPC, not later with SIGBUS on
/// first touch), or attach to an existing one.
pub fn shm_map(name: &str, create_len: Option<usize>) -> PyResult<Map> {
    shm_map_io(name, create_len).map_err(|e| match create_len {
        Some(n) => PyOSError::new_err(format!("reserving {n} bytes of shared memory ({name}): {e}")),
        None => PyOSError::new_err(format!("attaching shared memory {name}: {e}")),
    })
}

/// True for the errors that mean "shared memory is full right now".
pub fn is_no_space(e: &std::io::Error) -> bool {
    matches!(e.raw_os_error(), Some(libc::ENOSPC) | Some(libc::ENOMEM))
}

/// shm_map, keeping the OS error so callers can tell "full" from "broken".
pub fn shm_map_io(name: &str, create_len: Option<usize>) -> std::io::Result<Map> {
    let cname = CString::new(name).map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    unsafe {
        let flags = match create_len {
            Some(_) => libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
            None => libc::O_RDWR,
        };
        let fd = libc::shm_open(cname.as_ptr(), flags, 0o600 as libc::c_uint);
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let fail = |e: std::io::Error| {
            libc::close(fd);
            if create_len.is_some() {
                libc::shm_unlink(cname.as_ptr());
            }
            Err(e)
        };
        let len = match create_len {
            Some(n) => {
                while libc::ftruncate(fd, n as libc::off_t) != 0 {
                    if *errno() != libc::EINTR {
                        return fail(std::io::Error::last_os_error());
                    }
                }
                #[cfg(target_os = "linux")]
                {
                    // Block signals in this thread for the reservation: tmpfs
                    // abandons (and rolls back) a fallocate whenever a signal is
                    // pending, so under a steady trickle of signals a multi-MB
                    // call could retry forever. Pending signals (Ctrl+C too)
                    // are delivered as soon as the old mask is restored.
                    let mut all: libc::sigset_t = std::mem::zeroed();
                    let mut old: libc::sigset_t = std::mem::zeroed();
                    libc::sigfillset(&mut all);
                    libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut old);
                    let mut rc = libc::EINTR;
                    while rc == libc::EINTR {
                        rc = libc::posix_fallocate(fd, 0, n as libc::off_t);
                    }
                    libc::pthread_sigmask(libc::SIG_SETMASK, &old, std::ptr::null_mut());
                    if rc != 0 {
                        return fail(std::io::Error::from_raw_os_error(rc));
                    }
                }
                n
            }
            None => {
                let mut st: libc::stat = std::mem::zeroed();
                if libc::fstat(fd, &mut st) != 0 {
                    return fail(std::io::Error::last_os_error());
                }
                st.st_size as usize
            }
        };
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
        if p == libc::MAP_FAILED {
            return fail(std::io::Error::last_os_error());
        }
        libc::close(fd);
        Ok(Map { base: p.cast(), len })
    }
}

pub fn shm_unlink(name: &str) {
    if let Ok(c) = CString::new(name) {
        unsafe {
            libc::shm_unlink(c.as_ptr());
        }
    }
}

pub fn unique_name(tag: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
    // macOS caps shm names at 31 chars.
    format!("/{tag}{:x}_{:x}_{:x}", std::process::id(), n, t & 0xffff)
}
