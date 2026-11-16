//! Memory budgets shared by the renderer, the exporter and the worker.
//!
//! Budgets adapt to the machine: at most 1.5 GiB (soft) / 4.5 GiB (hard) per render band,
//! and at most 40 % (soft) / 80 % (hard) of both the available memory and the process
//! address-space limit. Memory figures come from `/proc` (Linux), `GlobalMemoryStatusEx`
//! (Windows) or `hw.memsize` (macOS, total only); where a figure is missing the fixed caps apply.
//!
//! Larger bands do not change the result (the coarse layer is global and bands overlap by a
//! pyramid pad), so the band caps trade nothing in quality; the process-wide ceiling in
//! [`memory_limit`] is what lets several large buffers (decoding, previews, export) coexist.

/// Physical memory in bytes (Linux `/proc`, Windows `GlobalMemoryStatusEx`, macOS
/// `hw.memsize`). `None` where the platform does not report it.
pub fn total_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        meminfo_kb("MemTotal:").map(|kb| kb * 1024)
    }
    #[cfg(windows)]
    {
        windows_memory_status().map(|m| m.ullTotalPhys)
    }
    #[cfg(target_os = "macos")]
    {
        let mut bytes: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: NUL-terminated name; `bytes`/`len` describe a writable u64.
        let rc = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && bytes > 0).then_some(bytes)
    }
    #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
    {
        None
    }
}

/// Memory the system can hand out right now without swapping (Linux `MemAvailable`, Windows
/// `ullAvailPhys`). `None` where no reliable live figure exists (macOS, others): callers
/// then fall back to fixed shares of [`total_memory`].
pub fn available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        meminfo_kb("MemAvailable:").map(|kb| kb * 1024)
    }
    #[cfg(windows)]
    {
        windows_memory_status().map(|m| m.ullAvailPhys)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn meminfo_kb(key: &str) -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
}

#[cfg(windows)]
fn windows_memory_status() -> Option<windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX>
{
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: zeroed plain-data struct with dwLength set, as the API requires.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

/// Soft address-space limit of this process (Linux `ulimit -v`, containers).
fn address_space_limit() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/limits")
            .ok()
            .and_then(|l| {
                l.lines()
                    .find(|x| x.starts_with("Max address space"))
                    .and_then(|x| x.split_whitespace().nth(3))
                    .and_then(|v| v.parse::<u64>().ok())
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// `(soft, hard)` budgets in bytes.
pub fn memory_budgets() -> (u64, u64) {
    let mut soft: u64 = 1536 * 1024 * 1024;
    let mut hard: u64 = 3 * 1536 * 1024 * 1024;
    // Without a live figure, assume half of physical memory is free.
    if let Some(avail) = available_memory().or_else(|| total_memory().map(|t| t / 2)) {
        soft = soft.min(avail * 2 / 5);
        hard = hard.min(avail * 4 / 5);
    }
    if let Some(bytes) = address_space_limit() {
        soft = soft.min(bytes * 2 / 5);
        hard = hard.min(bytes * 4 / 5);
    }
    (soft.max(64 * 1024 * 1024), hard.max(64 * 1024 * 1024))
}

/// Soft budget for one render band, the global coarse layer or the seam layout.
pub fn band_budget() -> u64 {
    memory_budgets().0
}

/// Hard ceiling for one render band.
pub fn band_hard_limit() -> u64 {
    memory_budgets().1
}

/// Process-wide ceiling for large working buffers (decoding, render bands, coarse layer, seam
/// layout, exact previews). Work that does not fit waits instead of pushing the machine into
/// swap. Default: 90 % of physical memory (75 % on platforms without live free-memory
/// figures; 60 % of an address-space rlimit, if lower);
/// `SLASHSTITCH_MEMORY_LIMIT_MB` overrides. On top of this fixed ceiling every reservation
/// also checks the live free memory (see [`headroom_floor`]), so other programs growing
/// meanwhile never get squeezed into swap.
pub fn memory_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if let Some(mb) = std::env::var("SLASHSTITCH_MEMORY_LIMIT_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            return mb.max(256) * 1024 * 1024;
        }
        // 90 % where every reservation is also checked against live free memory; 75 % where
        // the platform gives no live figure; 8 GiB if even the total is unknown.
        let mut limit = match (total_memory(), available_memory().is_some()) {
            (Some(t), true) => t / 10 * 9,
            (Some(t), false) => t / 4 * 3,
            (None, _) => 8 * 1024 * 1024 * 1024,
        };
        // Respect a process address-space limit (containers, ulimit -v).
        if let Some(bytes) = address_space_limit() {
            limit = limit.min(bytes * 3 / 5);
        }
        limit.max(256 * 1024 * 1024)
    })
}

struct Pool {
    used: std::sync::Mutex<u64>,
    freed: std::sync::Condvar,
}

fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Pool {
        used: std::sync::Mutex::new(0),
        freed: std::sync::Condvar::new(),
    })
}

/// Free memory that must remain for the OS and the desktop when a reservation is granted:
/// 5 % of physical memory, at least 1 GiB. Zero where the platform does not report it.
pub fn headroom_floor() -> u64 {
    total_memory().map_or(0, |t| (t / 20).max(1024 * 1024 * 1024))
}

/// Whether the machine currently has `bytes` to spare above [`headroom_floor`]. Platforms
/// without live memory figures rely on the fixed ceiling alone.
fn live_room_for(bytes: u64) -> bool {
    match available_memory() {
        Some(avail) => avail >= bytes.saturating_add(headroom_floor()),
        None => true,
    }
}

/// A share of the process memory ceiling, released on drop.
pub struct Reservation {
    bytes: u64,
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let p = pool();
        let mut used = p
            .used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *used = used.saturating_sub(self.bytes);
        p.freed.notify_all();
    }
}

/// Reserve `bytes` (clamped to the ceiling) from the process pool, waiting while other work
/// holds it. Returns `None` if `cancel` is raised while waiting. A request is always granted
/// when the pool is otherwise empty, so oversized work still runs — alone.
pub fn reserve(bytes: u64, cancel: Option<&std::sync::atomic::AtomicBool>) -> Option<Reservation> {
    let limit = memory_limit();
    let bytes = bytes.min(limit);
    let p = pool();
    let mut used = p
        .used
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        // A request is always granted when nothing else is reserved (progress guaranteed);
        // otherwise it must fit both the fixed ceiling and the live free memory.
        if *used == 0 || (*used + bytes <= limit && live_room_for(bytes)) {
            *used += bytes;
            return Some(Reservation { bytes });
        }
        if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
            return None;
        }
        used = p
            .freed
            .wait_timeout(used, std::time::Duration::from_millis(100))
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .0;
    }
}

/// Bytes currently reserved (diagnostics).
pub fn reserved_bytes() -> u64 {
    *pool()
        .used
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_wait_for_room_and_release_on_drop() {
        let limit = memory_limit();
        let big = reserve(limit, None).unwrap();
        assert_eq!(big.bytes(), limit);
        let cancel = std::sync::atomic::AtomicBool::new(true);
        // Pool full: a cancelled waiter gives up instead of exceeding the ceiling.
        assert!(reserve(1024, Some(&cancel)).is_none());
        let waiter = std::thread::spawn(|| reserve(1024, None).map(|r| r.bytes()));
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(!waiter.is_finished(), "must wait while the pool is full");
        drop(big);
        assert_eq!(waiter.join().unwrap(), Some(1024));
    }
}
