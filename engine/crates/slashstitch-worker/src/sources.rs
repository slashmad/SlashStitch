//! Source identity and the full-resolution source cache.
//!
//! * `SourceStamp` identifies the bytes a preview mip was built from (size + mtime). Reusing a
//!   mip, metadata or cached pyramid requires the current file to have the same stamp, so an
//!   edited source on disk is never shown from stale caches while the export reads new pixels.
//! * `SourceCache` keeps, per source and target profile, a disk-backed resolution pyramid of
//!   the colour-converted full-resolution image. It is built in parallel (bounded) on first
//!   use and reused by exact previews and every export in the session.

use rayon::prelude::*;
use slashstitch_core::raster::{
    convert_rgb16_profile, decode_source_raw_16, half_resolution_linear, MappedRgb16, Rgb16Image,
    Rgb16Raster,
};
use slashstitch_core::{SourceImage, TargetColorProfile, TargetTrc};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceStamp {
    pub len: u64,
    pub mtime_ns: i128,
    /// A camera RAW's Lightroom XMP sidecar (size, mtime): editing it changes the development.
    pub sidecar: Option<(u64, i128)>,
}

fn file_stamp(meta: &std::fs::Metadata) -> (u64, i128) {
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as i128);
    (meta.len(), mtime_ns)
}

impl SourceStamp {
    pub fn of(path: &Path) -> Result<Self, String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (len, mtime_ns) = file_stamp(&meta);
        let sidecar = slashstitch_core::rawdev::is_raw_path(path)
            .then(|| slashstitch_core::rawdev::find_sidecar(path))
            .flatten()
            .and_then(|side| std::fs::metadata(side).ok())
            .map(|m| file_stamp(&m));
        Ok(Self {
            len,
            mtime_ns,
            sidecar,
        })
    }
}

/// Verify that every source still has the stamp its cached data was built from.
pub fn verify_stamps(images: &[SourceImage], stamps: &[SourceStamp]) -> Result<(), String> {
    if images.len() != stamps.len() {
        return Err("Internal source identity mismatch".into());
    }
    for (im, stamp) in images.iter().zip(stamps) {
        let now = SourceStamp::of(Path::new(&im.path))?;
        if now != *stamp {
            return Err(format!(
                "Source changed on disk since it was loaded: {}. Re-open the project (or re-add the image) to use the new file.",
                im.path
            ));
        }
    }
    Ok(())
}

/// Disk-backed pyramid of one colour-converted source: level 0 is full resolution.
pub struct SourcePyramid {
    pub levels: Vec<MappedRgb16>,
}

type CacheKey = (String, SourceStamp, u64);

struct CacheEntry {
    pyramid: Arc<SourcePyramid>,
    bytes: u64,
    last_used: u64,
}

pub struct SourceCache {
    dir: PathBuf,
    entries: Mutex<HashMap<CacheKey, CacheEntry>>,
    build_lock: Mutex<()>,
    clock: AtomicUsize,
}

/// Free bytes on the file system holding `dir` (None where unknown).
fn free_disk_bytes(dir: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: valid NUL-terminated path and a zeroed out-parameter of the right type.
        if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        Some(st.f_bavail as u64 * st.f_frsize as u64)
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut free: u64 = 0;
        // SAFETY: NUL-terminated wide path; the unused out-parameters may be null.
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut free,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        (ok != 0).then_some(free)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = dir;
        None
    }
}

/// Bytes a source pyramid occupies on disk (6 bytes per pixel, all levels ~ 4/3).
fn pyramid_bytes_estimate(im: &SourceImage) -> u64 {
    u64::from(im.width) * u64::from(im.height) * 6 * 4 / 3
}

/// Stop building smaller levels once the long side is below this (preview mips cover it).
const MIN_LEVEL_LONG_SIDE: u32 = 1400;
/// Transient bytes per source pixel while decoding: 8-bit decode buffer, 16-bit working
/// copy, colour-converted copy and the first half-resolution level.
pub const DECODE_BYTES_PER_PX: u64 = 18;

/// Memory allowed for sources decoded concurrently. Large files are decoded fewer at a time
/// instead of risking memory exhaustion; `SLASHSTITCH_DECODE_BUDGET_MB` overrides.
pub fn decode_budget_bytes() -> u64 {
    if let Some(mb) = std::env::var("SLASHSTITCH_DECODE_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return mb.max(256) * 1024 * 1024;
    }
    let mut budget: u64 = 6 * 1024 * 1024 * 1024;
    let avail = slashstitch_core::memory::available_memory()
        .or_else(|| slashstitch_core::memory::total_memory().map(|t| t / 2));
    if let Some(avail) = avail {
        budget = budget.min(avail * 2 / 5);
    }
    budget.max(512 * 1024 * 1024)
}

/// Pixel count from the file header without decoding (falls back to a large estimate).
pub fn header_pixels(path: &Path) -> u64 {
    if slashstitch_core::rawdev::is_raw_path(path) {
        return decode_cost_pixels(path, slashstitch_core::rawdev::raw_pixels(path).max(1));
    }
    image::ImageReader::open(path)
        .ok()
        .and_then(|r| r.with_guessed_format().ok())
        .and_then(|r| r.into_dimensions().ok())
        .map_or(100_000_000, |(w, h)| u64::from(w) * u64::from(h))
}

/// Decode memory of a source expressed in "regular" pixels (`DECODE_BYTES_PER_PX` each):
/// camera RAW development needs more transient memory per output pixel.
pub fn decode_cost_pixels(path: &Path, pixels: u64) -> u64 {
    if slashstitch_core::rawdev::is_raw_path(path) {
        pixels * slashstitch_core::rawdev::RAW_DECODE_BYTES_PER_PX / DECODE_BYTES_PER_PX
    } else {
        pixels
    }
}

/// Disk ceiling for the source cache: up to 90 % of the space it could use (free space plus
/// what it already holds), always leaving at least 2 GiB (and 10 %) free on its filesystem.
fn cache_limit(free: Option<u64>, held: u64) -> u64 {
    match free {
        Some(free) => {
            let usable = free + held;
            usable.saturating_sub((usable / 10).max(2 * 1024 * 1024 * 1024))
        }
        None => 64 * 1024 * 1024 * 1024,
    }
}

/// Split work items (in order) into batches whose summed decode memory fits the budget;
/// every batch holds at least one item and at most the CPU thread count.
pub fn budget_batches(pixels: &[u64]) -> Vec<Vec<usize>> {
    let budget = decode_budget_bytes();
    let max_items = rayon::current_num_threads().max(1);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut used = 0u64;
    for (i, &px) in pixels.iter().enumerate() {
        let cost = px.saturating_mul(DECODE_BYTES_PER_PX);
        if !current.is_empty() && (used + cost > budget || current.len() >= max_items) {
            batches.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(i);
        used += cost;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

pub fn profile_signature(target: &TargetColorProfile) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    target.id().hash(&mut h);
    // Built-in profiles are identified by id: their encoded ICC carries a creation time and
    // differs between calls. Custom profiles are identified by their exact bytes.
    if let TargetColorProfile::CustomIcc { bytes, .. } = target {
        bytes.hash(&mut h);
    }
    h.finish()
}

impl SourceCache {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            entries: Mutex::new(HashMap::new()),
            build_lock: Mutex::new(()),
            clock: AtomicUsize::new(1),
        }
    }

    /// Upper limit for the cache on disk: `SLASHSTITCH_CACHE_LIMIT_GB`, otherwise half of the
    /// space available to the cache (free space plus what it already holds).
    fn disk_limit(&self, held: u64) -> u64 {
        if let Some(gb) = std::env::var("SLASHSTITCH_CACHE_LIMIT_GB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            return gb.max(1) * 1024 * 1024 * 1024;
        }
        cache_limit(free_disk_bytes(&self.dir), held)
    }

    /// Bytes currently held on disk.
    pub fn held_bytes(&self) -> u64 {
        self.entries.lock().unwrap().values().map(|e| e.bytes).sum()
    }

    /// Drop cached pyramids whose key is not in `keep` (sources removed or changed).
    pub fn retain_sources(&self, keep: &[(String, SourceStamp)]) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.retain(|(path, stamp, _), _| keep.iter().any(|(p, s)| p == path && s == stamp));
        }
    }

    /// Return full-resolution pyramids for all sources, building missing ones.
    /// `progress(done, total)` is called as sources finish.
    pub fn get_or_build(
        &self,
        images: &[SourceImage],
        stamps: &[SourceStamp],
        target: &TargetColorProfile,
        cancel: &AtomicBool,
        progress: &(dyn Fn(usize, usize) + Sync),
    ) -> Result<Vec<Arc<SourcePyramid>>, String> {
        let sig = profile_signature(target);
        let _guard = self
            .build_lock
            .lock()
            .map_err(|_| "Source cache poisoned")?;
        let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
        let missing: Vec<usize> = {
            let entries = self.entries.lock().unwrap();
            (0..images.len())
                .filter(|&i| !entries.contains_key(&(images[i].path.clone(), stamps[i], sig)))
                .collect()
        };
        // Disk ceiling: everything this request needs must fit; least recently used pyramids
        // that no render currently holds are evicted to make room.
        {
            let mut entries = self.entries.lock().unwrap();
            let wanted: Vec<CacheKey> = images
                .iter()
                .zip(stamps)
                .map(|(im, s)| (im.path.clone(), *s, sig))
                .collect();
            let needed: u64 = missing
                .iter()
                .map(|&i| pyramid_bytes_estimate(&images[i]))
                .sum();
            let request_total: u64 = needed
                + wanted
                    .iter()
                    .filter_map(|k| entries.get(k))
                    .map(|e| e.bytes)
                    .sum::<u64>();
            let held: u64 = entries.values().map(|e| e.bytes).sum();
            let limit = self.disk_limit(held);
            let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
            if request_total > limit {
                return Err(format!(
                    "The full-resolution source cache would need {:.1} GiB but its limit is {:.1} GiB (90 % of the usable disk space, or SLASHSTITCH_CACHE_LIMIT_GB). Free disk space or use fewer/smaller sources.",
                    gib(request_total),
                    gib(limit)
                ));
            }
            let mut held_now = held;
            while held_now + needed > limit {
                let victim = entries
                    .iter()
                    .filter(|(k, e)| !wanted.contains(k) && Arc::strong_count(&e.pyramid) == 1)
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(k, _)| k.clone());
                let Some(k) = victim else {
                    return Err(format!(
                        "The full-resolution source cache is full ({:.1} GiB of {:.1} GiB) and the remaining entries are in use; try again when the current render finishes.",
                        gib(held_now),
                        gib(limit)
                    ));
                };
                if let Some(e) = entries.remove(&k) {
                    held_now = held_now.saturating_sub(e.bytes);
                }
            }
        }
        let total = missing.len();
        let done = AtomicUsize::new(0);
        if total > 0 {
            progress(0, total);
        }
        let sizes: Vec<u64> = missing
            .iter()
            .map(|&i| {
                decode_cost_pixels(
                    Path::new(&images[i].path),
                    u64::from(images[i].width) * u64::from(images[i].height),
                )
            })
            .collect();
        for batch in budget_batches(&sizes) {
            // Decode memory for the whole batch, reserved before the parallel section (no
            // rayon worker may block on the ceiling: renders need those threads).
            let _memory = slashstitch_core::memory::reserve(
                batch.iter().map(|&k| sizes[k]).sum::<u64>() * DECODE_BYTES_PER_PX,
                Some(cancel),
            )
            .ok_or("Cancelled")?;
            let chunk: Vec<usize> = batch.iter().map(|&k| missing[k]).collect();
            let built: Vec<Result<(usize, SourcePyramid), String>> = chunk
                .par_iter()
                .map(|&i| {
                    if cancel.load(Ordering::Relaxed) {
                        return Err("Cancelled".into());
                    }
                    let im = &images[i];
                    let path = Path::new(&im.path);
                    if SourceStamp::of(path)? != stamps[i] {
                        return Err(format!(
                            "Source changed on disk since it was loaded: {}. Re-open the project to use the new file.",
                            im.path
                        ));
                    }
                    let raw = decode_source_raw_16(path)?;
                    if raw.image_rgb16.width() != im.width || raw.image_rgb16.height() != im.height {
                        return Err(format!("Source dimensions changed: {}", im.path));
                    }
                    let converted =
                        convert_rgb16_profile(&raw.image_rgb16, raw.icc_profile.as_deref(), target)?;
                    drop(raw);
                    let pyramid = build_pyramid(converted, &trc, &self.dir)?;
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress(n, total);
                    Ok((i, pyramid))
                })
                .collect();
            // Keep every pyramid that finished, even if a sibling failed or was cancelled.
            let mut entries = self.entries.lock().unwrap();
            let mut first_error = None;
            for result in built {
                match result {
                    Ok((i, pyramid)) => {
                        let bytes = pyramid
                            .levels
                            .iter()
                            .map(|l| {
                                u64::from(Rgb16Raster::width(l))
                                    * u64::from(Rgb16Raster::height(l))
                                    * 6
                            })
                            .sum();
                        let tick = self.clock.fetch_add(1, Ordering::Relaxed) as u64;
                        entries.insert(
                            (images[i].path.clone(), stamps[i], sig),
                            CacheEntry {
                                pyramid: Arc::new(pyramid),
                                bytes,
                                last_used: tick,
                            },
                        );
                    }
                    Err(e) => {
                        first_error.get_or_insert(e);
                    }
                }
            }
            if let Some(e) = first_error {
                return Err(e);
            }
        }
        let mut entries = self.entries.lock().unwrap();
        let tick = self.clock.fetch_add(1, Ordering::Relaxed) as u64;
        images
            .iter()
            .zip(stamps)
            .map(|(im, s)| {
                entries
                    .get_mut(&(im.path.clone(), *s, sig))
                    .map(|e| {
                        e.last_used = tick;
                        Arc::clone(&e.pyramid)
                    })
                    .ok_or_else(|| format!("Source cache missing {}", im.path))
            })
            .collect()
    }
}

fn build_pyramid(level0: Rgb16Image, trc: &TargetTrc, dir: &Path) -> Result<SourcePyramid, String> {
    let mut levels = Vec::new();
    let mut current = level0;
    loop {
        let long = current.width().max(current.height());
        let next = if long / 2 >= MIN_LEVEL_LONG_SIDE / 2 && long > MIN_LEVEL_LONG_SIDE {
            Some(half_resolution_linear(&current, trc))
        } else {
            None
        };
        levels.push(
            MappedRgb16::from_image_in(current, dir).map_err(|e| format!("Source cache: {e}"))?,
        );
        match next {
            Some(n) => current = n,
            None => break,
        }
    }
    Ok(SourcePyramid { levels })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn cache_limit_uses_most_of_the_disk_but_keeps_a_margin() {
        // Large disk: 90 % of free + held.
        assert_eq!(cache_limit(Some(900 * GIB), 100 * GIB), 900 * GIB);
        // Small disk: at least 2 GiB stays free.
        assert_eq!(cache_limit(Some(5 * GIB), 0), 3 * GIB);
        assert_eq!(cache_limit(Some(GIB), 0), 0);
        // Unknown free space: fixed fallback.
        assert_eq!(cache_limit(None, 0), 64 * GIB);
    }

    #[cfg(unix)]
    #[test]
    fn free_disk_space_is_reported_for_the_temp_dir() {
        assert!(free_disk_bytes(&std::env::temp_dir()).is_some_and(|b| b > 0));
    }
}
