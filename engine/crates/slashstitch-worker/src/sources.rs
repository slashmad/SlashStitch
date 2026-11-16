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
}

impl SourceStamp {
    pub fn of(path: &Path) -> Result<Self, String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as i128);
        Ok(Self {
            len: meta.len(),
            mtime_ns,
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

pub struct SourceCache {
    dir: PathBuf,
    entries: Mutex<HashMap<CacheKey, Arc<SourcePyramid>>>,
    build_lock: Mutex<()>,
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
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(kb) = info
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
        {
            budget = budget.min(kb * 1024 * 2 / 5);
        }
    }
    budget.max(512 * 1024 * 1024)
}

/// Pixel count from the file header without decoding (falls back to a large estimate).
pub fn header_pixels(path: &Path) -> u64 {
    image::ImageReader::open(path)
        .ok()
        .and_then(|r| r.with_guessed_format().ok())
        .and_then(|r| r.into_dimensions().ok())
        .map_or(100_000_000, |(w, h)| u64::from(w) * u64::from(h))
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
        }
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
        let total = missing.len();
        let done = AtomicUsize::new(0);
        if total > 0 {
            progress(0, total);
        }
        let sizes: Vec<u64> = missing
            .iter()
            .map(|&i| u64::from(images[i].width) * u64::from(images[i].height))
            .collect();
        for batch in budget_batches(&sizes) {
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
                        entries.insert((images[i].path.clone(), stamps[i], sig), Arc::new(pyramid));
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
        let entries = self.entries.lock().unwrap();
        images
            .iter()
            .zip(stamps)
            .map(|(im, s)| {
                entries
                    .get(&(im.path.clone(), *s, sig))
                    .cloned()
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
