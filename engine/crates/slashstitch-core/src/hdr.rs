//! Exposure brackets (HDR stacks): detection from EXIF.
//!
//! A stack is a run of consecutive frames from the same camera, lens and focal length that
//! differ in exposure and were taken as one burst: either EXIF marks them "Auto bracket", or
//! each frame follows the previous one within its exposure time plus a short gap. Frames of a
//! normal panorama shot in automatic exposure also vary in exposure, but they are separated by
//! the time it takes to turn the camera; requiring distinct, clearly different exposures inside
//! a stack and a total span of at least one stop keeps them apart.
//!
//! Merging stacks into one high-dynamic-range source is the next step (see the project's
//! HDR design notes); detection already lets the UI show what was recognized.

use crate::image_meta::SourceImageMeta;
use serde::{Deserialize, Serialize};

/// Position of a frame in a detected exposure bracket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketInfo {
    /// Stack number (0-based, in project order).
    pub stack: usize,
    /// Position inside the stack (0-based, in project order).
    pub index: usize,
    pub size: usize,
    /// Brightness of this frame relative to the stack's middle exposure, in stops
    /// (+2 = two stops more exposure).
    pub relative_ev: f64,
}

/// Largest bracket SlashStitch groups (cameras offer up to 9 frames).
const MAX_STACK: usize = 9;
/// Minimal exposure difference between frames of one stack (stops).
const MIN_STEP_EV: f64 = 0.65;
/// Minimal exposure span of a stack (stops).
const MIN_SPAN_EV: f64 = 1.0;
/// Extra time allowed between the end of one bracket frame and the next (seconds).
const MAX_GAP_S: f64 = 1.5;

/// Exposure of a frame in stops of light reaching the sensor, relative to an arbitrary
/// origin: log2(t / N²) + log2(ISO / 100). Higher = brighter. Needs the exposure time; the
/// aperture and ISO default to neutral values when unknown (stacks share them anyway).
pub fn exposure_stops(meta: &SourceImageMeta) -> Option<f64> {
    let t = meta.exposure_time_s.filter(|t| *t > 0.0)?;
    let n = meta.aperture.filter(|n| *n > 0.0).unwrap_or(1.0);
    let iso = meta.iso.filter(|i| *i > 0).map_or(100.0, f64::from);
    Some((t / (n * n)).log2() + (iso / 100.0).log2())
}

/// Seconds since an arbitrary epoch from "YYYY:MM:DD HH:MM:SS[.fff]".
pub fn capture_seconds(text: &str) -> Option<f64> {
    let (date, time) = text.trim().split_once(' ')?;
    let mut d = date.split(':').map(|v| v.parse::<i64>().ok());
    let (y, m, day) = (d.next()??, d.next()??, d.next()??);
    let (hms, frac) = match time.split_once('.') {
        Some((a, b)) => (a, format!("0.{b}").parse::<f64>().unwrap_or(0.0)),
        None => (time, 0.0),
    };
    let mut t = hms.split(':').map(|v| v.parse::<i64>().ok());
    let (hh, mm, ss) = (t.next()??, t.next()??, t.next()??);
    if !(1..=12).contains(&m) || !(1..=31).contains(&day) {
        return None;
    }
    // Days from civil date (Howard Hinnant's algorithm).
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe;
    Some((days * 86_400 + hh * 3600 + mm * 60 + ss) as f64 + frac)
}

fn same_setup(a: &SourceImageMeta, b: &SourceImageMeta) -> bool {
    a.camera_model == b.camera_model
        && a.lens_model == b.lens_model
        && match (a.focal_length_mm, b.focal_length_mm) {
            (Some(x), Some(y)) => (x - y).abs() <= 0.5,
            (None, None) => true,
            _ => false,
        }
        && (a.width, a.height) == (b.width, b.height)
}

/// Whether `next` continues the burst that `prev` belongs to.
fn continues_burst(prev: &SourceImageMeta, next: &SourceImageMeta) -> bool {
    if prev.auto_bracket && next.auto_bracket {
        return true;
    }
    let (Some(a), Some(b)) = (
        prev.capture_time.as_deref().and_then(capture_seconds),
        next.capture_time.as_deref().and_then(capture_seconds),
    ) else {
        return false;
    };
    let gap = b - a;
    gap >= -0.001 && gap <= prev.exposure_time_s.unwrap_or(0.0) + MAX_GAP_S
}

/// Detect exposure brackets among `metas` (project order). Frames outside a stack get `None`.
pub fn detect_exposure_stacks(metas: &[SourceImageMeta]) -> Vec<Option<BracketInfo>> {
    let ev: Vec<Option<f64>> = metas.iter().map(exposure_stops).collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for i in 0..metas.len() {
        let joins = current.last().is_some_and(|&last| {
            current.len() < MAX_STACK
                && same_setup(&metas[last], &metas[i])
                && continues_burst(&metas[last], &metas[i])
                && ev[i].is_some_and(|e| {
                    current
                        .iter()
                        .all(|&k| ev[k].is_some_and(|f| (e - f).abs() >= MIN_STEP_EV))
                })
        });
        if !joins {
            groups.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    groups.push(current);

    let mut out = vec![None; metas.len()];
    let mut stack = 0;
    for g in groups.into_iter().filter(|g| g.len() >= 2) {
        let mut evs: Vec<f64> = g.iter().filter_map(|&k| ev[k]).collect();
        evs.sort_by(f64::total_cmp);
        if evs.len() != g.len() || evs[evs.len() - 1] - evs[0] < MIN_SPAN_EV {
            continue;
        }
        let middle = if evs.len() % 2 == 1 {
            evs[evs.len() / 2]
        } else {
            0.5 * (evs[evs.len() / 2 - 1] + evs[evs.len() / 2])
        };
        for (index, &k) in g.iter().enumerate() {
            out[k] = Some(BracketInfo {
                stack,
                index,
                size: g.len(),
                relative_ev: ev[k].map_or(0.0, |e| e - middle),
            });
        }
        stack += 1;
    }
    out
}

/// Fill `bracket` in every meta from the current project order.
pub fn assign_brackets(metas: &mut [SourceImageMeta]) {
    let brackets = detect_exposure_stacks(metas);
    for (meta, b) in metas.iter_mut().zip(brackets) {
        meta.bracket = b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::LensProjection;

    fn frame(t: f64, time: &str) -> SourceImageMeta {
        SourceImageMeta {
            path: String::new(),
            file_name: String::new(),
            width: Some(6000),
            height: Some(4000),
            bit_depth: None,
            icc_profile_name: None,
            icc_preset_match: None,
            icc_status: String::new(),
            camera_make: Some("Sony".into()),
            camera_model: Some("ILCE-7RM3".into()),
            lens_model: Some("24mm".into()),
            focal_length_mm: Some(24.0),
            focal_length_35mm: None,
            aperture: Some(8.0),
            iso: Some(100),
            matched_crop_factor: None,
            matched_mount: None,
            matched_lens: None,
            matched_lens_type: None,
            estimated_lens_projection: LensProjection::Rectilinear,
            estimated_hfov_deg: None,
            warnings: vec![],
            exposure_time_s: Some(t),
            exposure_bias_ev: None,
            auto_bracket: false,
            capture_time: Some(time.into()),
            bracket: None,
        }
    }

    #[test]
    fn capture_time_parses_with_subseconds() {
        let a = capture_seconds("2025:04:15 23:59:59").unwrap();
        let b = capture_seconds("2025:04:16 00:00:00.25").unwrap();
        assert!((b - a - 1.25).abs() < 1e-9);
        assert!(capture_seconds("garbage").is_none());
    }

    #[test]
    fn three_frame_brackets_are_grouped_per_position() {
        // Two positions, each -2/0/+2 EV fired quickly; 6 s to turn the camera in between.
        let metas = vec![
            frame(1.0 / 500.0, "2025:04:15 12:00:00.00"),
            frame(1.0 / 125.0, "2025:04:15 12:00:00.40"),
            frame(1.0 / 30.0, "2025:04:15 12:00:00.80"),
            frame(1.0 / 500.0, "2025:04:15 12:00:07.00"),
            frame(1.0 / 125.0, "2025:04:15 12:00:07.40"),
            frame(1.0 / 30.0, "2025:04:15 12:00:07.80"),
        ];
        let b = detect_exposure_stacks(&metas);
        let stacks: Vec<_> = b
            .iter()
            .map(|x| x.as_ref().map(|x| (x.stack, x.index, x.size)))
            .collect();
        assert_eq!(
            stacks,
            vec![
                Some((0, 0, 3)),
                Some((0, 1, 3)),
                Some((0, 2, 3)),
                Some((1, 0, 3)),
                Some((1, 1, 3)),
                Some((1, 2, 3))
            ]
        );
        let rel = b[2].as_ref().unwrap().relative_ev;
        assert!((rel - 2.06).abs() < 0.05, "{rel}");
    }

    #[test]
    fn auto_exposure_panorama_frames_are_not_stacks() {
        // Different exposures, but 4 s apart (turning the camera) and no bracket flag.
        let metas = vec![
            frame(1.0 / 500.0, "2025:04:15 12:00:00"),
            frame(1.0 / 125.0, "2025:04:15 12:00:04"),
            frame(1.0 / 30.0, "2025:04:15 12:00:08"),
        ];
        assert!(detect_exposure_stacks(&metas).iter().all(Option::is_none));
        // Quick frames with (almost) equal exposure are not a bracket either.
        let same = vec![
            frame(1.0 / 250.0, "2025:04:15 12:00:00.0"),
            frame(1.0 / 200.0, "2025:04:15 12:00:00.3"),
        ];
        assert!(detect_exposure_stacks(&same).iter().all(Option::is_none));
    }

    #[test]
    fn exif_auto_bracket_flag_groups_long_night_exposures() {
        let mut metas = vec![
            frame(4.0, "2025:04:15 23:00:00"),
            frame(15.0, "2025:04:15 23:00:12"),
            frame(1.0, "2025:04:15 23:00:40"),
        ];
        for m in &mut metas {
            m.auto_bracket = true;
        }
        assert!(detect_exposure_stacks(&metas)
            .iter()
            .all(|b| b.as_ref().is_some_and(|b| b.size == 3)));
    }
}
