use crate::optics::{image_px_to_world_ray, rotation_matrix};
use crate::project::{ControlPoint, SourceImage};
use image::{imageops::FilterType, DynamicImage};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub struct Keypoint {
    pub x: f64,
    pub y: f64,
    pub score: f32,
    pub angle: f32,
    pub desc: [u64; 4],
}

pub type PairRotation = (usize, usize, Matrix3<f64>, usize);
pub type ControlPointMatches = (Vec<ControlPoint>, Vec<PairRotation>);
type PairMatch = (usize, usize, Vec<ControlPoint>, Option<Matrix3<f64>>);

/// Extract spatially distributed Harris keypoints + rotation-invariant 256-bit BRIEF descriptors.
pub fn detect_keypoints(
    img: &DynamicImage,
    orig_w: u32,
    orig_h: u32,
    max_points: usize,
) -> Vec<Keypoint> {
    let gray_full = img.to_luma8();
    let max_dim = gray_full.width().max(gray_full.height());
    let scale = if max_dim > 1024 {
        1024.0 / max_dim as f32
    } else {
        1.0
    };
    let w = ((gray_full.width() as f32 * scale).round() as u32).max(32);
    let h = ((gray_full.height() as f32 * scale).round() as u32).max(32);
    let resized = if (w, h) != (gray_full.width(), gray_full.height()) {
        image::imageops::resize(&gray_full, w, h, FilterType::Triangle)
    } else {
        gray_full
    };

    let wi = w as usize;
    let hi = h as usize;
    let raw = resized.as_raw();

    // Box-smoothed luma for stable descriptors and gradients.
    let mut smooth = vec![0.0f32; wi * hi];
    for y in 1..hi - 1 {
        for x in 1..wi - 1 {
            let mut acc = 0.0f32;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let idx = (y as isize + dy) as usize * wi + (x as isize + dx) as usize;
                    acc += raw[idx] as f32;
                }
            }
            smooth[y * wi + x] = acc / 9.0;
        }
    }

    // Gradients + structure tensor entries.
    let mut ixx = vec![0.0f32; wi * hi];
    let mut iyy = vec![0.0f32; wi * hi];
    let mut ixy = vec![0.0f32; wi * hi];
    let mut gx_arr = vec![0.0f32; wi * hi];
    let mut gy_arr = vec![0.0f32; wi * hi];

    for y in 2..hi - 2 {
        for x in 2..wi - 2 {
            let idx = y * wi + x;
            let gx = 0.5 * (smooth[idx + 1] - smooth[idx - 1]);
            let gy = 0.5 * (smooth[idx + wi] - smooth[idx - wi]);
            gx_arr[idx] = gx;
            gy_arr[idx] = gy;
            ixx[idx] = gx * gx;
            iyy[idx] = gy * gy;
            ixy[idx] = gx * gy;
        }
    }

    // Harris response integrated over 3x3 window, kept in grid buckets for even spatial coverage.
    let border = 18usize;
    if wi <= border * 2 || hi <= border * 2 {
        return Vec::new();
    }

    let grid_x = 8usize;
    let grid_y = 6usize;
    let per_cell = (max_points / (grid_x * grid_y)).max(4);
    let mut buckets: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); grid_x * grid_y];

    let mut responses = vec![0.0f32; wi * hi];
    for y in border..hi - border {
        for x in border..wi - border {
            let mut sxx = 0.0f32;
            let mut syy = 0.0f32;
            let mut sxy = 0.0f32;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let k = (y as isize + dy) as usize * wi + (x as isize + dx) as usize;
                    sxx += ixx[k];
                    syy += iyy[k];
                    sxy += ixy[k];
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let r = det - 0.04 * trace * trace;
            if r > 12.0 {
                responses[y * wi + x] = r;
            }
        }
    }

    for y in border..hi - border {
        for x in border..wi - border {
            let idx = y * wi + x;
            let r = responses[idx];
            if r <= 0.0 {
                continue;
            }
            // 3x3 non-maximum suppression.
            let mut is_max = true;
            'nms: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nb = (y as isize + dy) as usize * wi + (x as isize + dx) as usize;
                    if responses[nb] > r {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                let cx = ((x - border) * grid_x) / (wi - 2 * border).max(1);
                let cy = ((y - border) * grid_y) / (hi - 2 * border).max(1);
                let cell = cy.min(grid_y - 1) * grid_x + cx.min(grid_x - 1);
                buckets[cell].push((x, y, r));
            }
        }
    }

    let pairs = brief_pairs();
    let sx_full = orig_w as f64 / wi as f64;
    let sy_full = orig_h as f64 / hi as f64;
    let mut keypoints = Vec::new();

    for mut cell in buckets {
        cell.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        cell.truncate(per_cell);
        for (x, y, score) in cell {
            // Dominant gradient orientation over 7x7 window.
            let mut sum_gx = 0.0f32;
            let mut sum_gy = 0.0f32;
            for dy in -3isize..=3 {
                for dx in -3isize..=3 {
                    let k = (y as isize + dy) as usize * wi + (x as isize + dx) as usize;
                    sum_gx += gx_arr[k];
                    sum_gy += gy_arr[k];
                }
            }
            let angle = sum_gy.atan2(sum_gx);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let mut desc = [0u64; 4];
            for (bit_idx, &(ax, ay, bx, by)) in pairs.iter().enumerate() {
                let rx1 = (x as f32 + ax * cos_a - ay * sin_a).round() as isize;
                let ry1 = (y as f32 + ax * sin_a + ay * cos_a).round() as isize;
                let rx2 = (x as f32 + bx * cos_a - by * sin_a).round() as isize;
                let ry2 = (y as f32 + bx * sin_a + by * cos_a).round() as isize;
                let v1 = smooth[ry1.clamp(0, hi as isize - 1) as usize * wi
                    + rx1.clamp(0, wi as isize - 1) as usize];
                let v2 = smooth[ry2.clamp(0, hi as isize - 1) as usize * wi
                    + rx2.clamp(0, wi as isize - 1) as usize];
                if v1 < v2 {
                    desc[bit_idx / 64] |= 1u64 << (bit_idx % 64);
                }
            }

            // Subpixel quadratic refinement around 3x3 Harris peak.
            let center_idx = y * wi + x;
            let rl = responses[center_idx - 1];
            let rr = responses[center_idx + 1];
            let ru = responses[center_idx - wi];
            let rd = responses[center_idx + wi];
            let denom_x = 2.0 * (rl - 2.0 * score + rr);
            let denom_y = 2.0 * (ru - 2.0 * score + rd);
            let dx_sub = if denom_x.abs() > 1e-6 {
                ((rl - rr) / denom_x).clamp(-0.5, 0.5) as f64
            } else {
                0.0
            };
            let dy_sub = if denom_y.abs() > 1e-6 {
                ((ru - rd) / denom_y).clamp(-0.5, 0.5) as f64
            } else {
                0.0
            };

            keypoints.push(Keypoint {
                x: (x as f64 + dx_sub + 0.5) * sx_full,
                y: (y as f64 + dy_sub + 0.5) * sy_full,
                score,
                angle,
                desc,
            });
        }
    }

    keypoints
}

/// Match descriptors between two images and filter geometrically using 3D spherical rotation RANSAC.
pub fn match_image_pair(
    idx_a: usize,
    img_a: &SourceImage,
    kp_a: &[Keypoint],
    idx_b: usize,
    img_b: &SourceImage,
    kp_b: &[Keypoint],
) -> (Vec<ControlPoint>, Option<Matrix3<f64>>) {
    if kp_a.len() < 6 || kp_b.len() < 6 {
        return (Vec::new(), None);
    }

    let mut raw_matches: Vec<(usize, usize)> = Vec::new();
    for (ia, ka) in kp_a.iter().enumerate() {
        let mut best_d = u32::MAX;
        let mut second_d = u32::MAX;
        let mut best_ib = 0usize;
        for (ib, kb) in kp_b.iter().enumerate() {
            let d = hamming(&ka.desc, &kb.desc);
            if d < best_d {
                second_d = best_d;
                best_d = d;
                best_ib = ib;
            } else if d < second_d {
                second_d = d;
            }
        }
        if best_d <= 72 && (best_d as f32) < 0.82 * (second_d as f32) {
            // Symmetric cross-check: confirm best_ib also prefers ia.
            let mut rev_best_ia = ia;
            let mut rev_best_d = u32::MAX;
            for (ia2, ka2) in kp_a.iter().enumerate() {
                let d = hamming(&ka2.desc, &kp_b[best_ib].desc);
                if d < rev_best_d {
                    rev_best_d = d;
                    rev_best_ia = ia2;
                }
            }
            if rev_best_ia == ia {
                raw_matches.push((ia, best_ib));
            }
        }
    }

    if raw_matches.len() < 4 {
        return (Vec::new(), None);
    }

    // Compute camera-local unit rays (ignoring current yaw/pitch/roll so rotation RANSAC works from any initial pose).
    let mut local_a = img_a.clone();
    local_a.yaw_deg = 0.0;
    local_a.pitch_deg = 0.0;
    local_a.roll_deg = 0.0;

    let mut local_b = img_b.clone();
    local_b.yaw_deg = 0.0;
    local_b.pitch_deg = 0.0;
    local_b.roll_deg = 0.0;

    let ray_pairs: Vec<(usize, usize, Vector3<f64>, Vector3<f64>)> = raw_matches
        .iter()
        .filter_map(|&(ia, ib)| {
            let ra = image_px_to_world_ray(kp_a[ia].x, kp_a[ia].y, &local_a)?;
            let rb = image_px_to_world_ray(kp_b[ib].x, kp_b[ib].y, &local_b)?;
            Some((ia, ib, Vector3::from(ra), Vector3::from(rb)))
        })
        .collect();

    if ray_pairs.len() < 4 {
        return (Vec::new(), None);
    }

    // RANSAC over 2-ray-pair 3D rotations (Kabsch / triad).
    let cos_thresh = 2.0f64.to_radians().cos();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_rot: Option<Matrix3<f64>> = None;

    let n = ray_pairs.len();
    let step = ((n * (n - 1)) / 240).max(1);
    let mut sample_idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sample_idx += 1;
            if !sample_idx.is_multiple_of(step) && n > 20 {
                continue;
            }
            let r1a = ray_pairs[i].2;
            let r2a = ray_pairs[j].2;
            let r1b = ray_pairs[i].3;
            let r2b = ray_pairs[j].3;

            // Skip nearly collinear ray pairs.
            if r1a.dot(&r2a).abs() > 0.995 || r1b.dot(&r2b).abs() > 0.995 {
                continue;
            }
            let Some(rot) = rotation_from_two_rays(r1b, r2b, r1a, r2a) else {
                continue;
            };

            let inliers: Vec<usize> = ray_pairs
                .iter()
                .enumerate()
                .filter_map(|(k, (_, _, ra, rb))| {
                    let pred_a = rot * rb;
                    (pred_a.dot(ra) >= cos_thresh).then_some(k)
                })
                .collect();

            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_rot = Some(rot);
            }
        }
    }

    if best_inliers.len() < 4 {
        return (Vec::new(), None);
    }

    // Refine rotation across all inliers via SVD Kabsch, then re-verify against refined rotation.
    let refined_rot = kabsch_inliers(
        &best_inliers
            .iter()
            .map(|&k| (ray_pairs[k].3, ray_pairs[k].2))
            .collect::<Vec<_>>(),
    )
    .or(best_rot);

    let final_inliers: Vec<usize> = if let Some(rot) = refined_rot {
        let refined_cos = 1.6f64.to_radians().cos();
        let verified: Vec<usize> = ray_pairs
            .iter()
            .enumerate()
            .filter_map(|(k, (_, _, ra, rb))| ((rot * rb).dot(ra) >= refined_cos).then_some(k))
            .collect();
        if verified.len() >= 4 {
            verified
        } else {
            best_inliers
        }
    } else {
        best_inliers
    };

    if final_inliers.len() < 4 {
        return (Vec::new(), None);
    }

    let cps = final_inliers
        .into_iter()
        .map(|k| {
            let (ia, ib, _, _) = ray_pairs[k];
            ControlPoint {
                image_a: idx_a,
                x_a: kp_a[ia].x,
                y_a: kp_a[ia].y,
                image_b: idx_b,
                x_b: kp_b[ib].x,
                y_b: kp_b[ib].y,
                weight: 1.0,
            }
        })
        .collect();

    (cps, refined_rot)
}

/// Detect keypoints in parallel across all images and find verified pairwise matches.
pub fn find_all_control_points(
    images: &[SourceImage],
    loaded: &[DynamicImage],
) -> ControlPointMatches {
    find_all_control_points_cancel(images, loaded, &AtomicBool::new(false)).unwrap_or_default()
}

pub fn find_all_control_points_cancel(
    images: &[SourceImage],
    loaded: &[DynamicImage],
    cancel: &AtomicBool,
) -> Result<ControlPointMatches, String> {
    if images.len() != loaded.len() {
        return Err("Source count mismatch".into());
    }
    let keypoints: Vec<Vec<Keypoint>> = images
        .par_iter()
        .zip(loaded.par_iter())
        .map(|(meta, dyn_img)| {
            if cancel.load(Ordering::Relaxed) {
                Vec::new()
            } else {
                detect_keypoints(dyn_img, meta.width, meta.height, 720)
            }
        })
        .collect();

    let mut pairs = Vec::new();
    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            pairs.push((i, j));
        }
    }

    let mut results: Vec<PairMatch> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            if cancel.load(Ordering::Relaxed) {
                return (i, j, vec![], None);
            }
            let (cps, rot) =
                match_image_pair(i, &images[i], &keypoints[i], j, &images[j], &keypoints[j]);
            (i, j, cps, rot)
        })
        .collect();

    // Sort pairs strongest-first so union-find MST prefers high-confidence matches.
    results.sort_by_key(|entry| std::cmp::Reverse(entry.2.len()));

    let n = images.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find_root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    let strong_min = if n <= 2 { 4 } else { 8 };
    let mut all_cps = Vec::new();
    let mut pair_rots = Vec::new();
    for (i, j, cps, rot) in results {
        let count = cps.len();
        if count < 4 {
            continue;
        }
        let ra = find_root(&mut parent, i);
        let rb = find_root(&mut parent, j);
        let connects_new_component = ra != rb;
        if count >= strong_min || connects_new_component {
            if connects_new_component {
                parent[ra] = rb;
            }
            all_cps.extend(cps);
            if let Some(r) = rot {
                pair_rots.push((i, j, r, count));
            }
        }
    }
    if cancel.load(Ordering::Relaxed) {
        Err("Alignment cancelled".into())
    } else {
        Ok((all_cps, pair_rots))
    }
}

fn rotation_from_two_rays(
    b1: Vector3<f64>,
    b2: Vector3<f64>,
    a1: Vector3<f64>,
    a2: Vector3<f64>,
) -> Option<Matrix3<f64>> {
    let cross_b = b1.cross(&b2);
    let cross_a = a1.cross(&a2);
    if cross_b.norm() < 1e-6 || cross_a.norm() < 1e-6 {
        return None;
    }
    let b3 = cross_b.normalize();
    let a3 = cross_a.normalize();
    let b2_orth = b3.cross(&b1).normalize();
    let a2_orth = a3.cross(&a1).normalize();

    let mb = Matrix3::from_columns(&[b1.normalize(), b2_orth, b3]);
    let ma = Matrix3::from_columns(&[a1.normalize(), a2_orth, a3]);
    Some(ma * mb.transpose())
}

fn kabsch_inliers(pairs: &[(Vector3<f64>, Vector3<f64>)]) -> Option<Matrix3<f64>> {
    if pairs.is_empty() {
        return None;
    }
    let mut h = Matrix3::<f64>::zeros();
    for (b, a) in pairs {
        h += a * b.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut d = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Some(u * d * v_t)
}

#[inline]
fn hamming(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    (a[0] ^ b[0]).count_ones()
        + (a[1] ^ b[1]).count_ones()
        + (a[2] ^ b[2]).count_ones()
        + (a[3] ^ b[3]).count_ones()
}

/// Deterministic 256-bit sampling pattern inside radius 14 px.
fn brief_pairs() -> [(f32, f32, f32, f32); 256] {
    let mut out = [(0.0f32, 0.0, 0.0, 0.0); 256];
    let mut seed: u64 = 0x9e3779b97f4a7c15;
    let mut next_coord = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((seed >> 32) as i32 as f32) / (i32::MAX as f32);
        v * 14.0
    };
    for slot in &mut out {
        *slot = (next_coord(), next_coord(), next_coord(), next_coord());
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct StarPoint {
    x_full: f64,
    y_full: f64,
    strength: f32,
    is_sky_star: bool,
}

#[inline]
fn box_sum(sat: &[f64], stride: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    sat[y1 * stride + x1] - sat[y0 * stride + x1] - sat[y1 * stride + x0] + sat[y0 * stride + x0]
}

/// Detect subpixel pinpoint star / compact celestial keypoints using DoG + adaptive cell MAD.
fn detect_star_points(img: &DynamicImage, orig_w: u32, orig_h: u32) -> Vec<StarPoint> {
    let rgb = img.to_rgb8();
    let max_dim = rgb.width().max(rgb.height());
    let scale = if max_dim > 1024 {
        1024.0 / max_dim as f32
    } else {
        1.0
    };
    let w = ((rgb.width() as f32 * scale).round() as u32).max(32);
    let h = ((rgb.height() as f32 * scale).round() as u32).max(32);
    let resized = if (w, h) != (rgb.width(), rgb.height()) {
        image::imageops::resize(&rgb, w, h, FilterType::Triangle)
    } else {
        rgb
    };
    let wi = w as usize;
    let hi = h as usize;
    if wi < 20 || hi < 20 {
        return Vec::new();
    }

    let raw = resized.as_raw();
    let mut luma = vec![0.0f32; wi * hi];
    for i in 0..(wi * hi) {
        let p = &raw[i * 3..i * 3 + 3];
        luma[i] = 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
    }

    // Build summed-area table for O(1) center (3x3) and background (11x11) means.
    let stride = wi + 1;
    let mut sat = vec![0.0f64; (hi + 1) * stride];
    for y in 0..hi {
        let mut row_acc = 0.0f64;
        for x in 0..wi {
            row_acc += luma[y * wi + x] as f64;
            sat[(y + 1) * stride + (x + 1)] = sat[y * stride + (x + 1)] + row_acc;
        }
    }

    let mut dog = vec![0.0f32; wi * hi];
    for y in 5..(hi - 5) {
        for x in 5..(wi - 5) {
            let bg = box_sum(&sat, stride, x - 5, y - 5, x + 6, y + 6) / 121.0;
            let c3 = box_sum(&sat, stride, x - 1, y - 1, x + 2, y + 2) / 9.0;
            let center = 0.55 * (luma[y * wi + x] as f64) + 0.45 * c3;
            let d = (center - bg) as f32;
            if d > 0.0 {
                dog[y * wi + x] = d;
            }
        }
    }

    let grid_x = 8usize;
    let grid_y = 6usize;
    let border = 6usize;
    let mut cell_mads = vec![1.0f32; grid_x * grid_y];
    let mut scratch = Vec::with_capacity(2048);
    for gy in 0..grid_y {
        let y0 = border + (gy * (hi - 2 * border)) / grid_y;
        let y1 = border + ((gy + 1) * (hi - 2 * border)) / grid_y;
        for gx in 0..grid_x {
            let x0 = border + (gx * (wi - 2 * border)) / grid_x;
            let x1 = border + ((gx + 1) * (wi - 2 * border)) / grid_x;
            scratch.clear();
            for y in (y0..y1).step_by(2) {
                for x in (x0..x1).step_by(2) {
                    scratch.push(dog[y * wi + x]);
                }
            }
            if !scratch.is_empty() {
                scratch.sort_by(|a, b| a.total_cmp(b));
                let med = scratch[scratch.len() / 2];
                for v in &mut scratch {
                    *v = (*v - med).abs();
                }
                scratch.sort_by(|a, b| a.total_cmp(b));
                cell_mads[gy * grid_x + gx] = (scratch[scratch.len() / 2] * 1.4826).max(0.5);
            }
        }
    }

    let mut buckets: Vec<Vec<(usize, usize, f32, bool)>> = vec![Vec::new(); grid_x * grid_y];
    for y in border..(hi - border) {
        for x in border..(wi - border) {
            let idx = y * wi + x;
            let peak = dog[idx];
            let cx = ((x - border) * grid_x) / (wi - 2 * border).max(1);
            let cy = ((y - border) * grid_y) / (hi - 2 * border).max(1);
            let cell = cy.min(grid_y - 1) * grid_x + cx.min(grid_x - 1);
            let mad = cell_mads[cell];
            if peak <= (4.5 * mad).max(2.2) {
                continue;
            }
            // Reject isolated single-pixel hot pixels via 4-neighbor support.
            let max_nb4 = dog[idx - 1]
                .max(dog[idx + 1])
                .max(dog[idx - wi])
                .max(dog[idx + wi]);
            if max_nb4 <= 0.08 * peak {
                continue;
            }
            // 7x7 non-maximum suppression.
            let mut is_max = true;
            'nms: for dy in -3isize..=3 {
                for dx in -3isize..=3 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nb = (y as isize + dy) as usize * wi + (x as isize + dx) as usize;
                    if dog[nb] > peak {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if !is_max {
                continue;
            }
            // Check isotropic compactness: reject long straight ridge/edge lines as sky stars.
            let mut ring_max = 0.0f32;
            for d in [-3isize, 3isize] {
                let py = (y as isize + d) as usize;
                let px = (x as isize + d) as usize;
                ring_max = ring_max.max(dog[py * wi + x]).max(dog[y * wi + px]);
            }
            let is_sky_star = ring_max < 0.45 * peak && mad < 4.5;
            buckets[cell].push((x, y, peak, is_sky_star));
        }
    }

    let sx_full = orig_w as f64 / wi as f64;
    let sy_full = orig_h as f64 / hi as f64;
    let mut out = Vec::new();
    for mut cell in buckets {
        cell.sort_by(|a, b| b.2.total_cmp(&a.2));
        cell.truncate(12);
        for (x, y, strength, is_sky_star) in cell {
            // 5x5 intensity-weighted subpixel centroid above local patch minimum.
            let mut min_v = f32::MAX;
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let v = luma[(y as isize + dy) as usize * wi + (x as isize + dx) as usize];
                    min_v = min_v.min(v);
                }
            }
            let mut wsum = 0.0f64;
            let mut xsum = 0.0f64;
            let mut ysum = 0.0f64;
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let px = (x as isize + dx) as usize;
                    let py = (y as isize + dy) as usize;
                    let wt = (luma[py * wi + px] - min_v).max(0.0) as f64;
                    wsum += wt;
                    xsum += (px as f64 + 0.5) * wt;
                    ysum += (py as f64 + 0.5) * wt;
                }
            }
            let (cx, cy) = if wsum > 1e-6 {
                (xsum / wsum, ysum / wsum)
            } else {
                (x as f64 + 0.5, y as f64 + 0.5)
            };
            out.push(StarPoint {
                x_full: cx * sx_full,
                y_full: cy * sy_full,
                strength,
                is_sky_star,
            });
        }
    }
    out
}

/// Guided subpixel star/celestial control-point refinement using preliminary camera poses.
pub fn refine_star_control_points(
    images: &[SourceImage],
    loaded: &[DynamicImage],
    cancel: &AtomicBool,
) -> Result<Vec<ControlPoint>, String> {
    if images.len() != loaded.len() || images.len() < 2 {
        return Ok(Vec::new());
    }
    let stars: Vec<Vec<StarPoint>> = images
        .par_iter()
        .zip(loaded.par_iter())
        .map(|(meta, dyn_img)| {
            if cancel.load(Ordering::Relaxed) {
                Vec::new()
            } else {
                detect_star_points(dyn_img, meta.width, meta.height)
            }
        })
        .collect();

    if cancel.load(Ordering::Relaxed) {
        return Err("Alignment cancelled".into());
    }

    let world_rays: Vec<Vec<Option<Vector3<f64>>>> = images
        .par_iter()
        .zip(stars.par_iter())
        .map(|(im, pts)| {
            pts.iter()
                .map(|p| image_px_to_world_ray(p.x_full, p.y_full, im).map(Vector3::from))
                .collect()
        })
        .collect();

    let fwds: Vec<Vector3<f64>> = images
        .iter()
        .map(|im| {
            rotation_matrix(im.yaw_deg, im.pitch_deg, im.roll_deg) * Vector3::new(0.0, 0.0, 1.0)
        })
        .collect();

    let mut pairs = Vec::new();
    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            // Prioritize temporally nearby pairs (minimizes Earth star rotation drift)
            // while still checking strongly overlapping neighbors.
            let max_sep = (images[i].hfov_deg + images[j].hfov_deg) * 0.6;
            let sep = fwds[i].dot(&fwds[j]).clamp(-1.0, 1.0).acos().to_degrees();
            if sep <= max_sep
                && ((j - i) <= 3 || sep <= (images[i].hfov_deg + images[j].hfov_deg) * 0.4)
            {
                pairs.push((i, j));
            }
        }
    }

    let max_match_cos = 2.4f64.to_radians().cos();
    let ransac_cos = 0.75f64.to_radians().cos();

    let added: Vec<ControlPoint> = pairs
        .into_par_iter()
        .flat_map(|(i, j)| {
            if cancel.load(Ordering::Relaxed) {
                return Vec::new();
            }
            let sa = &stars[i];
            let sb = &stars[j];
            let ra = &world_rays[i];
            let rb = &world_rays[j];
            if sa.len() < 6 || sb.len() < 6 {
                return Vec::new();
            }
            let mut candidates: Vec<(usize, usize, Vector3<f64>, Vector3<f64>)> = Vec::new();
            for (ia, maybe_va) in ra.iter().enumerate() {
                let Some(va) = *maybe_va else { continue };
                let mut best_dot = -1.0f64;
                let mut second_dot = -1.0f64;
                let mut best_ib = 0usize;
                for (ib, maybe_vb) in rb.iter().enumerate() {
                    let Some(vb) = *maybe_vb else { continue };
                    let d = va.dot(&vb);
                    if d > best_dot {
                        second_dot = best_dot;
                        best_dot = d;
                        best_ib = ib;
                    } else if d > second_dot {
                        second_dot = d;
                    }
                }
                if best_dot < max_match_cos {
                    continue;
                }
                let ang1 = best_dot.clamp(-1.0, 1.0).acos();
                let ang2 = second_dot.clamp(-1.0, 1.0).acos();
                if ang1 > 0.65 * ang2 && ang2 < 4.0f64.to_radians() {
                    continue;
                }
                // Mutual best check.
                let vb = rb[best_ib].unwrap();
                let mut rev_best_ia = ia;
                let mut rev_best_dot = -1.0f64;
                for (ia2, opt_va2) in ra.iter().enumerate() {
                    let Some(va2) = opt_va2 else { continue };
                    let d = vb.dot(va2);
                    if d > rev_best_dot {
                        rev_best_dot = d;
                        rev_best_ia = ia2;
                    }
                }
                if rev_best_ia == ia {
                    candidates.push((ia, best_ib, va, vb));
                }
            }
            if candidates.len() < 6 {
                return Vec::new();
            }
            // Local camera rays for pairwise Kabsch verification.
            let mut local_a = images[i].clone();
            local_a.yaw_deg = 0.0;
            local_a.pitch_deg = 0.0;
            local_a.roll_deg = 0.0;
            let mut local_b = images[j].clone();
            local_b.yaw_deg = 0.0;
            local_b.pitch_deg = 0.0;
            local_b.roll_deg = 0.0;
            let local_pairs: Vec<(usize, usize, Vector3<f64>, Vector3<f64>)> = candidates
                .iter()
                .filter_map(|&(ia, ib, _, _)| {
                    let la = Vector3::from(image_px_to_world_ray(
                        sa[ia].x_full,
                        sa[ia].y_full,
                        &local_a,
                    )?);
                    let lb = Vector3::from(image_px_to_world_ray(
                        sb[ib].x_full,
                        sb[ib].y_full,
                        &local_b,
                    )?);
                    Some((ia, ib, la, lb))
                })
                .collect();
            if local_pairs.len() < 6 {
                return Vec::new();
            }
            let m = local_pairs.len();
            let step = ((m * (m - 1)) / 180).max(1);
            let mut sidx = 0usize;
            let mut best_inliers: Vec<usize> = Vec::new();
            for p in 0..m {
                for q in (p + 1)..m {
                    sidx += 1;
                    if !sidx.is_multiple_of(step) && m > 20 {
                        continue;
                    }
                    let (r1a, r2a) = (local_pairs[p].2, local_pairs[q].2);
                    let (r1b, r2b) = (local_pairs[p].3, local_pairs[q].3);
                    if r1a.dot(&r2a).abs() > 0.995 || r1b.dot(&r2b).abs() > 0.995 {
                        continue;
                    }
                    let Some(rot) = rotation_from_two_rays(r1b, r2b, r1a, r2a) else {
                        continue;
                    };
                    let inliers: Vec<usize> = local_pairs
                        .iter()
                        .enumerate()
                        .filter_map(|(k, (_, _, la, lb))| {
                            ((rot * lb).dot(la) >= ransac_cos).then_some(k)
                        })
                        .collect();
                    if inliers.len() > best_inliers.len() {
                        best_inliers = inliers;
                    }
                }
            }
            if best_inliers.len() < 6 {
                return Vec::new();
            }
            let Some(rot) = kabsch_inliers(
                &best_inliers
                    .iter()
                    .map(|&k| (local_pairs[k].3, local_pairs[k].2))
                    .collect::<Vec<_>>(),
            ) else {
                return Vec::new();
            };
            let mut pair_cps: Vec<(f32, ControlPoint)> = local_pairs
                .iter()
                .filter_map(|&(ia, ib, la, lb)| {
                    if (rot * lb).dot(&la) < ransac_cos {
                        return None;
                    }
                    let wt = if sa[ia].is_sky_star && sb[ib].is_sky_star {
                        3.0
                    } else {
                        1.5
                    };
                    Some((
                        sa[ia].strength + sb[ib].strength,
                        ControlPoint {
                            image_a: i,
                            x_a: sa[ia].x_full,
                            y_a: sa[ia].y_full,
                            image_b: j,
                            x_b: sb[ib].x_full,
                            y_b: sb[ib].y_full,
                            weight: wt,
                        },
                    ))
                })
                .collect();
            pair_cps.sort_by(|a, b| b.0.total_cmp(&a.0));
            pair_cps.truncate(80);
            pair_cps.into_iter().map(|(_, cp)| cp).collect()
        })
        .collect();

    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::ray_to_image_px;
    use crate::optimizer::{optimize_poses, seed_poses_from_pairs};
    use crate::project::LensProjection;
    use image::{DynamicImage, Rgb, RgbImage};

    #[test]
    fn detects_and_aligns_overlapping_textured_views() {
        // Synthesize two 320x240 views looking at a richly textured spherical scene
        // with true yaws 0 deg and +22 deg (about 60% overlap at 65 deg HFOV).
        let gt0 = SourceImage {
            id: 0,
            path: "a.png".into(),
            width: 320,
            height: 240,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 65.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let gt1 = SourceImage {
            id: 1,
            yaw_deg: 22.0,
            pitch_deg: -3.0,
            roll_deg: 2.0,
            ..gt0.clone()
        };

        let render_scene = |cam: &SourceImage| -> DynamicImage {
            let mut img = RgbImage::new(cam.width, cam.height);
            for y in 0..cam.height {
                for x in 0..cam.width {
                    let ray = image_px_to_world_ray(x as f64, y as f64, cam).unwrap();
                    let lon = ray[0].atan2(ray[2]);
                    let lat = ray[1].asin();
                    // Non-periodic multi-scale scene with unique landmarks.
                    let g1 = (lon * 17.3 + lat * 11.7).sin() * (lat * 31.1 - lon * 7.9).cos();
                    let g2 = (lon * 41.7 - 0.5).cos() * (lat * 37.3 + lon * 13.1).sin();
                    let g3 = ((lon * 7.0 + 1.2).sin() + (lat * 9.0 - 0.4).cos()) * 0.5;
                    let r = ((g1 * 0.5 + 0.5) * 230.0).clamp(0.0, 255.0) as u8;
                    let g = ((g2 * 0.5 + 0.5) * 230.0).clamp(0.0, 255.0) as u8;
                    let b = ((g3 * 0.5 + 0.5) * 230.0).clamp(0.0, 255.0) as u8;
                    img.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
            DynamicImage::ImageRgb8(img)
        };

        let dyn0 = render_scene(&gt0);
        let dyn1 = render_scene(&gt1);

        // Start both images unaligned at (0, 0, 0).
        let mut imgs = vec![
            gt0.clone(),
            SourceImage {
                yaw_deg: 0.0,
                pitch_deg: 0.0,
                roll_deg: 0.0,
                ..gt1.clone()
            },
        ];

        let (cps, pair_rots) = find_all_control_points(&imgs, &[dyn0, dyn1]);
        assert!(
            cps.len() >= 6,
            "Expected >=6 control points, got {}",
            cps.len()
        );
        seed_poses_from_pairs(&mut imgs, &pair_rots);
        let report = optimize_poses(&mut imgs, &cps, 25);
        assert!(report.final_rms_deg < 0.8, "rms={}", report.final_rms_deg);
        assert!(
            (imgs[1].yaw_deg - 22.0).abs() < 1.5,
            "yaw={}",
            imgs[1].yaw_deg
        );
        assert!(
            (imgs[1].pitch_deg - (-3.0)).abs() < 1.5,
            "pitch={}",
            imgs[1].pitch_deg
        );
        assert!(
            (imgs[1].roll_deg - 2.0).abs() < 1.5,
            "roll={}",
            imgs[1].roll_deg
        );
        let _ = ray_to_image_px([0.0, 0.0, 1.0], &gt0);
    }

    #[test]
    fn refines_isolated_sky_stars_with_guided_matching() {
        let gt0 = SourceImage {
            id: 0,
            path: "sky_a.png".into(),
            width: 480,
            height: 360,
            yaw_deg: 0.0,
            pitch_deg: 15.0,
            roll_deg: 0.0,
            hfov_deg: 75.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let gt1 = SourceImage {
            id: 1,
            yaw_deg: 22.0,
            pitch_deg: 15.5,
            roll_deg: -1.0,
            ..gt0.clone()
        };

        // Pseudo-random distributed stars on the upper sky sphere.
        let mut star_rays = Vec::new();
        for k in 0..70u32 {
            let f1 = ((k.wrapping_mul(2654435761) & 0xffff) as f64) / 65535.0;
            let f2 = (((k + 17).wrapping_mul(2246822519) & 0xffff) as f64) / 65535.0;
            let lon = (-18.0 + 58.0 * f1).to_radians();
            let lat = (-2.0 + 36.0 * f2).to_radians();
            star_rays.push([lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()]);
        }

        let render_sky = |cam: &SourceImage| -> DynamicImage {
            let mut buf = vec![12.0f32; (cam.width * cam.height) as usize];
            for &ray in &star_rays {
                if let Some(([px, py], _)) = ray_to_image_px(ray, cam) {
                    let ix0 = (px.floor() as i32 - 3).max(0) as u32;
                    let ix1 = (px.ceil() as i32 + 3).min(cam.width as i32 - 1) as u32;
                    let iy0 = (py.floor() as i32 - 3).max(0) as u32;
                    let iy1 = (py.ceil() as i32 + 3).min(cam.height as i32 - 1) as u32;
                    for yy in iy0..=iy1 {
                        for xx in ix0..=ix1 {
                            let dx = xx as f64 - px;
                            let dy = yy as f64 - py;
                            let val = (-(dx * dx + dy * dy) / (2.0 * 1.15 * 1.15)).exp() as f32;
                            let idx = (yy * cam.width + xx) as usize;
                            buf[idx] = (buf[idx] + 225.0 * val).min(255.0);
                        }
                    }
                }
            }
            let mut rgb = RgbImage::new(cam.width, cam.height);
            for y in 0..cam.height {
                for x in 0..cam.width {
                    let v = buf[(y * cam.width + x) as usize].round() as u8;
                    rgb.put_pixel(x, y, Rgb([v, v, v]));
                }
            }
            DynamicImage::ImageRgb8(rgb)
        };

        let dyn0 = render_sky(&gt0);
        let dyn1 = render_sky(&gt1);
        let init = vec![
            gt0.clone(),
            SourceImage {
                yaw_deg: 21.0,
                pitch_deg: 16.3,
                roll_deg: -0.3,
                ..gt1.clone()
            },
        ];
        let star_cps =
            refine_star_control_points(&init, &[dyn0, dyn1], &AtomicBool::new(false)).unwrap();
        assert!(
            star_cps.len() >= 12,
            "Expected >=12 guided star CPs, got {}",
            star_cps.len()
        );
        assert!(star_cps.iter().any(|cp| cp.weight > 2.5));
        let mut imgs = init;
        let report = optimize_poses(&mut imgs, &star_cps, 25);
        assert!(report.final_rms_deg < 0.1, "rms={}", report.final_rms_deg);
    }
}
