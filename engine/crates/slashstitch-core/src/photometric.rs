//! Photometric alignment: per-image, per-channel exposure/white-balance gains and a shared
//! radial vignetting model per lens group, estimated from overlapping source pixels.
//!
//! Model (linear light): the lens attenuates by `exp(-(k1 r² + k2 r⁴ + k3 r⁶))` (k > 0: darker
//! corners), so `corrected = gain[i][c] * observed * exp(+(k1 r² + k2 r⁴ + k3 r⁶))`,
//! where `r` is the distance from the image centre normalized by the half diagonal. The
//! unknowns are solved jointly in the log domain by robust (Huber-IRLS) least squares with a
//! weak prior pulling gains to 1 and vignetting to 0. Because the same parameters are applied
//! by the preview and export renderers, the preview shows the exported tonality.

use crate::color::TargetTrc;
use crate::optics::{image_px_to_world_ray, ImageProjector};
use crate::project::SourceImage;
use crate::raster::Rgb16Raster;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhotometricParams {
    /// Linear-light multiplier per image and channel.
    pub gains: Vec<[f32; 3]>,
    /// Vignetting attenuation polynomial `[k1, k2, k3]` per image (shared within a lens
    /// group): observed = scene · exp(-(k1 r² + k2 r⁴ + k3 r⁶)); positive values darken edges.
    pub vignetting: Vec<[f32; 3]>,
    /// Largest normalized squared radius with overlap evidence per image. The correction is
    /// held constant beyond it instead of extrapolating the polynomial into the corners.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vignetting_r2_max: Vec<f32>,
}

impl PhotometricParams {
    pub fn identity(n: usize) -> Self {
        Self {
            gains: vec![[1.0; 3]; n],
            vignetting: vec![[0.0; 3]; n],
            vignetting_r2_max: vec![1.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.gains.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gains.is_empty()
    }

    /// Validate shape and plausibility (used for documents loaded from disk).
    pub fn is_valid_for(&self, n: usize) -> bool {
        self.gains.len() == n
            && self.vignetting.len() == n
            && (self.vignetting_r2_max.is_empty() || self.vignetting_r2_max.len() == n)
            && self
                .vignetting_r2_max
                .iter()
                .all(|r| r.is_finite() && (0.0..=4.0).contains(r))
            && self
                .gains
                .iter()
                .flatten()
                .all(|g| g.is_finite() && (0.05..=20.0).contains(g))
            && self
                .vignetting
                .iter()
                .flatten()
                .all(|k| k.is_finite() && k.abs() <= 20.0)
    }

    /// Multiplicative correction for image `i` at normalized squared radius `r2`.
    #[inline]
    pub fn factor(&self, i: usize, r2: f32) -> [f32; 3] {
        let g = self.gains.get(i).copied().unwrap_or([1.0; 3]);
        let [k1, k2, k3] = self.vignetting.get(i).copied().unwrap_or([0.0; 3]);
        let r2 = r2.min(self.vignetting_r2_max.get(i).copied().unwrap_or(1.0));
        let log_att = r2 * (k1 + r2 * (k2 + r2 * k3));
        // Undo the attenuation.
        let v = log_att.exp();
        [g[0] * v, g[1] * v, g[2] * v]
    }
}

/// Squared radius normalization factor for an image: `1 / half_diagonal²` in full-res pixels.
#[inline]
pub fn inv_half_diag_sq(width: f64, height: f64) -> f64 {
    let hd = 0.5 * width.hypot(height);
    1.0 / (hd * hd).max(1.0)
}

struct Sample {
    i: usize,
    j: usize,
    r2_i: f64,
    r2_j: f64,
    log_i: [f64; 3],
    log_j: [f64; 3],
}

/// Area-average sample (box of `radius` pixels) in linear light from a small raster.
fn box_sample(img: &dyn Rgb16Raster, x: f64, y: f64, radius: i64, trc: &TargetTrc) -> [f64; 3] {
    let w = i64::from(img.width());
    let h = i64::from(img.height());
    let cx = x.round() as i64;
    let cy = y.round() as i64;
    let mut acc = [0.0f64; 3];
    let mut n = 0.0;
    for yy in (cy - radius).max(0)..=(cy + radius).min(h - 1) {
        for xx in (cx - radius).max(0)..=(cx + radius).min(w - 1) {
            let lin = trc.decode_rgb16_to_linear(img.sample_rgb16(xx as u32, yy as u32));
            acc[0] += f64::from(lin[0]);
            acc[1] += f64::from(lin[1]);
            acc[2] += f64::from(lin[2]);
            n += 1.0;
        }
    }
    if n > 0.0 {
        [acc[0] / n, acc[1] / n, acc[2] / n]
    } else {
        [0.0; 3]
    }
}

fn lens_group_key(im: &SourceImage) -> (u32, u32, i64, u8) {
    (
        im.width,
        im.height,
        (im.hfov_deg * 10.0).round() as i64,
        im.lens_projection as u8,
    )
}

/// Estimate photometric parameters from small (preview-size) rasters of every source.
///
/// `small[i]` may have any resolution; coordinates are scaled from the full-resolution
/// geometry in `images[i]`. Returns identity parameters when overlaps are insufficient.
pub fn estimate_photometric(
    images: &[SourceImage],
    small: &[&dyn Rgb16Raster],
    trc: &TargetTrc,
) -> PhotometricParams {
    let n = images.len().min(small.len());
    if n < 2 {
        return PhotometricParams::identity(images.len());
    }
    let projectors: Vec<ImageProjector> = images[..n].iter().map(ImageProjector::new).collect();

    // Collect overlap samples on a regular grid inside each image.
    let samples: Vec<Sample> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let im = &images[i];
            let raster_i = small[i];
            let sx_i = f64::from(raster_i.width()) / im.width as f64;
            let sy_i = f64::from(raster_i.height()) / im.height as f64;
            let inv_i = inv_half_diag_sq(im.width as f64, im.height as f64);
            let grid = 56usize;
            let mut out = Vec::new();
            for gy in 0..grid {
                for gx in 0..grid {
                    let px = (gx as f64 + 0.5) / grid as f64 * im.width as f64;
                    let py = (gy as f64 + 0.5) / grid as f64 * im.height as f64;
                    let Some(ray) = image_px_to_world_ray(px, py, im) else {
                        continue;
                    };
                    let Some((_, w_i)) = projectors[i].project(ray) else {
                        continue;
                    };
                    if w_i < 0.02 {
                        continue;
                    }
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let Some(([qx, qy], w_j)) = projectors[j].project(ray) else {
                            continue;
                        };
                        if w_j < 0.02 {
                            continue;
                        }
                        let raster_j = small[j];
                        let sx_j = f64::from(raster_j.width()) / images[j].width as f64;
                        let sy_j = f64::from(raster_j.height()) / images[j].height as f64;
                        let a = box_sample(raster_i, px * sx_i, py * sy_i, 2, trc);
                        let b = box_sample(raster_j, qx * sx_j, qy * sy_j, 2, trc);
                        let ok = |v: &[f64; 3]| v.iter().all(|&c| c > 2e-4 && c < 0.92);
                        if !ok(&a) || !ok(&b) {
                            continue;
                        }
                        let inv_j =
                            inv_half_diag_sq(images[j].width as f64, images[j].height as f64);
                        let dxi = px - im.width as f64 * 0.5;
                        let dyi = py - im.height as f64 * 0.5;
                        let dxj = qx - images[j].width as f64 * 0.5;
                        let dyj = qy - images[j].height as f64 * 0.5;
                        out.push(Sample {
                            i,
                            j,
                            r2_i: (dxi * dxi + dyi * dyi) * inv_i,
                            r2_j: (dxj * dxj + dyj * dyj) * inv_j,
                            log_i: [a[0].ln(), a[1].ln(), a[2].ln()],
                            log_j: [b[0].ln(), b[1].ln(), b[2].ln()],
                        });
                    }
                }
            }
            out
        })
        .collect();
    if samples.len() < 32 {
        return PhotometricParams::identity(images.len());
    }

    // Lens groups share one vignetting polynomial.
    let mut groups: Vec<(u32, u32, i64, u8)> = Vec::new();
    let group_of: Vec<usize> = images[..n]
        .iter()
        .map(|im| {
            let key = lens_group_key(im);
            match groups.iter().position(|g| *g == key) {
                Some(p) => p,
                None => {
                    groups.push(key);
                    groups.len() - 1
                }
            }
        })
        .collect();
    let ng = groups.len();
    // Unknown layout: [log gains: 3 * n][vignetting: 3 * ng]
    let dim = 3 * n + 3 * ng;
    let vig_off = 3 * n;
    let per_image = samples.len() as f64 / n as f64;
    let gain_prior = 0.02 * per_image.max(1.0);
    let vig_prior = 0.01 * samples.len() as f64;
    let mut sample_w = vec![1.0f64; samples.len() * 3];
    let mut solution = DVector::<f64>::zeros(dim);

    for _iter in 0..5 {
        let mut ata = DMatrix::<f64>::zeros(dim, dim);
        let mut atb = DVector::<f64>::zeros(dim);
        for (s_idx, s) in samples.iter().enumerate() {
            let gi = group_of[s.i];
            let gj = group_of[s.j];
            let phi_i = [s.r2_i, s.r2_i * s.r2_i, s.r2_i * s.r2_i * s.r2_i];
            let phi_j = [s.r2_j, s.r2_j * s.r2_j, s.r2_j * s.r2_j * s.r2_j];
            for c in 0..3 {
                let w = sample_w[s_idx * 3 + c];
                if w <= 0.0 {
                    continue;
                }
                // log corrected = lg + log I + k·phi must agree between i and j:
                // lg_i,c - lg_j,c + k_gi·phi_i - k_gj·phi_j = log_j - log_i
                let mut idx: Vec<(usize, f64)> = Vec::with_capacity(8);
                idx.push((3 * s.i + c, 1.0));
                idx.push((3 * s.j + c, -1.0));
                for p in 0..3 {
                    idx.push((vig_off + 3 * gi + p, phi_i[p]));
                    idx.push((vig_off + 3 * gj + p, -phi_j[p]));
                }
                let rhs = s.log_j[c] - s.log_i[c];
                for &(a, va) in &idx {
                    atb[a] += w * va * rhs;
                    for &(b, vb) in &idx {
                        ata[(a, b)] += w * va * vb;
                    }
                }
            }
        }
        for k in 0..3 * n {
            ata[(k, k)] += gain_prior;
        }
        for k in vig_off..dim {
            ata[(k, k)] += vig_prior;
        }
        let Some(chol) = ata.cholesky() else {
            return PhotometricParams::identity(images.len());
        };
        solution = chol.solve(&atb);
        // Huber reweighting on residuals (log units).
        let delta = 0.04;
        for (s_idx, s) in samples.iter().enumerate() {
            let gi = group_of[s.i];
            let gj = group_of[s.j];
            for c in 0..3 {
                let mut pred = solution[3 * s.i + c] - solution[3 * s.j + c];
                let mut powi = s.r2_i;
                let mut powj = s.r2_j;
                for p in 0..3 {
                    pred += solution[vig_off + 3 * gi + p] * powi;
                    pred -= solution[vig_off + 3 * gj + p] * powj;
                    powi *= s.r2_i;
                    powj *= s.r2_j;
                }
                let r = (pred - (s.log_j[c] - s.log_i[c])).abs();
                // Large disagreements are moving content (aurora, water, clouds); drop them.
                sample_w[s_idx * 3 + c] = if r > 0.6 {
                    0.0
                } else if r <= delta {
                    1.0
                } else {
                    delta / r
                };
            }
        }
    }

    // Radius range with evidence per lens group (98th percentile of sampled radii).
    let mut group_r2: Vec<Vec<f64>> = vec![Vec::new(); ng];
    for s in &samples {
        group_r2[group_of[s.i]].push(s.r2_i);
        group_r2[group_of[s.j]].push(s.r2_j);
    }
    let r2_max: Vec<f32> = group_r2
        .iter_mut()
        .map(|v| {
            if v.is_empty() {
                return 1.0;
            }
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * 0.98) as usize] as f32
        })
        .collect();

    let mut params = PhotometricParams::identity(images.len());
    for i in 0..n {
        for c in 0..3 {
            params.gains[i][c] = (solution[3 * i + c].clamp(-2.0, 2.0)).exp() as f32;
        }
        let g = group_of[i];
        let mut k = [
            solution[vig_off + 3 * g] as f32,
            solution[vig_off + 3 * g + 1] as f32,
            solution[vig_off + 3 * g + 2] as f32,
        ];
        // Physically plausible: attenuation at the corner within [-0.1, 2.5] log units and the
        // curve may not brighten the edge more than slightly.
        let at_corner = k[0] + k[1] + k[2];
        if !at_corner.is_finite() || !(-0.1..=2.5).contains(&at_corner) {
            k = [0.0; 3];
        }
        params.vignetting[i] = k;
        params.vignetting_r2_max[i] = r2_max[g];
    }
    // Normalize gains so the geometric mean of luminance gain is 1 (keeps overall exposure).
    let mean_log: f32 = params
        .gains
        .iter()
        .take(n)
        .map(|g| (0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]).ln())
        .sum::<f32>()
        / n as f32;
    let norm = (-mean_log).exp();
    for g in params.gains.iter_mut().take(n) {
        for c in g.iter_mut() {
            *c *= norm;
        }
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::LensProjection;
    use crate::raster::Rgb16Image;
    use image::Rgb;

    fn src(id: usize, yaw: f64) -> SourceImage {
        SourceImage {
            id,
            path: String::new(),
            width: 300,
            height: 200,
            yaw_deg: yaw,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 70.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        }
    }

    #[test]
    fn recovers_relative_exposure_between_overlapping_frames() {
        let trc = TargetTrc::srgb();
        // Same flat scene, second frame exposed 1.6x brighter in linear light.
        let base = 0.18f32;
        let make = |gain: f32| {
            let enc = trc.encode_linear_to_u16([base * gain; 3]);
            Rgb16Image::from_pixel(300, 200, Rgb(enc))
        };
        let a = make(1.0);
        let b = make(1.6);
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let params = estimate_photometric(&images, &[&a, &b], &trc);
        let ratio = params.gains[0][1] / params.gains[1][1];
        assert!((ratio - 1.6).abs() < 0.05, "gain ratio {ratio}");
        // Corrected linear values now agree.
        let ca = base * params.gains[0][1];
        let cb = base * 1.6 * params.gains[1][1];
        assert!((ca - cb).abs() / ca < 0.03);
    }

    #[test]
    fn recovers_known_vignetting_from_overlaps() {
        let trc = TargetTrc::srgb();
        // Smooth world scene seen by five overlapping frames with identical vignetting.
        let k_true = 0.45f64;
        let scene = |ray: [f64; 3]| 0.12 + 0.05 * (ray[0] * 2.0).sin() + 0.04 * ray[1];
        let images: Vec<SourceImage> = (0..5).map(|i| src(i, i as f64 * 30.0)).collect();
        let rasters: Vec<Rgb16Image> = images
            .iter()
            .map(|im| {
                let inv = inv_half_diag_sq(im.width as f64, im.height as f64);
                Rgb16Image::from_fn(im.width, im.height, |x, y| {
                    let ray = image_px_to_world_ray(x as f64, y as f64, im).unwrap();
                    let dx = x as f64 - im.width as f64 * 0.5;
                    let dy = y as f64 - im.height as f64 * 0.5;
                    let r2 = (dx * dx + dy * dy) * inv;
                    let v = (scene(ray) * (-k_true * r2).exp()) as f32;
                    Rgb(trc.encode_linear_to_u16([v; 3]))
                })
            })
            .collect();
        let refs: Vec<&dyn Rgb16Raster> = rasters.iter().map(|r| r as &dyn Rgb16Raster).collect();
        let params = estimate_photometric(&images, &refs, &trc);
        // Within the evidenced radius the correction undoes the true attenuation ...
        let r2_max = params.vignetting_r2_max[0];
        assert!(r2_max > 0.4 && r2_max <= 1.0, "r2_max {r2_max}");
        let edge = params.factor(0, r2_max)[0] / params.gains[0][0];
        let expected = (k_true * f64::from(r2_max)).exp() as f32;
        assert!(
            (edge / expected - 1.0).abs() < 0.05,
            "edge correction {edge} expected {expected}, k={:?}",
            params.vignetting[0]
        );
        // ... and is held constant beyond it (no polynomial extrapolation into corners).
        assert_eq!(params.factor(0, 1.0), params.factor(0, r2_max));
        // Mid-radius correction as well (shape, not only the corner value).
        let mid = params.factor(0, 0.25)[0] / params.gains[0][0];
        assert!(
            (mid / (k_true * 0.25).exp() as f32 - 1.0).abs() < 0.04,
            "mid {mid}"
        );
    }

    #[test]
    fn identity_factor_is_one() {
        let p = PhotometricParams::identity(2);
        assert_eq!(p.factor(1, 0.7), [1.0; 3]);
        assert!(p.is_valid_for(2));
        assert!(!p.is_valid_for(3));
    }
}
