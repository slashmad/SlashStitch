//! Shared panorama renderer used by both the exact preview and the exporter.
//!
//! Pipeline per output band:
//! 1. Map every output pixel to a world ray (projection, view rotation, crop/region).
//! 2. Project the ray into each covering source through a precomputed `ImageProjector`,
//!    sample the source level whose resolution matches the output scale (anti-aliasing),
//!    apply the photometric correction (gain + vignetting) in linear light.
//! 3. Blend: feathered weighted average, or Burt–Adelson multi-band blending with a seam
//!    label per pixel (highest edge-distance weight), per-image push–pull hole filling so no
//!    black borders bleed into the bands, and a level count derived from the output angular
//!    resolution so the preview and the export blend over the same angular width.
//!
//! Every image is processed only inside its own column range (ROI) of the band, which keeps
//! memory bounded by the panorama width rather than `width × image count`.

use crate::blend::BandViewParams;
use crate::color::TargetTrc;
use crate::optics::{
    image_px_to_world_ray, pano_uv_to_ray_shifted, rotation_matrix, ImageProjector,
};
use crate::photometric::{inv_half_diag_sq, PhotometricParams};
use crate::project::{PanoramaProjection, SourceImage};
use crate::raster::Rgb16Raster;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub type SourceRef<'a> = &'a (dyn Rgb16Raster + 'a);

/// Everything needed to render (part of) a panorama canvas.
pub struct RenderInput<'a> {
    pub out_w: u32,
    pub out_h: u32,
    pub projection: PanoramaProjection,
    pub pano_hfov_deg: f64,
    pub view: BandViewParams,
    pub images: &'a [SourceImage],
    /// Per image: every available raster of that source, at any resolution.
    pub sources: Vec<Vec<SourceRef<'a>>>,
    pub photometric: Option<&'a PhotometricParams>,
    pub multiband: bool,
    pub trc: &'a TargetTrc,
    /// Source resampling kernel.
    pub interpolation: Interpolation,
    /// Angular width of the coarsest multi-band transition (degrees).
    pub blend_width_deg: f64,
    /// Replace transient light seen by only one of >= 3 overlapping sources (multi-band).
    pub reject_transients: bool,
    /// Optimised seam layout for the whole canvas (same for preview, zoom and export).
    pub seams: Option<&'a SeamMap>,
}

/// Default coarsest multi-band transition width (degrees).
pub const DEFAULT_BLEND_WIDTH_DEG: f64 = 3.0;

/// Source resampling kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    /// Fastest; slightly soft (fast previews, drafts).
    Bilinear,
    /// Panorama Tools Spline36: sharp, minimal ringing (default).
    #[default]
    Spline36,
    /// Lanczos (a = 3): sharpest, slight ringing around point sources.
    Lanczos3,
}

/// Seam owner per cell of the full canvas (`u, v` in `[0, 1]`), independent of crop, region
/// and output size, so every render of a project places the seams identically.
#[derive(Debug, Clone)]
pub struct SeamMap {
    pub w: usize,
    pub h: usize,
    pub labels: Vec<u16>,
}

impl SeamMap {
    #[inline]
    pub fn label_at(&self, u: f64, v: f64) -> u16 {
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return u16::MAX;
        }
        let x = ((u * self.w as f64) as usize).min(self.w - 1);
        let y = ((v * self.h as f64) as usize).min(self.h - 1);
        self.labels[y * self.w + x]
    }
}

#[derive(Clone)]
struct CanvasMap {
    projection: PanoramaProjection,
    hfov: f64,
    v_shift: f64,
    cl: f64,
    ct: f64,
    cw: f64,
    ch: f64,
    aspect: f64,
    out_w: f64,
    out_h: f64,
    /// Full-resolution canvas size and the mapping from this render's pixel coordinates to
    /// full-resolution ones (identity except for the global coarse layer of a strip export).
    full_w: f64,
    full_h: f64,
    grid_scale: f64,
    grid_offset: f64,
    rot: Matrix3<f64>,
}

impl CanvasMap {
    /// Normalized canvas coordinates of a pixel position.
    #[inline]
    fn uv(&self, x: f64, y: f64) -> (f64, f64) {
        let xf = x * self.grid_scale + self.grid_offset;
        let yf = y * self.grid_scale + self.grid_offset;
        (
            self.cl + (xf / self.full_w) * self.cw,
            self.ct + (yf / self.full_h) * self.ch,
        )
    }

    #[inline]
    fn ray(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        let (u, v) = self.uv(x, y);
        let raw =
            pano_uv_to_ray_shifted(self.projection, u, v, self.hfov, self.aspect, self.v_shift)?;
        let r = self.rot * Vector3::from(raw);
        Some([r.x, r.y, r.z])
    }
}

/// Per-render precomputation shared by all bands.
pub struct RenderPlan {
    canvas: CanvasMap,
    projectors: Vec<ImageProjector>,
    /// Per image: available rasters as `(index, width fraction, height fraction)`, finest
    /// first. The level is chosen per sample from the local source footprint (trilinear).
    levels_of: Vec<Vec<(usize, f64, f64)>>,
    /// Any image offers more than one resolution (enables per-sample footprints).
    multi_level: bool,
    inv_hd2: Vec<f64>,
    photometric: Option<PhotometricParams>,
    /// Number of multi-band pyramid levels.
    pub levels: usize,
    /// Band/ROI alignment, `2^(levels-1)`.
    pub align: u32,
    /// Rows/columns of context needed around a band for strip-invariant blending.
    pub pad: u32,
    /// Output resolution in pixels per degree near the canvas centre.
    pub out_px_per_deg: f64,
    /// Largest ratio between the resolution an image needs and the best raster available
    /// (> 1 means the output is finer than the provided sources).
    pub resolution_deficit: f64,
    /// Globally blended coarse levels for strip-wise rendering of deep pyramids.
    coarse: Option<CoarseLayer>,
}

/// Multi-band result for pyramid levels `k..` of the whole canvas at `1/2^k` resolution.
/// Strips then only blend levels `0..k`, so their context padding stays small no matter how
/// deep the full pyramid is.
struct CoarseLayer {
    k: usize,
    w: usize,
    h: usize,
    rgb: Vec<[f32; 3]>,
}

/// Output of pass A for one band.
struct LabelBand {
    label: Vec<u16>,
    img_rgb: Vec<Vec<[f32; 3]>>,
    img_cov: Vec<Vec<f32>>,
    roi: Vec<Option<(usize, usize)>>,
}

/// Conservative bytes per band pixel of the multi-band renderer: labels, accumulators and
/// output, per-image ROI buffers and the transient pyramids of the image being blended.
pub fn band_bytes_per_px(image_count: usize) -> u64 {
    48 + 10 * image_count as u64 + 96
}

/// Fine pyramid levels blended per strip when a coarse layer is used.
const STRIP_FINE_LEVELS: usize = 4;

fn sanitized_crop(crop: [f64; 4]) -> (f64, f64, f64, f64) {
    let l = crop[0].clamp(0.0, 0.99);
    let t = crop[1].clamp(0.0, 0.99);
    let r = crop[2].clamp(l + 0.0001, 1.0);
    let b = crop[3].clamp(t + 0.0001, 1.0);
    (l, t, r, b)
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    let cx = a[1] * b[2] - a[2] * b[1];
    let cy = a[2] * b[0] - a[0] * b[2];
    let cz = a[0] * b[1] - a[1] * b[0];
    (cx * cx + cy * cy + cz * cz).sqrt().atan2(d)
}

/// Multi-band level count for a given output resolution (pixels per degree). The coarsest
/// band spacing is about three degrees, so preview and export blend over the same angle.
pub fn multiband_levels_for(
    px_per_deg: f64,
    blend_width_deg: f64,
    out_w: u32,
    out_h: u32,
) -> usize {
    let width = if blend_width_deg.is_finite() {
        blend_width_deg.clamp(0.25, 20.0)
    } else {
        DEFAULT_BLEND_WIDTH_DEG
    };
    let target = (px_per_deg.max(0.01) * width).log2().round() as i64 + 1;
    let max_by_size = (f64::from(out_w.min(out_h).max(2)) / 4.0).log2().floor() as i64;
    target.clamp(2, 9).min(max_by_size.max(1)) as usize
}

impl RenderPlan {
    pub fn new(input: &RenderInput<'_>) -> Self {
        Self::new_on_grid(input, None)
    }

    /// `grid = Some((full_w, full_h, scale))` renders the full canvas on a grid `scale` times
    /// coarser whose pixel `c` samples full-resolution pixel `c * scale` (the pyramid grid).
    fn new_on_grid(input: &RenderInput<'_>, grid: Option<(u32, u32, f64)>) -> Self {
        let (cl, ct, cr, cb) = sanitized_crop(input.view.crop);
        let cw = cr - cl;
        let ch = cb - ct;
        let (full_w, full_h, grid_scale) =
            grid.unwrap_or((input.out_w.max(1), input.out_h.max(1), 1.0));
        let aspect = input
            .view
            .canvas_aspect
            .unwrap_or((f64::from(full_w) / cw) / (f64::from(full_h) / ch));
        let canvas = CanvasMap {
            projection: input.projection,
            hfov: input.pano_hfov_deg,
            v_shift: input.view.view_v_shift,
            cl,
            ct,
            cw,
            ch,
            aspect,
            out_w: f64::from(input.out_w.max(1)),
            out_h: f64::from(input.out_h.max(1)),
            full_w: f64::from(full_w),
            full_h: f64::from(full_h),
            grid_scale,
            grid_offset: 0.5 - 0.5 * grid_scale,
            rot: rotation_matrix(input.view.view_yaw_deg, input.view.view_pitch_deg, 0.0),
        };
        let n = input.images.len().min(input.sources.len());
        let projectors: Vec<ImageProjector> =
            input.images[..n].iter().map(ImageProjector::new).collect();

        // Output angular step near the canvas centre (largest of the two axes).
        let mut rad_per_px = None;
        'search: for gy in [0.5, 0.35, 0.65, 0.2, 0.8] {
            for gx in [0.5, 0.35, 0.65, 0.2, 0.8, 0.05, 0.95] {
                let x = gx * canvas.out_w;
                let y = gy * canvas.out_h;
                if let (Some(a), Some(b), Some(c)) = (
                    canvas.ray(x, y),
                    canvas.ray(x + 1.0, y),
                    canvas.ray(x, y + 1.0),
                ) {
                    let step = angle_between(a, b).max(angle_between(a, c));
                    if step > 0.0 && step.is_finite() {
                        rad_per_px = Some(step);
                        break 'search;
                    }
                }
            }
        }
        let rad_per_px = rad_per_px.unwrap_or(std::f64::consts::TAU / canvas.out_w);
        let out_px_per_deg = 1.0 / rad_per_px.to_degrees();

        let mut resolution_deficit = 0.0f64;
        let mut levels_of = Vec::with_capacity(n);
        let mut inv_hd2 = Vec::with_capacity(n);
        for (i, im) in input.images[..n].iter().enumerate() {
            let w = im.width as f64;
            let h = im.height as f64;
            inv_hd2.push(inv_half_diag_sq(w, h));
            let mut lv: Vec<(usize, f64, f64)> = input.sources[i]
                .iter()
                .enumerate()
                .map(|(li, r)| (li, f64::from(r.width()) / w, f64::from(r.height()) / h))
                .collect();
            lv.sort_by(|a, b| b.1.total_cmp(&a.1));
            // Lowest source density (pixels per radian, full resolution) over the image
            // centre and its inner ring; used only to decide whether finer sources are needed.
            let mut src_px_per_rad = f64::INFINITY;
            for (fx, fy) in [(0.5, 0.5), (0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8)] {
                if let (Some(a), Some(b)) = (
                    image_px_to_world_ray(w * fx, h * fy, im),
                    image_px_to_world_ray(w * fx + 1.0, h * fy, im),
                ) {
                    src_px_per_rad = src_px_per_rad.min(1.0 / angle_between(a, b).max(1e-12));
                }
            }
            if !src_px_per_rad.is_finite() {
                src_px_per_rad = w / im.hfov_deg.to_radians().max(1e-6);
            }
            let needed = (rad_per_px.recip() / src_px_per_rad).min(1.0);
            if let Some(&(_, finest_frac, _)) = lv.first() {
                if finest_frac > 0.0 && finest_frac < needed * 0.999 {
                    resolution_deficit = resolution_deficit.max(needed / finest_frac);
                }
            }
            levels_of.push(lv);
        }
        let multi_level = levels_of.iter().any(|l| l.len() > 1);

        let levels = multiband_levels_for(
            out_px_per_deg,
            input.blend_width_deg,
            input.out_w,
            input.out_h,
        );
        let align = 1u32 << (levels - 1);
        let pad = (2u32 << levels).min(512);
        let photometric = input
            .photometric
            .filter(|p| p.is_valid_for(input.images.len()))
            .cloned();
        Self {
            canvas,
            projectors,
            levels_of,
            multi_level,
            inv_hd2,
            photometric,
            levels,
            align,
            pad,
            out_px_per_deg,
            resolution_deficit,
            coarse: None,
        }
    }

    /// For strip-wise rendering (export): when the pyramid is deeper than the strips can
    /// afford, blend the coarse levels once for the whole canvas. Afterwards `align`/`pad`
    /// describe the (much smaller) strip requirements. No-op for shallow pyramids.
    /// Estimated peak memory of the coarse layer when strips blend levels `0..k`.
    pub fn coarse_cost_bytes(&self, input: &RenderInput<'_>, k: usize) -> u64 {
        let scale = 1u64 << k.saturating_sub(2);
        let px = u64::from(input.out_w).div_ceil(scale) * u64::from(input.out_h).div_ceil(scale);
        px * band_bytes_per_px(self.projectors.len())
    }

    /// For strip-wise rendering (export) and identical exact previews: when the pyramid is
    /// deeper than strips can afford, blend the coarse levels once for the whole canvas.
    /// The split level is raised until the global layer fits `budget` bytes; if it never
    /// does, strips keep the full pyramid (and the caller's strip budget decides).
    /// Returns `None` when cancelled. Afterwards `align`/`pad` describe the strips.
    pub fn prepare_coarse(
        &mut self,
        input: &RenderInput<'_>,
        budget: u64,
        cancel: Option<&AtomicBool>,
    ) -> Option<()> {
        if !input.multiband || self.levels <= STRIP_FINE_LEVELS + 1 || self.projectors.len() < 2 {
            return Some(());
        }
        let mut k = STRIP_FINE_LEVELS;
        while k + 2 <= self.levels && self.coarse_cost_bytes(input, k) > budget {
            k += 1;
        }
        if k + 1 >= self.levels || self.coarse_cost_bytes(input, k) > budget {
            return Some(());
        }
        // Render the coarse layer on a grid 4x finer than level k, then Gaussian-reduce it
        // twice: an anti-aliased low-pass image regardless of which source levels exist.
        const OVERSAMPLE_LEVELS: usize = 2;
        let render_level = k - OVERSAMPLE_LEVELS;
        let scale = 1u32 << render_level;
        let rw = input.out_w.div_ceil(scale).max(1);
        let rh = input.out_h.div_ceil(scale).max(1);
        let coarse_input = RenderInput {
            out_w: rw,
            out_h: rh,
            projection: input.projection,
            pano_hfov_deg: input.pano_hfov_deg,
            view: input.view,
            images: input.images,
            sources: input.sources.clone(),
            photometric: input.photometric,
            multiband: true,
            trc: input.trc,
            interpolation: input.interpolation,
            blend_width_deg: input.blend_width_deg,
            reject_transients: input.reject_transients,
            seams: input.seams,
        };
        let mut plan = Self::new_on_grid(
            &coarse_input,
            Some((input.out_w, input.out_h, f64::from(scale))),
        );
        // Same angular blend width as the full pyramid for the levels this grid carries.
        plan.levels = (self.levels - render_level).clamp(1, plan.levels.max(1));
        plan.align = 1u32 << (plan.levels - 1);
        plan.pad = (2u32 << plan.levels).min(512);
        let (rgb, valid) = plan.render_rows(&coarse_input, 0, rh, cancel)?;
        let (mut w, mut h) = (rw as usize, rh as usize);
        let mut rgb = fill_holes(rgb, &valid, w, h);
        for _ in 0..OVERSAMPLE_LEVELS {
            rgb = reduce(&rgb, w, h, add_rgb);
            w = w.div_ceil(2).max(1);
            h = h.div_ceil(2).max(1);
        }
        let (cw, ch) = (w, h);
        self.coarse = Some(CoarseLayer {
            k,
            w: cw,
            h: ch,
            rgb,
        });
        self.levels = k + 1;
        self.align = 1u32 << k;
        self.pad = (2u32 << self.levels).min(512);
        Some(())
    }

    /// Raster level(s) for a sample whose footprint covers `footprint` full-resolution
    /// source pixels per output pixel: the finest level at or above one level pixel per
    /// output pixel, blended with the next coarser one (trilinear) for seamless transitions.
    /// Returns `(finer, Some((coarser, weight_of_coarser)))`.
    fn level_choice(&self, i: usize, footprint: f64) -> (usize, Option<(usize, f32)>) {
        let lv = &self.levels_of[i];
        let desired = if footprint > 1.0 {
            1.0 / footprint
        } else {
            1.0
        };
        let mut a = 0;
        while a + 1 < lv.len() && lv[a + 1].1 >= desired * 0.999 {
            a += 1;
        }
        if a + 1 >= lv.len() || lv[a].1 <= desired * 1.001 {
            return (a, None);
        }
        let t = ((lv[a].1 / desired).ln() / (lv[a].1 / lv[a + 1].1).ln()).clamp(0.0, 1.0) as f32;
        if t < 0.02 {
            (a, None)
        } else if t > 0.98 {
            (a + 1, None)
        } else {
            (a, Some((a + 1, t)))
        }
    }

    #[inline]
    fn sample_level(
        &self,
        input: &RenderInput<'_>,
        i: usize,
        level: usize,
        px: f64,
        py: f64,
    ) -> [f32; 3] {
        let (li, sx, sy) = self.levels_of[i][level];
        let raster = input.sources[i][li];
        let x = ((px + 0.5) * sx - 0.5).max(0.0);
        let y = ((py + 0.5) * sy - 0.5).max(0.0);
        match input.interpolation {
            Interpolation::Spline36 => kernel6_linear(raster, x, y, input.trc, spline36),
            Interpolation::Lanczos3 => kernel6_linear(raster, x, y, input.trc, lanczos3),
            Interpolation::Bilinear => bilinear_linear(raster, x, y, input.trc),
        }
    }

    /// Source footprint (full-resolution pixels per output pixel) of image `i` at `p`,
    /// from the projections of the neighbouring output pixels.
    #[inline]
    fn footprint(
        &self,
        i: usize,
        p: [f64; 2],
        ray_x: Option<[f64; 3]>,
        ray_y: Option<[f64; 3]>,
    ) -> f64 {
        let d = |r: Option<[f64; 3]>| {
            r.and_then(|r| self.projectors[i].project(r))
                .map(|([qx, qy], _)| (qx - p[0]).hypot(qy - p[1]))
        };
        match (d(ray_x), d(ray_y)) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => 1.0,
        }
    }

    #[inline]
    fn sample(
        &self,
        input: &RenderInput<'_>,
        i: usize,
        px: f64,
        py: f64,
        footprint: f64,
    ) -> [f32; 3] {
        let rgb = match self.level_choice(i, footprint) {
            (a, None) => self.sample_level(input, i, a, px, py),
            (a, Some((b, t))) => {
                let fa = self.sample_level(input, i, a, px, py);
                let fb = self.sample_level(input, i, b, px, py);
                [
                    fa[0] + (fb[0] - fa[0]) * t,
                    fa[1] + (fb[1] - fa[1]) * t,
                    fa[2] + (fb[2] - fa[2]) * t,
                ]
            }
        };
        match &self.photometric {
            Some(p) => {
                let im = &input.images[i];
                let dx = px - im.width as f64 * 0.5;
                let dy = py - im.height as f64 * 0.5;
                let r2 = ((dx * dx + dy * dy) * self.inv_hd2[i]) as f32;
                let f = p.factor(i, r2);
                [rgb[0] * f[0], rgb[1] * f[1], rgb[2] * f[2]]
            }
            None => rgb,
        }
    }

    /// Render output rows `[y_start, y_end)` in linear light. Returns pixels and coverage.
    pub fn render_rows(
        &self,
        input: &RenderInput<'_>,
        y_start: u32,
        y_end: u32,
        cancel: Option<&AtomicBool>,
    ) -> Option<(Vec<[f32; 3]>, Vec<bool>)> {
        let wi = input.out_w as usize;
        let strip_h = y_end.saturating_sub(y_start) as usize;
        if self.projectors.is_empty() || wi == 0 || strip_h == 0 {
            return Some((vec![[0.0; 3]; wi * strip_h], vec![false; wi * strip_h]));
        }
        if !input.multiband || self.projectors.len() == 1 || wi < 32 || input.out_h < 32 {
            return self.render_feather(input, y_start, y_end, cancel);
        }
        self.render_multiband(input, y_start, y_end, cancel)
    }

    /// Pass A of the multi-band renderer for rows `[band_y0, band_y1)`: per-pixel seam owner
    /// (after transient rejection) and each image's samples/weights inside its column ROI.
    fn label_band(
        &self,
        input: &RenderInput<'_>,
        band_y0: usize,
        band_y1: usize,
        cancel: Option<&AtomicBool>,
    ) -> Option<LabelBand> {
        let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        let wi = input.out_w as usize;
        let a = self.align as usize;
        let band_h = band_y1 - band_y0;
        let n = self.projectors.len();
        // Coarse coverage scan for per-image column ranges.
        let stride = 8usize;
        let mut ys: Vec<usize> = (band_y0..band_y1).step_by(stride).collect();
        ys.push(band_y1 - 1);
        let mut xs: Vec<usize> = (0..wi).step_by(stride).collect();
        xs.push(wi - 1);
        let ranges: Vec<Vec<(usize, usize)>> = ys
            .par_iter()
            .map(|&y| {
                let mut r = vec![(usize::MAX, 0usize); n];
                for &x in &xs {
                    if let Some(ray) = self.canvas.ray(x as f64 + 0.5, y as f64 + 0.5) {
                        for (i, proj) in self.projectors.iter().enumerate() {
                            if proj.project(ray).is_some() {
                                r[i].0 = r[i].0.min(x);
                                r[i].1 = r[i].1.max(x);
                            }
                        }
                    }
                }
                r
            })
            .collect();
        let mut roi: Vec<Option<(usize, usize)>> = vec![None; n];
        for i in 0..n {
            let (mut lo, mut hi) = (usize::MAX, 0usize);
            for r in &ranges {
                lo = lo.min(r[i].0);
                hi = hi.max(r[i].1);
            }
            if lo == usize::MAX {
                // The sparse grid can step over a source that is small on this canvas: scan it
                // densely (one projection per pixel, only for such small sources).
                let im = &input.images[i];
                let long = f64::from(im.width.max(im.height)) / f64::from(im.width.max(1));
                let size_px = im.hfov_deg * long * self.out_px_per_deg;
                if size_px >= 6.0 * stride as f64 {
                    continue;
                }
                let proj = &self.projectors[i];
                let (dlo, dhi) = (band_y0..band_y1)
                    .into_par_iter()
                    .map(|y| {
                        let mut r = (usize::MAX, 0usize);
                        for x in 0..wi {
                            if let Some(ray) = self.canvas.ray(x as f64 + 0.5, y as f64 + 0.5) {
                                if proj.project(ray).is_some() {
                                    r.0 = r.0.min(x);
                                    r.1 = r.1.max(x);
                                }
                            }
                        }
                        r
                    })
                    .reduce(|| (usize::MAX, 0), |p, q| (p.0.min(q.0), p.1.max(q.1)));
                if dlo == usize::MAX {
                    continue;
                }
                lo = dlo;
                hi = dhi;
            }
            let margin = 2 * stride + self.pad as usize;
            let x0 = lo.saturating_sub(margin) / a * a;
            let x1 = (hi + margin + 1).div_ceil(a) * a;
            roi[i] = Some((x0, x1.min(wi)));
        }

        // Pass A: label (seam owner) per pixel and per-image samples inside each ROI.
        let mut label = vec![u16::MAX; wi * band_h];
        let mut img_rgb: Vec<Vec<[f32; 3]>> = Vec::with_capacity(n);
        let mut img_cov: Vec<Vec<f32>> = Vec::with_capacity(n);
        for r in &roi {
            let len = r.map_or(0, |(x0, x1)| (x1 - x0) * band_h);
            img_rgb.push(vec![[0.0; 3]; len]);
            img_cov.push(vec![0.0; len]);
        }
        {
            type RowSlices<'s> = (
                &'s mut [u16],
                Vec<Option<(&'s mut [[f32; 3]], &'s mut [f32])>>,
            );
            let mut per_row: Vec<RowSlices<'_>> = label
                .chunks_mut(wi)
                .map(|l| (l, Vec::with_capacity(n)))
                .collect();
            for ((r, rgb_buf), cov_buf) in
                roi.iter().zip(img_rgb.iter_mut()).zip(img_cov.iter_mut())
            {
                match *r {
                    Some((x0, x1)) => {
                        let rw = x1 - x0;
                        for ((row, rgb), cov) in per_row
                            .iter_mut()
                            .zip(rgb_buf.chunks_mut(rw))
                            .zip(cov_buf.chunks_mut(rw))
                        {
                            row.1.push(Some((rgb, cov)));
                        }
                    }
                    None => {
                        for row in per_row.iter_mut() {
                            row.1.push(None);
                        }
                    }
                }
            }
            per_row
                .into_par_iter()
                .enumerate()
                .for_each(|(by, (lab, mut slots))| {
                    if cancelled() {
                        return;
                    }
                    let y = (band_y0 + by) as f64 + 0.5;
                    for (x, lab_px) in lab.iter_mut().enumerate() {
                        let Some(ray) = self.canvas.ray(x as f64 + 0.5, y) else {
                            continue;
                        };
                        let (ray_x, ray_y) = if self.multi_level {
                            (
                                self.canvas.ray(x as f64 + 1.5, y),
                                self.canvas.ray(x as f64 + 0.5, y + 1.0),
                            )
                        } else {
                            (None, None)
                        };
                        let mut best = u16::MAX;
                        let mut best_w = 0.0f64;
                        // Owner from the optimised seam map, if that source covers this pixel.
                        let wanted = input.seams.map_or(u16::MAX, |m| {
                            let (u, v) = self.canvas.uv(x as f64 + 0.5, y);
                            m.label_at(u, v)
                        });
                        let mut wanted_ok = false;
                        for i in 0..n {
                            let Some((x0, x1)) = roi[i] else {
                                continue;
                            };
                            if x < x0 || x >= x1 {
                                continue;
                            }
                            if let Some(([px, py], w)) = self.projectors[i].project(ray) {
                                let fp = if self.multi_level {
                                    self.footprint(i, [px, py], ray_x, ray_y)
                                } else {
                                    1.0
                                };
                                let rgb = self.sample(input, i, px, py, fp);
                                if let Some((rs, cs)) = slots[i].as_mut() {
                                    rs[x - x0] = rgb;
                                    // Coverage carries the seam weight (> 0) for relabelling.
                                    cs[x - x0] = (w as f32).max(1e-4);
                                }
                                // Seam owner: the source seeing this direction closest to its
                                // optical centre (sharpest, least vignetted), faded out near
                                // the frame border so seams never sit on an image edge.
                                let im = &input.images[i];
                                let dx = px - im.width as f64 * 0.5;
                                let dy = py - im.height as f64 * 0.5;
                                let r2 = (dx * dx + dy * dy) * self.inv_hd2[i];
                                let score = (1.0 - r2).max(0.0) * (w / 0.1).min(1.0) + 1e-6 * w;
                                if score > best_w {
                                    best_w = score;
                                    best = i as u16;
                                }
                                if i as u16 == wanted && w >= 1e-3 {
                                    wanted_ok = true;
                                }
                            }
                        }
                        *lab_px = if wanted_ok { wanted } else { best };
                    }
                });
        }
        if cancelled() {
            return None;
        }
        // Transient content (a light briefly hitting a rock, a passing car): where at least
        // three sources overlap and the seam owner disagrees strongly with the others at a
        // coarse scale, the area is taken from a source that agrees with the majority.
        if n >= 3 && input.reject_transients {
            self.reject_transients(&mut label, &img_rgb, &img_cov, &roi, wi, band_h, band_y0);
        }
        Some(LabelBand {
            label,
            img_rgb,
            img_cov,
            roi,
        })
    }

    /// Transparent RGBA overlay of the stitch structure for the rendered region: a coloured
    /// frame around the area each source covers and white lines where the multi-band seam
    /// owner changes (the same labels the export uses). `colors[i]` is the frame colour.
    pub fn seam_overlay(
        &self,
        input: &RenderInput<'_>,
        selected: Option<usize>,
        colors: &[[u8; 3]],
        cancel: Option<&AtomicBool>,
    ) -> Option<Vec<u8>> {
        let wi = input.out_w as usize;
        let h = input.out_h as usize;
        if wi == 0 || h == 0 || self.projectors.is_empty() {
            return Some(vec![0; wi * h * 4]);
        }
        let LabelBand {
            label,
            img_cov,
            roi,
            ..
        } = self.label_band(input, 0, h, cancel)?;
        let covers = |i: usize, x: usize, y: usize| -> bool {
            match roi[i] {
                Some((x0, x1)) if x >= x0 && x < x1 => img_cov[i][y * (x1 - x0) + (x - x0)] > 0.0,
                _ => false,
            }
        };
        let n = self.projectors.len();
        let mut out = vec![0u8; wi * h * 4];
        out.par_chunks_mut(wi * 4).enumerate().for_each(|(y, row)| {
            for x in 0..wi {
                let mut px: Option<([u8; 3], u8)> = None;
                // Source frames: coverage boundary of each image (thicker when selected).
                for i in 0..n {
                    let sel = selected == Some(i);
                    let r = if sel { 2isize } else { 1 };
                    let inside = covers(i, x, y);
                    let mut edge = false;
                    for (dx, dy) in [(r, 0), (-r, 0), (0, r), (0, -r)] {
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        let nb = nx >= 0
                            && ny >= 0
                            && (nx as usize) < wi
                            && (ny as usize) < h
                            && covers(i, nx as usize, ny as usize);
                        if inside && !nb {
                            edge = true;
                        }
                    }
                    if edge {
                        let c = colors.get(i).copied().unwrap_or([255, 200, 0]);
                        px = Some((c, if sel { 255 } else { 220 }));
                        if sel {
                            break;
                        }
                    }
                }
                // Seams: owner changes between neighbouring covered pixels.
                if px.is_none() {
                    let l = label[y * wi + x];
                    if l != u16::MAX {
                        let right = (x + 1 < wi).then(|| label[y * wi + x + 1]);
                        let down = (y + 1 < h).then(|| label[(y + 1) * wi + x]);
                        if [right, down]
                            .into_iter()
                            .flatten()
                            .any(|o| o != u16::MAX && o != l)
                        {
                            px = Some(([255, 255, 255], 235));
                        }
                    }
                }
                if let Some((c, alpha)) = px {
                    row[x * 4..x * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], alpha]);
                }
            }
        });
        Some(out)
    }

    /// Block size (pixels) for transient detection: about 0.4° so stars and noise average
    /// out while lit objects remain; a power of two dividing the band alignment.
    fn transient_block(&self) -> usize {
        let target = (self.out_px_per_deg * 0.4).max(1.0);
        let b = 1usize << (target.log2().round().clamp(2.0, 5.0) as u32);
        b.min(self.align.max(4) as usize)
    }

    #[allow(clippy::too_many_arguments)]
    fn reject_transients(
        &self,
        label: &mut [u16],
        img_rgb: &[Vec<[f32; 3]>],
        img_w: &[Vec<f32>],
        roi: &[Option<(usize, usize)>],
        wi: usize,
        band_h: usize,
        band_y0: usize,
    ) {
        const RATIO: f32 = 1.6;
        const DETAIL_RATIO: f32 = 1.4;
        const MIN_ABS: f32 = 0.004;
        const MIN_W: f32 = 0.02;
        let n = roi.len();
        let b = self.transient_block();
        // Absolute block rows keep decisions identical between strips.
        let first_block_row = band_y0 / b;
        let rows_b = (band_y0 + band_h).div_ceil(b) - first_block_row;
        let cols_b = wi.div_ceil(b);
        let luma = |p: [f32; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        // Per block: bit set of outlier images.
        let outliers: Vec<u128> = (0..rows_b * cols_b)
            .into_par_iter()
            .map(|blk| {
                let (br, bc) = (blk / cols_b, blk % cols_b);
                let y_lo = ((first_block_row + br) * b).max(band_y0) - band_y0;
                let y_hi = ((first_block_row + br + 1) * b).min(band_y0 + band_h) - band_y0;
                let (x_lo, x_hi) = (bc * b, ((bc + 1) * b).min(wi));
                let area = (y_hi.saturating_sub(y_lo) * (x_hi - x_lo)) as f32;
                if area <= 0.0 {
                    return 0;
                }
                // (image, mean luminance, mean absolute deviation = local detail)
                let mut means: Vec<(usize, f32, f32)> = Vec::new();
                for i in 0..n.min(128) {
                    let Some((x0, x1)) = roi[i] else {
                        continue;
                    };
                    if x_hi <= x0 || x_lo >= x1 {
                        continue;
                    }
                    let rw = x1 - x0;
                    let (mut sum, mut cnt) = (0.0f32, 0.0f32);
                    for y in y_lo..y_hi {
                        for x in x_lo.max(x0)..x_hi.min(x1) {
                            let k = y * rw + (x - x0);
                            if img_w[i][k] >= MIN_W {
                                sum += luma(img_rgb[i][k]);
                                cnt += 1.0;
                            }
                        }
                    }
                    if cnt >= 0.8 * area {
                        let mean = sum / cnt;
                        let mut dev = 0.0f32;
                        for y in y_lo..y_hi {
                            for x in x_lo.max(x0)..x_hi.min(x1) {
                                let k = y * rw + (x - x0);
                                if img_w[i][k] >= MIN_W {
                                    dev += (luma(img_rgb[i][k]) - mean).abs();
                                }
                            }
                        }
                        means.push((i, mean, dev / cnt));
                    }
                }
                if means.len() < 3 {
                    return 0;
                }
                let median_of = |f: fn(&(usize, f32, f32)) -> f32| {
                    let mut v: Vec<f32> = means.iter().map(f).collect();
                    v.sort_by(f32::total_cmp);
                    v[v.len() / 2]
                };
                let median = median_of(|m| m.1);
                let median_rel = median_of(|m| m.2 / m.1.max(1e-5));
                let mut mask = 0u128;
                // Transient light adds light AND raises the *relative* contrast (lit edges on
                // a dark object). Aurora that is simply brighter in one frame keeps its relative
                // structure and is left alone.
                for &(i, m, dev) in &means {
                    let rel = dev / m.max(1e-5);
                    if m > median.max(1e-5) * RATIO
                        && m - median > MIN_ABS
                        && rel > median_rel.max(0.02) * DETAIL_RATIO
                    {
                        mask |= 1u128 << i;
                    }
                }
                mask
            })
            .collect();
        if outliers.iter().all(|&m| m == 0) {
            return;
        }
        // Per offending image: grow its flagged blocks into contiguous regions and replace each
        // region by ONE source (the best-covering image that agrees with the majority), so a
        // near object seen with parallax is not stitched together from several viewpoints.
        let mut replace: Vec<Vec<(u16, u16)>> = vec![Vec::new(); rows_b * cols_b];
        for i in 0..n.min(128) {
            let bit = 1u128 << i;
            if !outliers.iter().any(|&m| m & bit != 0) {
                continue;
            }
            let mut grid: Vec<bool> = vec![false; rows_b * cols_b];
            for (blk, m) in outliers.iter().enumerate() {
                if m & bit == 0 {
                    continue;
                }
                let (br, bc) = ((blk / cols_b) as isize, (blk % cols_b) as isize);
                for dy in -2..=2isize {
                    for dx in -2..=2isize {
                        let (r, c) = (br + dy, bc + dx);
                        if r >= 0 && c >= 0 && (r as usize) < rows_b && (c as usize) < cols_b {
                            grid[r as usize * cols_b + c as usize] = true;
                        }
                    }
                }
            }
            let mut seen = vec![false; grid.len()];
            for start in 0..grid.len() {
                if !grid[start] || seen[start] {
                    continue;
                }
                let mut comp = vec![start];
                seen[start] = true;
                let mut k = 0;
                while k < comp.len() {
                    let blk = comp[k];
                    k += 1;
                    let (br, bc) = (blk / cols_b, blk % cols_b);
                    let mut push = |r: usize, c: usize| {
                        let q = r * cols_b + c;
                        if grid[q] && !seen[q] {
                            seen[q] = true;
                            comp.push(q);
                        }
                    };
                    if br > 0 {
                        push(br - 1, bc);
                    }
                    if br + 1 < rows_b {
                        push(br + 1, bc);
                    }
                    if bc > 0 {
                        push(br, bc - 1);
                    }
                    if bc + 1 < cols_b {
                        push(br, bc + 1);
                    }
                }
                // Score candidates over the pixels currently owned by image i.
                let mut score = vec![0.0f64; n];
                let mut covered = vec![0usize; n];
                let mut owned = 0usize;
                for &blk in &comp {
                    let (br, bc) = (blk / cols_b, blk % cols_b);
                    let y_lo = ((first_block_row + br) * b).max(band_y0) - band_y0;
                    let y_hi = ((first_block_row + br + 1) * b).min(band_y0 + band_h) - band_y0;
                    for y in y_lo..y_hi {
                        for x in bc * b..((bc + 1) * b).min(wi) {
                            if label[y * wi + x] as usize != i {
                                continue;
                            }
                            owned += 1;
                            for j in 0..n {
                                if j == i || outliers[blk] & (1u128 << j) != 0 {
                                    continue;
                                }
                                let Some((x0, x1)) = roi[j] else {
                                    continue;
                                };
                                if x < x0 || x >= x1 {
                                    continue;
                                }
                                let wv = img_w[j][y * (x1 - x0) + (x - x0)];
                                if wv >= MIN_W {
                                    score[j] += f64::from(wv);
                                    covered[j] += 1;
                                }
                            }
                        }
                    }
                }
                let best = (0..n)
                    .filter(|&j| j != i && owned > 0 && covered[j] * 100 >= owned * 95)
                    .max_by(|&a, &c| score[a].total_cmp(&score[c]));
                if let Some(j) = best {
                    for &blk in &comp {
                        replace[blk].push((i as u16, j as u16));
                    }
                }
            }
        }
        label.par_chunks_mut(wi).enumerate().for_each(|(y, row)| {
            let br = (band_y0 + y) / b - first_block_row;
            for (x, lab) in row.iter_mut().enumerate() {
                if *lab == u16::MAX {
                    continue;
                }
                let Some(&(_, to)) = replace[br * cols_b + x / b]
                    .iter()
                    .find(|(from, _)| *from == *lab)
                else {
                    continue;
                };
                let j = to as usize;
                if let Some((x0, x1)) = roi[j] {
                    if x >= x0 && x < x1 && img_w[j][y * (x1 - x0) + (x - x0)] >= MIN_W {
                        *lab = to;
                    }
                }
            }
        });
    }

    fn render_feather(
        &self,
        input: &RenderInput<'_>,
        y_start: u32,
        y_end: u32,
        cancel: Option<&AtomicBool>,
    ) -> Option<(Vec<[f32; 3]>, Vec<bool>)> {
        let wi = input.out_w as usize;
        let rows: Vec<(Vec<[f32; 3]>, Vec<bool>)> = (y_start..y_end)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![[0.0f32; 3]; wi];
                let mut valid = vec![false; wi];
                if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    return (row, valid);
                }
                for x in 0..wi {
                    let Some(ray) = self.canvas.ray(x as f64 + 0.5, f64::from(y) + 0.5) else {
                        continue;
                    };
                    let (ray_x, ray_y) = if self.multi_level {
                        (
                            self.canvas.ray(x as f64 + 1.5, f64::from(y) + 0.5),
                            self.canvas.ray(x as f64 + 0.5, f64::from(y) + 1.5),
                        )
                    } else {
                        (None, None)
                    };
                    let mut acc = [0.0f32; 3];
                    let mut wsum = 0.0f32;
                    for (i, proj) in self.projectors.iter().enumerate() {
                        if let Some(([px, py], w)) = proj.project(ray) {
                            let fp = if self.multi_level {
                                self.footprint(i, [px, py], ray_x, ray_y)
                            } else {
                                1.0
                            };
                            let rgb = self.sample(input, i, px, py, fp);
                            let wf = w as f32;
                            acc[0] += rgb[0] * wf;
                            acc[1] += rgb[1] * wf;
                            acc[2] += rgb[2] * wf;
                            wsum += wf;
                        }
                    }
                    if wsum > 1e-6 {
                        row[x] = [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum];
                        valid[x] = true;
                    }
                }
                (row, valid)
            })
            .collect();
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
        }
        let mut out = Vec::with_capacity(wi * rows.len());
        let mut valid_out = Vec::with_capacity(wi * rows.len());
        for (row, valid) in rows {
            out.extend_from_slice(&row);
            valid_out.extend_from_slice(&valid);
        }
        Some((out, valid_out))
    }

    fn render_multiband(
        &self,
        input: &RenderInput<'_>,
        y_start: u32,
        y_end: u32,
        cancel: Option<&AtomicBool>,
    ) -> Option<(Vec<[f32; 3]>, Vec<bool>)> {
        let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        let wi = input.out_w as usize;
        let a = self.align as usize;
        let levels = self.levels;
        let band_y0 = (y_start.saturating_sub(self.pad) as usize) / a * a;
        let band_y1 = ((y_end + self.pad) as usize).min(input.out_h as usize);
        let band_h = band_y1 - band_y0;
        let n = self.projectors.len();
        let LabelBand {
            label,
            mut img_rgb,
            mut img_cov,
            roi,
        } = self.label_band(input, band_y0, band_y1, cancel)?;

        // Pass B: per-image pyramids accumulated into the shared blend pyramid.
        let mut acc: Vec<Vec<[f32; 3]>> = Vec::with_capacity(levels);
        let mut wacc: Vec<Vec<f32>> = Vec::with_capacity(levels);
        for l in 0..levels {
            let (lw, lh) = level_dims(wi, band_h, l);
            acc.push(vec![[0.0; 3]; lw * lh]);
            wacc.push(vec![0.0; lw * lh]);
        }
        for i in 0..n {
            if cancelled() {
                return None;
            }
            let Some((x0, x1)) = roi[i] else {
                continue;
            };
            let rw = x1 - x0;
            let mask: Vec<f32> = (0..band_h)
                .into_par_iter()
                .flat_map_iter(|by| {
                    let row = &label[by * wi + x0..by * wi + x1];
                    row.iter()
                        .map(move |&l| if l as usize == i { 1.0 } else { 0.0 })
                })
                .collect();
            if !mask.iter().any(|&m| m > 0.0) {
                continue;
            }
            let rgb = std::mem::take(&mut img_rgb[i]);
            let cov: Vec<f32> = std::mem::take(&mut img_cov[i])
                .into_iter()
                .map(|c| if c > 0.0 { 1.0 } else { 0.0 })
                .collect();
            let lap = laplacian_pyramid_filled(rgb, cov, rw, band_h, levels);
            let mpyr = gaussian_pyramid_f32(mask, rw, band_h, levels);
            for l in 0..levels {
                let (aw, _) = level_dims(wi, band_h, l);
                let (lw, lh) = level_dims(rw, band_h, l);
                let ox = x0 >> l;
                let lap_l = &lap[l];
                let m_l = &mpyr[l];
                acc[l]
                    .par_chunks_mut(aw)
                    .zip(wacc[l].par_chunks_mut(aw))
                    .take(lh)
                    .enumerate()
                    .for_each(|(ry, (arow, wrow))| {
                        for c in 0..lw {
                            let m = m_l[ry * lw + c];
                            if m > 0.0 && ox + c < aw {
                                let v = lap_l[ry * lw + c];
                                let dst = &mut arow[ox + c];
                                dst[0] += v[0] * m;
                                dst[1] += v[1] * m;
                                dst[2] += v[2] * m;
                                wrow[ox + c] += m;
                            }
                        }
                    });
            }
        }

        for l in 0..levels {
            acc[l]
                .par_iter_mut()
                .zip(wacc[l].par_iter())
                .for_each(|(v, &w)| {
                    if w > 1e-6 {
                        v[0] /= w;
                        v[1] /= w;
                        v[2] /= w;
                    }
                });
        }
        if let Some(coarse) = &self.coarse {
            // The coarsest strip level is the low-pass residual: take it from the globally
            // blended coarse layer (same grid: level-k pixel c <-> canvas pixel c * 2^k).
            let top = levels - 1;
            debug_assert_eq!(top, coarse.k);
            let (aw, _) = level_dims(wi, band_h, top);
            let row0 = band_y0 >> top;
            acc[top]
                .par_chunks_mut(aw)
                .enumerate()
                .for_each(|(ry, row)| {
                    let cy = (row0 + ry).min(coarse.h - 1);
                    for (cx, v) in row.iter_mut().enumerate() {
                        *v = coarse.rgb[cy * coarse.w + cx.min(coarse.w - 1)];
                    }
                });
        }
        let collapsed = collapse(&acc, wi, band_h);
        let crop_y0 = y_start as usize - band_y0;
        let strip_h = (y_end - y_start) as usize;
        let mut out = vec![[0.0f32; 3]; wi * strip_h];
        let mut valid = vec![false; wi * strip_h];
        out.par_chunks_mut(wi)
            .zip(valid.par_chunks_mut(wi))
            .enumerate()
            .for_each(|(sy, (orow, vrow))| {
                let by = crop_y0 + sy;
                for x in 0..wi {
                    let idx = by * wi + x;
                    if label[idx] != u16::MAX {
                        let v = collapsed[idx];
                        orow[x] = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)];
                        vrow[x] = true;
                    }
                }
            });
        Some((out, valid))
    }
}

/// Corridor of one line: (crossing, low end, high end (exclusive), low-side label, high-side label).
type SeamSpan = (usize, usize, usize, u16, u16);

/// Compute optimised seams for the whole canvas of `template` (its region/size are ignored).
///
/// 1. Render pass A on a canvas grid about `grid_w` cells wide: initial owner = the source that
///    sees each direction closest to its optical centre.
/// 2. For every pair of neighbouring sources, re-route their shared boundary inside a corridor
///    (about +/- `corridor_deg`) along the path where the two sources agree best (dynamic
///    programming over rows or columns; cost = relative colour difference of the exposure-
///    corrected sources plus a small pull towards the original centre line).
pub fn compute_seam_map(
    template: &RenderInput<'_>,
    grid_w: usize,
    corridor_deg: f64,
    cancel: Option<&AtomicBool>,
) -> Option<SeamMap> {
    let aspect = template.view.canvas_aspect.unwrap_or(2.0).clamp(0.1, 20.0);
    // Same cell budget as a 2:1 canvas `grid_w` wide, whatever the canvas aspect.
    let cells = (grid_w * grid_w / 2) as f64;
    let gw = ((cells * aspect).sqrt().round() as usize).clamp(16, grid_w * 2);
    let gh = ((cells / aspect).sqrt().round() as usize).clamp(16, grid_w * 2);
    let input = RenderInput {
        out_w: gw as u32,
        out_h: gh as u32,
        projection: template.projection,
        pano_hfov_deg: template.pano_hfov_deg,
        view: BandViewParams {
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: Some(aspect),
            ..template.view
        },
        images: template.images,
        sources: template.sources.clone(),
        photometric: template.photometric,
        multiband: true,
        trc: template.trc,
        interpolation: Interpolation::Bilinear,
        blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        reject_transients: false,
        seams: None,
    };
    let plan = RenderPlan::new(&input);
    if plan.projectors.len() < 2 {
        return None;
    }
    let LabelBand {
        mut label,
        img_rgb,
        img_cov,
        roi,
    } = plan.label_band(&input, 0, gh, cancel)?;
    let cells_per_deg = plan.out_px_per_deg.max(0.1);
    let corridor = ((corridor_deg * cells_per_deg).round() as usize).clamp(4, gw / 4);
    // Sample access: weight and colour of source i at (x, y).
    let at = |i: usize, x: usize, y: usize| -> Option<[f32; 3]> {
        let (x0, x1) = roi[i]?;
        if x < x0 || x >= x1 {
            return None;
        }
        let k = y * (x1 - x0) + (x - x0);
        (img_cov[i][k] >= 0.03).then(|| img_rgb[i][k])
    };
    let diff = |a: [f32; 3], b: [f32; 3]| -> f32 {
        (0..3)
            .map(|c| (a[c] - b[c]).abs() / (a[c] + b[c] + 0.004))
            .sum::<f32>()
            / 3.0
    };
    // Neighbouring pairs and the dominant orientation of their boundary.
    let mut pairs: std::collections::HashMap<(u16, u16), (usize, usize)> =
        std::collections::HashMap::new();
    for y in 0..gh {
        for x in 0..gw {
            let l = label[y * gw + x];
            if l == u16::MAX {
                continue;
            }
            if x + 1 < gw {
                let r = label[y * gw + x + 1];
                if r != u16::MAX && r != l {
                    pairs.entry((l.min(r), l.max(r))).or_default().0 += 1;
                }
            }
            if y + 1 < gh {
                let d = label[(y + 1) * gw + x];
                if d != u16::MAX && d != l {
                    pairs.entry((l.min(d), l.max(d))).or_default().1 += 1;
                }
            }
        }
    }
    let mut pair_list: Vec<((u16, u16), (usize, usize))> = pairs.into_iter().collect();
    pair_list.sort_by_key(|p| std::cmp::Reverse(p.1 .0 + p.1 .1));
    for ((a, b), (horiz, vert)) in pair_list {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
        }
        if horiz + vert < 8 {
            continue;
        }
        // horiz counts left/right neighbours: a mostly vertical boundary -> DP over rows.
        let transpose = vert > horiz;
        let (lines, len) = if transpose { (gw, gh) } else { (gh, gw) };
        // Returns (x, y): rows are lines for a vertical boundary, columns otherwise.
        let idx = |line: usize, pos: usize| {
            if transpose {
                (line, pos)
            } else {
                (pos, line)
            }
        };
        // Per line: crossing position, corridor [lo, hi) and which label is on the low side.
        let mut spans: Vec<Option<SeamSpan>> = vec![None; lines];
        for (line, span) in spans.iter_mut().enumerate() {
            let mut cross = None;
            for pos in 0..len.saturating_sub(1) {
                let (x, y) = idx(line, pos);
                let (x2, y2) = idx(line, pos + 1);
                let (l0, l1) = (label[y * gw + x], label[y2 * gw + x2]);
                if l0 != l1 && (l0 == a || l0 == b) && (l1 == a || l1 == b) {
                    cross = Some((pos, l0, l1));
                    break;
                }
            }
            let Some((c, low, high)) = cross else {
                continue;
            };
            let ok = |pos: usize| {
                let (x, y) = idx(line, pos);
                let l = label[y * gw + x];
                (l == a || l == b)
                    && at(a as usize, x, y).is_some()
                    && at(b as usize, x, y).is_some()
            };
            if !ok(c) || !ok(c + 1) {
                continue;
            }
            let mut lo = c;
            while lo > 0 && c - (lo - 1) <= corridor && ok(lo - 1) {
                lo -= 1;
            }
            let mut hi = c + 1;
            while hi + 1 < len && hi - c < corridor && ok(hi + 1) {
                hi += 1;
            }
            *span = Some((c, lo, hi + 1, low, high));
        }
        // Dynamic programming over consecutive lines that have a corridor.
        let mut line = 0;
        while line < lines {
            if spans[line].is_none() {
                line += 1;
                continue;
            }
            let start = line;
            while line < lines && spans[line].is_some() {
                line += 1;
            }
            let seg = start..line;
            // cost[line][pos] for pos in span; back pointers.
            let mut costs: Vec<Vec<f32>> = Vec::with_capacity(seg.len());
            let mut back: Vec<Vec<i8>> = Vec::with_capacity(seg.len());
            for (k, l) in seg.clone().enumerate() {
                let (c, lo, hi, _, _) = spans[l].unwrap();
                let mut row = vec![f32::INFINITY; hi - lo];
                let mut brow = vec![0i8; hi - lo];
                for (j, slot) in row.iter_mut().enumerate() {
                    let pos = lo + j;
                    let (x, y) = idx(l, pos);
                    if at(a as usize, x, y).is_none() || at(b as usize, x, y).is_none() {
                        continue;
                    }
                    // Difference of local averages (5x5 cells): follow broad areas where the
                    // frames agree rather than noise or single stars.
                    let (mut sa, mut sb, mut cnt) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
                    for yy in y.saturating_sub(2)..(y + 3).min(gh) {
                        for xx in x.saturating_sub(2)..(x + 3).min(gw) {
                            if let (Some(qa), Some(qb)) =
                                (at(a as usize, xx, yy), at(b as usize, xx, yy))
                            {
                                for ch in 0..3 {
                                    sa[ch] += qa[ch];
                                    sb[ch] += qb[ch];
                                }
                                cnt += 1.0;
                            }
                        }
                    }
                    let (pa, pb) = (sa.map(|v| v / cnt), sb.map(|v| v / cnt));
                    let pull = 0.1 * (pos as f32 - c as f32).abs() / corridor as f32;
                    let local = diff(pa, pb) + pull;
                    if k == 0 {
                        *slot = local;
                        continue;
                    }
                    let (_, plo, _, _, _) = spans[l - 1].unwrap();
                    let prev = &costs[k - 1];
                    let mut best = f32::INFINITY;
                    let mut bd = 0i8;
                    for d in -1i64..=1 {
                        let pp = pos as i64 + d - plo as i64;
                        // A small price per sideways step keeps seams straight where the
                        // frames agree everywhere.
                        let step = if d == 0 { 0.0 } else { 0.01 };
                        if pp >= 0 && (pp as usize) < prev.len() && prev[pp as usize] + step < best
                        {
                            best = prev[pp as usize] + step;
                            bd = d as i8;
                        }
                    }
                    if best.is_finite() {
                        *slot = best + local;
                        brow[j] = bd;
                    }
                }
                costs.push(row);
                back.push(brow);
            }
            // Trace back the optimal seam position per line.
            let last = costs.len() - 1;
            let Some((mut j, _)) = costs[last]
                .iter()
                .enumerate()
                .filter(|(_, c)| c.is_finite())
                .min_by(|p, q| p.1.total_cmp(q.1))
            else {
                continue;
            };
            let mut seam = vec![0usize; costs.len()];
            for k in (0..costs.len()).rev() {
                let (_, lo, _, _, _) = spans[start + k].unwrap();
                seam[k] = lo + j;
                if k > 0 {
                    let (_, plo, _, _, _) = spans[start + k - 1].unwrap();
                    let prev_pos = (lo + j) as i64 + back[k][j] as i64;
                    j = (prev_pos - plo as i64).max(0) as usize;
                    j = j.min(costs[k - 1].len() - 1);
                }
            }
            // Relabel each corridor: low side up to the seam, high side after it.
            for (k, l) in seg.enumerate() {
                let (_, lo, hi, low, high) = spans[l].unwrap();
                for pos in lo..hi {
                    let (x, y) = idx(l, pos);
                    label[y * gw + x] = if pos <= seam[k] { low } else { high };
                }
            }
        }
    }
    Some(SeamMap {
        w: gw,
        h: gh,
        labels: label,
    })
}

/// Panorama Tools "Spline36" kernel: interpolating, mild negative lobes, sharper than
/// bilinear/bicubic without Lanczos-style ringing around point sources.
#[inline]
fn spline36(x: f32) -> f32 {
    if x < 1.0 {
        ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
    } else if x < 2.0 {
        let x = x - 1.0;
        ((-6.0 / 11.0 * x + 270.0 / 209.0) * x - 156.0 / 209.0) * x
    } else if x < 3.0 {
        let x = x - 2.0;
        ((1.0 / 11.0 * x - 45.0 / 209.0) * x + 26.0 / 209.0) * x
    } else {
        0.0
    }
}

/// Lanczos kernel with a = 3.
#[inline]
fn lanczos3(x: f32) -> f32 {
    if x < 1e-6 {
        1.0
    } else if x < 3.0 {
        let px = std::f32::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    } else {
        0.0
    }
}

#[inline]
fn kernel6_weights(t: f32, kernel: fn(f32) -> f32) -> [f32; 6] {
    let mut w = [0.0f32; 6];
    let mut sum = 0.0;
    for (k, slot) in w.iter_mut().enumerate() {
        *slot = kernel(((k as f32 - 2.0) - t).abs());
        sum += *slot;
    }
    for v in &mut w {
        *v /= sum;
    }
    w
}

/// 6×6 Spline36 interpolation in linear light (pixel centres at integer coordinates).
#[cfg(test)]
#[inline]
fn spline36_linear(img: &dyn Rgb16Raster, px: f64, py: f64, trc: &TargetTrc) -> [f32; 3] {
    kernel6_linear(img, px, py, trc, spline36)
}

/// 6×6 separable interpolation in linear light with the given kernel.
#[inline]
fn kernel6_linear(
    img: &dyn Rgb16Raster,
    px: f64,
    py: f64,
    trc: &TargetTrc,
    kernel: fn(f32) -> f32,
) -> [f32; 3] {
    let w = i64::from(img.width());
    let h = i64::from(img.height());
    let fx0 = px.floor();
    let fy0 = py.floor();
    let wx = kernel6_weights((px - fx0) as f32, kernel);
    let wy = kernel6_weights((py - fy0) as f32, kernel);
    let x0 = fx0 as i64 - 2;
    let y0 = fy0 as i64 - 2;
    let mut xs = [0u32; 6];
    for (k, slot) in xs.iter_mut().enumerate() {
        *slot = (x0 + k as i64).clamp(0, w - 1) as u32;
    }
    let mut out = [0.0f32; 3];
    for (j, wyj) in wy.iter().enumerate() {
        let yy = (y0 + j as i64).clamp(0, h - 1) as u32;
        let mut row = [0.0f32; 3];
        for (xx, wxi) in xs.iter().zip(wx.iter()) {
            let p = trc.decode_rgb16_to_linear(img.sample_rgb16(*xx, yy));
            row[0] += p[0] * wxi;
            row[1] += p[1] * wxi;
            row[2] += p[2] * wxi;
        }
        out[0] += row[0] * wyj;
        out[1] += row[1] * wyj;
        out[2] += row[2] * wyj;
    }
    [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
}

/// Fill uncovered pixels by push–pull from covered neighbours (smooth extrapolation).
fn fill_holes(rgb: Vec<[f32; 3]>, valid: &[bool], w: usize, h: usize) -> Vec<[f32; 3]> {
    if valid.iter().all(|&v| v) || !valid.iter().any(|&v| v) {
        return rgb;
    }
    let levels = {
        let mut l = 1;
        while (w.max(h) >> l) > 1 && l < 16 {
            l += 1;
        }
        l
    };
    let cov: Vec<f32> = valid.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    let lap = laplacian_pyramid_filled(rgb, cov, w, h, levels);
    collapse(&lap, w, h)
}

#[inline]
fn bilinear_linear(img: &dyn Rgb16Raster, px: f64, py: f64, trc: &TargetTrc) -> [f32; 3] {
    let w = img.width();
    let h = img.height();
    let x0 = (px.floor() as u32).min(w - 1);
    let y0 = (py.floor() as u32).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (px - f64::from(x0)).clamp(0.0, 1.0) as f32;
    let fy = (py - f64::from(y0)).clamp(0.0, 1.0) as f32;
    let p00 = trc.decode_rgb16_to_linear(img.sample_rgb16(x0, y0));
    let p10 = trc.decode_rgb16_to_linear(img.sample_rgb16(x1, y0));
    let p01 = trc.decode_rgb16_to_linear(img.sample_rgb16(x0, y1));
    let p11 = trc.decode_rgb16_to_linear(img.sample_rgb16(x1, y1));
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = p00[c] + (p10[c] - p00[c]) * fx;
        let bot = p01[c] + (p11[c] - p01[c]) * fx;
        out[c] = top + (bot - top) * fy;
    }
    out
}

pub(crate) fn level_dims(w: usize, h: usize, level: usize) -> (usize, usize) {
    let mut cw = w;
    let mut ch = h;
    for _ in 0..level {
        cw = cw.div_ceil(2).max(1);
        ch = ch.div_ceil(2).max(1);
    }
    (cw, ch)
}

const K: [(isize, f32); 3] = [(-1, 0.25), (0, 0.5), (1, 0.25)];

/// 2× reduce with a separable [1 2 1]/4 kernel (clamped borders), generic over pixel ops.
fn reduce<T: Copy + Send + Sync + Default>(
    src: &[T],
    w: usize,
    h: usize,
    add: impl Fn(T, T, f32) -> T + Sync,
) -> Vec<T> {
    let dw = w.div_ceil(2).max(1);
    let dh = h.div_ceil(2).max(1);
    // Horizontal pass on the rows that the vertical pass needs (all rows), then vertical.
    let mut tmp = vec![T::default(); dw * h];
    tmp.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..y * w + w];
        for (x, slot) in row.iter_mut().enumerate() {
            let sx = (x * 2).min(w - 1) as isize;
            let mut acc = T::default();
            for (dx, k) in K {
                let xx = (sx + dx).clamp(0, w as isize - 1) as usize;
                acc = add(acc, s[xx], k);
            }
            *slot = acc;
        }
    });
    let mut dst = vec![T::default(); dw * dh];
    dst.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let sy = (y * 2).min(h - 1) as isize;
        for (x, slot) in row.iter_mut().enumerate() {
            let mut acc = T::default();
            for (dy, k) in K {
                let yy = (sy + dy).clamp(0, h as isize - 1) as usize;
                acc = add(acc, tmp[yy * dw + x], k);
            }
            *slot = acc;
        }
    });
    dst
}

fn add_rgb(a: [f32; 3], b: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k]
}

fn add_f32(a: f32, b: f32, k: f32) -> f32 {
    a + b * k
}

/// Bilinear 2× expand from `(sw, sh)` to `(dw, dh)` (same convention as the reduce grid).
fn expand_rgb(src: &[[f32; 3]], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<[f32; 3]> {
    let mut dst = vec![[0.0f32; 3]; dw * dh];
    dst.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let fy = y as f32 * 0.5;
        let y0 = (fy.floor() as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f32;
        for (x, slot) in row.iter_mut().enumerate() {
            let fx = x as f32 * 0.5;
            let x0 = (fx.floor() as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f32;
            let p00 = src[y0 * sw + x0];
            let p10 = src[y0 * sw + x1];
            let p01 = src[y1 * sw + x0];
            let p11 = src[y1 * sw + x1];
            for c in 0..3 {
                let top = p00[c] + (p10[c] - p00[c]) * wx;
                let bot = p01[c] + (p11[c] - p01[c]) * wx;
                slot[c] = top + (bot - top) * wy;
            }
        }
    });
    dst
}

fn gaussian_pyramid_f32(base: Vec<f32>, w: usize, h: usize, levels: usize) -> Vec<Vec<f32>> {
    let mut pyr = Vec::with_capacity(levels);
    pyr.push(base);
    for l in 1..levels {
        let (pw, ph) = level_dims(w, h, l - 1);
        let next = reduce(pyr.last().unwrap(), pw, ph, add_f32);
        pyr.push(next);
    }
    pyr
}

/// Laplacian pyramid of an image that is only defined where `cov > 0`. Undefined pixels are
/// filled by push–pull (normalized reduction, then expansion from coarser levels) so that the
/// bands near the coverage border contain a smooth extrapolation instead of black.
fn laplacian_pyramid_filled(
    rgb: Vec<[f32; 3]>,
    cov: Vec<f32>,
    w: usize,
    h: usize,
    levels: usize,
) -> Vec<Vec<[f32; 3]>> {
    let mut g: Vec<Vec<[f32; 3]>> = Vec::with_capacity(levels);
    let mut c: Vec<Vec<f32>> = Vec::with_capacity(levels);
    g.push(rgb);
    c.push(cov);
    for l in 1..levels {
        let (pw, ph) = level_dims(w, h, l - 1);
        let premul: Vec<[f32; 3]> = g[l - 1]
            .par_iter()
            .zip(c[l - 1].par_iter())
            .map(|(v, &k)| [v[0] * k, v[1] * k, v[2] * k])
            .collect();
        let mut num = reduce(&premul, pw, ph, add_rgb);
        let den = reduce(&c[l - 1], pw, ph, add_f32);
        num.par_iter_mut().zip(den.par_iter()).for_each(|(v, &d)| {
            if d > 1e-6 {
                v[0] /= d;
                v[1] /= d;
                v[2] /= d;
            } else {
                *v = [0.0; 3];
            }
        });
        g.push(num);
        c.push(den);
    }
    // Fill the coarsest level's holes with the covered mean.
    {
        let top = levels - 1;
        let (mut s, mut n) = ([0.0f64; 3], 0.0f64);
        for (v, &k) in g[top].iter().zip(c[top].iter()) {
            if k > 1e-6 {
                s[0] += f64::from(v[0]);
                s[1] += f64::from(v[1]);
                s[2] += f64::from(v[2]);
                n += 1.0;
            }
        }
        let mean = if n > 0.0 {
            [(s[0] / n) as f32, (s[1] / n) as f32, (s[2] / n) as f32]
        } else {
            [0.0; 3]
        };
        for (v, &k) in g[top].iter_mut().zip(c[top].iter()) {
            if k <= 1e-6 {
                *v = mean;
            }
        }
    }
    // Pull: fill holes from the next coarser (already filled) level.
    for l in (0..levels - 1).rev() {
        let (lw, lh) = level_dims(w, h, l);
        let (nw, nh) = level_dims(w, h, l + 1);
        let up = expand_rgb(&g[l + 1], nw, nh, lw, lh);
        let cl = &c[l];
        g[l].par_iter_mut()
            .zip(up.par_iter())
            .zip(cl.par_iter())
            .for_each(|((v, u), &k)| {
                if k <= 1e-6 {
                    *v = *u;
                }
            });
    }
    // Laplacian in place, finest first (each level only reads the unmodified coarser one).
    for l in 0..levels - 1 {
        let (lw, lh) = level_dims(w, h, l);
        let (nw, nh) = level_dims(w, h, l + 1);
        let up = expand_rgb(&g[l + 1], nw, nh, lw, lh);
        g[l].par_iter_mut().zip(up.par_iter()).for_each(|(v, u)| {
            v[0] -= u[0];
            v[1] -= u[1];
            v[2] -= u[2];
        });
    }
    g
}

fn collapse(lap: &[Vec<[f32; 3]>], w: usize, h: usize) -> Vec<[f32; 3]> {
    let levels = lap.len();
    let mut cur = lap[levels - 1].clone();
    for l in (0..levels - 1).rev() {
        let (lw, lh) = level_dims(w, h, l);
        let (nw, nh) = level_dims(w, h, l + 1);
        let mut up = expand_rgb(&cur, nw, nh, lw, lh);
        up.par_iter_mut().zip(lap[l].par_iter()).for_each(|(u, v)| {
            u[0] += v[0];
            u[1] += v[1];
            u[2] += v[2];
        });
        cur = up;
    }
    cur
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
            width: 400,
            height: 300,
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

    fn input<'a>(
        images: &'a [SourceImage],
        rasters: &'a [Rgb16Image],
        trc: &'a TargetTrc,
        w: u32,
        h: u32,
        multiband: bool,
    ) -> RenderInput<'a> {
        RenderInput {
            out_w: w,
            out_h: h,
            projection: PanoramaProjection::Cylindrical,
            pano_hfov_deg: 120.0,
            view: BandViewParams {
                view_yaw_deg: 20.0,
                crop: [0.35, 0.3, 0.75, 0.7],
                canvas_aspect: Some(3.0),
                ..Default::default()
            },
            images,
            sources: rasters.iter().map(|r| vec![r as SourceRef<'a>]).collect(),
            photometric: None,
            multiband,
            trc,
            interpolation: Interpolation::Bilinear,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
            reject_transients: true,
            seams: None,
        }
    }

    fn textured(seed: u32, base: f32, trc: &TargetTrc) -> Rgb16Image {
        Rgb16Image::from_fn(400, 300, |x, y| {
            let n = ((x * 7 + y * 13 + seed) % 17) as f32 / 17.0;
            let v = base * (0.8 + 0.4 * n);
            Rgb(trc.encode_linear_to_u16([v, v * 0.9, v * 1.1]))
        })
    }

    #[test]
    fn strips_match_single_band_render() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let rasters = vec![textured(1, 0.2, &trc), textured(5, 0.3, &trc)];
        let inp = input(&images, &rasters, &trc, 512, 384, true);
        let plan = RenderPlan::new(&inp);
        let (full, fv) = plan.render_rows(&inp, 0, 384, None).unwrap();
        let mut worst = 0.0f32;
        let step = 64;
        for y0 in (0..384).step_by(step) {
            let (part, pv) = plan.render_rows(&inp, y0, y0 + step as u32, None).unwrap();
            for k in 0..part.len() {
                let gi = y0 as usize * 512 + k;
                assert_eq!(pv[k], fv[gi]);
                for c in 0..3 {
                    worst = worst.max((part[k][c] - full[gi][c]).abs());
                }
            }
        }
        assert!(worst < 1.0 / 1024.0, "strip seam {worst}");
    }

    #[test]
    fn multiband_never_bleeds_black_and_matches_flat_input() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let flat = |v: f32| Rgb16Image::from_pixel(400, 300, Rgb(trc.encode_linear_to_u16([v; 3])));
        let rasters = vec![flat(0.25), flat(0.25)];
        let inp = input(&images, &rasters, &trc, 512, 384, true);
        let plan = RenderPlan::new(&inp);
        let (px, valid) = plan.render_rows(&inp, 0, 384, None).unwrap();
        for (p, v) in px.iter().zip(valid.iter()) {
            if *v {
                assert!((p[0] - 0.25).abs() < 2e-3, "value {p:?}");
            }
        }
    }

    #[test]
    fn photometric_gain_removes_exposure_step() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let flat = |v: f32| Rgb16Image::from_pixel(400, 300, Rgb(trc.encode_linear_to_u16([v; 3])));
        let rasters = vec![flat(0.2), flat(0.4)];
        let mut p = PhotometricParams::identity(2);
        p.gains[1] = [0.5; 3];
        let mut inp = input(&images, &rasters, &trc, 512, 384, true);
        inp.photometric = Some(&p);
        let plan = RenderPlan::new(&inp);
        let (px, valid) = plan.render_rows(&inp, 0, 384, None).unwrap();
        let vals: Vec<f32> = px
            .iter()
            .zip(valid.iter())
            .filter(|(_, v)| **v)
            .map(|(p, _)| p[1])
            .collect();
        assert!(!vals.is_empty());
        assert!(vals.iter().all(|v| (v - 0.2).abs() < 3e-3));
    }

    #[test]
    fn lanczos3_interpolates_and_blend_width_controls_pyramid_depth() {
        let trc = TargetTrc::srgb();
        let img = textured(9, 0.25, &trc);
        let exact = trc.decode_rgb16_to_linear(img.sample_rgb16(120, 80));
        let l = kernel6_linear(&img, 120.0, 80.0, &trc, lanczos3);
        assert!((l[1] - exact[1]).abs() < 1e-5);
        let narrow = multiband_levels_for(20.0, 1.5, 4096, 2048);
        let normal = multiband_levels_for(20.0, DEFAULT_BLEND_WIDTH_DEG, 4096, 2048);
        let wide = multiband_levels_for(20.0, 6.0, 4096, 2048);
        assert!(narrow < normal && normal < wide, "{narrow} {normal} {wide}");
    }

    #[test]
    fn spline36_interpolates_exactly_and_is_sharper_than_bilinear() {
        let trc = TargetTrc::srgb();
        let img = textured(3, 0.3, &trc);
        for (x, y) in [(10u32, 12u32), (200, 150), (399, 299)] {
            let exact = trc.decode_rgb16_to_linear(img.sample_rgb16(x, y));
            let s = spline36_linear(&img, f64::from(x), f64::from(y), &trc);
            for c in 0..3 {
                assert!(
                    (s[c] - exact[c]).abs() < 1e-5,
                    "not interpolating at {x},{y}"
                );
            }
        }
        // Step edge: the spline transition is steeper than the bilinear one.
        let step = Rgb16Image::from_fn(64, 8, |x, _| {
            Rgb(trc.encode_linear_to_u16([if x < 32 { 0.1 } else { 0.6 }; 3]))
        });
        let at = |f: fn(&dyn Rgb16Raster, f64, f64, &TargetTrc) -> [f32; 3], x: f64| {
            f(&step, x, 4.0, &trc)[0]
        };
        let slope_spline = at(spline36_linear, 31.75) - at(spline36_linear, 31.25);
        let slope_bilinear = at(bilinear_linear, 31.75) - at(bilinear_linear, 31.25);
        assert!(
            slope_spline > slope_bilinear * 1.1,
            "{slope_spline} vs {slope_bilinear}"
        );
    }

    #[test]
    fn strips_with_global_coarse_layer_match_single_band_render() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let rasters = vec![textured(1, 0.2, &trc), textured(5, 0.35, &trc)];
        let mut inp = input(&images, &rasters, &trc, 1024, 512, true);
        inp.view.crop = [0.47, 0.3, 0.6, 0.7];
        inp.interpolation = Interpolation::Spline36;
        let full_plan = RenderPlan::new(&inp);
        assert!(
            full_plan.levels > STRIP_FINE_LEVELS + 1,
            "levels {}",
            full_plan.levels
        );
        let (full, fv) = full_plan.render_rows(&inp, 0, 512, None).unwrap();
        let mut plan = RenderPlan::new(&inp);
        plan.prepare_coarse(&inp, u64::MAX, None).unwrap();
        assert_eq!(plan.levels, STRIP_FINE_LEVELS + 1);
        let mut sum = 0.0f64;
        let mut worst = 0.0f32;
        let mut n = 0usize;
        for y0 in (0..512u32).step_by(96) {
            let y1 = (y0 + 96).min(512);
            let (part, pv) = plan.render_rows(&inp, y0, y1, None).unwrap();
            for k in 0..part.len() {
                let gi = y0 as usize * 1024 + k;
                assert_eq!(pv[k], fv[gi]);
                if pv[k] {
                    let d = (part[k][1] - full[gi][1]).abs() / full[gi][1].max(1e-3);
                    worst = worst.max(d);
                    sum += f64::from(d);
                    n += 1;
                }
            }
        }
        // The exact preview uses the same coarse model: single band == strips (N07).
        let (one, _) = plan.render_rows(&inp, 0, 512, None).unwrap();
        let mut again = Vec::new();
        for y0 in (0..512u32).step_by(96) {
            again.extend(
                plan.render_rows(&inp, y0, (y0 + 96).min(512), None)
                    .unwrap()
                    .0,
            );
        }
        let identical_worst = one
            .iter()
            .zip(again.iter())
            .map(|(p, q)| (p[1] - q[1]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            identical_worst < 1.0 / 2048.0,
            "exact vs export differ {identical_worst}"
        );
        let mean = sum / n.max(1) as f64;
        // Both are approximations of the ideal deep blend; differences are low-frequency and
        // concentrated near seams (a few 8-bit code values at most).
        assert!(
            mean < 0.01 && worst < 0.08,
            "relative mean {mean} worst {worst}"
        );
    }

    #[test]
    fn transient_light_in_one_of_three_sources_is_rejected() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 20.0), src(2, 40.0)];
        let flat = |v: f32| Rgb16Image::from_pixel(400, 300, Rgb(trc.encode_linear_to_u16([v; 3])));
        // A lit, textured object (like a rock hit by a torch) in the middle source only.
        let mut lit = flat(0.08);
        for y in 120..180 {
            for x in 170..230 {
                let v = if (x / 3 + y / 5) % 3 == 0 { 0.9 } else { 0.35 };
                lit.put_pixel(x, y, Rgb(trc.encode_linear_to_u16([v; 3])));
            }
        }
        let rasters = vec![flat(0.08), lit, flat(0.08)];
        let mut inp = input(&images, &rasters, &trc, 512, 256, true);
        inp.projection = PanoramaProjection::Equirectangular;
        inp.view = BandViewParams {
            view_yaw_deg: 20.0,
            crop: [0.42, 0.42, 0.58, 0.58],
            canvas_aspect: Some(2.0),
            ..Default::default()
        };
        let plan = RenderPlan::new(&inp);
        let (px, valid) = plan.render_rows(&inp, 0, 256, None).unwrap();
        let centre = 128 * 512 + 256;
        assert!(valid[centre]);
        assert!(px[centre][1] < 0.1, "transient kept: {:?}", px[centre]);
        // Without a majority (two sources) nothing is replaced.
        let two = vec![rasters[1].clone(), rasters[2].clone()];
        let two_images = vec![src(0, 20.0), src(1, 40.0)];
        let mut inp2 = input(&two_images, &two, &trc, 512, 256, true);
        inp2.projection = PanoramaProjection::Equirectangular;
        inp2.view = inp.view;
        let plan2 = RenderPlan::new(&inp2);
        let (px2, _) = plan2.render_rows(&inp2, 0, 256, None).unwrap();
        assert!(
            px2[centre][1] > 0.2,
            "two-source case must keep content: {:?}",
            px2[centre]
        );
    }

    #[test]
    fn optimised_seam_routes_around_content_that_differs() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let flat = |v: f32| Rgb16Image::from_pixel(400, 300, Rgb(trc.encode_linear_to_u16([v; 3])));
        // Something present only in B, exactly where the centre seam (yaw 20°) would run.
        let mut b = flat(0.1);
        for y in 0..300u32 {
            for x in 0..400u32 {
                let (dx, dy) = (x as f32 - 96.0, y as f32 - 150.0);
                if dx * dx + dy * dy < 30.0 * 30.0 {
                    b.put_pixel(x, y, Rgb(trc.encode_linear_to_u16([0.7; 3])));
                }
            }
        }
        let rasters = vec![flat(0.1), b];
        let mut inp = input(&images, &rasters, &trc, 1024, 512, true);
        inp.projection = PanoramaProjection::Equirectangular;
        inp.view = BandViewParams {
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: Some(2.0),
            ..Default::default()
        };
        let seam_cells_on_blob = |labels: &[u16], w: usize, h: usize| {
            let full = inp_full(&inp, w, h);
            let plan = RenderPlan::new(&full);
            let lb = plan.label_band(&full, 0, h, None).unwrap();
            let mut count = 0;
            for y in 0..h {
                for x in 0..w - 1 {
                    let (l, r) = (labels[y * w + x], labels[y * w + x + 1]);
                    if l != r && l != u16::MAX && r != u16::MAX {
                        if let Some((x0, x1)) = lb.roi[1] {
                            if x >= x0 && x < x1 && lb.img_rgb[1][y * (x1 - x0) + x - x0][1] > 0.4 {
                                count += 1;
                            }
                        }
                    }
                }
            }
            count
        };
        let base_plan = RenderPlan::new(&inp_full(&inp, 1024, 512));
        let base = base_plan
            .label_band(&inp_full(&inp, 1024, 512), 0, 512, None)
            .unwrap();
        let before = seam_cells_on_blob(&base.label, 1024, 512);
        let map = compute_seam_map(&inp, 1024, 6.0, None).unwrap();
        assert_eq!((map.w, map.h), (1024, 512));
        let after = seam_cells_on_blob(&map.labels, 1024, 512);
        assert!(
            before > 10,
            "test setup: centre seam must cross the blob ({before})"
        );
        assert_eq!(after, 0, "optimised seam still cuts the differing content");
    }

    fn inp_full<'a>(inp: &RenderInput<'a>, w: usize, h: usize) -> RenderInput<'a> {
        RenderInput {
            out_w: w as u32,
            out_h: h as u32,
            projection: inp.projection,
            pano_hfov_deg: inp.pano_hfov_deg,
            view: inp.view,
            images: inp.images,
            sources: inp.sources.clone(),
            photometric: inp.photometric,
            multiband: true,
            trc: inp.trc,
            interpolation: Interpolation::Bilinear,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
            reject_transients: false,
            seams: None,
        }
    }

    #[test]
    fn tiny_sources_are_not_lost_by_the_sparse_coverage_scan() {
        let trc = TargetTrc::srgb();
        let mut a = src(0, 5.0);
        let mut b = src(1, 95.0);
        a.hfov_deg = 5.0;
        b.hfov_deg = 5.0;
        let images = vec![a, b];
        let rasters = vec![textured(1, 0.3, &trc), textured(2, 0.3, &trc)];
        let mut feather = input(&images, &rasters, &trc, 256, 128, false);
        feather.projection = PanoramaProjection::Equirectangular;
        feather.view = BandViewParams {
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: Some(2.0),
            ..Default::default()
        };
        let count = |inp: &RenderInput<'_>| {
            let plan = RenderPlan::new(inp);
            let (_, valid) = plan.render_rows(inp, 0, 128, None).unwrap();
            valid.iter().filter(|v| **v).count()
        };
        let f = count(&feather);
        let mut mb = input(&images, &rasters, &trc, 256, 128, true);
        mb.projection = feather.projection;
        mb.view = feather.view;
        let m = count(&mb);
        assert!(f > 0, "setup: feather sees the sources");
        assert_eq!(m, f, "multi-band must cover the same pixels as feather");
    }

    #[test]
    fn samples_matching_pyramid_level_per_pixel() {
        let trc = TargetTrc::srgb();
        let images = vec![src(0, 0.0), src(1, 40.0)];
        let full = vec![textured(1, 0.2, &trc), textured(5, 0.3, &trc)];
        let half: Vec<Rgb16Image> = full
            .iter()
            .map(|r| crate::raster::half_resolution_linear(r, &trc))
            .collect();
        let render = |srcs: Vec<Vec<SourceRef<'_>>>, w: u32, h: u32| {
            let mut inp = input(&images, &full, &trc, w, h, false);
            inp.sources = srcs;
            let plan = RenderPlan::new(&inp);
            plan.render_rows(&inp, 0, h, None).unwrap().0
        };
        let both = || -> Vec<Vec<SourceRef<'_>>> {
            full.iter()
                .zip(half.iter())
                .map(|(f, h)| vec![f as SourceRef<'_>, h as SourceRef<'_>])
                .collect()
        };
        fn only(v: &[Rgb16Image]) -> Vec<Vec<SourceRef<'_>>> {
            v.iter().map(|r| vec![r as SourceRef<'_>]).collect()
        }
        let mean_diff = |a: &[[f32; 3]], b: &[[f32; 3]]| {
            a.iter()
                .zip(b)
                .map(|(p, q)| (p[1] - q[1]).abs())
                .sum::<f32>()
                / a.len() as f32
        };
        // Strong downscale (footprint >= 2): uses the half level, not aliased full resolution.
        let small_both = render(both(), 40, 30);
        assert!(mean_diff(&small_both, &render(only(&half), 40, 30)) < 1e-3);
        assert!(mean_diff(&small_both, &render(only(&full), 40, 30)) > 2e-3);
        // At or above 1:1 the full-resolution level is used unchanged.
        let big_both = render(both(), 1600, 1200);
        assert!(mean_diff(&big_both, &render(only(&full), 1600, 1200)) < 1e-6);
    }
}
