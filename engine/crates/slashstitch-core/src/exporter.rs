use crate::blend::BandViewParams;
use crate::color::{validate_bit_depth, TargetColorProfile, TargetTrc, DEFAULT_OUTPUT_BIT_DEPTH};
use crate::files::{protect_sources, publish, temporary_output};
use crate::photometric::PhotometricParams;
use crate::project::{PanoramaProjection, SourceImage};
use crate::raster::Rgb16Raster;
use crate::render::{Interpolation, RenderInput, RenderPlan, SourceRef, DEFAULT_BLEND_WIDTH_DEG};
use image::{ExtendedColorType, ImageEncoder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use thiserror::Error;
use tiff::encoder::{
    colortype::{RGB16, RGB8},
    TiffEncoder,
};

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("No source images to export")]
    NoImages,
    #[error("Invalid output dimensions {0}x{1}")]
    InvalidDimensions(u32, u32),
    #[error("Export exceeds the memory budget or uses an unsupported format: {0}")]
    ResourceLimit(String),
    #[error("Export cancelled")]
    Cancelled,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TIFF encoding error: {0}")]
    Tiff(#[from] tiff::TiffError),
    #[error("Image encoding error: {0}")]
    Image(#[from] image::ImageError),
}

fn default_crop() -> [f64; 4] {
    [0.0, 0.0, 1.0, 1.0]
}

fn default_bit_depth() -> u8 {
    DEFAULT_OUTPUT_BIT_DEPTH
}

fn default_color_profile() -> String {
    "srgb".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub projection: PanoramaProjection,
    pub pano_hfov_deg: f64,
    #[serde(default)]
    pub view_yaw_deg: f64,
    #[serde(default)]
    pub view_pitch_deg: f64,
    #[serde(default)]
    pub view_v_shift: f64,
    #[serde(default = "default_crop")]
    pub crop: [f64; 4],
    pub multiband: bool,
    pub force_bigtiff: bool,
    pub rows_per_strip: u32,
    #[serde(default)]
    pub canvas_aspect: Option<f64>,
    #[serde(default)]
    pub overwrite: bool,
    #[serde(default = "default_bit_depth")]
    pub bit_depth: u8,
    #[serde(default = "default_color_profile")]
    pub color_profile: String,
    #[serde(default)]
    pub custom_icc_name: Option<String>,
    #[serde(default)]
    pub custom_icc_base64: Option<String>,
    /// Exposure/white-balance gains and vignetting correction applied while rendering.
    #[serde(default)]
    pub photometric: Option<PhotometricParams>,
    /// Replace transient light seen by only one of several overlapping sources.
    #[serde(default)]
    pub reject_transients: bool,
    /// Source resampling kernel.
    #[serde(default)]
    pub interpolation: Interpolation,
    /// Coarsest multi-band transition width in degrees.
    #[serde(default = "default_blend_width")]
    pub blend_width_deg: f64,
}

fn default_blend_width() -> f64 {
    DEFAULT_BLEND_WIDTH_DEG
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSummary {
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub bigtiff: bool,
    pub bytes_written: u64,
    #[serde(default = "default_bit_depth")]
    pub bit_depth: u8,
    #[serde(default = "default_bit_depth")]
    pub container_bit_depth: u8,
    #[serde(default = "default_color_profile")]
    pub color_profile: String,
    #[serde(default)]
    pub color_profile_name: String,
    #[serde(default)]
    pub icc_embedded: bool,
}

pub fn validate_export_compatibility(
    output_path: &str,
    bit_depth: u8,
    color_profile: &str,
    custom_icc_name: Option<&str>,
    custom_icc_base64: Option<&str>,
) -> Result<(String, TargetColorProfile), String> {
    let bits = validate_bit_depth(bit_depth)?;
    let target = TargetColorProfile::from_parts(color_profile, custom_icc_name, custom_icc_base64)?;
    let ext = Path::new(output_path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if !["tif", "tiff", "png", "jpg", "jpeg"].contains(&ext.as_str()) {
        return Err("unsupported format: choose TIFF, PNG or JPEG with a file extension".into());
    }
    if (ext == "jpg" || ext == "jpeg") && bits != 8 {
        return Err(format!(
            "JPEG only supports 8-bit per channel; select 8-bit depth or export as TIFF/PNG for {bits}-bit precision"
        ));
    }
    Ok((ext, target))
}

/// Export panorama out-of-core in horizontal strips.
/// Writes TIFF/BigTIFF directly strip-by-strip without holding the full output image in RAM.
pub fn export_panorama<F>(
    options: &ExportOptions,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    on_progress: F,
) -> Result<ExportSummary, ExportError>
where
    F: FnMut(u32, u32) -> bool,
{
    let sources: Vec<Vec<SourceRef<'_>>> =
        loaded.iter().map(|r| vec![r as &dyn Rgb16Raster]).collect();
    export_panorama_with_sources(options, images, sources, None, None, on_progress)
}

/// Export using per-image raster pyramids (any resolutions). The renderer picks the level
/// matching the output scale, so downscaled exports are anti-aliased.
pub fn export_panorama_with_sources<F>(
    options: &ExportOptions,
    images: &[SourceImage],
    loaded: Vec<Vec<SourceRef<'_>>>,
    seams: Option<&crate::render::SeamMap>,
    cancel: Option<&AtomicBool>,
    mut on_progress: F,
) -> Result<ExportSummary, ExportError>
where
    F: FnMut(u32, u32) -> bool,
{
    if images.is_empty() || loaded.is_empty() || loaded.iter().any(Vec::is_empty) {
        return Err(ExportError::NoImages);
    }
    if options.width == 0
        || options.height == 0
        || options.width > 200_000
        || options.height > 200_000
    {
        return Err(ExportError::InvalidDimensions(
            options.width,
            options.height,
        ));
    }

    let out_path = Path::new(&options.output_path);
    if images.len() != loaded.len() {
        return Err(ExportError::ResourceLimit(
            "Source/raster count mismatch".into(),
        ));
    }
    let (ext, target_profile) = validate_export_compatibility(
        &options.output_path,
        options.bit_depth,
        &options.color_profile,
        options.custom_icc_name.as_deref(),
        options.custom_icc_base64.as_deref(),
    )
    .map_err(ExportError::ResourceLimit)?;

    let icc_bytes = target_profile
        .icc_bytes()
        .map_err(ExportError::ResourceLimit)?;
    let mox_profile = target_profile
        .to_moxcms_profile()
        .map_err(ExportError::ResourceLimit)?;
    let target_trc =
        TargetTrc::try_from_profile(&mox_profile).map_err(ExportError::ResourceLimit)?;

    if !options.pano_hfov_deg.is_finite()
        || !options.view_yaw_deg.is_finite()
        || !options.view_pitch_deg.is_finite()
        || !options.view_v_shift.is_finite()
        || options.view_v_shift.abs() > 2.0
        || options.crop.iter().any(|v| !v.is_finite())
        || options
            .canvas_aspect
            .is_some_and(|v| !v.is_finite() || !(0.1..=20.0).contains(&v))
    {
        return Err(ExportError::ResourceLimit(
            "Non-finite view parameters".into(),
        ));
    }
    protect_sources(out_path, images.iter().map(|im| Path::new(&im.path)))?;
    if out_path.exists() && !options.overwrite {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "Output exists; explicit overwrite approval is required",
        )
        .into());
    }
    let view = BandViewParams {
        view_yaw_deg: options.view_yaw_deg,
        view_pitch_deg: options.view_pitch_deg,
        view_v_shift: options.view_v_shift,
        crop: options.crop,
        canvas_aspect: options.canvas_aspect,
    };
    let input = RenderInput {
        out_w: options.width,
        out_h: options.height,
        projection: options.projection,
        pano_hfov_deg: options.pano_hfov_deg,
        view,
        images,
        sources: loaded,
        photometric: options.photometric.as_ref(),
        multiband: options.multiband,
        trc: &target_trc,
        interpolation: options.interpolation,
        blend_width_deg: options.blend_width_deg,
        reject_transients: options.reject_transients,
        seams,
    };
    // Every format, output-size and memory check happens before any large allocation.
    preflight_export(options, images.len())?;
    let container_bit_depth = if options.bit_depth > 8 { 16 } else { 8 };
    let bytes_per_pixel = if options.bit_depth > 8 { 6u64 } else { 3u64 };
    let estimated_raw_bytes = u64::from(options.width)
        .checked_mul(u64::from(options.height))
        .and_then(|px| px.checked_mul(bytes_per_pixel))
        .ok_or(ExportError::InvalidDimensions(
            options.width,
            options.height,
        ))?;
    if ext != "tif" && ext != "tiff" && estimated_raw_bytes > FULL_BUFFER_LIMIT {
        return Err(ExportError::ResourceLimit(
            "PNG/JPEG need a full output buffer; choose TIFF for exports above 512 MiB".into(),
        ));
    }
    let mut plan = RenderPlan::new(&input);
    if !on_progress(0, options.height) {
        return Err(ExportError::Cancelled);
    }
    // Deep pyramids: blend coarse levels once globally (within the memory budget) so strips
    // need little padding; cancellable like the strips themselves.
    plan.prepare_coarse(&input, band_budget(), cancel)
        .ok_or(ExportError::Cancelled)?;
    let strip_rows = choose_strip_rows(options, &plan, images.len())?;
    let renderer = StripRenderer {
        input: &input,
        plan: &plan,
        bit_depth: options.bit_depth,
        cancel,
    };
    if !on_progress(0, options.height) {
        return Err(ExportError::Cancelled);
    }
    let mut temporary = temporary_output(out_path)?;
    let use_bigtiff = options.force_bigtiff || estimated_raw_bytes > 3_500_000_000;

    if ext == "tif" || ext == "tiff" {
        {
            let mut writer = BufWriter::new(temporary.as_file_mut());
            if use_bigtiff {
                let mut enc = TiffEncoder::new_big(&mut writer)?;
                write_tiff_strips(
                    &mut enc,
                    options,
                    strip_rows,
                    &renderer,
                    &icc_bytes,
                    &mut on_progress,
                )?;
            } else {
                let mut enc = TiffEncoder::new(&mut writer)?;
                write_tiff_strips(
                    &mut enc,
                    options,
                    strip_rows,
                    &renderer,
                    &icc_bytes,
                    &mut on_progress,
                )?;
            }
            writer.flush()?;
        }
        if !on_progress(options.height, options.height) {
            return Err(ExportError::Cancelled);
        }
        protect_sources(out_path, images.iter().map(|im| Path::new(&im.path)))?;
        let bytes_written = temporary.as_file().metadata()?.len();
        publish(temporary, out_path, options.overwrite)?;
        return Ok(ExportSummary {
            output_path: options.output_path.clone(),
            width: options.width,
            height: options.height,
            bigtiff: use_bigtiff,
            bytes_written,
            bit_depth: options.bit_depth,
            container_bit_depth,
            color_profile: target_profile.id().to_string(),
            color_profile_name: target_profile.display_name(),
            icc_embedded: true,
        });
    }

    // PNG / JPEG: render strips and assemble into a bounded full output buffer (size
    // checked above, before planning).
    if options.bit_depth > 8 {
        let mut full_bytes = vec![
            0u8;
            usize::try_from(estimated_raw_bytes).map_err(|_| {
                ExportError::InvalidDimensions(options.width, options.height)
            })?
        ];
        let mut y = 0u32;
        while y < options.height {
            let y_end = (y + strip_rows).min(options.height);
            let strip = renderer.render_u16(y, y_end)?;
            let dst_off = y as usize * options.width as usize * 6;
            for (i, sample) in strip.into_iter().enumerate() {
                let bytes = sample.to_ne_bytes();
                full_bytes[dst_off + i * 2] = bytes[0];
                full_bytes[dst_off + i * 2 + 1] = bytes[1];
            }
            y = y_end;
            if !on_progress(y, options.height) {
                return Err(ExportError::Cancelled);
            }
        }
        let mut enc = image::codecs::png::PngEncoder::new(temporary.as_file_mut());
        enc.set_icc_profile(icc_bytes)
            .map_err(image::ImageError::Unsupported)?;
        enc.write_image(
            &full_bytes,
            options.width,
            options.height,
            ExtendedColorType::Rgb16,
        )?;
    } else {
        let mut full = vec![
            0u8;
            usize::try_from(estimated_raw_bytes).map_err(|_| {
                ExportError::InvalidDimensions(options.width, options.height)
            })?
        ];
        let mut y = 0u32;
        while y < options.height {
            let y_end = (y + strip_rows).min(options.height);
            let strip = renderer.render_u8(y, y_end)?;
            let dst_off = y as usize * options.width as usize * 3;
            full[dst_off..dst_off + strip.len()].copy_from_slice(&strip);
            y = y_end;
            if !on_progress(y, options.height) {
                return Err(ExportError::Cancelled);
            }
        }
        if ext == "png" {
            let mut enc = image::codecs::png::PngEncoder::new(temporary.as_file_mut());
            enc.set_icc_profile(icc_bytes)
                .map_err(image::ImageError::Unsupported)?;
            enc.write_image(
                &full,
                options.width,
                options.height,
                ExtendedColorType::Rgb8,
            )?;
        } else {
            let mut enc =
                image::codecs::jpeg::JpegEncoder::new_with_quality(temporary.as_file_mut(), 95);
            enc.set_icc_profile(icc_bytes)
                .map_err(image::ImageError::Unsupported)?;
            enc.write_image(
                &full,
                options.width,
                options.height,
                ExtendedColorType::Rgb8,
            )?;
        }
    }
    if !on_progress(options.height, options.height) {
        return Err(ExportError::Cancelled);
    }
    protect_sources(out_path, images.iter().map(|im| Path::new(&im.path)))?;
    let bytes_written = temporary.as_file().metadata()?.len();
    publish(temporary, out_path, options.overwrite)?;
    Ok(ExportSummary {
        output_path: options.output_path.clone(),
        width: options.width,
        height: options.height,
        bigtiff: false,
        bytes_written,
        bit_depth: options.bit_depth,
        container_bit_depth,
        color_profile: target_profile.id().to_string(),
        color_profile_name: target_profile.display_name(),
        icc_embedded: true,
    })
}

struct StripRenderer<'r, 'a> {
    input: &'r RenderInput<'a>,
    plan: &'r RenderPlan,
    bit_depth: u8,
    cancel: Option<&'r AtomicBool>,
}

impl StripRenderer<'_, '_> {
    fn linear(&self, y0: u32, y1: u32) -> Result<(Vec<[f32; 3]>, Vec<bool>), ExportError> {
        self.plan
            .render_rows(self.input, y0, y1, self.cancel)
            .ok_or(ExportError::Cancelled)
    }

    fn render_u8(&self, y0: u32, y1: u32) -> Result<Vec<u8>, ExportError> {
        let (px, valid) = self.linear(y0, y1)?;
        let trc = self.input.trc;
        let mut out = vec![0u8; px.len() * 3];
        out.par_chunks_mut(3)
            .zip(px.par_iter().zip(valid.par_iter()))
            .for_each(|(dst, (p, ok))| {
                if *ok {
                    dst.copy_from_slice(&trc.encode_linear_to_u8(*p));
                }
            });
        Ok(out)
    }

    fn render_u16(&self, y0: u32, y1: u32) -> Result<Vec<u16>, ExportError> {
        let (px, valid) = self.linear(y0, y1)?;
        let trc = self.input.trc;
        let ten = self.bit_depth == 10;
        let mut out = vec![0u16; px.len() * 3];
        out.par_chunks_mut(3)
            .zip(px.par_iter().zip(valid.par_iter()))
            .for_each(|(dst, (p, ok))| {
                if *ok {
                    dst.copy_from_slice(&if ten {
                        trc.encode_linear_to_10bit_u16(*p)
                    } else {
                        trc.encode_linear_to_u16(*p)
                    });
                }
            });
        Ok(out)
    }
}

/// Strip height: a multiple of the blend alignment, as tall as the memory budget allows so
/// the padding overhead of multi-band blending stays small.
/// Cheap size checks that need no sources: output buffer limit for PNG/JPEG and the minimum
/// band memory of a strip (smallest strip with the smallest padding a coarse layer allows).
/// Callers run this before decoding or caching full-resolution sources.
pub fn preflight_export(options: &ExportOptions, image_count: usize) -> Result<(), ExportError> {
    if options.width == 0 || options.height == 0 {
        return Err(ExportError::InvalidDimensions(
            options.width,
            options.height,
        ));
    }
    let ext = Path::new(&options.output_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let bytes_per_pixel = if options.bit_depth > 8 { 6u64 } else { 3u64 };
    let raw = u64::from(options.width) * u64::from(options.height) * bytes_per_pixel;
    if ext != "tif" && ext != "tiff" && raw > FULL_BUFFER_LIMIT {
        return Err(ExportError::ResourceLimit(
            "PNG/JPEG need a full output buffer; choose TIFF for exports above 512 MiB".into(),
        ));
    }
    let min_rows = 8 + if options.multiband { 2 * 64 } else { 0 };
    let min_band =
        u64::from(options.width) * min_rows * crate::render::band_bytes_per_px(image_count);
    if min_band > band_hard_limit() {
        return Err(ExportError::ResourceLimit(format!(
            "Export width {} needs more than the band memory budget; reduce width or disable multi-band",
            options.width
        )));
    }
    Ok(())
}

/// Memory budget for one render band (and for the global coarse layer): at most 1.5 GiB and
/// at most 40 % of both the available memory and the process address-space limit (Linux),
/// so a constrained machine or container refuses an export instead of aborting.
pub fn band_budget() -> u64 {
    memory_budgets().0
}

/// Hard ceiling for one band: never more than 80 % of available memory / address space.
pub fn band_hard_limit() -> u64 {
    memory_budgets().1
}

/// `(soft, hard)` band budgets from MemAvailable and the address-space rlimit (Linux).
fn memory_budgets() -> (u64, u64) {
    let mut budget: u64 = 1536 * 1024 * 1024;
    let mut hard: u64 = 3 * 1536 * 1024 * 1024;
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(kb) = info
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
        {
            budget = budget.min(kb * 1024 * 2 / 5);
            hard = hard.min(kb * 1024 * 4 / 5);
        }
    }
    if let Ok(limits) = std::fs::read_to_string("/proc/self/limits") {
        if let Some(bytes) = limits
            .lines()
            .find(|l| l.starts_with("Max address space"))
            .and_then(|l| l.split_whitespace().nth(3))
            .and_then(|v| v.parse::<u64>().ok())
        {
            budget = budget.min(bytes * 2 / 5);
            hard = hard.min(bytes * 4 / 5);
        }
    }
    (budget.max(64 * 1024 * 1024), hard.max(64 * 1024 * 1024))
}
/// Largest full output buffer for formats that cannot be written in strips.
const FULL_BUFFER_LIMIT: u64 = 512 * 1024 * 1024;

fn choose_strip_rows(
    options: &ExportOptions,
    plan: &RenderPlan,
    image_count: usize,
) -> Result<u32, ExportError> {
    let budget = band_budget();
    let align = plan.align.max(8);
    let pad = if options.multiband { plan.pad } else { 0 };
    // label + accumulators + output per band pixel, plus per-image ROI buffers and the
    // transient pyramids of the image currently being blended (coverage ~ 1/2 of width).
    let per_px = crate::render::band_bytes_per_px(image_count);
    let width = u64::from(options.width);
    let rows_for = |rows: u32| width * u64::from(rows + 2 * pad) * per_px;
    let mut rows = (options.rows_per_strip.max(align) / align * align).max(align);
    let target = (2048 / align).max(1) * align;
    while rows + align <= target.max(rows) && rows_for(rows + align) <= budget {
        rows += align;
    }
    while rows > align && rows_for(rows) > budget {
        rows -= align;
    }
    if rows_for(rows) > band_hard_limit() {
        return Err(ExportError::ResourceLimit(format!(
            "Export width {} needs more than the band memory budget; reduce width or disable multi-band",
            options.width
        )));
    }
    Ok(rows.min(options.height.max(1)))
}

fn write_tiff_strips<W: std::io::Write + std::io::Seek, K: tiff::encoder::TiffKind, F>(
    enc: &mut TiffEncoder<W, K>,
    options: &ExportOptions,
    strip_rows: u32,
    renderer: &StripRenderer<'_, '_>,
    icc_bytes: &[u8],
    on_progress: &mut F,
) -> Result<(), ExportError>
where
    F: FnMut(u32, u32) -> bool,
{
    if options.bit_depth > 8 {
        let mut img_enc = enc.new_image::<RGB16>(options.width, options.height)?;
        img_enc
            .encoder()
            .write_tag(tiff::tags::Tag::IccProfile, icc_bytes)?;
        img_enc.rows_per_strip(strip_rows)?;
        let mut y = 0u32;
        while y < options.height {
            let y_end = (y + strip_rows).min(options.height);
            let strip = renderer.render_u16(y, y_end)?;
            img_enc.write_strip(&strip)?;
            y = y_end;
            if !on_progress(y, options.height) {
                return Err(ExportError::Cancelled);
            }
        }
        img_enc.finish()?;
    } else {
        let mut img_enc = enc.new_image::<RGB8>(options.width, options.height)?;
        img_enc
            .encoder()
            .write_tag(tiff::tags::Tag::IccProfile, icc_bytes)?;
        img_enc.rows_per_strip(strip_rows)?;
        let mut y = 0u32;
        while y < options.height {
            let y_end = (y + strip_rows).min(options.height);
            let strip = renderer.render_u8(y, y_end)?;
            img_enc.write_strip(&strip)?;
            y = y_end;
            if !on_progress(y, options.height) {
                return Err(ExportError::Cancelled);
            }
        }
        img_enc.finish()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::LensProjection;
    use crate::raster::Rgb16Image;
    use image::{ImageDecoder, ImageReader, Rgb, RgbImage};

    #[test]
    fn impossible_exports_are_rejected_before_any_large_allocation() {
        let img = RgbImage::from_pixel(160, 120, Rgb([90, 90, 90]));
        let src = |id: usize, yaw: f64| SourceImage {
            id,
            path: String::new(),
            width: 160,
            height: 120,
            yaw_deg: yaw,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 70.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let dir = tempfile::tempdir().unwrap();
        let mut images = [src(0, -20.0), src(1, 20.0)];
        for (k, im) in images.iter_mut().enumerate() {
            let path = dir.path().join(format!("src{k}.png"));
            img.save(&path).unwrap();
            im.path = path.to_string_lossy().into();
        }
        let mut opts = ExportOptions {
            output_path: dir.path().join("huge.png").to_string_lossy().into(),
            width: 50_000,
            height: 50_000,
            projection: PanoramaProjection::Cylindrical,
            pano_hfov_deg: 120.0,
            view_yaw_deg: 0.0,
            view_pitch_deg: 0.0,
            view_v_shift: 0.0,
            crop: [0.0, 0.0, 1.0, 1.0],
            multiband: true,
            force_bigtiff: false,
            rows_per_strip: 128,
            canvas_aspect: Some(1.0),
            overwrite: false,
            bit_depth: 8,
            color_profile: "srgb".into(),
            custom_icc_name: None,
            custom_icc_base64: None,
            photometric: None,
            reject_transients: false,
            interpolation: Interpolation::Spline36,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        };
        let started = std::time::Instant::now();
        let r = export_panorama(&opts, &images, &[img.clone(), img.clone()], |_, _| true);
        assert!(matches!(r, Err(ExportError::ResourceLimit(_))), "{r:?}");
        // TIFF of an extreme width: refused by the band budget, not by an allocation abort.
        opts.output_path = dir.path().join("huge.tif").to_string_lossy().into();
        opts.width = 200_000;
        opts.height = 200_000;
        let r = export_panorama(&opts, &images, &[img.clone(), img], |_, _| true);
        assert!(matches!(r, Err(ExportError::ResourceLimit(_))), "{r:?}");
        assert!(started.elapsed().as_secs() < 5);
        assert!(!dir.path().join("huge.png").exists() && !dir.path().join("huge.tif").exists());
    }

    #[test]
    fn exports_striped_bigtiff_multiband() {
        let mut left = RgbImage::new(160, 120);
        let mut right = RgbImage::new(160, 120);
        for y in 0..120 {
            for x in 0..160 {
                left.put_pixel(x, y, Rgb([40, (x % 200) as u8, 180]));
                right.put_pixel(x, y, Rgb([180, (y % 200) as u8, 60]));
            }
        }
        let images = vec![
            SourceImage {
                id: 0,
                path: "left.png".into(),
                width: 160,
                height: 120,
                yaw_deg: -18.0,
                pitch_deg: 0.0,
                roll_deg: 0.0,
                hfov_deg: 65.0,
                lens_projection: LensProjection::Rectilinear,
                distortion: [0.0; 3],
                distortion_poly5: None,
                distortion_scale: 1.0,
            },
            SourceImage {
                id: 1,
                path: "right.png".into(),
                width: 160,
                height: 120,
                yaw_deg: 18.0,
                pitch_deg: 0.0,
                roll_deg: 0.0,
                hfov_deg: 65.0,
                lens_projection: LensProjection::Rectilinear,
                distortion: [0.0; 3],
                distortion_poly5: None,
                distortion_scale: 1.0,
            },
        ];
        let tmp = std::env::temp_dir().join("slashstitch_test_bigtiff.tif");
        let _ = std::fs::remove_file(&tmp);
        let opts = ExportOptions {
            output_path: tmp.to_string_lossy().to_string(),
            width: 320,
            height: 160,
            projection: PanoramaProjection::Equirectangular,
            pano_hfov_deg: 120.0,
            view_yaw_deg: 0.0,
            view_pitch_deg: 0.0,
            view_v_shift: 0.0,
            crop: [0.0, 0.0, 1.0, 1.0],
            multiband: true,
            force_bigtiff: true,
            rows_per_strip: 32,
            canvas_aspect: None,
            overwrite: false,
            bit_depth: 8,
            color_profile: "srgb".into(),
            custom_icc_name: None,
            custom_icc_base64: None,
            photometric: None,
            reject_transients: false,
            interpolation: Interpolation::Spline36,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        };
        let summary = export_panorama(&opts, &images, &[left, right], |_, _| true).unwrap();
        assert!(summary.bigtiff);
        assert!(summary.bytes_written > 320 * 160 * 3);
        let dims = image::image_dimensions(&tmp).unwrap();
        assert_eq!(dims, (320, 160));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn exports_cropped_shifted_view() {
        let mut img = RgbImage::new(160, 120);
        for y in 0..120 {
            for x in 0..160 {
                img.put_pixel(x, y, Rgb([200, 120, 60]));
            }
        }
        let images = vec![SourceImage {
            id: 0,
            path: "only.png".into(),
            width: 160,
            height: 120,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 75.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        }];
        let tmp = std::env::temp_dir().join("slashstitch_test_cropped.png");
        let _ = std::fs::remove_file(&tmp);
        let opts = ExportOptions {
            output_path: tmp.to_string_lossy().to_string(),
            width: 120,
            height: 80,
            projection: PanoramaProjection::Equirectangular,
            pano_hfov_deg: 120.0,
            view_yaw_deg: 0.0,
            view_pitch_deg: 0.0,
            view_v_shift: 0.0,
            crop: [0.42, 0.35, 0.58, 0.65],
            multiband: false,
            force_bigtiff: false,
            rows_per_strip: 32,
            canvas_aspect: None,
            overwrite: false,
            bit_depth: 8,
            color_profile: "srgb".into(),
            custom_icc_name: None,
            custom_icc_base64: None,
            photometric: None,
            reject_transients: false,
            interpolation: Interpolation::Spline36,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        };
        let summary = export_panorama(&opts, &images, &[img], |_, _| true).unwrap();
        assert_eq!((summary.width, summary.height), (120, 80));
        let rendered = image::open(&tmp).unwrap().to_rgb8();
        let center_px = rendered.get_pixel(60, 40).0;
        assert!(
            center_px[0] > 180 && center_px[1] > 100,
            "center={center_px:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn exports_10bit_gradient_preserving_more_than_256_levels_and_adobe_icc() {
        let mut grad = Rgb16Image::new(1024, 64);
        for y in 0..64 {
            for x in 0..1024 {
                let v = ((x * 65535) / 1023) as u16;
                grad.put_pixel(x, y, Rgb([v, v / 2, 65535 - v]));
            }
        }
        let images = vec![SourceImage {
            id: 0,
            path: "grad.tif".into(),
            width: 1024,
            height: 64,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 90.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        }];
        let tif_path = std::env::temp_dir().join("slashstitch_test_10bit_adobe.tif");
        let _ = std::fs::remove_file(&tif_path);
        let mut opts = ExportOptions {
            output_path: tif_path.to_string_lossy().to_string(),
            width: 1024,
            height: 64,
            projection: PanoramaProjection::Rectilinear,
            pano_hfov_deg: 60.0,
            view_yaw_deg: 0.0,
            view_pitch_deg: 0.0,
            view_v_shift: 0.0,
            crop: [0.0, 0.0, 1.0, 1.0],
            multiband: false,
            force_bigtiff: false,
            rows_per_strip: 64,
            canvas_aspect: Some(16.0),
            overwrite: true,
            bit_depth: 10,
            color_profile: "adobe_rgb".into(),
            custom_icc_name: None,
            custom_icc_base64: None,
            photometric: None,
            reject_transients: false,
            interpolation: Interpolation::Spline36,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        };
        let summary =
            export_panorama(&opts, &images, std::slice::from_ref(&grad), |_, _| true).unwrap();
        assert_eq!(summary.bit_depth, 10);
        assert_eq!(summary.container_bit_depth, 16);
        assert!(summary.icc_embedded);
        let mut dec = ImageReader::open(&tif_path)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        let icc = dec.icc_profile().unwrap().expect("TIFF ICC embedded");
        let parsed = moxcms::ColorProfile::new_from_slice(&icc).unwrap();
        assert_eq!(
            crate::color::detect_preset_match(&parsed, &icc),
            Some("adobe_rgb")
        );
        let out_16 = image::DynamicImage::from_decoder(dec).unwrap().into_rgb16();
        let unique_r: std::collections::HashSet<u16> =
            (0..1024).map(|x| out_16.get_pixel(x, 32).0[0]).collect();
        assert!(
            unique_r.len() > 350,
            "Expected 10-bit gradient (>256 levels), got {} unique values",
            unique_r.len()
        );
        let _ = std::fs::remove_file(&tif_path);

        // Full 16-bit export preserves >1024 tonal levels across the gradient.
        let mut grad16 = Rgb16Image::new(2048, 128);
        for y in 0..128 {
            for x in 0..2048 {
                let v = ((x * 65535) / 2047) as u16;
                grad16.put_pixel(x, y, Rgb([v, v / 2, 65535 - v]));
            }
        }
        let images16 = vec![SourceImage {
            width: 2048,
            height: 128,
            ..images[0].clone()
        }];
        let tif16_path = std::env::temp_dir().join("slashstitch_test_16bit_prophoto.tif");
        let _ = std::fs::remove_file(&tif16_path);
        opts.output_path = tif16_path.to_string_lossy().to_string();
        opts.width = 2048;
        opts.height = 128;
        opts.rows_per_strip = 128;
        opts.bit_depth = 16;
        opts.color_profile = "prophoto_rgb".into();
        let sum16 =
            export_panorama(&opts, &images16, std::slice::from_ref(&grad16), |_, _| true).unwrap();
        assert_eq!(sum16.bit_depth, 16);
        assert_eq!(sum16.container_bit_depth, 16);
        assert_eq!(sum16.color_profile, "prophoto_rgb");
        let dec16 = ImageReader::open(&tif16_path)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        let out_full16 = image::DynamicImage::from_decoder(dec16)
            .unwrap()
            .into_rgb16();
        let unique16: std::collections::HashSet<u16> = (0..2048)
            .map(|x| out_full16.get_pixel(x, 64).0[0])
            .collect();
        assert!(
            unique16.len() > 1200,
            "Expected full 16-bit gradient (>1024 levels), got {} unique values",
            unique16.len()
        );
        let _ = std::fs::remove_file(&tif16_path);

        // JPEG must reject both 10-bit and 16-bit export cleanly.
        let jpg_path = std::env::temp_dir().join("slashstitch_test_highbit_reject.jpg");
        opts.output_path = jpg_path.to_string_lossy().to_string();
        opts.bit_depth = 10;
        assert!(export_panorama(&opts, &images, &[grad], |_, _| true).is_err());
        opts.bit_depth = 16;
        assert!(export_panorama(&opts, &images16, &[grad16], |_, _| true).is_err());
    }
}
