use crate::color::{detect_preset_match, extract_icc_description};
use crate::lensfun::LensDatabase;
use crate::project::LensProjection;
use exif::{In, Reader, Tag, Value};
use moxcms::{ColorProfile, DataColorSpace};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceImageMeta {
    pub path: String,
    pub file_name: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    #[serde(default)]
    pub bit_depth: Option<u8>,
    #[serde(default)]
    pub icc_profile_name: Option<String>,
    #[serde(default)]
    pub icc_preset_match: Option<String>,
    #[serde(default)]
    pub icc_status: String,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens_model: Option<String>,
    pub focal_length_mm: Option<f64>,
    pub focal_length_35mm: Option<f64>,
    pub aperture: Option<f64>,
    pub iso: Option<u32>,
    pub matched_crop_factor: Option<f64>,
    pub matched_mount: Option<String>,
    pub matched_lens: Option<String>,
    pub matched_lens_type: Option<String>,
    pub estimated_lens_projection: LensProjection,
    pub estimated_hfov_deg: Option<f64>,
    pub warnings: Vec<String>,
    /// Exposure time in seconds (EXIF ExposureTime).
    #[serde(default)]
    pub exposure_time_s: Option<f64>,
    /// Exposure compensation in EV (EXIF ExposureBiasValue).
    #[serde(default)]
    pub exposure_bias_ev: Option<f64>,
    /// EXIF ExposureMode is "Auto bracket".
    #[serde(default)]
    pub auto_bracket: bool,
    /// EXIF DateTimeOriginal with sub-seconds, "YYYY:MM:DD HH:MM:SS[.fff]".
    #[serde(default)]
    pub capture_time: Option<String>,
    /// Membership in a detected exposure bracket (HDR stack); derived for the whole project.
    #[serde(default)]
    pub bracket: Option<crate::hdr::BracketInfo>,
}

/// Exposure-related EXIF fields shared by the raster and RAW inspectors.
#[derive(Default)]
struct ExposureExif {
    exposure_time_s: Option<f64>,
    exposure_bias_ev: Option<f64>,
    auto_bracket: bool,
    capture_time: Option<String>,
}

fn read_exposure(exif: &exif::Exif) -> ExposureExif {
    let bias = exif
        .get_field(Tag::ExposureBiasValue, In::PRIMARY)
        .and_then(|f| match &f.value {
            Value::SRational(v) => v
                .first()
                .filter(|r| r.denom != 0)
                .map(|r| f64::from(r.num) / f64::from(r.denom)),
            Value::Rational(v) => v
                .first()
                .filter(|r| r.denom != 0)
                .map(|r| f64::from(r.num) / f64::from(r.denom)),
            _ => None,
        });
    let capture_time = read_ascii(exif, Tag::DateTimeOriginal).map(|t| {
        match read_ascii(exif, Tag::SubSecTimeOriginal)
            .filter(|s| s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty())
        {
            Some(sub) => format!("{t}.{sub}"),
            None => t,
        }
    });
    ExposureExif {
        exposure_time_s: read_rational(exif, Tag::ExposureTime).filter(|t| *t > 0.0),
        exposure_bias_ev: bias,
        auto_bracket: read_u32(exif, Tag::ExposureMode) == Some(2),
        capture_time,
    }
}

pub fn inspect_images(paths: &[String], db: &LensDatabase) -> Vec<SourceImageMeta> {
    paths
        .par_iter()
        .map(|p| inspect_image(Path::new(p), db))
        .collect()
}

pub fn inspect_image(path: &Path, db: &LensDatabase) -> SourceImageMeta {
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());

    let mut warnings = Vec::new();
    if crate::rawdev::is_raw_path(path) {
        return inspect_raw(path, file_name, db);
    }
    let (mut width, mut height) = match image::image_dimensions(path) {
        Ok((w, h)) => (Some(w), Some(h)),
        Err(err) => {
            warnings.push(format!("Could not read dimensions directly: {err}"));
            (None, None)
        }
    };

    let mut bit_depth = None;
    let mut icc_profile_name = None;
    let mut icc_preset_match = None;
    let mut icc_status = "missing".to_string();

    if let Ok(reader) = image::ImageReader::open(path) {
        if let Ok(reader) = reader.with_guessed_format() {
            if let Ok(mut decoder) = reader.into_decoder() {
                use image::ImageDecoder;
                let color = decoder.color_type();
                let ch = u16::from(color.channel_count()).max(1);
                bit_depth = Some((color.bits_per_pixel() / ch) as u8);
                match decoder.icc_profile() {
                    Ok(Some(bytes)) => match ColorProfile::new_from_slice(&bytes) {
                        Ok(profile) if profile.color_space == DataColorSpace::Rgb => {
                            icc_status = "embedded".to_string();
                            icc_preset_match = Some(
                                detect_preset_match(&profile, &bytes)
                                    .unwrap_or("custom")
                                    .to_string(),
                            );
                            icc_profile_name = extract_icc_description(&profile)
                                .or_else(|| Some("Embedded RGB ICC".to_string()));
                        }
                        Ok(_) => {
                            icc_status = "unsupported_colorspace".to_string();
                            warnings.push(
                                "Embedded non-RGB ICC profile is unsupported; convert this source to RGB first".into(),
                            );
                        }
                        Err(err) => {
                            icc_status = "invalid".to_string();
                            warnings.push(format!("Invalid embedded ICC profile: {err}"));
                        }
                    },
                    Ok(None) => {
                        icc_status = "missing".to_string();
                    }
                    Err(err) => {
                        icc_status = "invalid".to_string();
                        warnings.push(format!("Could not read embedded ICC profile: {err}"));
                    }
                }
            }
        }
    }
    let mut camera_make = None;
    let mut camera_model = None;
    let mut lens_model = None;
    let mut focal_length_mm = None;
    let mut focal_length_35mm = None;
    let mut aperture = None;
    let mut iso = None;
    let mut orientation = 1;
    let mut exposure = ExposureExif::default();

    if let Ok(file) = File::open(path) {
        let mut reader = BufReader::new(file);
        if let Ok(exif) = Reader::new().read_from_container(&mut reader) {
            camera_make = read_ascii(&exif, Tag::Make);
            camera_model = read_ascii(&exif, Tag::Model);
            lens_model = read_ascii(&exif, Tag::LensModel);
            focal_length_mm = read_rational(&exif, Tag::FocalLength);
            focal_length_35mm = read_u32(&exif, Tag::FocalLengthIn35mmFilm).map(|v| v as f64);
            aperture = read_rational(&exif, Tag::FNumber);
            orientation = read_u32(&exif, Tag::Orientation).unwrap_or(1);
            iso = read_u32(&exif, Tag::PhotographicSensitivity);
            exposure = read_exposure(&exif);
        }
    } else {
        warnings.push("Could not open file for EXIF reading".to_string());
    }

    if (5..=8).contains(&orientation) {
        std::mem::swap(&mut width, &mut height);
    }
    let matched_cam = db.find_camera(camera_make.as_deref(), camera_model.as_deref());
    let matched_crop_factor = matched_cam.map(|c| c.crop_factor);
    let matched_mount = matched_cam.map(|c| c.mount.clone());
    let matched_lens_profile = db.find_lens(lens_model.as_deref(), matched_mount.as_deref());

    let matched_lens_type = matched_lens_profile.map(|l| l.lens_type.clone());
    let estimated_lens_projection =
        infer_lens_projection(matched_lens_type.as_deref(), lens_model.as_deref());

    let estimated_hfov_deg = estimate_hfov_deg(
        focal_length_mm,
        focal_length_35mm,
        matched_crop_factor.or_else(|| matched_lens_profile.map(|l| l.crop_factor)),
        width,
        height,
        estimated_lens_projection,
    );

    SourceImageMeta {
        path: path.to_string_lossy().to_string(),
        file_name,
        width,
        height,
        bit_depth,
        icc_profile_name,
        icc_preset_match,
        icc_status,
        camera_make,
        camera_model,
        lens_model,
        focal_length_mm,
        focal_length_35mm,
        aperture,
        iso,
        matched_crop_factor,
        matched_mount,
        matched_lens: matched_lens_profile.map(|l| format!("{} {}", l.maker, l.model)),
        matched_lens_type,
        estimated_lens_projection,
        estimated_hfov_deg,
        warnings,
        exposure_time_s: exposure.exposure_time_s,
        exposure_bias_ev: exposure.exposure_bias_ev,
        auto_bracket: exposure.auto_bracket,
        capture_time: exposure.capture_time,
        bracket: None,
    }
}

/// Metadata of a camera RAW/DNG source: developed size and bit depth from the RAW decoder,
/// camera/lens EXIF (TIFF-based RAWs) and what the Lightroom settings will change.
fn inspect_raw(path: &Path, file_name: String, db: &LensDatabase) -> SourceImageMeta {
    let mut warnings = Vec::new();
    let info = match crate::rawdev::raw_info(path) {
        Ok(info) => Some(info),
        Err(err) => {
            warnings.push(err);
            None
        }
    };
    let (mut camera_make, mut camera_model, mut lens_model) = (None, None, None);
    let (mut focal_length_mm, mut focal_length_35mm, mut aperture, mut iso) =
        (None, None, None, None);
    let mut exposure = ExposureExif::default();
    if let Ok(file) = File::open(path) {
        if let Ok(exif) = Reader::new().read_from_container(&mut BufReader::new(file)) {
            camera_make = read_ascii(&exif, Tag::Make);
            camera_model = read_ascii(&exif, Tag::Model);
            lens_model = read_ascii(&exif, Tag::LensModel);
            focal_length_mm = read_rational(&exif, Tag::FocalLength);
            focal_length_35mm = read_u32(&exif, Tag::FocalLengthIn35mmFilm).map(f64::from);
            aperture = read_rational(&exif, Tag::FNumber);
            iso = read_u32(&exif, Tag::PhotographicSensitivity);
            exposure = read_exposure(&exif);
        }
    }
    if let Some(info) = &info {
        camera_make = camera_make.or_else(|| Some(info.make.clone()).filter(|m| !m.is_empty()));
        camera_model = camera_model.or_else(|| Some(info.model.clone()).filter(|m| !m.is_empty()));
        if info.enhanced {
            warnings.push(
                "This DNG contains an enhanced (e.g. Lightroom Denoise) image; it is used when it can be decoded".into(),
            );
        }
        match &info.xmp {
            Some(x) => {
                let mut applied = Vec::new();
                if let Some((t, tint)) = x.temperature_tint {
                    applied.push(format!("white balance {t:.0} K / tint {tint:+.0}"));
                }
                if x.exposure_ev.abs() > 1e-6 {
                    applied.push(format!("exposure {:+.2} EV", x.exposure_ev));
                }
                if x.orientation.is_some_and(|o| o != 1) {
                    applied.push("orientation".into());
                }
                warnings.push(format!(
                    "Lightroom settings from {}: {}",
                    x.origin,
                    if applied.is_empty() {
                        "nothing to apply (as shot)".into()
                    } else {
                        format!("applied {}", applied.join(", "))
                    }
                ));
                if !x.ignored.is_empty() {
                    warnings.push(format!(
                        "Lightroom settings not reproduced: {}",
                        x.ignored.join(", ")
                    ));
                }
            }
            None => warnings.push(match info.as_shot {
                Some((t, tint)) => format!(
                    "Developed with the camera's as-shot white balance (about {t:.0} K / tint {tint:+.0}); no Lightroom XMP found"
                ),
                None => "Developed with the camera's as-shot white balance (no Lightroom XMP found)".into(),
            }),
        }
    }
    let (width, height) = info
        .as_ref()
        .map_or((None, None), |i| (Some(i.width), Some(i.height)));
    let matched_cam = db.find_camera(camera_make.as_deref(), camera_model.as_deref());
    let matched_crop_factor = matched_cam.map(|c| c.crop_factor);
    let matched_mount = matched_cam.map(|c| c.mount.clone());
    let matched_lens_profile = db.find_lens(lens_model.as_deref(), matched_mount.as_deref());
    let matched_lens_type = matched_lens_profile.map(|l| l.lens_type.clone());
    let estimated_lens_projection =
        infer_lens_projection(matched_lens_type.as_deref(), lens_model.as_deref());
    let estimated_hfov_deg = estimate_hfov_deg(
        focal_length_mm,
        focal_length_35mm,
        matched_crop_factor.or_else(|| matched_lens_profile.map(|l| l.crop_factor)),
        width,
        height,
        estimated_lens_projection,
    );
    SourceImageMeta {
        path: path.to_string_lossy().to_string(),
        file_name,
        width,
        height,
        bit_depth: info.as_ref().map(|i| i.bits),
        icc_profile_name: Some("Camera RAW (linear Rec.2020)".into()),
        icc_preset_match: None,
        icc_status: "raw".into(),
        camera_make,
        camera_model,
        lens_model,
        focal_length_mm,
        focal_length_35mm,
        aperture,
        iso,
        matched_crop_factor,
        matched_mount,
        matched_lens: matched_lens_profile.map(|l| format!("{} {}", l.maker, l.model)),
        matched_lens_type,
        estimated_lens_projection,
        estimated_hfov_deg,
        warnings,
        exposure_time_s: exposure.exposure_time_s,
        exposure_bias_ev: exposure.exposure_bias_ev,
        auto_bracket: exposure.auto_bracket,
        capture_time: exposure.capture_time,
        bracket: None,
    }
}

fn infer_lens_projection(lensfun_type: Option<&str>, exif_lens: Option<&str>) -> LensProjection {
    if let Some(lt) = lensfun_type {
        let lower = lt.to_ascii_lowercase();
        if lower == "equirectangular" {
            return LensProjection::Equirectangular;
        }
        if lower == "equisolid" {
            return LensProjection::FisheyeEquisolid;
        }
        if lower.contains("fisheye") {
            return LensProjection::FisheyeFullframe;
        }
    }
    if let Some(ex) = exif_lens {
        let lower = ex.to_ascii_lowercase();
        if lower.contains("diagonal fisheye") {
            return LensProjection::FisheyeEquisolid;
        }
        if lower.contains("circular fisheye") {
            return LensProjection::FisheyeCircular;
        }
        if lower.contains("fisheye") {
            return LensProjection::FisheyeFullframe;
        }
    }
    LensProjection::Rectilinear
}

fn estimate_hfov_deg(
    focal_mm: Option<f64>,
    focal_35mm: Option<f64>,
    crop_factor: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    lens_proj: LensProjection,
) -> Option<f64> {
    let eff_focal = if let Some(f35) = focal_35mm.filter(|v| *v > 0.0) {
        f35
    } else {
        let f = focal_mm.filter(|v| *v > 0.0)?;
        f * crop_factor.unwrap_or(1.0)
    };

    let (w, h) = match (width, height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w as f64, h as f64),
        _ => (3.0, 2.0),
    };
    let sensor_w = if w >= h { 36.0 } else { 24.0 };
    let fov_rad = match lens_proj {
        LensProjection::Rectilinear => 2.0 * (sensor_w / (2.0 * eff_focal)).atan(),
        LensProjection::FisheyeFullframe | LensProjection::FisheyeCircular => sensor_w / eff_focal,
        LensProjection::FisheyeEquisolid => {
            4.0 * (sensor_w / (4.0 * eff_focal)).clamp(-0.9999, 0.9999).asin()
        }
        LensProjection::Equirectangular => sensor_w / eff_focal,
    };
    Some(fov_rad.to_degrees())
}

fn read_ascii(exif: &exif::Exif, tag: Tag) -> Option<String> {
    let field = exif.get_field(tag, In::PRIMARY)?;
    match &field.value {
        Value::Ascii(items) => items
            .first()
            .map(|bytes| {
                String::from_utf8_lossy(bytes)
                    .trim_matches('\0')
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty()),
        _ => Some(field.display_value().to_string().trim().to_string()).filter(|s| !s.is_empty()),
    }
}

fn read_rational(exif: &exif::Exif, tag: Tag) -> Option<f64> {
    let field = exif.get_field(tag, In::PRIMARY)?;
    match &field.value {
        Value::Rational(items) => items
            .first()
            .filter(|r| r.denom != 0)
            .map(|r| r.num as f64 / r.denom as f64),
        _ => None,
    }
}

fn read_u32(exif: &exif::Exif, tag: Tag) -> Option<u32> {
    let field = exif.get_field(tag, In::PRIMARY)?;
    field.value.get_uint(0)
}
