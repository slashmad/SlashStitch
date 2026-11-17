//! Camera RAW / DNG development into the 16-bit working format.
//!
//! `rawler` decodes the sensor data and its metadata (black/white levels, CFA layout, camera
//! colour matrix, as-shot white balance, crops, orientation). Development happens here, in
//! floating point:
//!
//! 1. black/white scaling to 0..1;
//! 2. white balance on the mosaic, then a clip at the common saturation point (all channels
//!    saturate together, so blown highlights stay neutral instead of turning magenta);
//! 3. demosaicing (RCD for Bayer, see `demosaic.rs`; rawler's bilinear for X-Trans) and the
//!    active/default crop;
//! 4. camera RGB to linear Rec.2020 (rows normalized so white-balanced neutrals stay neutral;
//!    no gamut clipping beyond negative values);
//! 5. exposure (DNG baseline exposure plus Lightroom/ACR `Exposure2012` from XMP) with a smooth
//!    highlight shoulder, so positive exposure compresses instead of hard-clipping highlights.
//!
//! The result is a 16-bit linear Rec.2020 image with an embedded linear Rec.2020 ICC profile;
//! the regular colour pipeline converts it to the project profile like any other source.

use crate::raster::{RawDecodedSource, Rgb16Image, MAX_SOURCE_BYTES};
use moxcms::{ColorProfile, ProfileText, ToneReprCurve};
use nalgebra::Matrix3;
use rawler::decoders::{Orientation, RawDecodeParams};
use rawler::formats::tiff::reader::TiffReader;
use rawler::formats::tiff::{GenericTiffReader, IFD};
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::imgop::sensor::SensorType;
use rawler::imgop::xyz::Illuminant;
use rawler::imgop::{Dim2, Point, Rect};
use rawler::rawimage::{RawImage, RawImageData, RawPhotometricInterpretation};
use rawler::rawsource::RawSource;
use rawler::tags::{DngTag, TiffCommonTag};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// File extensions decoded as camera RAW (checked case-insensitively). RAW containers are
/// TIFF-like, so content sniffing would find only the embedded thumbnail.
pub const RAW_EXTENSIONS: [&str; 25] = [
    "3fr", "ari", "arw", "cr2", "cr3", "crw", "dcr", "dcs", "dng", "erf", "iiq", "kdc", "mef",
    "mos", "mrw", "nef", "nrw", "orf", "pef", "raf", "rw2", "sr2", "srf", "srw", "x3f",
];

/// Transient bytes per output pixel while developing (16-bit and f32 mosaic, RCD planes and
/// direction maps, f32 RGB, 16-bit output).
pub const RAW_DECODE_BYTES_PER_PX: u64 = 40;

pub fn is_raw_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| RAW_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Linear Rec.2020 → XYZ (D65).
const REC2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.636_958_048, 0.144_616_904, 0.168_880_975],
    [0.262_700_212, 0.677_998_072, 0.059_301_716],
    [0.0, 0.028_072_693, 1.060_985_058],
];

/// The ICC profile attached to developed RAW sources: Rec.2020 primaries, linear TRC.
pub fn raw_working_icc() -> &'static [u8] {
    static ICC: OnceLock<Vec<u8>> = OnceLock::new();
    ICC.get_or_init(|| {
        let mut p = ColorProfile::new_bt2020();
        let linear = ToneReprCurve::Parametric(vec![1.0]);
        p.red_trc = Some(linear.clone());
        p.green_trc = Some(linear.clone());
        p.blue_trc = Some(linear);
        p.description = Some(ProfileText::PlainString(
            "SlashStitch RAW linear Rec.2020".into(),
        ));
        let raw = p.encode().expect("linear Rec.2020 profile encodes");
        crate::color::normalize_icc(&raw).unwrap_or(raw)
    })
}

/// Whether `icc` is the RAW working profile (such sources are never adopted as the project's
/// export profile: a linear profile is unsuitable for 8-bit output).
pub fn is_raw_working_icc(icc: &[u8]) -> bool {
    icc == raw_working_icc()
}

// ---------------------------------------------------------------------------------------------
// Lightroom / Adobe Camera Raw settings (XMP)

const CRS_NS: &str = "http://ns.adobe.com/camera-raw-settings/1.0/";
const TIFF_NS: &str = "http://ns.adobe.com/tiff/1.0/";

/// The subset of Lightroom/ACR develop settings SlashStitch applies to RAW sources, plus the
/// non-default settings it found but cannot reproduce (reported to the user).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct XmpDevelop {
    /// Where the settings came from (sidecar path or "embedded XMP").
    pub origin: String,
    /// `Some((kelvin, tint))` for a white balance other than "As Shot".
    pub temperature_tint: Option<(f64, f64)>,
    pub exposure_ev: f64,
    /// EXIF-style orientation (1–8) from `tiff:Orientation` (includes rotations made in
    /// Lightroom).
    pub orientation: Option<u8>,
    pub ignored: Vec<String>,
}

/// Numeric Lightroom adjustments whose neutral value is 0 and which SlashStitch does not
/// reproduce (global tone, presence and colour tools).
const UNSUPPORTED_ZERO_DEFAULT: [(&str, &str); 22] = [
    ("Contrast2012", "Contrast"),
    ("Highlights2012", "Highlights"),
    ("Shadows2012", "Shadows"),
    ("Whites2012", "Whites"),
    ("Blacks2012", "Blacks"),
    ("Texture", "Texture"),
    ("Clarity2012", "Clarity"),
    ("Dehaze", "Dehaze"),
    ("Vibrance", "Vibrance"),
    ("Saturation", "Saturation"),
    ("ParametricShadows", "Tone curve"),
    ("ParametricDarks", "Tone curve"),
    ("ParametricLights", "Tone curve"),
    ("ParametricHighlights", "Tone curve"),
    ("SplitToningShadowSaturation", "Split toning"),
    ("SplitToningHighlightSaturation", "Split toning"),
    ("ColorGradeMidtoneSat", "Color grading"),
    ("ColorGradeGlobalSat", "Color grading"),
    ("VignetteAmount", "Lens vignetting (manual)"),
    ("PostCropVignetteAmount", "Post-crop vignette"),
    ("GrainAmount", "Grain"),
    ("PerspectiveUpright", "Upright"),
];

/// Parse Lightroom/ACR settings from an XMP packet. `None` if it holds no `crs:` settings.
pub fn parse_xmp(xml: &str, origin: &str) -> Option<XmpDevelop> {
    let doc = roxmltree::Document::parse(xml).ok()?;
    let mut values: Vec<(String, String)> = Vec::new();
    let mut local_corrections = false;
    let mut orientation = None;
    for node in doc.descendants() {
        for attr in node.attributes() {
            if attr.namespace() == Some(CRS_NS) {
                values.push((attr.name().to_string(), attr.value().to_string()));
            }
            if attr.namespace() == Some(TIFF_NS) && attr.name() == "Orientation" {
                orientation = attr.value().trim().parse::<u8>().ok();
            }
        }
        if node.is_element()
            && node.tag_name().namespace() == Some(TIFF_NS)
            && node.tag_name().name() == "Orientation"
        {
            orientation = node.text().and_then(|t| t.trim().parse::<u8>().ok());
        }
        if node.is_element() && node.tag_name().namespace() == Some(CRS_NS) {
            let name = node.tag_name().name();
            if name.ends_with("Corrections") && node.has_children() {
                local_corrections = true;
            }
            if let Some(text) = node.text().map(str::trim).filter(|t| !t.is_empty()) {
                values.push((name.to_string(), text.to_string()));
            }
        }
    }
    if values.is_empty() {
        return None;
    }
    let get = |key: &str| {
        values
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    let num = |key: &str| get(key).and_then(|v| v.trim_start_matches('+').parse::<f64>().ok());
    let mut out = XmpDevelop {
        origin: origin.to_string(),
        exposure_ev: num("Exposure2012")
            .or_else(|| num("Exposure"))
            .unwrap_or(0.0),
        orientation: orientation.filter(|o| (1..=8).contains(o)),
        ..XmpDevelop::default()
    };
    let wb = get("WhiteBalance").unwrap_or("As Shot");
    if wb != "As Shot" {
        if let (Some(t), tint) = (num("Temperature"), num("Tint").unwrap_or(0.0)) {
            if (1500.0..=50_000.0).contains(&t) {
                out.temperature_tint = Some((t, tint));
            }
        }
    }
    for (key, label) in UNSUPPORTED_ZERO_DEFAULT {
        if num(key).is_some_and(|v| v.abs() > 1e-9) && !out.ignored.iter().any(|l| l == label) {
            out.ignored.push(label.to_string());
        }
    }
    let hsl = values.iter().any(|(k, v)| {
        (k.starts_with("HueAdjustment")
            || k.starts_with("SaturationAdjustment")
            || k.starts_with("LuminanceAdjustment"))
            && v.trim_start_matches('+')
                .parse::<f64>()
                .is_ok_and(|x| x != 0.0)
    });
    if hsl {
        out.ignored.push("HSL".into());
    }
    if get("HasCrop") == Some("True") {
        out.ignored
            .push("Crop (sources keep their full frame for alignment)".into());
    }
    if get("ConvertToGrayscale") == Some("True") {
        out.ignored.push("Black & White".into());
    }
    if let Some(name) = get("ToneCurveName2012").filter(|n| *n != "Linear" && *n != "Custom") {
        out.ignored.push(format!("Tone curve \"{name}\""));
    }
    if let Some(profile) = get("CameraProfile").filter(|p| *p != "Adobe Standard") {
        out.ignored.push(format!(
            "Camera profile \"{profile}\" (the camera colour matrix is used)"
        ));
    }
    if local_corrections {
        out.ignored.push("Local adjustments / masks".into());
    }
    Some(out)
}

/// Candidate sidecar paths Lightroom/ACR write for `raw`: `name.xmp` and `name.EXT.xmp`.
fn sidecar_candidates(raw: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for ext in ["xmp", "XMP"] {
        out.push(raw.with_extension(ext));
        if let Some(name) = raw.file_name() {
            let mut n = name.to_os_string();
            n.push(".");
            n.push(ext);
            out.push(raw.with_file_name(n));
        }
    }
    out
}

/// The XMP sidecar that belongs to `raw`, if any.
pub fn find_sidecar(raw: &Path) -> Option<PathBuf> {
    sidecar_candidates(raw).into_iter().find(|p| p.is_file())
}

/// Lightroom/ACR settings for `raw`: a sidecar wins over XMP embedded in the file (DNG).
pub fn read_xmp(raw: &Path) -> Option<XmpDevelop> {
    if let Some(side) = find_sidecar(raw) {
        if let Ok(text) = std::fs::read_to_string(&side) {
            let origin = side.file_name().map_or_else(
                || side.display().to_string(),
                |n| n.to_string_lossy().into(),
            );
            if let Some(x) = parse_xmp(&text, &origin) {
                return Some(x);
            }
        }
    }
    let packet = tiff_tag_bytes(raw, 700)?;
    let text = String::from_utf8_lossy(&packet);
    parse_xmp(&text, "embedded XMP")
}

/// Raw bytes of a root-IFD TIFF tag (XMP packet, DNG BaselineExposure source data).
fn tiff_tag_bytes(path: &Path, tag: u16) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).ok()?;
    decoder.get_tag_u8_vec(tiff::tags::Tag::Unknown(tag)).ok()
}

/// DNG `BaselineExposure` + `BaselineExposureOffset` (EV) from the root IFD, 0 when absent.
/// The offset belongs to the embedded camera profile; together they give the profile's
/// intended exposure, which keeps DNGs consistent with their original camera RAW files.
fn dng_baseline_exposure(path: &Path) -> f64 {
    let Ok(file) = std::fs::File::open(path) else {
        return 0.0;
    };
    let Ok(mut decoder) = tiff::decoder::Decoder::new(std::io::BufReader::new(file)) else {
        return 0.0;
    };
    let mut ev = 0.0;
    for tag in [50730, 51109] {
        ev += match decoder.find_tag(tiff::tags::Tag::Unknown(tag)) {
            Ok(Some(tiff::decoder::ifd::Value::SRational(n, d))) if d != 0 => {
                f64::from(n) / f64::from(d)
            }
            Ok(Some(tiff::decoder::ifd::Value::Rational(n, d))) if d != 0 => {
                f64::from(n) / f64::from(d)
            }
            _ => 0.0,
        };
    }
    ev.clamp(-4.0, 4.0)
}

/// The IFD of a DNG's "enhanced" image (NewSubFileType 16, e.g. Lightroom Denoise): linear
/// camera RGB that replaces the mosaic when present.
fn enhanced_ifd(tiff: &GenericTiffReader) -> Option<&IFD> {
    tiff.find_ifds_with_tag(TiffCommonTag::NewSubFileType)
        .into_iter()
        .find(|ifd| {
            let get = |t: TiffCommonTag| ifd.get_entry(t).and_then(|e| e.get_u32(0).ok().flatten());
            get(TiffCommonTag::NewSubFileType) == Some(16)
                && get(TiffCommonTag::SamplesPerPixel) == Some(3)
                && get(TiffCommonTag::PhotometricInt) == Some(34892)
        })
}

/// Whether `path` is a DNG with an enhanced (e.g. denoised) image.
fn has_enhanced_image(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("dng"))
        && RawSource::new(path).ok().is_some_and(|source| {
            GenericTiffReader::new(&mut source.reader(), 0, 0, None, &[])
                .ok()
                .is_some_and(|t| enhanced_ifd(&t).is_some())
        })
}

/// Decode a DNG's enhanced image as normalized (black 0, white 1) linear camera RGB.
fn enhanced_dng_rgb(path: &Path) -> Result<Option<(usize, usize, Vec<f32>)>, String> {
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("dng"))
    {
        return Ok(None);
    }
    let source = RawSource::new(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let Ok(tiff) = GenericTiffReader::new(&mut source.reader(), 0, 0, None, &[]) else {
        return Ok(None);
    };
    let Some(ifd) = enhanced_ifd(&tiff) else {
        return Ok(None);
    };
    let num = |e: Option<&rawler::formats::tiff::Entry>, i: usize| {
        e.and_then(|e| e.get_f32(i).ok().flatten())
    };
    let w = num(ifd.get_entry(TiffCommonTag::ImageWidth), 0).unwrap_or(0.0) as usize;
    let h = num(ifd.get_entry(TiffCommonTag::ImageLength), 0).unwrap_or(0.0) as usize;
    let bits = num(ifd.get_entry(TiffCommonTag::BitsPerSample), 0)
        .unwrap_or(16.0)
        .clamp(1.0, 32.0);
    let black_entry = ifd.get_entry(DngTag::BlackLevel);
    let black: [f32; 3] = std::array::from_fn(|c| {
        num(black_entry, c)
            .or_else(|| num(black_entry, 0))
            .unwrap_or(0.0)
    });
    let white = num(ifd.get_entry(DngTag::WhiteLevel), 0).unwrap_or(2f32.powf(bits) - 1.0);
    let data = rawler::decoders::plain_image_from_ifd(ifd, &source)
        .map_err(|e| format!("Enhanced DNG image: {e}"))?;
    let mut v = data.as_f32().into_owned();
    if w == 0 || h == 0 || v.len() != w * h * 3 {
        return Err("Enhanced DNG image has an unexpected size".into());
    }
    {
        use rayon::prelude::*;
        v.par_chunks_mut(3).for_each(|px| {
            for c in 0..3 {
                px[c] = (px[c] - black[c]).max(0.0) / (white - black[c]).max(1.0);
            }
        });
    }
    Ok(Some((w, h, v)))
}

// ---------------------------------------------------------------------------------------------
// Colour temperature (Robertson isotherms, as in the DNG SDK)

/// (1e6/T, u, v, isotherm slope) — Robertson 1968.
const ROBERTSON: [(f64, f64, f64, f64); 31] = [
    (0.0, 0.18006, 0.26352, -0.24341),
    (10.0, 0.18066, 0.26589, -0.25479),
    (20.0, 0.18133, 0.26846, -0.26876),
    (30.0, 0.18208, 0.27119, -0.28539),
    (40.0, 0.18293, 0.27407, -0.30470),
    (50.0, 0.18388, 0.27709, -0.32675),
    (60.0, 0.18494, 0.28021, -0.35156),
    (70.0, 0.18611, 0.28342, -0.37915),
    (80.0, 0.18740, 0.28668, -0.40955),
    (90.0, 0.18880, 0.28997, -0.44278),
    (100.0, 0.19032, 0.29326, -0.47888),
    (125.0, 0.19462, 0.30141, -0.58204),
    (150.0, 0.19962, 0.30921, -0.70471),
    (175.0, 0.20525, 0.31647, -0.84901),
    (200.0, 0.21142, 0.32312, -1.0182),
    (225.0, 0.21807, 0.32909, -1.2168),
    (250.0, 0.22511, 0.33439, -1.4512),
    (275.0, 0.23247, 0.33904, -1.7298),
    (300.0, 0.24010, 0.34308, -2.0637),
    (325.0, 0.24702, 0.34655, -2.4681),
    (350.0, 0.25591, 0.34951, -2.9641),
    (375.0, 0.26400, 0.35200, -3.5814),
    (400.0, 0.27218, 0.35407, -4.3633),
    (425.0, 0.28039, 0.35577, -5.3762),
    (450.0, 0.28863, 0.35714, -6.7262),
    (475.0, 0.29685, 0.35823, -8.5955),
    (500.0, 0.30505, 0.35907, -11.324),
    (525.0, 0.31320, 0.35968, -15.628),
    (550.0, 0.32129, 0.36011, -23.325),
    (575.0, 0.32931, 0.36038, -40.770),
    (600.0, 0.33724, 0.36051, -116.45),
];

/// CIE xy chromaticity of a correlated colour temperature with a Lightroom-style tint.
pub fn temperature_tint_to_xy(kelvin: f64, tint: f64) -> (f64, f64) {
    const TINT_SCALE: f64 = -3000.0;
    let r = 1.0e6 / kelvin.clamp(1500.0, 50_000.0);
    let offset = tint / TINT_SCALE;
    let mut index = 0;
    while index < 29 && r >= ROBERTSON[index + 1].0 {
        index += 1;
    }
    let (r0, u0, v0, t0) = ROBERTSON[index];
    let (r1, u1, v1, t1) = ROBERTSON[index + 1];
    let f = (r1 - r) / (r1 - r0);
    let mut u = u0 * f + u1 * (1.0 - f);
    let mut v = v0 * f + v1 * (1.0 - f);
    let (len0, len1) = ((1.0 + t0 * t0).sqrt(), (1.0 + t1 * t1).sqrt());
    let (uu0, vv0) = (1.0 / len0, t0 / len0);
    let (uu1, vv1) = (1.0 / len1, t1 / len1);
    let (mut uu, mut vv) = (uu0 * f + uu1 * (1.0 - f), vv0 * f + vv1 * (1.0 - f));
    let len = (uu * uu + vv * vv).sqrt();
    uu /= len;
    vv /= len;
    u += uu * offset;
    v += vv * offset;
    let d = u - 4.0 * v + 2.0;
    (1.5 * u / d, v / d)
}

/// Inverse of [`temperature_tint_to_xy`] (DNG SDK `dng_temperature::Set_xy_coord`).
pub fn xy_to_temperature_tint(x: f64, y: f64) -> (f64, f64) {
    const TINT_SCALE: f64 = -3000.0;
    let d = 1.5 - x + 6.0 * y;
    let (u, v) = (2.0 * x / d, 3.0 * y / d);
    let (mut last_dt, mut last_du, mut last_dv) = (0.0, 0.0, 0.0);
    for index in 1..=30 {
        let (_, ui, vi, ti) = ROBERTSON[index];
        let len = (1.0 + ti * ti).sqrt();
        let (mut du, mut dv) = (1.0 / len, ti / len);
        let (uu, vv) = (u - ui, v - vi);
        let mut dt = -uu * dv + vv * du;
        if dt <= 0.0 || index == 30 {
            dt = (-dt).max(0.0);
            let f = if index == 1 { 0.0 } else { dt / (last_dt + dt) };
            let (r0, u0, v0, _) = ROBERTSON[index - 1];
            let (r1, u1, v1, _) = ROBERTSON[index];
            let kelvin = 1.0e6 / (r0 * f + r1 * (1.0 - f));
            let uu = u - (u0 * f + u1 * (1.0 - f));
            let vv = v - (v0 * f + v1 * (1.0 - f));
            du = du * (1.0 - f) + last_du * f;
            dv = dv * (1.0 - f) + last_dv * f;
            let len = (du * du + dv * dv).sqrt();
            let tint = (uu * du / len + vv * dv / len) * TINT_SCALE;
            return (kelvin, tint);
        }
        last_dt = dt;
        last_du = du;
        last_dv = dv;
    }
    (6500.0, 0.0)
}

// ---------------------------------------------------------------------------------------------
// Development

/// What a RAW source is developed with (reported in the image info panel).
#[derive(Debug, Clone, Default)]
pub struct RawDevelopInfo {
    pub width: u32,
    pub height: u32,
    pub bits: u8,
    pub make: String,
    pub model: String,
    pub xmp: Option<XmpDevelop>,
    pub baseline_exposure_ev: f64,
    /// The DNG holds an enhanced (e.g. Lightroom Denoise) image, which is used.
    pub enhanced: bool,
    /// The camera's as-shot white balance as (kelvin, tint), when it can be computed.
    pub as_shot: Option<(f64, f64)>,
}

fn open_raw(path: &Path, dummy: bool) -> Result<RawImage, String> {
    let source = RawSource::new(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let decoder = rawler::get_decoder(&source).map_err(|e| format!("Unsupported RAW file: {e}"))?;
    decoder
        .raw_image(&source, &RawDecodeParams::default(), dummy)
        .map_err(|e| format!("RAW decode failed: {e}"))
}

/// EXIF orientation (1–8) of a RAW file: Lightroom's `tiff:Orientation` first, then the
/// root-IFD Orientation tag of TIFF-based RAWs (rawler does not read it for every format),
/// then rawler's own value.
fn raw_orientation(path: &Path, raw: &RawImage, xmp: Option<&XmpDevelop>) -> u8 {
    if let Some(o) = xmp.and_then(|x| x.orientation) {
        return o;
    }
    let from_tiff = std::fs::File::open(path).ok().and_then(|f| {
        let mut d = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).ok()?;
        d.find_tag(tiff::tags::Tag::Orientation)
            .ok()
            .flatten()
            .and_then(|v| v.into_u16().ok())
            .and_then(|v| u8::try_from(v).ok())
            .filter(|o| (1..=8).contains(o))
    });
    from_tiff.unwrap_or(match raw.orientation {
        Orientation::HorizontalFlip => 2,
        Orientation::Rotate180 => 3,
        Orientation::VerticalFlip => 4,
        Orientation::Transpose => 5,
        Orientation::Rotate90 => 6,
        Orientation::Transverse => 7,
        Orientation::Rotate270 => 8,
        Orientation::Normal | Orientation::Unknown => 1,
    })
}

/// Output size of a developed RAW (default crop, orientation applied), without decoding.
fn developed_size(raw: &RawImage, orientation: u8) -> (u32, u32) {
    let (w, h) = raw
        .crop_area
        .or(raw.active_area)
        .map_or((raw.width, raw.height), |r| (r.d.w, r.d.h));
    let (w, h) = (w as u32, h as u32);
    if orientation >= 5 {
        (h, w)
    } else {
        (w, h)
    }
}

/// Size, bit depth, camera and the Lightroom settings that will be applied — cheap (no
/// pixel decoding).
pub fn raw_info(path: &Path) -> Result<RawDevelopInfo, String> {
    let raw = open_raw(path, true)?;
    let xmp = read_xmp(path);
    let (width, height) = developed_size(&raw, raw_orientation(path, &raw, xmp.as_ref()));
    Ok(RawDevelopInfo {
        width,
        height,
        bits: raw.bps.clamp(1, 16) as u8,
        make: raw.clean_make.clone(),
        model: raw.clean_model.clone(),
        xmp,
        baseline_exposure_ev: dng_baseline_exposure(path),
        enhanced: has_enhanced_image(path),
        as_shot: as_shot_temperature(&raw),
    })
}

/// The camera's as-shot white balance expressed as Lightroom-style temperature and tint.
fn as_shot_temperature(raw: &RawImage) -> Option<(f64, f64)> {
    let w = raw.wb_coeffs;
    if !w[..3].iter().all(|v| v.is_finite() && *v > 0.0) {
        return None;
    }
    let neutral =
        nalgebra::Vector3::new(1.0 / f64::from(w[0]), 1.0, 1.0 / f64::from(w[2])) * f64::from(w[1]);
    let ((x, y), _) = CameraMatrices::of(raw).ok()?.white_of_neutral(neutral)?;
    Some(xy_to_temperature_tint(x, y))
}

/// Output pixel count of a RAW file (0 if it cannot be read).
pub fn raw_pixels(path: &Path) -> u64 {
    open_raw(path, true).map_or(0, |raw| {
        let (w, h) = developed_size(&raw, 1);
        u64::from(w) * u64::from(h)
    })
}

/// Correlated colour temperature the DNG SDK assigns to a calibration illuminant.
fn illuminant_kelvin(i: Illuminant) -> Option<f64> {
    Some(match i {
        Illuminant::A | Illuminant::Tungsten => 2850.0,
        Illuminant::IsoStudioTungsten => 3200.0,
        Illuminant::WhiteFluorescent => 3450.0,
        Illuminant::CoolWhiteFluorescent | Illuminant::Fluorescent => 4150.0,
        Illuminant::D50 | Illuminant::DaylightWhiteFluorescent => 5000.0,
        Illuminant::D55
        | Illuminant::Daylight
        | Illuminant::FineWeather
        | Illuminant::Flash
        | Illuminant::B => 5500.0,
        Illuminant::DaylightFluorescent => 6430.0,
        Illuminant::D65 | Illuminant::C | Illuminant::CloudyWeather => 6500.0,
        Illuminant::D75 | Illuminant::Shade => 7500.0,
        Illuminant::Unknown => return None,
    })
}

/// The camera's XYZ→camera matrices: one, or the coolest and warmest calibration pair.
struct CameraMatrices {
    /// (kelvin, matrix), warmest (lowest temperature) first.
    pairs: Vec<(f64, Matrix3<f64>)>,
}

impl CameraMatrices {
    fn of(raw: &RawImage) -> Result<Self, String> {
        let mut all: Vec<(f64, Matrix3<f64>)> = Vec::new();
        for (illuminant, m) in &raw.color_matrix {
            if m.len() < 9 {
                return Err("Unsupported (four-colour) RAW colour matrix".into());
            }
            let kelvin = illuminant_kelvin(*illuminant).unwrap_or(6500.0);
            all.push((kelvin, Matrix3::from_fn(|r, c| f64::from(m[r * 3 + c]))));
        }
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.dedup_by(|a, b| (a.0 - b.0).abs() < 1.0);
        let pairs = match all.len() {
            0 => {
                return Err(
                    "This camera has no colour matrix; RAW development is not possible".into(),
                )
            }
            1 => all,
            n => vec![all[0], all[n - 1]],
        };
        Ok(Self { pairs })
    }

    /// XYZ→camera at a white point temperature: linear in inverse temperature between the
    /// two calibrations, clamped outside them (DNG specification).
    fn at(&self, kelvin: f64) -> Matrix3<f64> {
        match self.pairs.as_slice() {
            [(_, m)] => *m,
            [(t1, m1), (t2, m2), ..] => {
                let g = if kelvin <= *t1 {
                    1.0
                } else if kelvin >= *t2 {
                    0.0
                } else {
                    (1.0 / kelvin - 1.0 / t2) / (1.0 / t1 - 1.0 / t2)
                };
                m1 * g + m2 * (1.0 - g)
            }
            [] => Matrix3::identity(),
        }
    }

    /// White point (xy and temperature) of a camera neutral, iterated as in the DNG SDK: the
    /// matrix depends on the temperature, which depends on the matrix.
    fn white_of_neutral(&self, neutral: nalgebra::Vector3<f64>) -> Option<((f64, f64), f64)> {
        let mut xy = (0.3457, 0.3585); // D50 start
        let mut kelvin = 5000.0;
        for _ in 0..30 {
            let xyz = self.at(kelvin).try_inverse()? * neutral;
            let sum = xyz.sum();
            if !(sum.is_finite() && sum > 1e-9) {
                return None;
            }
            let next = (xyz[0] / sum, xyz[1] / sum);
            let (t, _) = xy_to_temperature_tint(next.0, next.1);
            let done = (next.0 - xy.0).abs() + (next.1 - xy.1).abs() < 1e-7;
            xy = next;
            kelvin = t;
            if done {
                break;
            }
        }
        Some((xy, kelvin))
    }
}

/// Camera RGB → linear Rec.2020, for white-balanced camera data (dcraw convention: rows of
/// camera←Rec.2020 normalized to 1, so a balanced neutral maps to a neutral).
fn camera_to_rec2020(xyz_to_cam: &Matrix3<f64>) -> Result<Matrix3<f64>, String> {
    let rgb_to_xyz = Matrix3::from_fn(|r, c| REC2020_TO_XYZ[r][c]);
    let mut cam_from_rgb = xyz_to_cam * rgb_to_xyz;
    for r in 0..3 {
        let sum: f64 = cam_from_rgb.row(r).sum();
        if sum.abs() < 1e-9 {
            return Err("Degenerate RAW colour matrix".into());
        }
        for c in 0..3 {
            cam_from_rgb[(r, c)] /= sum;
        }
    }
    cam_from_rgb
        .try_inverse()
        .ok_or_else(|| "Singular RAW colour matrix".into())
}

/// White-balance multipliers (R, G, B; minimum 1) and the XYZ→camera matrix for that white
/// point: Lightroom temperature/tint if given, else the camera's as-shot balance, else D65.
fn white_balance(
    raw: &RawImage,
    matrices: &CameraMatrices,
    xmp: Option<&XmpDevelop>,
) -> ([f64; 3], Matrix3<f64>) {
    let from_neutral = |n: nalgebra::Vector3<f64>| -> Option<[f64; 3]> {
        (n.iter().all(|v| v.is_finite() && *v > 1e-6)).then(|| [1.0 / n[0], 1.0 / n[1], 1.0 / n[2]])
    };
    let xyz = |(x, y): (f64, f64)| nalgebra::Vector3::new(x / y, 1.0, (1.0 - x - y) / y);
    let normalize = |mul: [f64; 3]| {
        let min = mul.iter().copied().fold(f64::INFINITY, f64::min);
        mul.map(|m| m / min)
    };
    if let Some((t, tint)) = xmp.and_then(|x| x.temperature_tint) {
        let m = matrices.at(t);
        if let Some(mul) = from_neutral(m * xyz(temperature_tint_to_xy(t, tint))) {
            return (normalize(mul), m);
        }
    }
    let w = raw.wb_coeffs;
    if w[..3].iter().all(|v| v.is_finite() && *v > 0.0) {
        let mul = [f64::from(w[0]), f64::from(w[1]), f64::from(w[2])];
        let neutral = nalgebra::Vector3::new(1.0 / mul[0], 1.0 / mul[1], 1.0 / mul[2]);
        let m = matrices
            .white_of_neutral(neutral)
            .map_or_else(|| matrices.at(6500.0), |(_, k)| matrices.at(k));
        return (normalize(mul), m);
    }
    let m = matrices.at(6500.0);
    let mul = from_neutral(m * xyz((0.3127, 0.3290))).unwrap_or([1.0; 3]);
    (normalize(mul), m)
}

/// Exposure with a highlight shoulder: linear up to `KNEE`, then a rational curve that reaches
/// 1.0 exactly at the brightest possible input (`white`) with continuous slope. With no positive
/// exposure (`white` <= 1) it is the identity.
fn tone(x: f32, gain: f32, white: f32) -> f32 {
    const KNEE: f32 = 0.75;
    let v = x * gain;
    if white <= 1.0 || v <= KNEE {
        return v.clamp(0.0, 1.0);
    }
    let s = (white - KNEE) / (1.0 - KNEE);
    let a = ((v - KNEE) / (white - KNEE)).min(1.0);
    (KNEE + (1.0 - KNEE) * (s * a / (1.0 + (s - 1.0) * a))).min(1.0)
}

/// RCD demosaicing of a white-balanced, normalized Bayer mosaic over its active area, then the
/// default crop. `None` for layouts it does not handle (X-Trans, Fuji rotation, linear data).
fn rcd_bayer(raw: &RawImage) -> Option<(u32, u32, Vec<f32>)> {
    let RawPhotometricInterpretation::Cfa(config) = &raw.photometric else {
        return None;
    };
    if config.sensor != SensorType::Bayer
        || !config.cfa.is_rgb()
        || raw.fuji_rotation_width.is_some()
        || raw.cpp != 1
    {
        return None;
    }
    let RawImageData::Float(data) = &raw.data else {
        return None;
    };
    let full = Rect::new(Point::zero(), Dim2::new(raw.width, raw.height));
    let roi = raw.active_area.unwrap_or(full);
    if roi.p.x + roi.d.w > raw.width || roi.p.y + roi.d.h > raw.height {
        return None;
    }
    let (rw, rh) = (roi.d.w, roi.d.h);
    let mut mosaic = vec![0f32; rw * rh];
    for (y, row) in mosaic.chunks_mut(rw).enumerate() {
        let start = (roi.p.y + y) * raw.width + roi.p.x;
        row.copy_from_slice(&data[start..start + rw]);
    }
    let cfa = config.cfa.shift(roi.p.x, roi.p.y);
    // Bayer patterns repeat every 2×2 sites: a table avoids rawler's general 48×48 lookup.
    let table = [
        [cfa.color_at(0, 0).min(2), cfa.color_at(0, 1).min(2)],
        [cfa.color_at(1, 0).min(2), cfa.color_at(1, 1).min(2)],
    ];
    let periodic = cfa.width == 2 && cfa.height == 2;
    let colour = |y: usize, x: usize| {
        if periodic {
            table[y & 1][x & 1]
        } else {
            cfa.color_at(y, x).min(2)
        }
    };
    let planes = crate::demosaic::rcd(&mosaic, rw, rh, &colour);
    drop(mosaic);
    // The default crop is given in full-sensor coordinates.
    let crop = match raw.crop_area {
        Some(c)
            if c.p.x >= roi.p.x
                && c.p.y >= roi.p.y
                && c.p.x + c.d.w <= roi.p.x + rw
                && c.p.y + c.d.h <= roi.p.y + rh =>
        {
            Rect::new(Point::new(c.p.x - roi.p.x, c.p.y - roi.p.y), c.d)
        }
        _ => Rect::new(Point::zero(), Dim2::new(rw, rh)),
    };
    let mut flat = vec![0f32; crop.d.w * crop.d.h * 3];
    {
        use rayon::prelude::*;
        flat.par_chunks_mut(crop.d.w * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let base = (crop.p.y + y) * rw + crop.p.x;
                for (x, px) in row.chunks_mut(3).enumerate() {
                    for c in 0..3 {
                        px[c] = planes[c][base + x];
                    }
                }
            });
    }
    Some((crop.d.w as u32, crop.d.h as u32, flat))
}

/// Develop a RAW/DNG file into the 16-bit working format (see module docs).
pub fn decode_raw_16(path: &Path) -> Result<RawDecodedSource, String> {
    let mut raw = open_raw(path, false)?;
    let xmp = read_xmp(path);
    let orientation = raw_orientation(path, &raw, xmp.as_ref());
    let (ow, oh) = developed_size(&raw, orientation);
    if ow < 2 || oh < 2 || u64::from(ow) * u64::from(oh) * 16 > MAX_SOURCE_BYTES {
        return Err(format!(
            "RAW dimensions {ow}×{oh} exceed the per-source decode budget"
        ));
    }
    let bits = raw.bps.clamp(1, 16) as u8;
    let matrices = CameraMatrices::of(&raw)?;
    let (mul, xyz_to_cam) = white_balance(&raw, &matrices, xmp.as_ref());
    let cam_to_rgb = camera_to_rec2020(&xyz_to_cam)?;
    let exposure_ev = dng_baseline_exposure(path) + xmp.as_ref().map_or(0.0, |x| x.exposure_ev);

    let mut warnings = Vec::new();
    match enhanced_dng_rgb(path) {
        // A Lightroom-enhanced DNG: its denoised linear RGB replaces the mosaic (same
        // geometry, so the main image's default crop still applies).
        Ok(Some((w, h, v))) if w == raw.width && h == raw.height => {
            raw.data = RawImageData::Float(v);
            raw.cpp = 3;
            raw.photometric = RawPhotometricInterpretation::LinearRaw;
        }
        other => {
            if let Err(e) = other {
                warnings.push(format!(
                    "The DNG's enhanced (e.g. Lightroom Denoise) image could not be decoded ({e}); the original sensor data is developed instead"
                ));
            }
            raw.apply_scaling()
                .map_err(|e| format!("RAW scaling failed: {e}"))?;
        }
    }
    let mul32 = mul.map(|m| m as f32);
    let RawImageData::Float(ref mut data) = raw.data else {
        return Err("RAW scaling produced integer data".into());
    };
    match &raw.photometric {
        RawPhotometricInterpretation::Cfa(config) => {
            if !config.cfa.is_rgb() {
                return Err("Only RGB colour-filter sensors are supported".into());
            }
            let (w, cfa) = (raw.width, config.cfa.clone());
            data.chunks_mut(w).enumerate().for_each(|(row, line)| {
                for (col, v) in line.iter_mut().enumerate() {
                    let c = cfa.color_at(row, col).min(2);
                    // Saturation is common to all channels after balancing (clip at 1).
                    *v = (*v * mul32[c]).min(1.0);
                }
            });
        }
        RawPhotometricInterpretation::LinearRaw => {
            if raw.cpp != 3 {
                return Err(format!("Unsupported linear RAW with {} channels", raw.cpp));
            }
            for px in data.chunks_mut(3) {
                for c in 0..3 {
                    px[c] = (px[c] * mul32[c]).min(1.0);
                }
            }
        }
        RawPhotometricInterpretation::BlackIsZero => {
            return Err("Monochrome RAW files are not supported".into());
        }
    }
    let (w, h, flat) = match rcd_bayer(&raw) {
        Some(result) => result,
        None => {
            // X-Trans, Fuji-rotated and linear (demosaiced) data: rawler's pipeline.
            let develop = RawDevelop::new_with(&[
                ProcessingStep::Demosaic,
                ProcessingStep::FujiRotate,
                ProcessingStep::CropActiveArea,
                ProcessingStep::CropDefault,
            ]);
            let rgb = match develop
                .develop_intermediate(&raw)
                .map_err(|e| format!("RAW development failed: {e}"))?
            {
                Intermediate::ThreeColor(px) => px,
                _ => return Err("Unsupported RAW colour layout".into()),
            };
            let dim = rgb.dim();
            (dim.w as u32, dim.h as u32, rgb.flatten())
        }
    };
    drop(raw.data);

    let m: [[f32; 3]; 3] =
        std::array::from_fn(|r| std::array::from_fn(|c| cam_to_rgb[(r, c)] as f32));
    let gain = 2f32.powf(exposure_ev as f32);
    let white = gain.max(1.0);
    let mut out = vec![0u16; flat.len()];
    {
        use rayon::prelude::*;
        out.par_chunks_mut(3)
            .zip(flat.par_chunks(3))
            .for_each(|(dst, src)| {
                for (r, d) in dst.iter_mut().enumerate() {
                    let lin = m[r][0] * src[0] + m[r][1] * src[1] + m[r][2] * src[2];
                    *d = (tone(lin.max(0.0), gain, white) * 65535.0 + 0.5) as u16;
                }
            });
    }
    drop(flat);
    let image = Rgb16Image::from_raw(w, h, out).ok_or("RAW buffer size mismatch")?;
    let mut dynamic = image::DynamicImage::ImageRgb16(image);
    if let Some(o) = image::metadata::Orientation::from_exif(orientation) {
        dynamic.apply_orientation(o);
    }
    Ok(RawDecodedSource {
        image_rgb16: dynamic.into_rgb16(),
        icc_profile: Some(raw_working_icc().to_vec()),
        source_bit_depth: bits,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_extensions_are_recognized_case_insensitively() {
        assert!(is_raw_path(Path::new("/a/_DSC8336.ARW")));
        assert!(is_raw_path(Path::new("x.dng")));
        assert!(is_raw_path(Path::new("x.Cr3")));
        assert!(!is_raw_path(Path::new("x.tif")));
        assert!(!is_raw_path(Path::new("x")));
    }

    #[test]
    fn temperature_conversion_follows_the_planckian_locus() {
        // D65 is close to 6504 K on the locus; 2856 K is illuminant A.
        let (x, y) = temperature_tint_to_xy(6504.0, 0.0);
        assert!(
            (x - 0.3135).abs() < 0.003 && (y - 0.3237).abs() < 0.003,
            "{x} {y}"
        );
        let (x, y) = temperature_tint_to_xy(2856.0, 0.0);
        assert!(
            (x - 0.4476).abs() < 0.003 && (y - 0.4074).abs() < 0.003,
            "{x} {y}"
        );
        // Tint describes the light: a positive (magenta-slider) tint means a greener
        // illuminant, above the locus, which the balance then neutralizes toward magenta.
        let (_, y_tinted) = temperature_tint_to_xy(5000.0, 30.0);
        let (_, y_neutral) = temperature_tint_to_xy(5000.0, 0.0);
        assert!(y_tinted > y_neutral);
    }

    #[test]
    fn temperature_and_tint_round_trip() {
        for (t, tint) in [
            (2850.0, 0.0),
            (3200.0, -15.0),
            (4770.0, 10.0),
            (6500.0, 0.0),
            (9000.0, 40.0),
        ] {
            let (x, y) = temperature_tint_to_xy(t, tint);
            let (t2, tint2) = xy_to_temperature_tint(x, y);
            assert!((t2 - t).abs() < t * 1e-3, "{t} → {t2}");
            assert!((tint2 - tint).abs() < 0.2, "{tint} → {tint2}");
        }
    }

    #[test]
    fn dual_illuminant_matrices_interpolate_in_inverse_temperature() {
        let a = Matrix3::from_element(1.0);
        let d65 = Matrix3::from_element(3.0);
        let m = CameraMatrices {
            pairs: vec![(2850.0, a), (6500.0, d65)],
        };
        assert_eq!(m.at(2000.0), a);
        assert_eq!(m.at(8000.0), d65);
        // Halfway in mired between 2850 K and 6500 K.
        let mid = 2.0 / (1.0 / 2850.0 + 1.0 / 6500.0);
        assert!((m.at(mid)[(0, 0)] - 2.0).abs() < 1e-9);
        // A neutral computed at a temperature is found again by the iteration.
        let cams = CameraMatrices {
            pairs: vec![
                (
                    2850.0,
                    Matrix3::new(
                        0.7683, -0.3276, 0.0299, -0.3630, 1.0892, 0.3018, -0.0337, 0.0965, 0.6553,
                    ),
                ),
                (
                    6500.0,
                    Matrix3::new(
                        0.6640, -0.1847, -0.0503, -0.5238, 1.3010, 0.2474, -0.0993, 0.1673, 0.6527,
                    ),
                ),
            ],
        };
        let (x, y) = temperature_tint_to_xy(4000.0, 0.0);
        let neutral = cams.at(4000.0) * nalgebra::Vector3::new(x / y, 1.0, (1.0 - x - y) / y);
        let (_, kelvin) = cams.white_of_neutral(neutral).unwrap();
        assert!((kelvin - 4000.0).abs() < 5.0, "{kelvin}");
    }

    #[test]
    fn tone_is_identity_without_positive_exposure_and_smooth_with_it() {
        for x in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert!((tone(x, 1.0, 1.0) - x).abs() < 1e-6);
        }
        let g = 2f32.powf(0.7);
        let w = g;
        assert!((tone(1.0, g, w) - 1.0).abs() < 1e-5, "raw white maps to 1");
        assert!(
            (tone(0.3, g, w) - 0.3 * g).abs() < 1e-6,
            "linear below the knee"
        );
        let mut prev = 0.0;
        for i in 0..=1000 {
            let y = tone(i as f32 / 1000.0, g, w);
            assert!(y >= prev - 1e-7 && y <= 1.0);
            prev = y;
        }
        // Slope is continuous at the knee.
        let k = 0.75 / g;
        let left = (tone(k, g, w) - tone(k - 1e-3, g, w)) / 1e-3;
        let right = (tone(k + 1e-3, g, w) - tone(k, g, w)) / 1e-3;
        assert!((left - right).abs() < 0.02 * left, "{left} {right}");
    }

    #[test]
    fn lightroom_xmp_settings_are_parsed_and_unsupported_ones_reported() {
        let xmp = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
          <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
            crs:WhiteBalance="Custom" crs:Temperature="4770" crs:Tint="+10" crs:Exposure2012="+0.68"
            crs:Contrast2012="+20" crs:Highlights2012="-75" crs:Texture="0" crs:HasCrop="False"
            crs:CameraProfile="Camera Vivid" crs:HueAdjustmentRed="0" crs:SaturationAdjustmentBlue="5">
            <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li></rdf:Seq></crs:ToneCurvePV2012>
          </rdf:Description></rdf:RDF></x:xmpmeta>"#;
        let d = parse_xmp(xmp, "test.xmp").unwrap();
        assert_eq!(d.temperature_tint, Some((4770.0, 10.0)));
        assert!((d.exposure_ev - 0.68).abs() < 1e-9);
        for label in ["Contrast", "Highlights", "HSL"] {
            assert!(
                d.ignored.iter().any(|l| l == label),
                "{label}: {:?}",
                d.ignored
            );
        }
        assert!(!d.ignored.iter().any(|l| l == "Texture"));
        assert!(d.ignored.iter().any(|l| l.contains("Camera Vivid")));
        // "As Shot" keeps the camera's balance even when a temperature is written.
        let as_shot = xmp.replace(
            "crs:WhiteBalance=\"Custom\"",
            "crs:WhiteBalance=\"As Shot\"",
        );
        assert_eq!(parse_xmp(&as_shot, "t").unwrap().temperature_tint, None);
        assert!(parse_xmp("<x/>", "t").is_none());
    }

    #[test]
    fn rec2020_conversion_keeps_balanced_neutrals_neutral() {
        // A plausible camera matrix (Sony-like XYZ→camera).
        let xyz_to_cam = Matrix3::new(
            0.6640, -0.1847, -0.0503, -0.5238, 1.3010, 0.2474, -0.0993, 0.1673, 0.6527,
        );
        let m = camera_to_rec2020(&xyz_to_cam).unwrap();
        let grey = m * nalgebra::Vector3::new(0.4, 0.4, 0.4);
        assert!((grey - nalgebra::Vector3::new(0.4, 0.4, 0.4)).norm() < 1e-9);
    }

    /// Write a small linear (demosaiced) DNG whose camera space is linear Rec.2020, so the
    /// development must return the stored values (plus the requested exposure).
    fn synthetic_dng(path: &Path, w: usize, h: usize, pixel: impl Fn(usize, usize) -> [f64; 3]) {
        use rawler::dng::writer::DngWriter;
        use rawler::formats::tiff::{Rational, SRational};
        let mut data = vec![0u16; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let p = pixel(x, y);
                for c in 0..3 {
                    data[(y * w + x) * 3 + c] = (p[c] * 65535.0).round() as u16;
                }
            }
        }
        let xyz_to_cam = Matrix3::from_fn(|r, c| REC2020_TO_XYZ[r][c])
            .try_inverse()
            .unwrap();
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        let mut dng = DngWriter::new(&mut file, rawler::dng::DNG_VERSION_V1_4).unwrap();
        let matrix: Vec<SRational> = (0..9)
            .map(|i| SRational::new((xyz_to_cam[(i / 3, i % 3)] * 1e6).round() as i32, 1_000_000))
            .collect();
        dng.color_matrix(1, Illuminant::D65, matrix);
        dng.as_shot_neutral([
            Rational::new(1, 1),
            Rational::new(1, 1),
            Rational::new(1, 1),
        ]);
        let mut frame = dng.subframe_on_root(0);
        frame
            .rgb_image_u16(&data, w, h, rawler::dng::DngCompression::Uncompressed, 1)
            .unwrap();
        frame.finalize().unwrap();
        dng.close().unwrap();
    }

    #[test]
    fn synthetic_linear_dng_develops_to_its_linear_rec2020_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic.dng");
        let (w, h) = (64, 48);
        let colour = |x: usize, _y: usize| {
            if x < 32 {
                [0.18; 3]
            } else {
                [0.40, 0.20, 0.10]
            }
        };
        synthetic_dng(&path, w, h, colour);
        let info = raw_info(&path).unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        let out = decode_raw_16(&path).unwrap();
        assert!(is_raw_working_icc(out.icc_profile.as_deref().unwrap()));
        assert_eq!(out.image_rgb16.dimensions(), (64, 48));
        for (x, expect) in [(10, [0.18; 3]), (50, [0.40, 0.20, 0.10])] {
            let px = out.image_rgb16.get_pixel(x, 20).0;
            for c in 0..3 {
                let got = f64::from(px[c]) / 65535.0;
                assert!(
                    (got - expect[c]).abs() < 0.004,
                    "x={x} c={c}: {got} vs {}",
                    expect[c]
                );
            }
        }
    }

    #[test]
    fn xmp_sidecar_exposure_and_orientation_are_applied() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame.dng");
        // Top-left quadrant marked, so the rotation is observable.
        synthetic_dng(&path, 40, 20, |x, y| {
            if x < 10 && y < 5 {
                [0.2; 3]
            } else {
                [0.05; 3]
            }
        });
        std::fs::write(
            dir.path().join("frame.xmp"),
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
              xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6" crs:Exposure2012="+1.00"/>
            </rdf:RDF></x:xmpmeta>"#,
        )
        .unwrap();
        let out = decode_raw_16(&path).unwrap();
        // Orientation 6 rotates 90° clockwise: 40×20 becomes 20×40, the mark moves top-right.
        assert_eq!(out.image_rgb16.dimensions(), (20, 40));
        let v = |x, y| f64::from(out.image_rgb16.get_pixel(x, y).0[1]) / 65535.0;
        assert!(
            (v(17, 3) - 0.4).abs() < 0.005,
            "marked, +1 EV: {}",
            v(17, 3)
        );
        assert!(
            (v(5, 30) - 0.1).abs() < 0.005,
            "background, +1 EV: {}",
            v(5, 30)
        );
    }

    #[test]
    fn raw_working_profile_is_linear_rec2020_and_recognized() {
        let icc = raw_working_icc();
        assert!(is_raw_working_icc(icc));
        let p = ColorProfile::new_from_slice(icc).unwrap();
        assert_eq!(crate::color::detect_preset_match(&p, icc), None);
    }
}
