//! One consistent decode path for import, features and export; disk-backed full-size sources.
use crate::color::{TargetColorProfile, TargetTrc};
use image::{DynamicImage, ImageBuffer, ImageDecoder, ImageReader, Rgb, RgbImage};
use memmap2::Mmap;
use moxcms::{ColorProfile, DataColorSpace, Layout, TransformOptions};
use std::io::Write;
use std::path::Path;

/// Decode guard per source (16 bytes per pixel of transient decode memory): about 268 MP,
/// enough for 100+ MP medium-format files. The process memory ceiling (`memory::reserve`)
/// governs how many large sources decode at once; this only rejects absurd/hostile headers.
pub const MAX_SOURCE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub type Rgb16Image = ImageBuffer<Rgb<u16>, Vec<u16>>;

pub struct RawDecodedSource {
    pub image_rgb16: Rgb16Image,
    pub icc_profile: Option<Vec<u8>>,
    pub source_bit_depth: u8,
    /// Non-fatal notes for the image info panel (e.g. a RAW fallback).
    pub warnings: Vec<String>,
}

/// Decode a source file at full precision (preserved up to 16 bits/ch) along with its validated
/// embedded RGB ICC profile bytes if present. Rejects non-opaque alpha and non-RGB ICC profiles.
pub fn decode_source_raw_16(path: &Path) -> Result<RawDecodedSource, String> {
    if crate::rawdev::is_raw_path(path) {
        return crate::rawdev::decode_raw_16(path);
    }
    let mut reader = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    // The image crate's default 512 MiB allocation limit would reject 16-bit sources above
    // ~90 MP; use this module's own guard instead.
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_SOURCE_BYTES);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(|e| e.to_string())?;
    let (w, h) = decoder.dimensions();
    if w < 2 || h < 2 || u64::from(w) * u64::from(h) * 16 > MAX_SOURCE_BYTES {
        return Err(format!(
            "Image dimensions {w}×{h} exceed the per-source decode budget"
        ));
    }
    let color = decoder.color_type();
    let source_bit_depth = (color.bits_per_pixel() / u16::from(color.channel_count()).max(1)) as u8;
    let orientation = decoder.orientation().map_err(|e| e.to_string())?;
    let icc_profile = decoder.icc_profile().map_err(|e| e.to_string())?;
    if let Some(ref icc) = icc_profile {
        let profile =
            ColorProfile::new_from_slice(icc).map_err(|e| format!("Invalid ICC profile: {e}"))?;
        if profile.color_space != DataColorSpace::Rgb {
            return Err(
                "Embedded non-RGB ICC profile is unsupported; convert this source to RGB first"
                    .into(),
            );
        }
    }
    let mut decoded = DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    decoded.apply_orientation(orientation);
    if decoded.color().has_alpha() && decoded.to_rgba16().pixels().any(|pixel| pixel[3] != 65535) {
        return Err("Transparent sources are unsupported by the opaque working format; flatten this source first".into());
    }
    Ok(RawDecodedSource {
        image_rgb16: decoded.into_rgb16(),
        icc_profile,
        source_bit_depth,
        warnings: Vec::new(),
    })
}

/// Convert a 16-bit source image from its embedded ICC (or assumed sRGB if absent) into `target`.
pub fn convert_rgb16_profile(
    src: &Rgb16Image,
    src_icc: Option<&[u8]>,
    target: &TargetColorProfile,
) -> Result<Rgb16Image, String> {
    if src_icc.is_none() && matches!(target, TargetColorProfile::Srgb) {
        return Ok(src.clone());
    }
    if let (
        Some(src_bytes),
        TargetColorProfile::CustomIcc {
            bytes: dst_bytes, ..
        },
    ) = (src_icc, target)
    {
        if src_bytes == dst_bytes.as_slice() {
            return Ok(src.clone());
        }
    }
    let src_profile = match src_icc {
        Some(icc) => {
            let p = ColorProfile::new_from_slice(icc)
                .map_err(|e| format!("Invalid ICC profile: {e}"))?;
            if p.color_space != DataColorSpace::Rgb {
                return Err(
                    "Embedded non-RGB ICC profile is unsupported; convert this source to RGB first"
                        .into(),
                );
            }
            p
        }
        None => ColorProfile::new_srgb(),
    };
    let dst_profile = target.to_moxcms_profile()?;
    let transform = src_profile
        .create_transform_16bit(
            Layout::Rgb,
            &dst_profile,
            Layout::Rgb,
            TransformOptions::default(),
        )
        .map_err(|e| format!("ICC 16-bit transform: {e}"))?;
    let mut dest = Rgb16Image::new(src.width(), src.height());
    transform
        .transform(src.as_raw(), dest.as_mut())
        .map_err(|e| format!("ICC 16-bit conversion: {e}"))?;
    Ok(dest)
}

/// Downscale a 16-bit RGB image preserving 16-bit precision for cached preview mipmaps.
pub fn thumbnail_rgb16(src: &Rgb16Image, max_dim: u32) -> Rgb16Image {
    let (w, h) = src.dimensions();
    let scale = (f64::from(max_dim) / f64::from(w.max(h))).min(1.0);
    let tw = ((f64::from(w) * scale).round() as u32).max(1);
    let th = ((f64::from(h) * scale).round() as u32).max(1);
    // Area-averaging thumbnail directly on the buffer (no full-size copy of the source).
    image::imageops::thumbnail(src, tw, th)
}

pub fn decode_source(path: &Path) -> Result<DynamicImage, String> {
    let raw = decode_source_raw_16(path)?;
    let srgb16 = convert_rgb16_profile(
        &raw.image_rgb16,
        raw.icc_profile.as_deref(),
        &TargetColorProfile::Srgb,
    )?;
    let (w, h) = srgb16.dimensions();
    let bytes: Vec<u8> = srgb16
        .as_raw()
        .iter()
        .map(|&v| ((u32::from(v) + 128) / 257).min(255) as u8)
        .collect();
    let rgb8 = RgbImage::from_raw(w, h, bytes).ok_or("Invalid RGB8 dimensions")?;
    Ok(DynamicImage::ImageRgb8(rgb8))
}

pub trait RgbRaster: Sync {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn rgb_bytes(&self) -> &[u8];
}

impl RgbRaster for RgbImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn rgb_bytes(&self) -> &[u8] {
        self.as_raw()
    }
}

pub struct MappedRgb {
    width: u32,
    height: u32,
    map: Mmap,
    // The private backing file must remain alive and immutable for the mapping.
    _file: tempfile::NamedTempFile,
}

impl MappedRgb {
    pub fn from_image_in(image: RgbImage, directory: &Path) -> std::io::Result<Self> {
        let mut file = tempfile::NamedTempFile::new_in(directory)?;
        file.write_all(image.as_raw())?;
        file.flush()?;
        // SAFETY: this process exclusively owns this private tempfile, never writes
        // it after mapping, and retains it until after the mapping is dropped.
        let map = unsafe { Mmap::map(file.as_file())? };
        Ok(Self {
            width: image.width(),
            height: image.height(),
            map,
            _file: file,
        })
    }
}

impl RgbRaster for MappedRgb {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn rgb_bytes(&self) -> &[u8] {
        &self.map
    }
}

pub trait Rgb16Raster: Sync {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn sample_rgb16(&self, x: u32, y: u32) -> [u16; 3];
}

impl Rgb16Raster for RgbImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn sample_rgb16(&self, x: u32, y: u32) -> [u16; 3] {
        let off = (y as usize * self.width() as usize + x as usize) * 3;
        let raw = self.as_raw();
        [
            u16::from(raw[off]) * 257,
            u16::from(raw[off + 1]) * 257,
            u16::from(raw[off + 2]) * 257,
        ]
    }
}

impl Rgb16Raster for Rgb16Image {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn sample_rgb16(&self, x: u32, y: u32) -> [u16; 3] {
        let off = (y as usize * self.width() as usize + x as usize) * 3;
        let raw = self.as_raw();
        [raw[off], raw[off + 1], raw[off + 2]]
    }
}

pub struct MappedRgb16 {
    width: u32,
    height: u32,
    map: Mmap,
    // The private backing file must remain alive and immutable for the mapping.
    _file: tempfile::NamedTempFile,
}

impl MappedRgb16 {
    pub fn from_image_in(image: Rgb16Image, directory: &Path) -> std::io::Result<Self> {
        let (width, height) = image.dimensions();
        let mut file = tempfile::NamedTempFile::new_in(directory)?;
        {
            // Convert in large chunks; per-sample writes dominated the cache build time.
            let mut buf = vec![0u8; 1 << 22];
            for chunk in image.as_raw().chunks(buf.len() / 2) {
                let bytes = &mut buf[..chunk.len() * 2];
                for (dst, &v) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(chunk) {
                    *dst = v.to_ne_bytes();
                }
                file.write_all(bytes)?;
            }
            file.flush()?;
        }
        // SAFETY: this process exclusively owns this private tempfile, never writes
        // it after mapping, and retains it until after the mapping is dropped.
        let map = unsafe { Mmap::map(file.as_file())? };
        Ok(Self {
            width,
            height,
            map,
            _file: file,
        })
    }
}

impl Rgb16Raster for image::RgbaImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn sample_rgb16(&self, x: u32, y: u32) -> [u16; 3] {
        let p = self.get_pixel(x, y).0;
        [
            u16::from(p[0]) * 257,
            u16::from(p[1]) * 257,
            u16::from(p[2]) * 257,
        ]
    }
}

/// Halve a 16-bit encoded image with a 2×2 box filter evaluated in linear light, so bright
/// point sources (stars) keep their energy instead of darkening as with gamma-space averaging.
pub fn half_resolution_linear(src: &Rgb16Image, trc: &TargetTrc) -> Rgb16Image {
    use rayon::prelude::*;
    let (w, h) = src.dimensions();
    let dw = w.div_ceil(2).max(1);
    let dh = h.div_ceil(2).max(1);
    let raw = src.as_raw();
    let mut out = vec![0u16; dw as usize * dh as usize * 3];
    out.par_chunks_mut(dw as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = (y * 2).min(h as usize - 1);
            let y1 = (y * 2 + 1).min(h as usize - 1);
            for x in 0..dw as usize {
                let x0 = (x * 2).min(w as usize - 1);
                let x1 = (x * 2 + 1).min(w as usize - 1);
                let mut acc = [0.0f32; 3];
                for (xx, yy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                    let o = (yy * w as usize + xx) * 3;
                    let lin = trc.decode_rgb16_to_linear([raw[o], raw[o + 1], raw[o + 2]]);
                    acc[0] += lin[0];
                    acc[1] += lin[1];
                    acc[2] += lin[2];
                }
                let enc = trc.encode_linear_to_u16([acc[0] * 0.25, acc[1] * 0.25, acc[2] * 0.25]);
                row[x * 3..x * 3 + 3].copy_from_slice(&enc);
            }
        });
    Rgb16Image::from_raw(dw, dh, out).expect("half-resolution buffer matches dimensions")
}

impl Rgb16Raster for MappedRgb16 {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn sample_rgb16(&self, x: u32, y: u32) -> [u16; 3] {
        let off = (y as usize * self.width as usize + x as usize) * 6;
        [
            u16::from_ne_bytes([self.map[off], self.map[off + 1]]),
            u16::from_ne_bytes([self.map[off + 2], self.map[off + 3]]),
            u16::from_ne_bytes([self.map[off + 4], self.map[off + 5]]),
        ]
    }
}

#[inline]
pub fn srgb_to_linear(value: f32) -> f32 {
    static LUT: std::sync::OnceLock<TargetTrc> = std::sync::OnceLock::new();
    let trc = LUT.get_or_init(TargetTrc::srgb);
    let v = (value.clamp(0.0, 255.0).round() as u16) * 257;
    trc.decode_rgb16_to_linear([v, 0, 0])[0] * 255.0
}

#[inline]
pub fn linear_to_srgb(value: f32) -> u8 {
    static LUT: std::sync::OnceLock<TargetTrc> = std::sync::OnceLock::new();
    let trc = LUT.get_or_init(TargetTrc::srgb);
    trc.encode_linear_to_u8([value / 255.0, 0.0, 0.0])[0]
}
