use moxcms::{ColorProfile, DataColorSpace, ProfileText, ToneReprCurve};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const DEFAULT_OUTPUT_BIT_DEPTH: u8 = 8;
pub const MAX_ICC_BYTES: usize = 4 * 1024 * 1024;
pub const SUPPORTED_BIT_DEPTHS: [u8; 3] = [8, 10, 16];
pub const SUPPORTED_COLOR_PROFILE_IDS: [&str; 6] = [
    "srgb",
    "adobe_rgb",
    "display_p3",
    "prophoto_rgb",
    "bt2020",
    "custom_icc",
];

pub fn validate_bit_depth(bits: u8) -> Result<u8, String> {
    if matches!(bits, 8 | 10 | 16) {
        Ok(bits)
    } else {
        Err(format!(
            "Unsupported bit depth {bits}; choose 8, 10 or 16 bits per channel"
        ))
    }
}

/// Represent 10-bit precision (`0..=1023`) inside a standard 16-bit unsigned integer container
/// (`0..=65535`) via bit replication: `0 -> 0`, `1023 -> 65535`, monotonically reversible
/// with `sample >> 6`.
#[inline]
pub fn quantize_10bit_to_u16(encoded_0_to_1: f32) -> u16 {
    let q = (encoded_0_to_1.clamp(0.0, 1.0) * 1023.0).round() as u16;
    (q << 6) | (q >> 4)
}

#[inline]
pub fn u16_to_10bit_level(sample: u16) -> u16 {
    sample >> 6
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColorProfileConfig {
    pub profile: String,
    #[serde(default)]
    pub custom_name: Option<String>,
    #[serde(default)]
    pub custom_icc_base64: Option<String>,
}

impl Default for ColorProfileConfig {
    fn default() -> Self {
        Self {
            profile: "srgb".to_string(),
            custom_name: None,
            custom_icc_base64: None,
        }
    }
}

#[derive(Clone)]
pub enum TargetColorProfile {
    Srgb,
    AdobeRgb,
    DisplayP3,
    ProPhotoRgb,
    Bt2020,
    CustomIcc { name: String, bytes: Vec<u8> },
}

impl TargetColorProfile {
    pub fn from_parts(
        id: &str,
        custom_name: Option<&str>,
        custom_base64: Option<&str>,
    ) -> Result<Self, String> {
        match id {
            "" | "srgb" => Ok(Self::Srgb),
            "adobe_rgb" => Ok(Self::AdobeRgb),
            "display_p3" => Ok(Self::DisplayP3),
            "prophoto_rgb" => Ok(Self::ProPhotoRgb),
            "bt2020" => Ok(Self::Bt2020),
            "custom_icc" => {
                let b64 = custom_base64
                    .filter(|s| !s.trim().is_empty())
                    .ok_or("Custom ICC profile requires embedded ICC data")?;
                let bytes = decode_base64(b64)?;
                if bytes.is_empty() || bytes.len() > MAX_ICC_BYTES {
                    return Err("Custom ICC profile exceeds size budget or is empty".into());
                }
                let parsed = ColorProfile::new_from_slice(&bytes)
                    .map_err(|e| format!("Invalid custom ICC profile: {e}"))?;
                if parsed.color_space != DataColorSpace::Rgb {
                    return Err(
                        "Only RGB ICC profiles are supported as target panorama profiles".into(),
                    );
                }
                TargetTrc::try_from_profile(&parsed)?;
                let derived_name = custom_name
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .or_else(|| extract_icc_description(&parsed))
                    .unwrap_or_else(|| "Custom ICC Profile".to_string());
                Ok(Self::CustomIcc {
                    name: derived_name,
                    bytes,
                })
            }
            other => Err(format!(
                "Unsupported color profile '{other}'; expected srgb, adobe_rgb, display_p3, prophoto_rgb, bt2020 or custom_icc"
            )),
        }
    }

    pub fn from_raw_icc(name: Option<&str>, bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > MAX_ICC_BYTES {
            return Err("ICC profile exceeds size budget or is empty".into());
        }
        let parsed = ColorProfile::new_from_slice(&bytes)
            .map_err(|e| format!("Invalid ICC profile: {e}"))?;
        if parsed.color_space != DataColorSpace::Rgb {
            return Err("Only RGB ICC profiles are supported".into());
        }
        TargetTrc::try_from_profile(&parsed)?;
        if let Some(preset) = detect_preset_match(&parsed, &bytes) {
            return Self::from_parts(preset, None, None);
        }
        let label = name
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| extract_icc_description(&parsed))
            .unwrap_or_else(|| "Custom ICC Profile".to_string());
        Ok(Self::CustomIcc { name: label, bytes })
    }

    pub fn id(&self) -> &'static str {
        match self {
            Self::Srgb => "srgb",
            Self::AdobeRgb => "adobe_rgb",
            Self::DisplayP3 => "display_p3",
            Self::ProPhotoRgb => "prophoto_rgb",
            Self::Bt2020 => "bt2020",
            Self::CustomIcc { .. } => "custom_icc",
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Srgb => "sRGB IEC61966-2.1".to_string(),
            Self::AdobeRgb => "Adobe RGB (1998)".to_string(),
            Self::DisplayP3 => "Display P3".to_string(),
            Self::ProPhotoRgb => "ProPhoto RGB".to_string(),
            Self::Bt2020 => "Rec.2020 (BT.2020)".to_string(),
            Self::CustomIcc { name, .. } => name.clone(),
        }
    }

    pub fn custom_icc_base64(&self) -> Option<String> {
        match self {
            Self::CustomIcc { bytes, .. } => Some(encode_base64(bytes)),
            _ => None,
        }
    }

    pub fn to_moxcms_profile(&self) -> Result<ColorProfile, String> {
        match self {
            Self::Srgb => Ok(ColorProfile::new_srgb()),
            Self::AdobeRgb => Ok(ColorProfile::new_adobe_rgb()),
            Self::DisplayP3 => Ok(ColorProfile::new_display_p3()),
            Self::ProPhotoRgb => Ok(ColorProfile::new_pro_photo_rgb()),
            Self::Bt2020 => Ok(ColorProfile::new_bt2020()),
            Self::CustomIcc { bytes, .. } => {
                ColorProfile::new_from_slice(bytes).map_err(|e| format!("Invalid custom ICC: {e}"))
            }
        }
    }

    pub fn icc_bytes(&self) -> Result<Vec<u8>, String> {
        match self {
            // User/source profiles are embedded byte-for-byte.
            Self::CustomIcc { bytes, .. } => Ok(bytes.clone()),
            _ => self
                .to_moxcms_profile()?
                .encode()
                .map(|raw| normalize_icc(&raw).unwrap_or(raw))
                .map_err(|e| format!("Could not encode ICC profile {}: {e}", self.display_name())),
        }
    }
}

/// Rewrite an encoded ICC profile so it satisfies the structural rules strict readers check
/// (libpng, Adobe): every tag starts on a 4-byte boundary, the PCS illuminant is exactly D50
/// and the profile ID is zeroed (not computed). Tag contents are unchanged; shared tags stay
/// shared. Returns `None` if the input is not a well-formed tag table.
pub fn normalize_icc(raw: &[u8]) -> Option<Vec<u8>> {
    let be32 = |b: &[u8], o: usize| -> Option<u32> {
        Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?))
    };
    let count = be32(raw, 128)? as usize;
    if count == 0 || count > 1000 {
        return None;
    }
    let table_end = 132 + 12 * count;
    let mut tags = Vec::with_capacity(count);
    for k in 0..count {
        let base = 132 + 12 * k;
        let sig = be32(raw, base)?;
        let off = be32(raw, base + 4)? as usize;
        let len = be32(raw, base + 8)? as usize;
        if off < table_end || off.checked_add(len)? > raw.len() {
            return None;
        }
        tags.push((sig, off, len));
    }
    let mut out = raw.get(..table_end)?.to_vec();
    while out.len() % 4 != 0 {
        out.push(0);
    }
    let mut placed: Vec<((usize, usize), usize)> = Vec::new();
    for (k, &(_, off, len)) in tags.iter().enumerate() {
        let new_off = match placed.iter().find(|(key, _)| *key == (off, len)) {
            Some((_, o)) => *o,
            None => {
                let o = out.len();
                out.extend_from_slice(&raw[off..off + len]);
                while out.len() % 4 != 0 {
                    out.push(0);
                }
                placed.push(((off, len), o));
                o
            }
        };
        let base = 132 + 12 * k;
        out[base + 4..base + 8].copy_from_slice(&(new_off as u32).to_be_bytes());
    }
    let size = out.len() as u32;
    out[0..4].copy_from_slice(&size.to_be_bytes());
    // D50 PCS illuminant in s15Fixed16: X 0.9642, Y 1.0, Z 0.8249.
    out[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    out[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    out[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());
    out[84..100].fill(0);
    Some(out)
}

pub fn extract_icc_description(profile: &ColorProfile) -> Option<String> {
    let text = profile.description.as_ref()?;
    let raw = match text {
        ProfileText::PlainString(s) => s.clone(),
        ProfileText::Localizable(items) => items
            .iter()
            .find(|item| item.language == "en" && !item.value.trim().is_empty())
            .or_else(|| items.iter().find(|item| !item.value.trim().is_empty()))
            .map(|item| item.value.clone())
            .unwrap_or_default(),
        ProfileText::Description(desc) => {
            if !desc.ascii_string.trim().is_empty() {
                desc.ascii_string.clone()
            } else {
                desc.unicode_string.clone()
            }
        }
    };
    let clean = raw.trim().to_string();
    if clean.is_empty() {
        None
    } else {
        Some(clean)
    }
}

/// Built-in preset for raw ICC bytes (see `detect_preset_match`); `None` if unparsable.
pub fn preset_match_for_icc(raw_icc: &[u8]) -> Option<&'static str> {
    let parsed = ColorProfile::new_from_slice(raw_icc).ok()?;
    detect_preset_match(&parsed, raw_icc)
}

/// Return the built-in preset that is colorimetrically identical to `profile`.
///
/// A preset is only reported when the ICC bytes are byte-identical to the encoded preset, or
/// when primaries, white point and every tone curve agree within half a 10-bit code value.
/// The profile description is never used as evidence: a "Linear sRGB" profile has sRGB
/// primaries but a different tone curve and must stay a custom profile.
pub fn detect_preset_match(profile: &ColorProfile, raw_icc: &[u8]) -> Option<&'static str> {
    let presets = [
        ("srgb", ColorProfile::new_srgb()),
        ("adobe_rgb", ColorProfile::new_adobe_rgb()),
        ("display_p3", ColorProfile::new_display_p3()),
        ("prophoto_rgb", ColorProfile::new_pro_photo_rgb()),
        ("bt2020", ColorProfile::new_bt2020()),
    ];
    for (id, preset) in &presets {
        if let Ok(encoded) = preset.encode() {
            if encoded == raw_icc {
                return Some(id);
            }
        }
    }
    if profile.color_space != DataColorSpace::Rgb {
        return None;
    }
    let candidate = TargetTrc::try_from_profile(profile).ok()?;
    presets
        .iter()
        .find(|(_, preset)| colorimetrically_equal(profile, &candidate, preset))
        .map(|(id, _)| *id)
}

fn xyz_close(a: &moxcms::Xyzd, b: &moxcms::Xyzd) -> bool {
    const TOL: f64 = 0.0025;
    (a.x - b.x).abs() <= TOL && (a.y - b.y).abs() <= TOL && (a.z - b.z).abs() <= TOL
}

fn colorimetrically_equal(
    profile: &ColorProfile,
    candidate: &TargetTrc,
    preset: &ColorProfile,
) -> bool {
    if !(xyz_close(&profile.red_colorant, &preset.red_colorant)
        && xyz_close(&profile.green_colorant, &preset.green_colorant)
        && xyz_close(&profile.blue_colorant, &preset.blue_colorant)
        && xyz_close(&profile.white_point, &preset.white_point))
    {
        return false;
    }
    let Ok(reference) = TargetTrc::try_from_profile(preset) else {
        return false;
    };
    // Dense, black-weighted comparison in the encoded domain (log + linear spacing).
    let probes = (0..=400)
        .map(|i| i as f32 / 400.0)
        .chain((0..=200).map(|i| 10f32.powf(-6.0 + 6.0 * i as f32 / 200.0)));
    for lin in probes {
        let a = candidate.encode_linear_to_f32([lin; 3]);
        let b = reference.encode_linear_to_f32([lin; 3]);
        if a.iter()
            .zip(b.iter())
            .any(|(x, y)| (x - y).abs() > 0.5 / 1023.0)
        {
            return false;
        }
    }
    true
}

/// Number of guide buckets used to accelerate the exact inverse lookup.
const ENCODE_GUIDE_STEPS: usize = 4096;

/// Per-channel tone reproduction tables.
///
/// `decode_u16` is the exact forward curve sampled at every 16-bit code value. The inverse is
/// computed from that same table by monotone search plus linear interpolation between adjacent
/// codes, so encode(decode(code)) returns `code` for every 16-bit value regardless of how steep
/// the curve is near black (pure gamma curves such as Adobe RGB and ProPhoto have an infinite
/// slope at zero, which a uniformly sampled inverse LUT cannot represent).
struct ChannelTrcLut {
    decode_u16: Vec<f32>,
    /// `guide[b]` is the first code whose linear value is `>= b / ENCODE_GUIDE_STEPS`.
    guide: Vec<u32>,
}

impl ChannelTrcLut {
    fn build(trc: &ToneReprCurve) -> Result<Arc<Self>, String> {
        let lin = trc
            .make_linear_evaluator()
            .map_err(|e| format!("Unsupported tone curve in target profile: {e:?}"))?;
        let mut decode_u16: Vec<f32> = Vec::with_capacity(65536);
        let mut running = 0.0f32;
        for i in 0..=65535u32 {
            let x = i as f32 / 65535.0;
            let mut v = lin.evaluate_value(x);
            if !v.is_finite() {
                v = 0.0;
            }
            // Enforce a monotone table so the inverse search is well defined.
            running = running.max(v.clamp(0.0, 1.0));
            decode_u16.push(running);
        }
        let mut guide = Vec::with_capacity(ENCODE_GUIDE_STEPS + 2);
        let mut code = 0usize;
        for b in 0..=ENCODE_GUIDE_STEPS {
            let threshold = b as f32 / ENCODE_GUIDE_STEPS as f32;
            while code < 65535 && decode_u16[code] < threshold {
                code += 1;
            }
            guide.push(code as u32);
        }
        guide.push(65535);
        Ok(Arc::new(Self { decode_u16, guide }))
    }

    #[inline]
    fn decode_u16(&self, v: u16) -> f32 {
        self.decode_u16[v as usize]
    }

    /// Exact inverse of `decode_u16` with sub-code interpolation; returns encoded `[0, 1]`.
    #[inline]
    fn encode(&self, linear: f32) -> f32 {
        let l = if linear.is_finite() {
            linear.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let t = &self.decode_u16;
        if l <= t[0] {
            return 0.0;
        }
        if l >= t[65535] {
            return 1.0;
        }
        let b = ((l * ENCODE_GUIDE_STEPS as f32) as usize).min(ENCODE_GUIDE_STEPS - 1);
        let lo = (self.guide[b] as usize).saturating_sub(1);
        let hi = (self.guide[b + 1] as usize + 1).min(65535);
        // First code in [lo, hi] whose value is >= l.
        let idx = lo + t[lo..=hi].partition_point(|&v| v < l);
        let idx = idx.clamp(1, 65535);
        let v0 = t[idx - 1];
        let v1 = t[idx];
        let frac = if v1 > v0 { (l - v0) / (v1 - v0) } else { 0.0 };
        ((idx - 1) as f32 + frac.clamp(0.0, 1.0)) / 65535.0
    }
}

/// Precomputed linearization and inverse OETF LUTs for a target ICC profile.
#[derive(Clone)]
pub struct TargetTrc {
    channels: [Arc<ChannelTrcLut>; 3],
}

impl TargetTrc {
    pub fn srgb() -> Self {
        Self::try_from_profile(&ColorProfile::new_srgb()).expect("built-in sRGB has a valid TRC")
    }

    /// Build exact tone tables for a matrix/TRC RGB profile. Profiles without per-channel
    /// tone curves (LUT-only ICC) are rejected instead of silently assuming gamma 2.2.
    pub fn try_from_profile(profile: &ColorProfile) -> Result<Self, String> {
        let r_trc = profile.red_trc.as_ref();
        let g_trc = profile.green_trc.as_ref();
        let b_trc = profile.blue_trc.as_ref();
        let (Some(r_curve), Some(g_curve), Some(b_curve)) = (r_trc, g_trc, b_trc) else {
            return Err(
                "Target ICC profile has no RGB tone curves (LUT-only profiles are not supported as panorama target)"
                    .into(),
            );
        };
        let r = ChannelTrcLut::build(r_curve)?;
        let g = if g_trc == r_trc {
            Arc::clone(&r)
        } else {
            ChannelTrcLut::build(g_curve)?
        };
        let b = if b_trc == r_trc {
            Arc::clone(&r)
        } else if b_trc == g_trc {
            Arc::clone(&g)
        } else {
            ChannelTrcLut::build(b_curve)?
        };
        Ok(Self {
            channels: [r, g, b],
        })
    }

    #[inline]
    pub fn decode_rgb16_to_linear(&self, rgb: [u16; 3]) -> [f32; 3] {
        [
            self.channels[0].decode_u16(rgb[0]),
            self.channels[1].decode_u16(rgb[1]),
            self.channels[2].decode_u16(rgb[2]),
        ]
    }

    #[inline]
    pub fn decode_rgb8_to_linear(&self, rgb: [u8; 3]) -> [f32; 3] {
        [
            self.channels[0].decode_u16(u16::from(rgb[0]) * 257),
            self.channels[1].decode_u16(u16::from(rgb[1]) * 257),
            self.channels[2].decode_u16(u16::from(rgb[2]) * 257),
        ]
    }

    #[inline]
    pub fn encode_linear_to_f32(&self, linear: [f32; 3]) -> [f32; 3] {
        [
            self.channels[0].encode(linear[0]),
            self.channels[1].encode(linear[1]),
            self.channels[2].encode(linear[2]),
        ]
    }

    #[inline]
    pub fn encode_linear_to_u8(&self, linear: [f32; 3]) -> [u8; 3] {
        let enc = self.encode_linear_to_f32(linear);
        [
            (enc[0] * 255.0).round().clamp(0.0, 255.0) as u8,
            (enc[1] * 255.0).round().clamp(0.0, 255.0) as u8,
            (enc[2] * 255.0).round().clamp(0.0, 255.0) as u8,
        ]
    }

    #[inline]
    pub fn encode_linear_to_10bit_u16(&self, linear: [f32; 3]) -> [u16; 3] {
        let enc = self.encode_linear_to_f32(linear);
        [
            quantize_10bit_to_u16(enc[0]),
            quantize_10bit_to_u16(enc[1]),
            quantize_10bit_to_u16(enc[2]),
        ]
    }

    #[inline]
    pub fn encode_linear_to_u16(&self, linear: [f32; 3]) -> [u16; 3] {
        let enc = self.encode_linear_to_f32(linear);
        [
            (enc[0] * 65535.0).round().clamp(0.0, 65535.0) as u16,
            (enc[1] * 65535.0).round().clamp(0.0, 65535.0) as u16,
            (enc[2] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ]
    }

    /// Generate 8-bit preview GPU LUTs: `e2l` (`256` entries of `vec4<f32>` RGB linear `[0..1]`)
    /// and `thr` (`256` entries): `thr[k]` is the linear value at which the exactly rounded 8-bit
    /// output switches from code `k - 1` to `k` (`thr[0] = 0`). The shader counts thresholds by
    /// binary search, which reproduces the CPU `encode_linear_to_u8` quantization.
    pub fn gpu_tables(&self) -> ([[f32; 4]; 256], [[f32; 4]; 256]) {
        let mut e2l = [[0.0f32; 4]; 256];
        for (i, slot) in e2l.iter_mut().enumerate() {
            let v16 = (i as u16) * 257;
            *slot = [
                self.channels[0].decode_u16(v16),
                self.channels[1].decode_u16(v16),
                self.channels[2].decode_u16(v16),
                0.0,
            ];
        }
        let mut thr = [[0.0f32; 4]; 256];
        for (k, slot) in thr.iter_mut().enumerate().skip(1) {
            // Encoded midpoint between code k-1 and k, mapped back to linear light.
            let mid = ((k as f32 - 0.5) / 255.0 * 65535.0).clamp(0.0, 65535.0);
            let i0 = mid.floor() as usize;
            let i1 = (i0 + 1).min(65535);
            let f = mid - i0 as f32;
            for (dst, channel) in slot.iter_mut().zip(self.channels.iter()) {
                let t = &channel.decode_u16;
                *dst = t[i0] * (1.0 - f) + t[i1] * f;
            }
        }
        (e2l, thr)
    }
}

pub fn encode_base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

pub fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let mut vals = Vec::with_capacity(input.len());
    let mut padding = 0usize;
    for byte in input.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            padding += 1;
            vals.push(0u8);
            continue;
        }
        if padding > 0 {
            return Err("Invalid base64 padding".into());
        }
        let val = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err("Invalid character in base64 ICC payload".into()),
        };
        vals.push(val);
    }
    if vals.len() % 4 != 0 || padding > 2 {
        return Err("Malformed base64 ICC payload".into());
    }
    let mut out = Vec::with_capacity((vals.len() / 4) * 3);
    let chunks = vals.len() / 4;
    for (idx, chunk) in vals.as_chunks::<4>().0.iter().enumerate() {
        let n = (u32::from(chunk[0]) << 18)
            | (u32::from(chunk[1]) << 12)
            | (u32::from(chunk[2]) << 6)
            | u32::from(chunk[3]);
        out.push((n >> 16) as u8);
        let pad_here = if idx + 1 == chunks { padding } else { 0 };
        if pad_here < 2 {
            out.push((n >> 8) as u8);
        }
        if pad_here == 0 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_10bit_has_1024_reversible_levels() {
        let mut seen = std::collections::HashSet::new();
        for q in 0..1024u16 {
            let v16 = quantize_10bit_to_u16(q as f32 / 1023.0);
            assert_eq!(u16_to_10bit_level(v16), q);
            seen.insert(v16);
        }
        assert_eq!(seen.len(), 1024);
        assert_eq!(quantize_10bit_to_u16(0.0), 0);
        assert_eq!(quantize_10bit_to_u16(1.0), 65535);
    }

    #[test]
    fn built_in_profiles_encode_and_roundtrip_trc() {
        for id in ["srgb", "adobe_rgb", "display_p3", "prophoto_rgb", "bt2020"] {
            let target = TargetColorProfile::from_parts(id, None, None).unwrap();
            let prof = target.to_moxcms_profile().unwrap();
            let encoded = target.icc_bytes().unwrap();
            assert!(!encoded.is_empty());
            assert_eq!(detect_preset_match(&prof, &encoded), Some(id));
            let trc = TargetTrc::try_from_profile(&prof).unwrap();
            for probe in [0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
                let v16 = (probe * 65535.0).round() as u16;
                let lin = trc.decode_rgb16_to_linear([v16, v16, v16]);
                let back = trc.encode_linear_to_f32(lin);
                assert!(
                    (back[0] - probe).abs() < 1e-3,
                    "profile={id} probe={probe} back={}",
                    back[0]
                );
            }
        }
    }

    #[test]
    fn validates_supported_bit_depths_and_full_16bit_encoding() {
        for bits in [8u8, 10, 16] {
            assert_eq!(validate_bit_depth(bits).unwrap(), bits);
        }
        assert!(validate_bit_depth(12).is_err());
        let trc = TargetTrc::srgb();
        assert_eq!(trc.encode_linear_to_u16([0.0, 0.5, 1.0])[0], 0);
        assert_eq!(trc.encode_linear_to_u16([0.0, 0.5, 1.0])[2], 65535);
    }

    #[test]
    fn exact_inverse_trc_roundtrips_every_16bit_code_near_black() {
        for id in ["srgb", "adobe_rgb", "display_p3", "prophoto_rgb", "bt2020"] {
            let prof = TargetColorProfile::from_parts(id, None, None)
                .unwrap()
                .to_moxcms_profile()
                .unwrap();
            let trc = TargetTrc::try_from_profile(&prof).unwrap();
            let mut worst = 0i32;
            for code in (0..=65535u32).step_by(7).chain(0..4096) {
                let v = code as u16;
                let lin = trc.decode_rgb16_to_linear([v; 3]);
                let back = trc.encode_linear_to_u16(lin)[0];
                // Flat segments (identical linear values) may legitimately map to the first code.
                let back_lin = trc.decode_rgb16_to_linear([back; 3])[0];
                if back_lin != lin[0] {
                    worst = worst.max((i32::from(back) - code as i32).abs());
                }
            }
            assert!(
                worst <= 1,
                "profile={id} worst 16-bit roundtrip error {worst}"
            );
            for v8 in 0..=255u8 {
                let lin = trc.decode_rgb8_to_linear([v8; 3]);
                assert_eq!(
                    trc.encode_linear_to_u8(lin)[0],
                    v8,
                    "profile={id} rgb8={v8}"
                );
            }
        }
    }

    #[test]
    fn gpu_thresholds_reproduce_cpu_8bit_quantization() {
        for id in ["srgb", "adobe_rgb", "prophoto_rgb"] {
            let prof = TargetColorProfile::from_parts(id, None, None)
                .unwrap()
                .to_moxcms_profile()
                .unwrap();
            let trc = TargetTrc::try_from_profile(&prof).unwrap();
            let (_, thr) = trc.gpu_tables();
            for i in 0..=20000u32 {
                let lin = (i as f32 / 20000.0).powi(3);
                let gpu = thr.iter().skip(1).filter(|t| t[0] <= lin).count() as i32;
                let cpu = i32::from(trc.encode_linear_to_u8([lin; 3])[0]);
                assert!(
                    (gpu - cpu).abs() <= 1,
                    "profile={id} lin={lin} gpu={gpu} cpu={cpu}"
                );
            }
        }
    }

    #[test]
    fn linear_srgb_icc_is_not_replaced_by_srgb_preset() {
        let mut linear = ColorProfile::new_srgb();
        let gamma_one = ToneReprCurve::Parametric(vec![1.0]);
        linear.red_trc = Some(gamma_one.clone());
        linear.green_trc = Some(gamma_one.clone());
        linear.blue_trc = Some(gamma_one);
        let bytes = linear.encode().unwrap();
        let parsed = ColorProfile::new_from_slice(&bytes).unwrap();
        assert_eq!(detect_preset_match(&parsed, &bytes), None);
        let target =
            TargetColorProfile::from_raw_icc(Some("Linear sRGB working space"), bytes.clone())
                .unwrap();
        assert_eq!(target.id(), "custom_icc");
        assert_eq!(target.icc_bytes().unwrap(), bytes);
        // A re-encoded true sRGB profile is still recognized colorimetrically.
        let srgb_bytes = ColorProfile::new_srgb().encode().unwrap();
        let srgb = ColorProfile::new_from_slice(&srgb_bytes).unwrap();
        assert_eq!(detect_preset_match(&srgb, &srgb_bytes), Some("srgb"));
    }

    #[test]
    fn built_in_icc_is_structurally_normalized() {
        for id in ["srgb", "adobe_rgb", "display_p3", "prophoto_rgb", "bt2020"] {
            let target = TargetColorProfile::from_parts(id, None, None).unwrap();
            let icc = target.icc_bytes().unwrap();
            let be = |o: usize| u32::from_be_bytes(icc[o..o + 4].try_into().unwrap());
            assert_eq!(be(0) as usize, icc.len());
            assert_eq!(
                &icc[68..80],
                &[0, 0, 0xF6, 0xD6, 0, 1, 0, 0, 0, 0, 0xD3, 0x2D]
            );
            for k in 0..be(128) as usize {
                assert_eq!(
                    be(132 + 12 * k + 4) % 4,
                    0,
                    "profile={id} tag {k} misaligned"
                );
            }
            let parsed = ColorProfile::new_from_slice(&icc).unwrap();
            assert_eq!(detect_preset_match(&parsed, &icc), Some(id));
        }
    }

    #[test]
    fn base64_roundtrips_and_rejects_bad_input() {
        let data = b"Hello ICC profile \x00\xff\x80!";
        let b64 = encode_base64(data);
        assert_eq!(decode_base64(&b64).unwrap(), data);
        assert!(decode_base64("abc").is_err());
        assert!(decode_base64("ab=c").is_err());
    }
}
