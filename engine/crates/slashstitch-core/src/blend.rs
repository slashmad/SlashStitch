use crate::color::TargetTrc;
use crate::project::{PanoramaProjection, SourceImage};
use crate::raster::Rgb16Raster;
use crate::render::{Interpolation, RenderInput, RenderPlan, DEFAULT_BLEND_WIDTH_DEG};

#[derive(Debug, Clone, Copy)]
pub struct BandViewParams {
    pub view_yaw_deg: f64,
    pub view_pitch_deg: f64,
    pub view_v_shift: f64,
    /// Normalized panorama crop box [left, top, right, bottom] in [0, 1]^4.
    pub crop: [f64; 4],
    pub canvas_aspect: Option<f64>,
}

impl Default for BandViewParams {
    fn default() -> Self {
        Self {
            view_yaw_deg: 0.0,
            view_pitch_deg: 0.0,
            view_v_shift: 0.0,
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: None,
        }
    }
}

/// Sample a 16-bit-capable RGB image with bilinear interpolation in linear light `[0.0, 1.0]`.
#[inline]
pub fn sample_bilinear_rgb16(
    img: &impl Rgb16Raster,
    px: f64,
    py: f64,
    trc: &TargetTrc,
) -> [f32; 3] {
    let w = img.width();
    let h = img.height();
    let x0 = (px.floor() as u32).min(w - 1);
    let y0 = (py.floor() as u32).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (px - x0 as f64) as f32;
    let fy = (py - y0 as f64) as f32;

    let pixel = |x: u32, y: u32| trc.decode_rgb16_to_linear(img.sample_rgb16(x, y));
    let p00 = pixel(x0, y0);
    let p10 = pixel(x1, y0);
    let p01 = pixel(x0, y1);
    let p11 = pixel(x1, y1);

    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Render a horizontal band `[y_start .. y_end)` of the panorama into sRGB RGB8 bytes.
#[allow(clippy::too_many_arguments)]
pub fn render_band(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
) -> Vec<u8> {
    render_band_8bit_with_view(
        out_w,
        out_h,
        y_start,
        y_end,
        projection,
        pano_hfov_deg,
        &BandViewParams::default(),
        images,
        loaded,
        multiband,
        &TargetTrc::srgb(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn render_band_with_view(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view: &BandViewParams,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
) -> Vec<u8> {
    render_band_8bit_with_view(
        out_w,
        out_h,
        y_start,
        y_end,
        projection,
        pano_hfov_deg,
        view,
        images,
        loaded,
        multiband,
        &TargetTrc::srgb(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn render_band_8bit_with_view(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view: &BandViewParams,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
    trc: &TargetTrc,
) -> Vec<u8> {
    let strip_h = (y_end - y_start) as usize;
    let wi = out_w as usize;
    let (linear, valid) = render_band_linear(
        out_w,
        out_h,
        y_start,
        y_end,
        projection,
        pano_hfov_deg,
        view,
        images,
        loaded,
        multiband,
        trc,
    );
    let mut out = vec![0u8; wi * strip_h * 3];
    for (idx, (&px, &ok)) in linear.iter().zip(valid.iter()).enumerate() {
        if ok {
            let encoded = trc.encode_linear_to_u8(px);
            let dst = idx * 3;
            out[dst] = encoded[0];
            out[dst + 1] = encoded[1];
            out[dst + 2] = encoded[2];
        }
    }
    out
}

/// Render a horizontal band `[y_start .. y_end)` into 16-bit unsigned RGB words (`0..=65535`)
/// quantized to either 10-bit (`1024` levels/ch) or full 16-bit (`65536` levels/ch).
#[allow(clippy::too_many_arguments)]
pub fn render_band_u16_with_view(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view: &BandViewParams,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
    bit_depth: u8,
    trc: &TargetTrc,
) -> Vec<u16> {
    let strip_h = (y_end - y_start) as usize;
    let wi = out_w as usize;
    let (linear, valid) = render_band_linear(
        out_w,
        out_h,
        y_start,
        y_end,
        projection,
        pano_hfov_deg,
        view,
        images,
        loaded,
        multiband,
        trc,
    );
    let mut out = vec![0u16; wi * strip_h * 3];
    for (idx, (&px, &ok)) in linear.iter().zip(valid.iter()).enumerate() {
        if ok {
            let encoded = if bit_depth == 10 {
                trc.encode_linear_to_10bit_u16(px)
            } else {
                trc.encode_linear_to_u16(px)
            };
            let dst = idx * 3;
            out[dst] = encoded[0];
            out[dst + 1] = encoded[1];
            out[dst + 2] = encoded[2];
        }
    }
    out
}

/// Render a horizontal band `[y_start .. y_end)` quantized to 10-bit precision (`1024` levels/ch)
/// stored inside 16-bit unsigned RGB words (`0..=65535`).
#[allow(clippy::too_many_arguments)]
pub fn render_band_10bit_with_view(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view: &BandViewParams,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
    trc: &TargetTrc,
) -> Vec<u16> {
    render_band_u16_with_view(
        out_w,
        out_h,
        y_start,
        y_end,
        projection,
        pano_hfov_deg,
        view,
        images,
        loaded,
        multiband,
        10,
        trc,
    )
}

/// Render a horizontal band `[y_start .. y_end)` with full 16-bit precision (`65536` levels/ch).
#[allow(clippy::too_many_arguments)]
pub fn render_band_16bit_with_view(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view: &BandViewParams,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
    trc: &TargetTrc,
) -> Vec<u16> {
    render_band_u16_with_view(
        out_w,
        out_h,
        y_start,
        y_end,
        projection,
        pano_hfov_deg,
        view,
        images,
        loaded,
        multiband,
        16,
        trc,
    )
}

#[allow(clippy::too_many_arguments)]
fn render_band_linear(
    out_w: u32,
    out_h: u32,
    y_start: u32,
    y_end: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view: &BandViewParams,
    images: &[SourceImage],
    loaded: &[impl Rgb16Raster],
    multiband: bool,
    trc: &TargetTrc,
) -> (Vec<[f32; 3]>, Vec<bool>) {
    let n = images.len().min(loaded.len());
    let input = RenderInput {
        out_w,
        out_h,
        projection,
        pano_hfov_deg,
        view: *view,
        images: &images[..n],
        sources: loaded[..n]
            .iter()
            .map(|r| vec![r as &dyn Rgb16Raster])
            .collect(),
        photometric: None,
        multiband,
        trc,
        interpolation: Interpolation::Spline36,
        blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        reject_transients: false,
        seams: None,
    };
    let plan = RenderPlan::new(&input);
    plan.render_rows(&input, y_start, y_end, None)
        .unwrap_or_else(|| {
            let len = out_w as usize * (y_end - y_start) as usize;
            (vec![[0.0; 3]; len], vec![false; len])
        })
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::project::LensProjection;
    use image::{Rgb, RgbImage};
    fn source() -> SourceImage {
        SourceImage {
            id: 0,
            path: String::new(),
            width: 240,
            height: 160,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 95.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        }
    }
    #[test]
    fn feather_blends_in_linear_light() {
        let source = source();
        let black = RgbImage::from_pixel(240, 160, Rgb([0, 0, 0]));
        let white = RgbImage::from_pixel(240, 160, Rgb([255, 255, 255]));
        let result = render_band(
            64,
            64,
            0,
            64,
            PanoramaProjection::Rectilinear,
            45.0,
            &[source.clone(), source],
            &[black, white],
            false,
        );
        let at = (32 * 64 + 32) * 3;
        assert!(
            (187..=189).contains(&result[at]),
            "linear mix should be ~188 sRGB, got {}",
            result[at]
        );
    }
    #[test]
    fn exported_geometry_uses_preview_canvas_aspect() {
        let image = RgbImage::from_fn(240, 160, |x, y| Rgb([x as u8, y as u8, 100]));
        let source = source();
        let view = BandViewParams {
            canvas_aspect: Some(1.5),
            ..Default::default()
        };
        let one = render_band_with_view(
            101,
            101,
            0,
            101,
            PanoramaProjection::Rectilinear,
            70.0,
            &view,
            std::slice::from_ref(&source),
            std::slice::from_ref(&image),
            false,
        );
        let two = render_band_with_view(
            101,
            201,
            0,
            201,
            PanoramaProjection::Rectilinear,
            70.0,
            &view,
            std::slice::from_ref(&source),
            std::slice::from_ref(&image),
            false,
        );
        let a = (35 * 101 + 60) * 3;
        let b = (70 * 101 + 60) * 3;
        assert!(one[a].abs_diff(two[b]) <= 2 && one[a + 1].abs_diff(two[b + 1]) <= 2);
    }
}
