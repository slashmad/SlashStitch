use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PanoramaProjection {
    #[default]
    Equirectangular,
    Cylindrical,
    Mercator,
    Rectilinear,
    Vedutismo,
    Stereographic,
    StereographicDown,
    CircularFisheye,
    FullframeFisheye,
    TransverseEquirectangular,
    TransverseCylindrical,
    TransverseMercator,
    TransverseVedutismo,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LensProjection {
    #[default]
    Rectilinear,
    FisheyeFullframe,
    FisheyeCircular,
    FisheyeEquisolid,
    Equirectangular,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceImage {
    pub id: usize,
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    pub roll_deg: f64,
    pub hfov_deg: f64,
    pub lens_projection: LensProjection,
    /// PTLens radial distortion parameters [a, b, c].
    #[serde(default)]
    pub distortion: [f64; 3],
    /// Poly5 is not a PTLens polynomial; preserve both coefficients.
    #[serde(default)]
    pub distortion_poly5: Option<[f64; 2]>,
    #[serde(default = "unit_scale")]
    pub distortion_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPoint {
    pub image_a: usize,
    pub x_a: f64,
    pub y_a: f64,
    pub image_b: usize,
    pub x_b: f64,
    pub y_b: f64,
    /// Relative optimization weight (1.0 default; higher for infinity/star matches).
    #[serde(default = "unit_scale", skip_serializing_if = "is_unit_weight")]
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StitchProject {
    pub name: String,
    pub projection: PanoramaProjection,
    pub output_width: u32,
    pub output_height: u32,
    pub images: Vec<SourceImage>,
    pub control_points: Vec<ControlPoint>,
}

fn unit_scale() -> f64 {
    1.0
}

fn is_unit_weight(w: &f64) -> bool {
    (*w - 1.0).abs() < 1e-12
}

/// Convert normalized equirectangular coordinates u in [0,1], v in [0,1] into a 3D unit ray.
/// Convention: +Z forward (yaw=0, pitch=0), +X right, +Y up.
pub fn equirect_uv_to_ray(u: f64, v: f64) -> [f64; 3] {
    let lon = (u - 0.5) * 2.0 * PI;
    let lat = (0.5 - v) * PI;
    let cos_lat = lat.cos();
    [cos_lat * lon.sin(), lat.sin(), cos_lat * lon.cos()]
}

/// Convert a 3D unit ray into normalized equirectangular coordinates (u, v) in [0, 1].
pub fn ray_to_equirect_uv(ray: [f64; 3]) -> [f64; 2] {
    let [x, y, z] = ray;
    let lon = x.atan2(z);
    let lat = y.clamp(-1.0, 1.0).asin();
    let u = lon / (2.0 * PI) + 0.5;
    let v = 0.5 - lat / PI;
    [u, v]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equirect_roundtrip_center() {
        let ray = equirect_uv_to_ray(0.5, 0.5);
        assert!((ray[0]).abs() < 1e-9);
        assert!((ray[1]).abs() < 1e-9);
        assert!((ray[2] - 1.0).abs() < 1e-9);
        let uv = ray_to_equirect_uv(ray);
        assert!((uv[0] - 0.5).abs() < 1e-9);
        assert!((uv[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn equirect_roundtrip_quarter() {
        let ray = equirect_uv_to_ray(0.75, 0.25);
        let uv = ray_to_equirect_uv(ray);
        assert!((uv[0] - 0.75).abs() < 1e-9);
        assert!((uv[1] - 0.25).abs() < 1e-9);
    }
}
