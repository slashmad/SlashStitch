use crate::project::{equirect_uv_to_ray, LensProjection, PanoramaProjection, SourceImage};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// Build camera-to-world rotation matrix from yaw, pitch, roll in degrees.
/// Order: R = Ry(yaw) * Rx(-pitch) * Rz(roll) so positive yaw turns right (+X),
/// positive pitch looks up (+Y), positive roll tilts clockwise.
pub fn rotation_matrix(yaw_deg: f64, pitch_deg: f64, roll_deg: f64) -> Matrix3<f64> {
    let y = yaw_deg.to_radians();
    let p = (-pitch_deg).to_radians();
    let r = roll_deg.to_radians();

    let ry = Matrix3::new(y.cos(), 0.0, y.sin(), 0.0, 1.0, 0.0, -y.sin(), 0.0, y.cos());
    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, p.cos(), -p.sin(), 0.0, p.sin(), p.cos());
    let rz = Matrix3::new(r.cos(), -r.sin(), 0.0, r.sin(), r.cos(), 0.0, 0.0, 0.0, 1.0);
    ry * rx * rz
}

/// Rotate a world-space unit ray by viewport yaw and pitch offsets (degrees).
#[inline]
pub fn rotate_view_ray(ray: [f64; 3], yaw_deg: f64, pitch_deg: f64) -> [f64; 3] {
    if yaw_deg == 0.0 && pitch_deg == 0.0 {
        return ray;
    }
    let v = rotation_matrix(yaw_deg, pitch_deg, 0.0) * Vector3::from(ray);
    [v.x, v.y, v.z]
}

/// Shift normalized vertical canvas coordinate for horizon-preserving projections.
#[inline]
pub fn shift_pano_v(projection: PanoramaProjection, v: f64, view_v_shift: f64) -> f64 {
    if view_v_shift == 0.0 {
        return v;
    }
    match projection {
        PanoramaProjection::Cylindrical
        | PanoramaProjection::Mercator
        | PanoramaProjection::Rectilinear
        | PanoramaProjection::Vedutismo
        | PanoramaProjection::Stereographic
        | PanoramaProjection::TransverseCylindrical
        | PanoramaProjection::TransverseMercator
        | PanoramaProjection::TransverseVedutismo => v - view_v_shift,
        PanoramaProjection::Equirectangular
        | PanoramaProjection::StereographicDown
        | PanoramaProjection::CircularFisheye
        | PanoramaProjection::FullframeFisheye
        | PanoramaProjection::TransverseEquirectangular => v,
    }
}

/// Map normalized panorama canvas coordinates (u, v) in [0, 1]^2 with vertical shift to a world ray.
#[inline]
pub fn pano_uv_to_ray_shifted(
    projection: PanoramaProjection,
    u: f64,
    v: f64,
    pano_hfov_deg: f64,
    aspect: f64,
    view_v_shift: f64,
) -> Option<[f64; 3]> {
    pano_uv_to_ray(
        projection,
        u,
        shift_pano_v(projection, v, view_v_shift),
        pano_hfov_deg,
        aspect,
    )
}

/// Map normalized panorama canvas coordinates (u, v) in [0, 1]^2 to a world 3D unit ray.
pub fn pano_uv_to_ray(
    projection: PanoramaProjection,
    u: f64,
    v: f64,
    pano_hfov_deg: f64,
    aspect: f64,
) -> Option<[f64; 3]> {
    let asp = aspect.max(0.1);
    match projection {
        PanoramaProjection::Equirectangular => Some(equirect_uv_to_ray(u, v)),
        PanoramaProjection::Cylindrical => {
            let lon = (u - 0.5) * 2.0 * PI;
            let y_cyl = (0.5 - v) * (2.0 * PI / asp);
            let norm = (1.0 + y_cyl * y_cyl).sqrt();
            Some([lon.sin() / norm, y_cyl / norm, lon.cos() / norm])
        }
        PanoramaProjection::Mercator => {
            let lon = (u - 0.5) * 2.0 * PI;
            let y_m = (0.5 - v) * (2.0 * PI / asp);
            let cos_lat = 1.0 / y_m.cosh();
            let sin_lat = y_m.tanh();
            Some([cos_lat * lon.sin(), sin_lat, cos_lat * lon.cos()])
        }
        PanoramaProjection::Rectilinear => {
            let hfov = pano_hfov_deg.clamp(10.0, 150.0).to_radians();
            let half_w = (hfov * 0.5).tan();
            let half_h = half_w / asp;
            let x = (u - 0.5) * 2.0 * half_w;
            let y = (0.5 - v) * 2.0 * half_h;
            let norm = (x * x + y * y + 1.0).sqrt();
            Some([x / norm, y / norm, 1.0 / norm])
        }
        PanoramaProjection::Vedutismo => {
            // Panini d = 1 (Vedutismo): preserves vertical lines and all straight lines through center.
            let scale = (pano_hfov_deg.clamp(20.0, 320.0).to_radians() * 0.25).tan();
            let sx = (u - 0.5) * 2.0 * scale;
            let sy = (0.5 - v) * 2.0 * scale / asp;
            let den = 1.0 + sx * sx;
            let rx = (2.0 * sx) / den;
            let rz = (1.0 - sx * sx) / den;
            let ry = (2.0 * sy) / den;
            let norm = (rx * rx + ry * ry + rz * rz).sqrt();
            Some([rx / norm, ry / norm, rz / norm])
        }
        PanoramaProjection::Stereographic => {
            Some(stereographic_uv_to_ray(u, v, pano_hfov_deg, asp))
        }
        PanoramaProjection::StereographicDown => {
            let [rx, ry, rz] = stereographic_uv_to_ray(u, v, pano_hfov_deg, asp);
            // Rotate -90 deg around X so center looks straight down at nadir (Little Planet).
            Some([rx, -rz, ry])
        }
        PanoramaProjection::CircularFisheye => {
            let half_rad = pano_hfov_deg.clamp(20.0, 360.0).to_radians() * 0.5;
            let min_span = asp.min(1.0);
            let px = (u - 0.5) * 2.0 * asp / min_span;
            let py = (0.5 - v) * 2.0 / min_span;
            let r = px.hypot(py);
            if r > 1.0 {
                return None;
            }
            if r < 1e-12 {
                return Some([0.0, 0.0, 1.0]);
            }
            let theta = r * half_rad;
            let s = theta.sin() / r;
            Some([px * s, py * s, theta.cos()])
        }
        PanoramaProjection::FullframeFisheye => {
            let half_rad = pano_hfov_deg.clamp(20.0, 360.0).to_radians() * 0.5;
            let px = (u - 0.5) * 2.0;
            let py = (0.5 - v) * 2.0 / asp;
            let r = px.hypot(py);
            let theta = r * half_rad;
            if theta > PI {
                return None;
            }
            if r < 1e-12 {
                return Some([0.0, 0.0, 1.0]);
            }
            let s = theta.sin() / r;
            Some([px * s, py * s, theta.cos()])
        }
        PanoramaProjection::TransverseEquirectangular => transverse_wrap(
            PanoramaProjection::Equirectangular,
            u,
            v,
            pano_hfov_deg,
            asp,
        ),
        PanoramaProjection::TransverseCylindrical => {
            transverse_wrap(PanoramaProjection::Cylindrical, u, v, pano_hfov_deg, asp)
        }
        PanoramaProjection::TransverseMercator => {
            transverse_wrap(PanoramaProjection::Mercator, u, v, pano_hfov_deg, asp)
        }
        PanoramaProjection::TransverseVedutismo => {
            transverse_wrap(PanoramaProjection::Vedutismo, u, v, pano_hfov_deg, asp)
        }
    }
}

#[inline]
fn stereographic_uv_to_ray(u: f64, v: f64, pano_hfov_deg: f64, aspect: f64) -> [f64; 3] {
    let scale = (pano_hfov_deg.clamp(20.0, 340.0).to_radians() * 0.25).tan();
    let sx = (u - 0.5) * 2.0 * scale;
    let sy = (0.5 - v) * 2.0 * scale / aspect;
    let r2 = sx * sx + sy * sy;
    let denom = 1.0 + r2;
    [2.0 * sx / denom, 2.0 * sy / denom, (1.0 - r2) / denom]
}

#[inline]
fn transverse_wrap(
    base: PanoramaProjection,
    u: f64,
    v: f64,
    pano_hfov_deg: f64,
    aspect: f64,
) -> Option<[f64; 3]> {
    let [rx, ry, rz] = pano_uv_to_ray(base, 1.0 - v, u, pano_hfov_deg, 1.0 / aspect)?;
    Some([-ry, rx, rz])
}

/// Apply PTLens radial distortion: r_src = r * (a*r^3 + b*r^2 + c*r + (1 - a - b - c)).
#[inline]
pub fn apply_ptlens_distortion(r_norm: f64, [a, b, c]: [f64; 3]) -> f64 {
    if a == 0.0 && b == 0.0 && c == 0.0 {
        return r_norm;
    }
    let d = 1.0 - a - b - c;
    let r2 = r_norm * r_norm;
    let r3 = r2 * r_norm;
    r_norm * (a * r3 + b * r2 + c * r_norm + d)
}

/// Inverse PTLens distortion using 4 Newton-Raphson iterations.
#[inline]
pub fn invert_ptlens_distortion(r_src_norm: f64, [a, b, c]: [f64; 3]) -> f64 {
    if a == 0.0 && b == 0.0 && c == 0.0 {
        return r_src_norm;
    }
    let d = 1.0 - a - b - c;
    let mut r = r_src_norm;
    for _ in 0..4 {
        let r2 = r * r;
        let r3 = r2 * r;
        let f = r * (a * r3 + b * r2 + c * r + d) - r_src_norm;
        let df = 4.0 * a * r3 + 3.0 * b * r2 + 2.0 * c * r + d;
        if df.abs() < 1e-8 {
            break;
        }
        r -= f / df;
    }
    r
}

/// Evaluate distortion in its calibration's normalized coordinate system.
pub fn distort_radius(r: f64, img: &SourceImage) -> f64 {
    let scale = img.distortion_scale.max(1e-6);
    let rc = r * scale;
    match img.distortion_poly5 {
        Some([k1, k2]) => r * (1.0 + k1 * rc * rc + k2 * rc.powi(4)),
        None => apply_ptlens_distortion(rc, img.distortion) / scale,
    }
}

pub fn undistort_radius(radius: f64, img: &SourceImage) -> f64 {
    let mut r = radius;
    for _ in 0..12 {
        let eps = 1e-6;
        let f = distort_radius(r, img) - radius;
        let df = (distort_radius(r + eps, img) - distort_radius(r - eps, img)) / (2.0 * eps);
        if df.abs() < 1e-10 {
            break;
        }
        let next = r - f / df;
        if !next.is_finite() || next < 0.0 {
            break;
        }
        r = next;
        if f.abs() < 1e-10 {
            break;
        }
    }
    r
}

/// Precomputed world-to-image projection for one source image.
///
/// Construction evaluates the rotation matrix and lens constants once; `project` is then a
/// handful of multiplications plus the lens-specific angle math. Results are identical to the
/// historical per-call `ray_to_image_px`.
#[derive(Debug, Clone)]
pub struct ImageProjector {
    inv: [[f64; 3]; 3],
    lens: LensProjection,
    hfov: f64,
    w: f64,
    h: f64,
    half_min: f64,
    f_rect: f64,
    f_equisolid: f64,
    max_theta: f64,
    distortion: [f64; 3],
    poly5: Option<[f64; 2]>,
    distortion_scale: f64,
    trivial_distortion: bool,
}

impl ImageProjector {
    pub fn new(img: &SourceImage) -> Self {
        let r = rotation_matrix(img.yaw_deg, img.pitch_deg, img.roll_deg).transpose();
        let w = img.width as f64;
        let h = img.height as f64;
        let hfov = img
            .hfov_deg
            .clamp(
                1.0,
                if img.lens_projection == LensProjection::Rectilinear {
                    175.0
                } else {
                    359.0
                },
            )
            .to_radians();
        Self {
            inv: [
                [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
                [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
                [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
            ],
            lens: img.lens_projection,
            hfov,
            w,
            h,
            half_min: (w.min(h) * 0.5).max(1.0),
            f_rect: 0.5 / (hfov * 0.5).tan(),
            f_equisolid: 0.5 / (2.0 * (hfov * 0.25).sin()).max(1e-6),
            max_theta: (hfov * 0.5 * (w.hypot(h) / w.max(1.0)).max(1.0) * 1.05).min(PI * 0.99),
            distortion: img.distortion,
            poly5: img.distortion_poly5,
            distortion_scale: img.distortion_scale.max(1e-6),
            trivial_distortion: img.distortion_poly5.is_none() && img.distortion == [0.0, 0.0, 0.0],
        }
    }

    pub fn width(&self) -> f64 {
        self.w
    }

    pub fn height(&self) -> f64 {
        self.h
    }

    #[inline]
    fn distort_scale(&self, r_norm: f64) -> f64 {
        if self.trivial_distortion || r_norm <= 1e-9 {
            return 1.0;
        }
        let rc = r_norm * self.distortion_scale;
        let distorted = match self.poly5 {
            Some([k1, k2]) => r_norm * (1.0 + k1 * rc * rc + k2 * rc.powi(4)),
            None => apply_ptlens_distortion(rc, self.distortion) / self.distortion_scale,
        };
        distorted / r_norm
    }

    /// Project a world unit ray to source pixel coordinates and an edge feather weight.
    #[inline]
    pub fn project(&self, ray_world: [f64; 3]) -> Option<([f64; 2], f64)> {
        let m = &self.inv;
        let cx = m[0][0] * ray_world[0] + m[0][1] * ray_world[1] + m[0][2] * ray_world[2];
        let cy = m[1][0] * ray_world[0] + m[1][1] * ray_world[1] + m[1][2] * ray_world[2];
        let cz = m[2][0] * ray_world[0] + m[2][1] * ray_world[1] + m[2][2] * ray_world[2];
        let (w, h) = (self.w, self.h);
        if w <= 1.0 || h <= 1.0 {
            return None;
        }
        let (nx, ny) = match self.lens {
            LensProjection::Rectilinear => {
                if cz <= 1e-4 {
                    return None;
                }
                (self.f_rect * cx / cz, self.f_rect * cy / cz)
            }
            LensProjection::FisheyeFullframe | LensProjection::FisheyeCircular => {
                let phi_r = cx.hypot(cy);
                let theta = phi_r.atan2(cz);
                if theta > self.max_theta {
                    return None;
                }
                if phi_r < 1e-12 {
                    (0.0, 0.0)
                } else {
                    let r = theta / self.hfov;
                    (r * cx / phi_r, r * cy / phi_r)
                }
            }
            LensProjection::FisheyeEquisolid => {
                let phi_r = cx.hypot(cy);
                if phi_r < 1e-12 {
                    if cz <= 0.0 {
                        return None;
                    }
                    (0.0, 0.0)
                } else {
                    let theta = phi_r.atan2(cz);
                    let r = 2.0 * self.f_equisolid * (theta * 0.5).sin();
                    (r * cx / phi_r, r * cy / phi_r)
                }
            }
            LensProjection::Equirectangular => {
                let lon = cx.atan2(cz);
                let lat = cy.clamp(-1.0, 1.0).asin();
                (lon / self.hfov, lat / self.hfov)
            }
        };

        let px_un = nx * w;
        let py_un = -ny * w;
        let r_norm = px_un.hypot(py_un) / self.half_min;
        let scale = self.distort_scale(r_norm);
        let px = w * 0.5 + px_un * scale;
        let py = h * 0.5 + py_un * scale;

        if px < 0.0 || py < 0.0 || px >= w - 1.0 || py >= h - 1.0 {
            return None;
        }
        if self.lens == LensProjection::FisheyeCircular {
            let dist_center = (px - w * 0.5).hypot(py - h * 0.5);
            if dist_center > self.half_min * 0.99 {
                return None;
            }
        }
        let dx = (px / w).min(1.0 - px / w) * 2.0;
        let dy = (py / h).min(1.0 - py / h) * 2.0;
        let edge = dx.min(dy).clamp(0.0, 1.0);
        let weight = edge * edge * (3.0 - 2.0 * edge);
        Some(([px, py], weight.max(1e-4)))
    }
}

/// Project a world unit ray into a source image's pixel coordinates (px, py) and edge feather weight.
pub fn ray_to_image_px(ray_world: [f64; 3], img: &SourceImage) -> Option<([f64; 2], f64)> {
    ImageProjector::new(img).project(ray_world)
}

/// Convert a pixel (px, py) in a source image into a 3D unit ray in world space.
pub fn image_px_to_world_ray(px: f64, py: f64, img: &SourceImage) -> Option<[f64; 3]> {
    let w = img.width as f64;
    let h = img.height as f64;
    if w <= 1.0 || h <= 1.0 {
        return None;
    }
    let half_min = (w.min(h) * 0.5).max(1.0);
    let dx = px - w * 0.5;
    let dy = py - h * 0.5;
    let r_src_norm = dx.hypot(dy) / half_min;
    let scale = if r_src_norm > 1e-9 {
        undistort_radius(r_src_norm, img) / r_src_norm
    } else {
        1.0
    };

    let nx = (dx * scale) / w;
    let ny = -(dy * scale) / w;
    let hfov = img
        .hfov_deg
        .clamp(
            1.0,
            if img.lens_projection == LensProjection::Rectilinear {
                175.0
            } else {
                359.0
            },
        )
        .to_radians();

    let cam_ray = match img.lens_projection {
        LensProjection::Rectilinear => {
            let f = 0.5 / (hfov * 0.5).tan();
            Vector3::new(nx, ny, f).normalize()
        }
        LensProjection::FisheyeFullframe | LensProjection::FisheyeCircular => {
            let r = nx.hypot(ny);
            if r < 1e-12 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                let theta = r * hfov;
                let s = theta.sin() / r;
                Vector3::new(nx * s, ny * s, theta.cos())
            }
        }
        LensProjection::FisheyeEquisolid => {
            let r = nx.hypot(ny);
            if r < 1e-12 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                let f_norm = 0.5 / (2.0 * (hfov * 0.25).sin()).max(1e-6);
                let arg = (r / (2.0 * f_norm)).clamp(-0.9999, 0.9999);
                let theta = 2.0 * arg.asin();
                let s = theta.sin() / r;
                Vector3::new(nx * s, ny * s, theta.cos())
            }
        }
        LensProjection::Equirectangular => {
            let lon = nx * hfov;
            let lat = (ny * hfov).clamp(-PI * 0.499, PI * 0.499);
            Vector3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos())
        }
    };

    let world = rotation_matrix(img.yaw_deg, img.pitch_deg, img.roll_deg) * cam_ray;
    let n = world.normalize();
    Some([n.x, n.y, n.z])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_img() -> SourceImage {
        SourceImage {
            id: 0,
            path: String::new(),
            width: 1000,
            height: 800,
            yaw_deg: 25.0,
            pitch_deg: -10.0,
            roll_deg: 5.0,
            hfov_deg: 65.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.005, -0.015, 0.002],
            distortion_poly5: None,
            distortion_scale: 1.0,
        }
    }

    #[test]
    fn image_ray_roundtrip_with_distortion() {
        let img = test_img();
        let ray = image_px_to_world_ray(640.0, 310.0, &img).unwrap();
        let ([px, py], _) = ray_to_image_px(ray, &img).unwrap();
        assert!((px - 640.0).abs() < 1e-2, "px={px}");
        assert!((py - 310.0).abs() < 1e-2, "py={py}");
    }

    #[test]
    fn fisheye_roundtrip() {
        let mut img = test_img();
        img.lens_projection = LensProjection::FisheyeFullframe;
        img.hfov_deg = 135.0;
        img.distortion = [0.0, 0.0, 0.0];
        let ray = image_px_to_world_ray(300.0, 250.0, &img).unwrap();
        let ([px, py], _) = ray_to_image_px(ray, &img).unwrap();
        assert!((px - 300.0).abs() < 1e-3);
        assert!((py - 250.0).abs() < 1e-3);
    }

    #[test]
    fn equisolid_fisheye_roundtrip() {
        let mut img = test_img();
        img.width = 5304;
        img.height = 7952;
        img.lens_projection = LensProjection::FisheyeEquisolid;
        img.hfov_deg = 94.31;
        img.distortion = [0.0, 0.0, 0.0];
        let ray = image_px_to_world_ray(4500.0, 6800.0, &img).unwrap();
        let ([px, py], _) = ray_to_image_px(ray, &img).unwrap();
        assert!((px - 4500.0).abs() < 1e-2, "px={px}");
        assert!((py - 6800.0).abs() < 1e-2, "py={py}");
    }

    #[test]
    fn poly5_calibration_scale_and_inverse_are_preserved() {
        let mut img = test_img();
        img.distortion = [0.0; 3];
        img.distortion_poly5 = Some([0.01, 0.02]);
        img.distortion_scale = 1.0;
        let expected = 1.2 * (1.0 + 0.01 * 1.2f64.powi(2) + 0.02 * 1.2f64.powi(4));
        assert!((distort_radius(1.2, &img) - expected).abs() < 1e-12);
        assert!((undistort_radius(expected, &img) - 1.2).abs() < 1e-8);
        img.distortion_scale = 0.5;
        let expected = 1.2 * (1.0 + 0.01 * 0.6f64.powi(2) + 0.02 * 0.6f64.powi(4));
        assert!((distort_radius(1.2, &img) - expected).abs() < 1e-12);
        let ray = image_px_to_world_ray(640.0, 310.0, &img).unwrap();
        let ([x, y], _) = ray_to_image_px(ray, &img).unwrap();
        assert!((x - 640.0).abs() < 0.01 && (y - 310.0).abs() < 0.01);
    }

    #[test]
    fn ptgui_panorama_projections_produce_unit_rays_and_symmetries() {
        let all = [
            PanoramaProjection::Equirectangular,
            PanoramaProjection::Cylindrical,
            PanoramaProjection::Mercator,
            PanoramaProjection::Rectilinear,
            PanoramaProjection::Vedutismo,
            PanoramaProjection::Stereographic,
            PanoramaProjection::StereographicDown,
            PanoramaProjection::CircularFisheye,
            PanoramaProjection::FullframeFisheye,
            PanoramaProjection::TransverseEquirectangular,
            PanoramaProjection::TransverseCylindrical,
            PanoramaProjection::TransverseMercator,
            PanoramaProjection::TransverseVedutismo,
        ];
        for proj in all {
            let center = pano_uv_to_ray(proj, 0.5, 0.5, 160.0, 2.0).unwrap();
            let norm =
                (center[0] * center[0] + center[1] * center[1] + center[2] * center[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "proj={proj:?} norm={norm}");
            if proj == PanoramaProjection::StereographicDown {
                assert!((center[1] + 1.0).abs() < 1e-6, "nadir center={center:?}");
            } else {
                assert!(
                    (center[2] - 1.0).abs() < 1e-6,
                    "proj={proj:?} center={center:?}"
                );
            }
            let r_right = pano_uv_to_ray(proj, 0.65, 0.5, 160.0, 2.0).unwrap();
            assert!(r_right[0] > 0.0, "proj={proj:?} right={r_right:?}");
        }
        // Circular fisheye corner outside inscribed circle is masked out.
        assert!(
            pano_uv_to_ray(PanoramaProjection::CircularFisheye, 0.02, 0.02, 180.0, 2.0).is_none()
        );
    }

    #[test]
    fn shift_pano_v_shifts_horizon_projections_without_touching_spherical() {
        let base_cyl =
            pano_uv_to_ray(PanoramaProjection::Cylindrical, 0.5, 0.5, 160.0, 2.0).unwrap();
        let shifted_cyl =
            pano_uv_to_ray_shifted(PanoramaProjection::Cylindrical, 0.5, 0.3, 160.0, 2.0, -0.2)
                .unwrap();
        assert!((base_cyl[1] - shifted_cyl[1]).abs() < 1e-12);
        assert!((base_cyl[2] - shifted_cyl[2]).abs() < 1e-12);

        let base_eq =
            pano_uv_to_ray(PanoramaProjection::Equirectangular, 0.5, 0.3, 160.0, 2.0).unwrap();
        let shifted_eq = pano_uv_to_ray_shifted(
            PanoramaProjection::Equirectangular,
            0.5,
            0.3,
            160.0,
            2.0,
            -0.2,
        )
        .unwrap();
        assert!((base_eq[1] - shifted_eq[1]).abs() < 1e-12);
    }
}
