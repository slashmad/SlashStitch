use crate::optics::{image_px_to_world_ray, rotation_matrix};
use crate::project::{ControlPoint, SourceImage};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OptimizeOptions {
    pub optimize_hfov: bool,
    pub level_horizon: bool,
    #[serde(default = "default_true")]
    pub optimize_distortion: bool,
    #[serde(default = "default_true")]
    pub prioritize_sky: bool,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            optimize_hfov: true,
            level_horizon: true,
            optimize_distortion: true,
            prioritize_sky: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub initial_rms_deg: f64,
    pub final_rms_deg: f64,
    pub max_error_deg: f64,
    pub iterations: usize,
    pub control_points_used: usize,
    pub control_points_rejected: usize,
    pub optimized_hfov_deg: Option<f64>,
    #[serde(default)]
    pub optimized_distortion: Option<[f64; 3]>,
}

/// Seed initial (yaw, pitch, roll) by propagating pairwise rotations via BFS from image 0.
pub fn seed_poses_from_pairs(
    images: &mut [SourceImage],
    pair_rots: &[(usize, usize, Matrix3<f64>, usize)],
) {
    let n = images.len();
    if n <= 1 || pair_rots.is_empty() {
        return;
    }

    let mut adj: Vec<Vec<(usize, Matrix3<f64>, usize)>> = vec![Vec::new(); n];
    for &(i, j, rot_j_to_i, weight) in pair_rots {
        if i < n && j < n {
            adj[i].push((j, rot_j_to_i, weight));
            adj[j].push((i, rot_j_to_i.transpose(), weight));
        }
    }

    let mut visited = vec![false; n];
    let mut rots = vec![Matrix3::<f64>::identity(); n];
    rots[0] = rotation_matrix(images[0].yaw_deg, images[0].pitch_deg, images[0].roll_deg);
    visited[0] = true;

    let mut queue = VecDeque::new();
    queue.push_back(0usize);

    while let Some(u) = queue.pop_front() {
        let mut neighbors = adj[u].clone();
        neighbors.sort_by_key(|entry| std::cmp::Reverse(entry.2));
        for (v, rel_rot, _) in neighbors {
            if !visited[v] {
                visited[v] = true;
                rots[v] = rots[u] * rel_rot;
                let (yaw, pitch, roll) = matrix_to_ypr_deg(&rots[v]);
                images[v].yaw_deg = yaw;
                images[v].pitch_deg = pitch;
                images[v].roll_deg = roll;
                queue.push_back(v);
            }
        }
    }
}

/// Extract (yaw_deg, pitch_deg, roll_deg) matching `rotation_matrix(yaw, pitch, roll)`.
pub fn matrix_to_ypr_deg(r: &Matrix3<f64>) -> (f64, f64, f64) {
    let sin_p = r[(1, 2)].clamp(-1.0, 1.0);
    let pitch = sin_p.asin();
    let cos_p = pitch.cos();
    let (yaw, roll) = if cos_p.abs() > 1e-6 {
        let yaw = r[(0, 2)].atan2(r[(2, 2)]);
        let roll = r[(1, 0)].atan2(r[(1, 1)]);
        (yaw, roll)
    } else {
        let yaw = (-r[(2, 0)]).atan2(r[(0, 0)]);
        (yaw, 0.0)
    };
    (yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees())
}

/// Optimize per-image (yaw, pitch, roll) for images 1..N (anchoring image 0)
/// with default options (used by existing tests / quick callers without changing HFOV).
pub fn optimize_poses(
    images: &mut [SourceImage],
    control_points: &[ControlPoint],
    max_iterations: usize,
) -> OptimizationReport {
    optimize_poses_with_options(
        images,
        control_points,
        max_iterations,
        OptimizeOptions {
            optimize_hfov: false,
            level_horizon: false,
            optimize_distortion: false,
            prioritize_sky: true,
        },
    )
}

/// Optimize poses (and optionally shared HFOV + horizon leveling) via Levenberg-Marquardt.
pub fn optimize_poses_with_options(
    images: &mut [SourceImage],
    control_points: &[ControlPoint],
    max_iterations: usize,
    options: OptimizeOptions,
) -> OptimizationReport {
    optimize_poses_cancel(
        images,
        control_points,
        max_iterations,
        options,
        &AtomicBool::new(false),
    )
    .unwrap()
}

pub fn optimize_poses_cancel(
    images: &mut [SourceImage],
    control_points: &[ControlPoint],
    max_iterations: usize,
    mut options: OptimizeOptions,
    cancel: &AtomicBool,
) -> Result<OptimizationReport, String> {
    if control_points
        .iter()
        .any(|p| p.image_a >= images.len() || p.image_b >= images.len())
    {
        return Err("Invalid control-point image index".into());
    }
    // A shared focal / distortion correction is only meaningful for identical lens geometry.
    if let Some(first) = images.first() {
        let same_geom = images.iter().all(|im| {
            im.lens_projection == first.lens_projection
                && (im.width as f64 / im.height as f64 - first.width as f64 / first.height as f64)
                    .abs()
                    < 0.001
                && (im.hfov_deg - first.hfov_deg).abs() < 0.01
        });
        options.optimize_hfov &= same_geom;
        let same_dist = same_geom
            && images.iter().all(|im| {
                im.distortion_poly5.is_none()
                    && (im.distortion_scale - first.distortion_scale).abs() < 1e-5
                    && (0..3).all(|k| (im.distortion[k] - first.distortion[k]).abs() < 1e-5)
            });
        // Require sufficient radial spread before fitting radial distortion [a, b, c].
        let mut inner_cnt = 0usize;
        let mut outer_cnt = 0usize;
        for cp in control_points {
            let im = &images[cp.image_a];
            let half_min = (im.width.min(im.height) as f64 * 0.5).max(1.0);
            let r =
                (cp.x_a - im.width as f64 * 0.5).hypot(cp.y_a - im.height as f64 * 0.5) / half_min;
            if r < 0.55 {
                inner_cnt += 1;
            }
            if r > 0.65 {
                outer_cnt += 1;
            }
        }
        options.optimize_distortion &=
            same_dist && control_points.len() >= 24 && inner_cnt >= 6 && outer_cnt >= 10;
    }
    let n = images.len();
    if n <= 1 || control_points.is_empty() {
        return Ok(OptimizationReport {
            initial_rms_deg: f64::NAN,
            final_rms_deg: f64::NAN,
            max_error_deg: f64::NAN,
            iterations: 0,
            control_points_used: control_points.len(),
            control_points_rejected: 0,
            optimized_hfov_deg: images.first().map(|i| i.hfov_deg),
            optimized_distortion: images.first().map(|i| i.distortion),
        });
    }

    let initial_rms_deg = compute_rms_error_deg(images, control_points).0;
    let hfov_param = if options.optimize_hfov { 1 } else { 0 };
    let dist_params = if options.optimize_distortion { 3 } else { 0 };
    let params_len = (n - 1) * 3 + hfov_param + dist_params;
    let mut lambda = 1e-3f64;
    let mut iterations = 0usize;

    for iter in 0..max_iterations {
        if cancel.load(Ordering::Relaxed) {
            return Err("Optimization cancelled".into());
        }
        iterations = iter + 1;
        let raw = compute_residual_vec(images, control_points);
        let errs: Vec<f64> = raw
            .as_slice()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| (r.iter().map(|v| v * v).sum::<f64>()).sqrt())
            .collect();
        let mut sorted_errs = errs.clone();
        sorted_errs.sort_by(f64::total_cmp);
        let med_err = sorted_errs
            .get(sorted_errs.len() / 2)
            .copied()
            .unwrap_or(0.0);
        let huber_rad = if iter < 6 {
            0.50f64.to_radians()
        } else {
            (med_err * 2.0).clamp(0.10f64.to_radians(), 0.35f64.to_radians())
        };
        let weights: Vec<f64> = errs
            .iter()
            .enumerate()
            .map(|(idx, &err)| {
                let huber = (huber_rad / err.max(1e-12)).min(1.0).sqrt();
                let sky_w = if options.prioritize_sky {
                    control_points[idx].weight.clamp(0.25, 5.0).sqrt()
                } else {
                    1.0
                };
                huber * sky_w
            })
            .collect();
        let residuals = weighted_residual_vec(images, control_points, &weights);
        let cost = residuals.norm_squared();
        if cost < 1e-10 {
            break;
        }

        let m = residuals.len();
        let mut j_mat = DMatrix::<f64>::zeros(m, params_len);
        let eps = 1e-4f64;

        for img_idx in 1..n {
            if cancel.load(Ordering::Relaxed) {
                return Err("Optimization cancelled".into());
            }
            for comp in 0..3 {
                let col = (img_idx - 1) * 3 + comp;
                let orig = get_param(&images[img_idx], comp);

                set_param(&mut images[img_idx], comp, orig + eps);
                let r_plus = weighted_residual_vec(images, control_points, &weights);

                set_param(&mut images[img_idx], comp, orig - eps);
                let r_minus = weighted_residual_vec(images, control_points, &weights);

                set_param(&mut images[img_idx], comp, orig);

                let deriv = (r_plus - r_minus) / (2.0 * eps);
                j_mat.set_column(col, &deriv);
            }
        }

        if options.optimize_hfov {
            let col = (n - 1) * 3;
            let orig_hfovs: Vec<f64> = images.iter().map(|im| im.hfov_deg).collect();
            for im in images.iter_mut() {
                im.hfov_deg = (im.hfov_deg + eps).clamp(5.0, 350.0);
            }
            let r_plus = weighted_residual_vec(images, control_points, &weights);
            for (idx, im) in images.iter_mut().enumerate() {
                im.hfov_deg = (orig_hfovs[idx] - eps).clamp(5.0, 350.0);
            }
            let r_minus = weighted_residual_vec(images, control_points, &weights);
            for (idx, im) in images.iter_mut().enumerate() {
                im.hfov_deg = orig_hfovs[idx];
            }
            let deriv = (r_plus - r_minus) / (2.0 * eps);
            j_mat.set_column(col, &deriv);
        }

        if options.optimize_distortion {
            let base_col = (n - 1) * 3 + hfov_param;
            let dist_eps = 1e-5f64;
            let orig_dists: Vec<[f64; 3]> = images.iter().map(|im| im.distortion).collect();
            for k in 0..3 {
                for (im, orig) in images.iter_mut().zip(&orig_dists) {
                    im.distortion[k] = orig[k] + dist_eps;
                }
                let r_plus = weighted_residual_vec(images, control_points, &weights);
                for (im, orig) in images.iter_mut().zip(&orig_dists) {
                    im.distortion[k] = orig[k] - dist_eps;
                }
                let r_minus = weighted_residual_vec(images, control_points, &weights);
                for (im, orig) in images.iter_mut().zip(&orig_dists) {
                    im.distortion[k] = orig[k];
                }
                let deriv = (r_plus - r_minus) / (2.0 * dist_eps);
                j_mat.set_column(base_col + k, &deriv);
            }
        }

        let jt = j_mat.transpose();
        let mut a = &jt * &j_mat;
        for d in 0..params_len {
            let reg = if options.optimize_distortion && d >= (n - 1) * 3 + hfov_param {
                0.02
            } else {
                0.0
            };
            a[(d, d)] += lambda * a[(d, d)].max(1e-6) + reg;
        }
        let b = -(&jt * &residuals);

        let Some(chol) = a.cholesky() else {
            lambda *= 10.0;
            continue;
        };
        let delta = chol.solve(&b);
        if delta.norm() < 1e-6 {
            break;
        }

        let backup: Vec<(f64, f64, f64, f64, [f64; 3])> = images
            .iter()
            .map(|im| {
                (
                    im.yaw_deg,
                    im.pitch_deg,
                    im.roll_deg,
                    im.hfov_deg,
                    im.distortion,
                )
            })
            .collect();

        for (img_idx, image) in images.iter_mut().enumerate().skip(1) {
            let base = (img_idx - 1) * 3;
            image.yaw_deg += delta[base].clamp(-15.0, 15.0);
            image.pitch_deg += delta[base + 1].clamp(-15.0, 15.0);
            image.roll_deg += delta[base + 2].clamp(-15.0, 15.0);
        }
        if options.optimize_hfov {
            let d_hfov = delta[(n - 1) * 3].clamp(-5.0, 5.0);
            for im in images.iter_mut() {
                im.hfov_deg = (im.hfov_deg + d_hfov).clamp(10.0, 350.0);
            }
        }
        if options.optimize_distortion {
            let base_col = (n - 1) * 3 + hfov_param;
            let da = delta[base_col].clamp(-0.02, 0.02);
            let db = delta[base_col + 1].clamp(-0.03, 0.03);
            let dc = delta[base_col + 2].clamp(-0.02, 0.02);
            for im in images.iter_mut() {
                let mut a_v = (im.distortion[0] + da).clamp(-0.15, 0.15);
                let mut b_v = (im.distortion[1] + db).clamp(-0.25, 0.25);
                let mut c_v = (im.distortion[2] + dc).clamp(-0.15, 0.15);
                let sum_abs = a_v.abs() + b_v.abs() + c_v.abs();
                if sum_abs > 0.35 {
                    let s = 0.35 / sum_abs;
                    a_v *= s;
                    b_v *= s;
                    c_v *= s;
                }
                im.distortion = [a_v, b_v, c_v];
            }
        }

        let new_cost = weighted_residual_vec(images, control_points, &weights).norm_squared();
        if new_cost < cost {
            lambda = (lambda * 0.3).max(1e-7);
            if (cost - new_cost) / cost.max(1e-12) < 1e-6 {
                break;
            }
        } else {
            for (idx, (y, p, r, f, d)) in backup.into_iter().enumerate() {
                images[idx].yaw_deg = y;
                images[idx].pitch_deg = p;
                images[idx].roll_deg = r;
                images[idx].hfov_deg = f;
                images[idx].distortion = d;
            }
            lambda *= 5.0;
            if lambda > 1e6 {
                break;
            }
        }
    }

    if options.level_horizon {
        level_horizon_poses(images);
    }

    let (final_rms_deg, max_error_deg) = compute_rms_error_deg(images, control_points);
    Ok(OptimizationReport {
        initial_rms_deg,
        final_rms_deg,
        max_error_deg,
        iterations,
        control_points_used: control_points.len(),
        control_points_rejected: 0,
        optimized_hfov_deg: images.first().map(|i| i.hfov_deg),
        optimized_distortion: images.first().map(|i| i.distortion),
    })
}

/// Level the global panorama coordinate frame so the swept row sits flat on the horizon
/// and the middle frame is centered around yaw = 0 deg.
pub fn level_horizon_poses(images: &mut [SourceImage]) {
    if images.len() < 2 {
        return;
    }

    let mut forwards = Vec::with_capacity(images.len());
    let mut ups = Vec::with_capacity(images.len());
    let mut rots = Vec::with_capacity(images.len());
    for img in images.iter() {
        let r = rotation_matrix(img.yaw_deg, img.pitch_deg, img.roll_deg);
        forwards.push(r * Vector3::new(0.0, 0.0, 1.0));
        ups.push(r * Vector3::new(0.0, 1.0, 0.0));
        rots.push(r);
    }

    let Some(avg_up) = ups.iter().sum::<Vector3<f64>>().try_normalize(1e-8) else {
        return;
    };
    let Some(avg_fwd) = forwards.iter().sum::<Vector3<f64>>().try_normalize(1e-8) else {
        return;
    };

    // If cameras span >= 3 points, fit plane normal to centered forward vectors (smallest eigenvector of sum df df^T)
    // so upward- or downward-pitched sweeps still recover the true vertical axis of the horizon.
    let world_up = if images.len() >= 3 {
        let mean_fwd = forwards.iter().sum::<Vector3<f64>>() / (images.len() as f64);
        let mut cov = Matrix3::<f64>::zeros();
        for f in &forwards {
            let df = f - mean_fwd;
            cov += df * df.transpose();
        }
        let eig = cov.symmetric_eigen();
        let (min_idx, _) = eig
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let mut normal = eig.eigenvectors.column(min_idx).into_owned().normalize();
        if normal.dot(&avg_up) < 0.0 {
            normal = -normal;
        }
        // Only trust plane fit if the sweep actually has strong horizontal spread (largest/smallest eig > 4).
        let max_ev = eig.eigenvalues.iter().copied().fold(0.0f64, f64::max);
        let min_ev = eig.eigenvalues[min_idx].max(1e-9);
        let mut eigenvalues = eig.eigenvalues.as_slice().to_vec();
        eigenvalues.sort_by(f64::total_cmp);
        if max_ev > 1e-6 && eigenvalues[1] > max_ev * 1e-4 && max_ev / min_ev > 4.0 {
            normal
        } else {
            avg_up
        }
    } else {
        avg_up
    };

    // Build global alignment rotation G that maps world_up -> (0, 1, 0) and avg_fwd -> +Z plane.
    let new_y = world_up;
    let Some(proj_fwd) = (avg_fwd - new_y * avg_fwd.dot(&new_y)).try_normalize(1e-8) else {
        return;
    };
    if proj_fwd.norm() < 1e-6 {
        return;
    }
    let new_x = new_y.cross(&proj_fwd).normalize();
    let new_z = new_x.cross(&new_y).normalize();
    let basis = Matrix3::from_columns(&[new_x, new_y, new_z]);
    let g = basis.transpose();

    for (idx, img) in images.iter_mut().enumerate() {
        let leveled = g * rots[idx];
        let (y, p, r) = matrix_to_ypr_deg(&leveled);
        img.yaw_deg = y;
        img.pitch_deg = p;
        img.roll_deg = r;
    }
}

fn get_param(img: &SourceImage, comp: usize) -> f64 {
    match comp {
        0 => img.yaw_deg,
        1 => img.pitch_deg,
        _ => img.roll_deg,
    }
}

fn set_param(img: &mut SourceImage, comp: usize, val: f64) {
    match comp {
        0 => img.yaw_deg = val,
        1 => img.pitch_deg = val,
        _ => img.roll_deg = val,
    }
}

fn compute_residual_vec(images: &[SourceImage], cps: &[ControlPoint]) -> DVector<f64> {
    let mut vals = Vec::with_capacity(cps.len() * 3);
    for cp in cps {
        let ra = image_px_to_world_ray(cp.x_a, cp.y_a, &images[cp.image_a]).unwrap_or([0.0; 3]);
        let rb = image_px_to_world_ray(cp.x_b, cp.y_b, &images[cp.image_b]).unwrap_or([0.0; 3]);
        let delta = Vector3::from(ra) - Vector3::from(rb);
        vals.extend(delta.iter().copied());
    }
    DVector::from_vec(vals)
}

fn weighted_residual_vec(
    images: &[SourceImage],
    cps: &[ControlPoint],
    weights: &[f64],
) -> DVector<f64> {
    let mut residual = compute_residual_vec(images, cps);
    for (i, v) in residual.iter_mut().enumerate() {
        *v *= weights[i / 3];
    }
    residual
}

fn compute_rms_error_deg(images: &[SourceImage], cps: &[ControlPoint]) -> (f64, f64) {
    if cps.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0;
    let mut max_err = 0.0f64;
    let mut count = 0usize;
    for cp in cps {
        let Some(ra) = image_px_to_world_ray(cp.x_a, cp.y_a, &images[cp.image_a]) else {
            continue;
        };
        let Some(rb) = image_px_to_world_ray(cp.x_b, cp.y_b, &images[cp.image_b]) else {
            continue;
        };
        let dot = Vector3::from(ra).dot(&Vector3::from(rb)).clamp(-1.0, 1.0);
        let deg = dot.acos().to_degrees();
        sum_sq += deg * deg;
        max_err = max_err.max(deg);
        count += 1;
    }
    if count == 0 {
        (0.0, 0.0)
    } else {
        ((sum_sq / count as f64).sqrt(), max_err)
    }
}

pub fn connected_components(n: usize, cps: &[ControlPoint]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; n];
    let mut result = vec![];
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut group = vec![];
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(i) = queue.pop_front() {
            group.push(i);
            for cp in cps {
                let other = if cp.image_a == i {
                    cp.image_b
                } else if cp.image_b == i {
                    cp.image_a
                } else {
                    continue;
                };
                if other < n && !seen[other] {
                    seen[other] = true;
                    queue.push_back(other);
                }
            }
        }
        result.push(group);
    }
    result
}

pub fn control_point_error_deg(images: &[SourceImage], cp: &ControlPoint) -> Option<f64> {
    let a = images.get(cp.image_a)?;
    let b = images.get(cp.image_b)?;
    let ra = image_px_to_world_ray(cp.x_a, cp.y_a, a)?;
    let rb = image_px_to_world_ray(cp.x_b, cp.y_b, b)?;
    Some(
        Vector3::from(ra)
            .dot(&Vector3::from(rb))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::ray_to_image_px;
    use crate::project::LensProjection;

    #[test]
    fn matrix_ypr_roundtrip() {
        let m = rotation_matrix(32.5, -12.0, 7.25);
        let (y, p, r) = matrix_to_ypr_deg(&m);
        assert!((y - 32.5).abs() < 1e-6, "y={y}");
        assert!((p - (-12.0)).abs() < 1e-6, "p={p}");
        assert!((r - 7.25).abs() < 1e-6, "r={r}");
    }

    #[test]
    fn lm_recovers_misaligned_camera_pose() {
        let gt_img0 = SourceImage {
            id: 0,
            path: String::new(),
            width: 1200,
            height: 800,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 70.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let gt_img1 = SourceImage {
            id: 1,
            yaw_deg: 28.0,
            pitch_deg: -4.0,
            roll_deg: 3.0,
            ..gt_img0.clone()
        };

        let mut cps = Vec::new();
        for &lon in &[-5.0f64, 5.0, 12.0, 18.0, 22.0] {
            for &lat in &[-10.0f64, 0.0, 10.0] {
                let lr = lon.to_radians();
                let ar = lat.to_radians();
                let ray = [ar.cos() * lr.sin(), ar.sin(), ar.cos() * lr.cos()];
                if let (Some(([xa, ya], _)), Some(([xb, yb], _))) = (
                    ray_to_image_px(ray, &gt_img0),
                    ray_to_image_px(ray, &gt_img1),
                ) {
                    cps.push(ControlPoint {
                        image_a: 0,
                        x_a: xa,
                        y_a: ya,
                        image_b: 1,
                        x_b: xb,
                        y_b: yb,
                        weight: 1.0,
                    });
                }
            }
        }
        assert!(cps.len() >= 8);

        let mut imgs = vec![
            gt_img0,
            SourceImage {
                yaw_deg: 15.0,
                pitch_deg: 2.0,
                roll_deg: -2.0,
                ..gt_img1
            },
        ];
        let report = optimize_poses(&mut imgs, &cps, 25);
        assert!(report.final_rms_deg < 0.05, "rms={}", report.final_rms_deg);
        assert!((imgs[1].yaw_deg - 28.0).abs() < 0.1);
        assert!((imgs[1].pitch_deg - (-4.0)).abs() < 0.1);
        assert!((imgs[1].roll_deg - 3.0).abs() < 0.1);
    }

    #[test]
    fn lm_optimizes_radial_distortion_on_fisheye() {
        let gt0 = SourceImage {
            id: 0,
            path: String::new(),
            width: 1000,
            height: 1500,
            yaw_deg: -10.0,
            pitch_deg: 12.0,
            roll_deg: 0.0,
            hfov_deg: 91.5,
            lens_projection: LensProjection::FisheyeEquisolid,
            distortion: [-0.014, 0.032, -0.018],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let gt1 = SourceImage {
            id: 1,
            yaw_deg: 10.0,
            pitch_deg: 12.2,
            roll_deg: -0.4,
            ..gt0.clone()
        };

        let mut cps = Vec::new();
        for ix in -7..=7 {
            for iy in -6..=10 {
                let lon = (ix as f64 * 5.0).to_radians();
                let lat = (12.0 + iy as f64 * 4.5).to_radians();
                let ray = [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()];
                if let (Some(([xa, ya], _)), Some(([xb, yb], _))) =
                    (ray_to_image_px(ray, &gt0), ray_to_image_px(ray, &gt1))
                {
                    cps.push(ControlPoint {
                        image_a: 0,
                        x_a: xa,
                        y_a: ya,
                        image_b: 1,
                        x_b: xb,
                        y_b: yb,
                        weight: if lat > 0.25 { 3.0 } else { 1.0 },
                    });
                }
            }
        }
        assert!(cps.len() >= 30, "cps={}", cps.len());

        let mut imgs = vec![
            SourceImage {
                hfov_deg: 87.5,
                distortion: [0.0; 3],
                ..gt0.clone()
            },
            SourceImage {
                yaw_deg: 11.2,
                hfov_deg: 87.5,
                distortion: [0.0; 3],
                ..gt1.clone()
            },
        ];
        let opts = OptimizeOptions {
            optimize_hfov: true,
            level_horizon: false,
            optimize_distortion: true,
            prioritize_sky: true,
        };
        let report = optimize_poses_with_options(&mut imgs, &cps, 40, opts);
        assert!(report.final_rms_deg < 0.05, "rms={}", report.final_rms_deg);
        assert!(report.optimized_distortion.is_some());
        assert!(
            (imgs[0].hfov_deg - 91.5).abs() < 1.5,
            "hfov={}",
            imgs[0].hfov_deg
        );
    }
}
