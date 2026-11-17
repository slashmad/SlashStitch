//! Bayer demosaicing: RCD ("Ratio Corrected Demosaicing", Luis Sanz Rodríguez, v2.3 — the
//! default in RawTherapee and darktable).
//!
//! RCD estimates green at red/blue sites from colour *ratios* against a local low-pass image,
//! weighted by directional gradients and a vertical/horizontal discrimination built from a
//! high-pass statistic; red/blue are then filled from colour *differences* along diagonals and
//! cardinals. Compared with PPG it keeps fine detail (stars, foliage) with much less zipper and
//! false colour. A bilinear pass provides the 4-pixel border, which the default crop removes.
//!
//! Input: white-balanced mosaic in 0..1 (row-major, `w`×`h`) and the CFA colour (0 = R,
//! 1 = G, 2 = B) of each site. Output: three planes, clamped to 0..1.

// Stencil code: neighbours are addressed by index offsets around `x`, which iterator chains
// would only obscure.
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;

const EPS: f32 = 1e-5;
const EPS_SQ: f32 = 1e-10;

/// Weighted interpolation `a·b + (1 − a)·c`.
#[inline]
fn intp(a: f32, b: f32, c: f32) -> f32 {
    a * (b - c) + c
}

/// Bilinear demosaicing (also the border fallback of RCD). With `border` > 0, pixels farther
/// than `border` from every edge only receive their native sample: RCD overwrites the rest.
fn bilinear(
    cfa: &[f32],
    w: usize,
    h: usize,
    color: &(dyn Fn(usize, usize) -> usize + Sync),
    border: usize,
) -> [Vec<f32>; 3] {
    let mut planes = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
    let [r, g, b] = &mut planes;
    r.par_chunks_mut(w)
        .zip(g.par_chunks_mut(w))
        .zip(b.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((r, g), b))| {
            let inner_row = border > 0 && y >= border && y + border < h;
            for x in 0..w {
                if inner_row && x >= border && x + border < w {
                    let v = cfa[y * w + x];
                    match color(y, x) {
                        0 => r[x] = v,
                        1 => g[x] = v,
                        _ => b[x] = v,
                    }
                    continue;
                }
                let mut sum = [0f32; 3];
                let mut n = [0u32; 3];
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        let c = color(yy, xx);
                        sum[c] += cfa[yy * w + xx];
                        n[c] += 1;
                    }
                }
                let native = color(y, x);
                let v = |c: usize| {
                    if c == native {
                        cfa[y * w + x]
                    } else if n[c] > 0 {
                        sum[c] / n[c] as f32
                    } else {
                        0.0
                    }
                };
                r[x] = v(0);
                g[x] = v(1);
                b[x] = v(2);
            }
        });
    planes
}

/// Apply `f(y, row)` to every row of `out` in parallel.
fn rows(out: &mut [f32], w: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    out.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| f(y, row));
}

/// RCD demosaicing of a Bayer mosaic; see the module documentation.
pub fn rcd(
    cfa: &[f32],
    w: usize,
    h: usize,
    color: &(dyn Fn(usize, usize) -> usize + Sync),
) -> [Vec<f32>; 3] {
    assert_eq!(cfa.len(), w * h);
    if w < 12 || h < 12 {
        return bilinear(cfa, w, h, color, 0);
    }
    let [mut red, mut green, mut blue] = bilinear(cfa, w, h, color, 4);
    let (w1, w2, w3, w4) = (w, 2 * w, 3 * w, 4 * w);
    let sq = |v: f32| v * v;

    // Step 1: vertical/horizontal discrimination from a squared high-pass statistic.
    let mut buf_v = vec![0f32; w * h];
    rows(&mut buf_v, w, |y, row| {
        if (3..h - 3).contains(&y) {
            for (x, out) in row.iter_mut().enumerate() {
                let i = y * w + x;
                *out = sq((cfa[i - w3] - cfa[i - w1] - cfa[i + w1] + cfa[i + w3])
                    - 3.0 * (cfa[i - w2] + cfa[i + w2])
                    + 6.0 * cfa[i]);
            }
        }
    });
    let mut buf_h = vec![0f32; w * h];
    rows(&mut buf_h, w, |y, row| {
        for x in 3..w - 3 {
            let i = y * w + x;
            row[x] = sq((cfa[i - 3] - cfa[i - 1] - cfa[i + 1] + cfa[i + 3])
                - 3.0 * (cfa[i - 2] + cfa[i + 2])
                + 6.0 * cfa[i]);
        }
    });
    let mut vh = vec![0.5f32; w * h];
    rows(&mut vh, w, |y, row| {
        if (4..h - 4).contains(&y) {
            for x in 4..w - 4 {
                let i = y * w + x;
                let v = (buf_v[i - w1] + buf_v[i] + buf_v[i + w1]).max(EPS_SQ);
                let hh = (buf_h[i - 1] + buf_h[i] + buf_h[i + 1]).max(EPS_SQ);
                row[x] = v / (v + hh);
            }
        }
    });
    drop(buf_v);
    drop(buf_h);

    // Step 2: low-pass image at red/blue sites.
    let mut lpf = vec![0f32; w * h];
    rows(&mut lpf, w, |y, row| {
        if (2..h - 2).contains(&y) {
            for x in 2..w - 2 {
                if color(y, x) == 1 {
                    continue;
                }
                let i = y * w + x;
                row[x] = cfa[i]
                    + 0.5 * (cfa[i - w1] + cfa[i + w1] + cfa[i - 1] + cfa[i + 1])
                    + 0.25
                        * (cfa[i - w1 - 1] + cfa[i - w1 + 1] + cfa[i + w1 - 1] + cfa[i + w1 + 1]);
            }
        }
    });

    // Step 3: green at red/blue sites.
    let disc = |dir: &[f32], i: usize| {
        let central = dir[i];
        let neighbourhood =
            0.25 * (dir[i - w1 - 1] + dir[i - w1 + 1] + dir[i + w1 - 1] + dir[i + w1 + 1]);
        if (0.5 - central).abs() < (0.5 - neighbourhood).abs() {
            neighbourhood
        } else {
            central
        }
    };
    rows(&mut green, w, |y, row| {
        if !(4..h - 4).contains(&y) {
            return;
        }
        for x in 4..w - 4 {
            if color(y, x) == 1 {
                continue;
            }
            let i = y * w + x;
            let c = cfa[i];
            let n_grad = EPS
                + (cfa[i - w1] - cfa[i + w1]).abs()
                + (c - cfa[i - w2]).abs()
                + (cfa[i - w1] - cfa[i - w3]).abs()
                + (cfa[i - w2] - cfa[i - w4]).abs();
            let s_grad = EPS
                + (cfa[i + w1] - cfa[i - w1]).abs()
                + (c - cfa[i + w2]).abs()
                + (cfa[i + w1] - cfa[i + w3]).abs()
                + (cfa[i + w2] - cfa[i + w4]).abs();
            let w_grad = EPS
                + (cfa[i - 1] - cfa[i + 1]).abs()
                + (c - cfa[i - 2]).abs()
                + (cfa[i - 1] - cfa[i - 3]).abs()
                + (cfa[i - 2] - cfa[i - 4]).abs();
            let e_grad = EPS
                + (cfa[i + 1] - cfa[i - 1]).abs()
                + (c - cfa[i + 2]).abs()
                + (cfa[i + 1] - cfa[i + 3]).abs()
                + (cfa[i + 2] - cfa[i + 4]).abs();
            let l = lpf[i];
            let n_est = cfa[i - w1] * 2.0 * l / (EPS + l + lpf[i - w2]);
            let s_est = cfa[i + w1] * 2.0 * l / (EPS + l + lpf[i + w2]);
            let w_est = cfa[i - 1] * 2.0 * l / (EPS + l + lpf[i - 2]);
            let e_est = cfa[i + 1] * 2.0 * l / (EPS + l + lpf[i + 2]);
            let v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
            let h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);
            row[x] = intp(disc(&vh, i), h_est, v_est).clamp(0.0, 1.0);
        }
    });
    drop(lpf);

    // Step 4.1: diagonal discrimination.
    let mut buf_p = vec![0f32; w * h];
    let mut buf_q = vec![0f32; w * h];
    rows(&mut buf_p, w, |y, row| {
        if (3..h - 3).contains(&y) {
            for x in 3..w - 3 {
                let i = y * w + x;
                row[x] = sq((cfa[i - w3 - 3] - cfa[i - w1 - 1] - cfa[i + w1 + 1]
                    + cfa[i + w3 + 3])
                    - 3.0 * (cfa[i - w2 - 2] + cfa[i + w2 + 2])
                    + 6.0 * cfa[i]);
            }
        }
    });
    rows(&mut buf_q, w, |y, row| {
        if (3..h - 3).contains(&y) {
            for x in 3..w - 3 {
                let i = y * w + x;
                row[x] = sq((cfa[i - w3 + 3] - cfa[i - w1 + 1] - cfa[i + w1 - 1]
                    + cfa[i + w3 - 3])
                    - 3.0 * (cfa[i - w2 + 2] + cfa[i + w2 - 2])
                    + 6.0 * cfa[i]);
            }
        }
    });
    let mut pq = vec![0.5f32; w * h];
    rows(&mut pq, w, |y, row| {
        if (4..h - 4).contains(&y) {
            for x in 4..w - 4 {
                let i = y * w + x;
                let p = (buf_p[i - w1 - 1] + buf_p[i] + buf_p[i + w1 + 1]).max(EPS_SQ);
                let q = (buf_q[i - w1 + 1] + buf_q[i] + buf_q[i + w1 - 1]).max(EPS_SQ);
                row[x] = p / (p + q);
            }
        }
    });
    drop(buf_p);
    drop(buf_q);

    // Step 4.2: red at blue sites and blue at red sites (diagonal colour differences).
    let diag = |plane: &[f32], g: &[f32], y: usize, x: usize| -> f32 {
        let i = y * w + x;
        let gi = g[i];
        let nw_grad = EPS
            + (plane[i - w1 - 1] - plane[i + w1 + 1]).abs()
            + (plane[i - w1 - 1] - plane[i - w3 - 3]).abs()
            + (gi - g[i - w2 - 2]).abs();
        let ne_grad = EPS
            + (plane[i - w1 + 1] - plane[i + w1 - 1]).abs()
            + (plane[i - w1 + 1] - plane[i - w3 + 3]).abs()
            + (gi - g[i - w2 + 2]).abs();
        let sw_grad = EPS
            + (plane[i + w1 - 1] - plane[i - w1 + 1]).abs()
            + (plane[i + w1 - 1] - plane[i + w3 - 3]).abs()
            + (gi - g[i + w2 - 2]).abs();
        let se_grad = EPS
            + (plane[i + w1 + 1] - plane[i - w1 - 1]).abs()
            + (plane[i + w1 + 1] - plane[i + w3 + 3]).abs()
            + (gi - g[i + w2 + 2]).abs();
        let nw = plane[i - w1 - 1] - g[i - w1 - 1];
        let ne = plane[i - w1 + 1] - g[i - w1 + 1];
        let sw = plane[i + w1 - 1] - g[i + w1 - 1];
        let se = plane[i + w1 + 1] - g[i + w1 + 1];
        let p_est = (nw_grad * se + se_grad * nw) / (nw_grad + se_grad);
        let q_est = (ne_grad * sw + sw_grad * ne) / (ne_grad + sw_grad);
        (gi + intp(disc(&pq, i), q_est, p_est)).clamp(0.0, 1.0)
    };
    let fill_diag = |target: &mut Vec<f32>, own: usize| {
        let src = target.clone();
        rows(target, w, |y, row| {
            if !(4..h - 4).contains(&y) {
                return;
            }
            for x in 4..w - 4 {
                // The opposite chroma site: this plane is missing there.
                let c = color(y, x);
                if c != 1 && c != own {
                    row[x] = diag(&src, &green, y, x);
                }
            }
        });
    };
    fill_diag(&mut red, 0);
    fill_diag(&mut blue, 2);
    drop(pq);

    // Step 4.3: red and blue at green sites (cardinal colour differences).
    let card = |plane: &[f32], g: &[f32], y: usize, x: usize| -> f32 {
        let i = y * w + x;
        let gi = g[i];
        let n_grad = EPS
            + (gi - g[i - w2]).abs()
            + (plane[i - w1] - plane[i + w1]).abs()
            + (plane[i - w1] - plane[i - w3]).abs();
        let s_grad = EPS
            + (gi - g[i + w2]).abs()
            + (plane[i + w1] - plane[i - w1]).abs()
            + (plane[i + w1] - plane[i + w3]).abs();
        let w_grad = EPS
            + (gi - g[i - 2]).abs()
            + (plane[i - 1] - plane[i + 1]).abs()
            + (plane[i - 1] - plane[i - 3]).abs();
        let e_grad = EPS
            + (gi - g[i + 2]).abs()
            + (plane[i + 1] - plane[i - 1]).abs()
            + (plane[i + 1] - plane[i + 3]).abs();
        let n = plane[i - w1] - g[i - w1];
        let s = plane[i + w1] - g[i + w1];
        let ww = plane[i - 1] - g[i - 1];
        let e = plane[i + 1] - g[i + 1];
        let v_est = (n_grad * s + s_grad * n) / (n_grad + s_grad);
        let h_est = (e_grad * ww + w_grad * e) / (e_grad + w_grad);
        (gi + intp(disc(&vh, i), h_est, v_est)).clamp(0.0, 1.0)
    };
    let fill_card = |target: &mut Vec<f32>| {
        let src = target.clone();
        rows(target, w, |y, row| {
            if !(4..h - 4).contains(&y) {
                return;
            }
            for x in 4..w - 4 {
                if color(y, x) == 1 {
                    row[x] = card(&src, &green, y, x);
                }
            }
        });
    };
    fill_card(&mut red);
    fill_card(&mut blue);
    [red, green, blue]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rggb(y: usize, x: usize) -> usize {
        match (y % 2, x % 2) {
            (0, 0) => 0,
            (1, 1) => 2,
            _ => 1,
        }
    }

    fn mosaic(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 3]) -> Vec<f32> {
        (0..w * h)
            .map(|i| f(i / w, i % w)[rggb(i / w, i % w)])
            .collect()
    }

    fn rmse(
        planes: &[Vec<f32>; 3],
        w: usize,
        h: usize,
        f: impl Fn(usize, usize) -> [f32; 3],
    ) -> f64 {
        let mut s = 0.0;
        let mut n = 0.0;
        for y in 8..h - 8 {
            for x in 8..w - 8 {
                let t = f(y, x);
                for c in 0..3 {
                    s += f64::from(planes[c][y * w + x] - t[c]).powi(2);
                    n += 1.0;
                }
            }
        }
        (s / n).sqrt()
    }

    #[test]
    fn flat_colour_is_reproduced_exactly() {
        let (w, h) = (40, 32);
        let f = |_: usize, _: usize| [0.42, 0.31, 0.12];
        let out = rcd(&mosaic(w, h, f), w, h, &rggb);
        assert!(rmse(&out, w, h, f) < 1e-5);
    }

    #[test]
    fn rcd_beats_bilinear_on_fine_detail() {
        let (w, h) = (96, 80);
        // Oblique fine stripes with a colour cast plus a smooth gradient.
        let f = |y: usize, x: usize| {
            let t = (x as f32 * 0.9 + y as f32 * 0.35).sin() * 0.2 + 0.5;
            let g = 0.1 + 0.3 * x as f32 / w as f32;
            [t * 0.9 + g * 0.1, t * 0.8 + g * 0.2, t * 0.6 + g * 0.3]
        };
        let m = mosaic(w, h, f);
        let r = rmse(&rcd(&m, w, h, &rggb), w, h, f);
        let b = rmse(&bilinear(&m, w, h, &rggb, 0), w, h, f);
        assert!(r < 0.6 * b, "rcd {r} vs bilinear {b}");
    }

    #[test]
    fn output_stays_in_range_at_hard_edges() {
        let (w, h) = (48, 40);
        let f = |y: usize, x: usize| {
            if (x / 5 + y / 7).is_multiple_of(2) {
                [1.0, 1.0, 1.0]
            } else {
                [0.0, 0.02, 0.0]
            }
        };
        let out = rcd(&mosaic(w, h, f), w, h, &rggb);
        assert!(out.iter().flatten().all(|v| (0.0..=1.0).contains(v)));
    }
}
