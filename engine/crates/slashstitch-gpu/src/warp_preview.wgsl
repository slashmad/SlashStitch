struct Params {
  out_w: u32,
  out_h: u32,
  num_images: u32,
  projection: u32,
  pano_hfov_rad: f32,
  view_yaw_rad: f32,
  view_pitch_rad: f32,
  view_v_shift: f32,
  canvas_aspect: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  // Visible canvas region: left, top, width, height in normalized canvas units.
  region: vec4<f32>,
};

struct ImageUniform {
  inv_r0: vec4<f32>,
  inv_r1: vec4<f32>,
  inv_r2: vec4<f32>,
  width: u32,
  height: u32,
  lens_proj: u32,
  pixel_offset: u32,
  hfov_rad: f32,
  dist_a: f32,
  dist_b: f32,
  dist_c: f32,
  poly5: vec2<f32>,
  distortion_scale: f32,
  distortion_model: u32,
  // RGB gain and 1 / half_diagonal^2 (mip pixels).
  gain: vec4<f32>,
  // Vignetting attenuation polynomial k1, k2, k3 (positive: darker corners); w = r2 limit.
  vig: vec4<f32>,
};

struct TrcLut {
  e2l: array<vec4<f32>, 256>,
  thr: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> images: array<ImageUniform>;
@group(0) @binding(2) var<storage, read> src_pixels: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_pixels: array<u32>;
@group(0) @binding(4) var<uniform> trc_lut: TrcLut;

const PI: f32 = 3.14159265359;

fn unpack_rgba(packed: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(packed & 255u),
    f32((packed >> 8u) & 255u),
    f32((packed >> 16u) & 255u),
    f32((packed >> 24u) & 255u)
  );
}

fn pack_rgba(c: vec4<f32>) -> u32 {
  let r = u32(clamp(round(c.r), 0.0, 255.0));
  let g = u32(clamp(round(c.g), 0.0, 255.0));
  let b = u32(clamp(round(c.b), 0.0, 255.0));
  let a = u32(clamp(round(c.a), 0.0, 255.0));
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
  let ri = min(u32(clamp(round(c.r), 0.0, 255.0)), 255u);
  let gi = min(u32(clamp(round(c.g), 0.0, 255.0)), 255u);
  let bi = min(u32(clamp(round(c.b), 0.0, 255.0)), 255u);
  return vec3<f32>(trc_lut.e2l[ri].x, trc_lut.e2l[gi].y, trc_lut.e2l[bi].z);
}

// Exact 8-bit quantization: count how many code thresholds (linear light) lie at or below
// the value. Mirrors the CPU exact inverse TRC (`TargetTrc::encode_linear_to_u8`).
fn quantize_channel(lin: f32, ch: u32) -> f32 {
  var lo: u32 = 0u;
  var hi: u32 = 255u;
  // Find the largest k in [0, 255] with thr[k] <= lin (thr[0] == 0).
  for (var step: u32 = 0u; step < 8u; step = step + 1u) {
    let mid = (lo + hi + 1u) / 2u;
    if (trc_lut.thr[mid][ch] <= lin) { lo = mid; } else { hi = mid - 1u; }
  }
  return f32(lo);
}

fn from_linear(lin: vec3<f32>) -> vec3<f32> {
  let c = clamp(lin, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec3<f32>(quantize_channel(c.r, 0u), quantize_channel(c.g, 1u), quantize_channel(c.b, 2u));
}
fn sample_bilinear(img: ImageUniform, px: f32, py: f32) -> vec3<f32> {
  let w = img.width;
  let h = img.height;
  let x0 = min(u32(floor(px)), w - 1u);
  let y0 = min(u32(floor(py)), h - 1u);
  let x1 = min(x0 + 1u, w - 1u);
  let y1 = min(y0 + 1u, h - 1u);
  let fx = px - f32(x0);
  let fy = py - f32(y0);

  let p00 = to_linear(unpack_rgba(src_pixels[img.pixel_offset + y0 * w + x0]).rgb);
  let p10 = to_linear(unpack_rgba(src_pixels[img.pixel_offset + y0 * w + x1]).rgb);
  let p01 = to_linear(unpack_rgba(src_pixels[img.pixel_offset + y1 * w + x0]).rgb);
  let p11 = to_linear(unpack_rgba(src_pixels[img.pixel_offset + y1 * w + x1]).rgb);

  let top = mix(p00, p10, fx);
  let bot = mix(p01, p11, fx);
  return mix(top, bot, fy);
}

fn stereographic_ray(u: f32, v: f32, asp: f32) -> vec3<f32> {
  let scale = tan(clamp(params.pano_hfov_rad, 20.0*PI/180.0, 340.0*PI/180.0) * 0.25);
  let sx = (u - 0.5) * 2.0 * scale;
  let sy = (0.5 - v) * 2.0 * scale / max(asp, 0.1);
  let r2 = sx * sx + sy * sy;
  let denom = 1.0 + r2;
  return vec3<f32>(2.0 * sx / denom, 2.0 * sy / denom, (1.0 - r2) / denom);
}

fn pano_base_ray(proj: u32, u: f32, v: f32, asp: f32) -> vec4<f32> {
  let a = max(asp, 0.1);
  if (proj == 0u) {
    // Equirectangular
    let lon = (u - 0.5) * 2.0 * PI;
    let lat = (0.5 - v) * PI;
    let cl = cos(lat);
    return vec4<f32>(cl * sin(lon), sin(lat), cl * cos(lon), 1.0);
  } else if (proj == 1u) {
    // Cylindrical
    let lon = (u - 0.5) * 2.0 * PI;
    let y_cyl = (0.5 - v) * (2.0 * PI / a);
    let inv = inverseSqrt(1.0 + y_cyl * y_cyl);
    return vec4<f32>(sin(lon) * inv, y_cyl * inv, cos(lon) * inv, 1.0);
  } else if (proj == 2u) {
    // Mercator
    let lon = (u - 0.5) * 2.0 * PI;
    let y_m = (0.5 - v) * (2.0 * PI / a);
    let cos_lat = 1.0 / cosh(y_m);
    let sin_lat = tanh(y_m);
    return vec4<f32>(cos_lat * sin(lon), sin_lat, cos_lat * cos(lon), 1.0);
  } else if (proj == 3u) {
    // Rectilinear
    let hfov = clamp(params.pano_hfov_rad, 10.0*PI/180.0, 150.0*PI/180.0);
    let hw = tan(hfov * 0.5);
    let hh = hw / a;
    let x = (u - 0.5) * 2.0 * hw;
    let y = (0.5 - v) * 2.0 * hh;
    return vec4<f32>(normalize(vec3<f32>(x, y, 1.0)), 1.0);
  } else if (proj == 4u) {
    // Vedutismo (Panini d=1)
    let scale = tan(clamp(params.pano_hfov_rad, 20.0*PI/180.0, 320.0*PI/180.0) * 0.25);
    let sx = (u - 0.5) * 2.0 * scale;
    let sy = (0.5 - v) * 2.0 * scale / a;
    let den = 1.0 + sx * sx;
    let rx = (2.0 * sx) / den;
    let rz = (1.0 - sx * sx) / den;
    let ry = (2.0 * sy) / den;
    return vec4<f32>(normalize(vec3<f32>(rx, ry, rz)), 1.0);
  } else if (proj == 5u) {
    // Stereographic
    return vec4<f32>(stereographic_ray(u, v, a), 1.0);
  } else if (proj == 6u) {
    // Stereographic Down (Little Planet)
    let r = stereographic_ray(u, v, a);
    return vec4<f32>(r.x, -r.z, r.y, 1.0);
  } else if (proj == 7u) {
    // Circular Fisheye
    let half_rad = clamp(params.pano_hfov_rad, 20.0*PI/180.0, 2.0*PI) * 0.5;
    let min_span = min(a, 1.0);
    let px = (u - 0.5) * 2.0 * a / min_span;
    let py = (0.5 - v) * 2.0 / min_span;
    let r = length(vec2<f32>(px, py));
    if (r > 1.0) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (r < 1e-6) {
      return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
    let theta = r * half_rad;
    let s = sin(theta) / r;
    return vec4<f32>(px * s, py * s, cos(theta), 1.0);
  } else {
    // Fullframe Fisheye (8u)
    let half_rad = clamp(params.pano_hfov_rad, 20.0*PI/180.0, 2.0*PI) * 0.5;
    let px = (u - 0.5) * 2.0;
    let py = (0.5 - v) * 2.0 / a;
    let r = length(vec2<f32>(px, py));
    let theta = r * half_rad;
    if (theta > PI) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (r < 1e-6) {
      return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
    let s = sin(theta) / r;
    return vec4<f32>(px * s, py * s, cos(theta), 1.0);
  }
}

fn pano_uv_to_ray(u: f32, v: f32, aspect: f32) -> vec4<f32> {
  let p = params.projection;
  let supports_shift = (p >= 1u && p <= 5u) || (p >= 10u && p <= 12u);
  let v_eff = select(v, v - params.view_v_shift, supports_shift);
  var res = vec4<f32>(0.0, 0.0, 1.0, 1.0);
  if (p <= 8u) {
    res = pano_base_ray(p, u, v_eff, aspect);
  } else {
    var base_proj: u32 = 0u;
    if (p == 9u) { base_proj = 0u; }
    else if (p == 10u) { base_proj = 1u; }
    else if (p == 11u) { base_proj = 2u; }
    else { base_proj = 4u; }
    let r = pano_base_ray(base_proj, 1.0 - v_eff, u, 1.0 / max(aspect, 0.1));
    res = vec4<f32>(-r.y, r.x, r.z, r.w);
  }
  if (res.w < 0.5) {
    return res;
  }

  // Interactive viewport yaw/pitch offset.
  let ray = res.xyz;
  let cy = cos(params.view_yaw_rad);
  let sy = sin(params.view_yaw_rad);
  let cp = cos(-params.view_pitch_rad);
  let sp = sin(-params.view_pitch_rad);
  let r_p = vec3<f32>(ray.x, ray.y * cp - ray.z * sp, ray.y * sp + ray.z * cp);
  return vec4<f32>(r_p.x * cy + r_p.z * sy, r_p.y, -r_p.x * sy + r_p.z * cy, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.out_w || gid.y >= params.out_h) {
    return;
  }

  let u = params.region.x + (f32(gid.x) + 0.5) / f32(params.out_w) * params.region.z;
  let v = params.region.y + (f32(gid.y) + 0.5) / f32(params.out_h) * params.region.w;
  let aspect = params.canvas_aspect;
  let ray_res = pano_uv_to_ray(u, v, aspect);
  if (ray_res.w < 0.5) {
    let checker = ((gid.x >> 4u) ^ (gid.y >> 4u)) & 1u;
    let bg = select(22.0, 28.0, checker == 1u);
    out_pixels[gid.y * params.out_w + gid.x] = pack_rgba(vec4<f32>(bg,bg+2.0,bg+6.0,255.0));
    return;
  }
  let ray = ray_res.xyz;

  var acc = vec3<f32>(0.0, 0.0, 0.0);
  var wsum: f32 = 0.0;

  for (var i: u32 = 0u; i < params.num_images; i = i + 1u) {
    let img = images[i];
    let cam = vec3<f32>(
      dot(img.inv_r0.xyz, ray),
      dot(img.inv_r1.xyz, ray),
      dot(img.inv_r2.xyz, ray)
    );

    let w = f32(img.width);
    let h = f32(img.height);
    if (w <= 1.0 || h <= 1.0) {
      continue;
    }

    var nx: f32 = 0.0;
    var ny: f32 = 0.0;
    var valid = true;

    if (img.lens_proj == 0u) {
      if (cam.z <= 0.0001) {
        valid = false;
      } else {
        let f = 0.5 / tan(img.hfov_rad * 0.5);
        nx = f * cam.x / cam.z;
        ny = f * cam.y / cam.z;
      }
    } else if (img.lens_proj == 1u || img.lens_proj == 2u) {
      let phi_r = length(vec2<f32>(cam.x, cam.y));
      let theta = atan2(phi_r, cam.z);
      if (theta > min(img.hfov_rad * 0.5 * max(length(vec2<f32>(w,h))/w,1.0) * 1.05, PI*0.99)) {
        valid = false;
      } else if (phi_r > 1e-6) {
        let r = theta / img.hfov_rad;
        nx = r * cam.x / phi_r;
        ny = r * cam.y / phi_r;
      }
    } else if (img.lens_proj == 4u) {
      // Equisolid fisheye: r = 2 * f_norm * sin(theta / 2)
      let phi_r = length(vec2<f32>(cam.x, cam.y));
      let theta = atan2(phi_r, cam.z);
      let f_norm = 0.5 / max(2.0 * sin(img.hfov_rad * 0.25), 1e-6);
      if (phi_r > 1e-6) {
        let r = 2.0 * f_norm * sin(theta * 0.5);
        nx = r * cam.x / phi_r;
        ny = r * cam.y / phi_r;
      }
    } else {
      let lon = atan2(cam.x, cam.z);
      let lat = asin(clamp(cam.y, -1.0, 1.0));
      nx = lon / img.hfov_rad;
      ny = lat / img.hfov_rad;
    }

    if (!valid) {
      continue;
    }

    let half_min = max(min(w, h) * 0.5, 1.0);
    let px_un = nx * w;
    let py_un = -ny * w;
    let r_norm = length(vec2<f32>(px_un, py_un)) / half_min * max(img.distortion_scale,0.000001);
    var scale: f32 = 1.0;
    if (img.distortion_model == 1u) {
      let r2=r_norm*r_norm;
      scale=1.0+img.poly5.x*r2+img.poly5.y*r2*r2;
    } else if (r_norm > 1e-6 && (img.dist_a != 0.0 || img.dist_b != 0.0 || img.dist_c != 0.0)) {
      let d = 1.0 - img.dist_a - img.dist_b - img.dist_c;
      let r2 = r_norm * r_norm;
      let r3 = r2 * r_norm;
      scale = img.dist_a * r3 + img.dist_b * r2 + img.dist_c * r_norm + d;
    }

    let px = w * 0.5 + px_un * scale;
    let py = h * 0.5 + py_un * scale;

    if (px < 0.0 || py < 0.0 || px >= w - 1.0 || py >= h - 1.0) {
      continue;
    }
    if (img.lens_proj == 2u && length(vec2<f32>(px - w * 0.5, py - h * 0.5)) > half_min * 0.99) {
      continue;
    }

    let dx = min(px / w, 1.0 - px / w) * 2.0;
    let dy = min(py / h, 1.0 - py / h) * 2.0;
    let edge = clamp(min(dx, dy), 0.0, 1.0);
    let weight = max(edge * edge * (3.0 - 2.0 * edge), 0.0001);

    let ddx = px - w * 0.5;
    let ddy = py - h * 0.5;
    let r2 = min((ddx * ddx + ddy * ddy) * img.gain.w, img.vig.w);
    // Undo lens attenuation exp(-(k1 r^2 + k2 r^4 + k3 r^6)).
    let devignette = exp(r2 * (img.vig.x + r2 * (img.vig.y + r2 * img.vig.z)));
    let rgb = sample_bilinear(img, px, py) * img.gain.xyz * devignette;
    acc += rgb * weight;
    wsum += weight;
  }

  let out_idx = gid.y * params.out_w + gid.x;
  if (wsum > 1e-6) {
    out_pixels[out_idx] = pack_rgba(vec4<f32>(from_linear(acc / wsum), 255.0));
  } else {
    // Subtle dark checkerboard background for uncovered sky/nadir.
    let checker = ((gid.x >> 4u) ^ (gid.y >> 4u)) & 1u;
    let bg = select(22.0, 28.0, checker == 1u);
    out_pixels[out_idx] = pack_rgba(vec4<f32>(bg, bg + 2.0, bg + 6.0, 255.0));
  }
}
