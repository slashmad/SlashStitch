//! GPU-accelerated panorama preview renderer using wgpu compute shaders, with automatic
//! CPU fallback through the shared core renderer when no GPU adapter is available.
//!
//! This is the *fast* interactive preview (feathered blend of preview mips). The exact
//! preview, identical in algorithm to the export, is produced by `slashstitch_core::render`.

use bytemuck::{Pod, Zeroable};
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use slashstitch_core::{
    photometric::{inv_half_diag_sq, PhotometricParams},
    raster::Rgb16Raster,
    rotation_matrix, BandViewParams, Interpolation, LensProjection, PanoramaProjection,
    RenderInput, RenderPlan, SourceImage, TargetTrc, DEFAULT_BLEND_WIDTH_DEG,
};
use std::sync::Mutex;
use std::time::Instant;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderParams {
    out_w: u32,
    out_h: u32,
    num_images: u32,
    projection: u32,
    pano_hfov_rad: f32,
    view_yaw_rad: f32,
    view_pitch_rad: f32,
    view_v_shift: f32,
    canvas_aspect: f32,
    _pad: [u32; 3],
    region: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuImageUniform {
    inv_r0: [f32; 4],
    inv_r1: [f32; 4],
    inv_r2: [f32; 4],
    width: u32,
    height: u32,
    lens_proj: u32,
    pixel_offset: u32,
    hfov_rad: f32,
    dist_a: f32,
    dist_b: f32,
    dist_c: f32,
    poly5: [f32; 2],
    distortion_scale: f32,
    distortion_model: u32,
    /// RGB gain and `1 / half_diagonal²` in mip pixels.
    gain: [f32; 4],
    /// Vignetting attenuation polynomial `[k1, k2, k3, r2_max]` (positive: darker corners;
    /// the correction is held constant beyond the evidenced radius `r2_max`).
    vig: [f32; 4],
}

type SourceCacheEntry = (u64, wgpu::Buffer, Vec<u32>);

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub adapter_name: String,
    pub backend_name: String,
    failed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Packed source mips kept resident between renders, keyed by the caller's generation.
    src_cache: Mutex<Option<SourceCacheEntry>>,
}

pub struct PreviewRenderer {
    gpu: Option<GpuContext>,
}

pub struct PreviewOutput {
    pub image: RgbaImage,
    pub backend_used: String,
    pub render_ms: f64,
}

/// One fast-preview render request.
pub struct PreviewRequest<'a> {
    pub out_w: u32,
    pub out_h: u32,
    pub projection: PanoramaProjection,
    pub pano_hfov_deg: f64,
    /// View rotation, vertical shift, visible canvas region (`crop`) and canvas aspect.
    pub view: BandViewParams,
    pub images: &'a [SourceImage],
    pub mips: &'a [&'a RgbaImage],
    pub photometric: Option<&'a PhotometricParams>,
    pub trc: &'a TargetTrc,
    /// Identity of `mips`; equal keys let the GPU reuse the uploaded source buffer.
    pub source_key: Option<u64>,
}

impl PreviewRenderer {
    pub fn new() -> Self {
        Self {
            gpu: if std::env::var_os("SLASHSTITCH_FORCE_CPU").is_some() {
                None
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(init_wgpu))
                    .ok()
                    .and_then(Result::ok)
            },
        }
    }

    pub fn backend_label(&self) -> String {
        match &self.gpu {
            Some(g) if !g.failed.load(std::sync::atomic::Ordering::Relaxed) => {
                format!("{} ({})", g.adapter_name, g.backend_name)
            }
            Some(_) => "CPU fallback (GPU disconnected)".into(),
            None => "CPU Rayon fallback".to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_preview(
        &self,
        out_w: u32,
        out_h: u32,
        projection: PanoramaProjection,
        pano_hfov_deg: f64,
        view_yaw_deg: f64,
        view_pitch_deg: f64,
        images: &[SourceImage],
        mip_rgba: &[&RgbaImage],
    ) -> PreviewOutput {
        self.render_preview_with_view(
            out_w,
            out_h,
            projection,
            pano_hfov_deg,
            view_yaw_deg,
            view_pitch_deg,
            0.0,
            images,
            mip_rgba,
            &TargetTrc::srgb(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_preview_with_view(
        &self,
        out_w: u32,
        out_h: u32,
        projection: PanoramaProjection,
        pano_hfov_deg: f64,
        view_yaw_deg: f64,
        view_pitch_deg: f64,
        view_v_shift: f64,
        images: &[SourceImage],
        mip_rgba: &[&RgbaImage],
        trc: &TargetTrc,
    ) -> PreviewOutput {
        self.render(&PreviewRequest {
            out_w,
            out_h,
            projection,
            pano_hfov_deg,
            view: BandViewParams {
                view_yaw_deg,
                view_pitch_deg,
                view_v_shift,
                crop: [0.0, 0.0, 1.0, 1.0],
                canvas_aspect: Some(f64::from(out_w) / f64::from(out_h.max(1))),
            },
            images,
            mips: mip_rgba,
            photometric: None,
            trc,
            source_key: None,
        })
    }

    pub fn render(&self, req: &PreviewRequest<'_>) -> PreviewOutput {
        let t0 = Instant::now();
        if let Some(gpu) = &self.gpu {
            if !req.images.is_empty() && !req.mips.is_empty() {
                if let Ok(Ok(img)) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    render_wgpu_req(gpu, req)
                })) {
                    return PreviewOutput {
                        image: img,
                        backend_used: self.backend_label(),
                        render_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    };
                }
            }
        }
        let img = render_cpu_req(req);
        PreviewOutput {
            image: img,
            backend_used: "CPU Rayon".to_string(),
            render_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

impl Default for PreviewRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Subtle dark checkerboard for uncovered canvas, identical on GPU and CPU.
#[inline]
pub fn checker_rgba(x: u32, y: u32) -> [u8; 4] {
    let checker = ((x >> 4) ^ (y >> 4)) & 1;
    let bg = if checker == 1 { 28u8 } else { 22u8 };
    [bg, bg + 2, bg + 6, 255]
}

fn region_of(view: &BandViewParams) -> [f64; 4] {
    let l = view.crop[0].clamp(0.0, 0.9999);
    let t = view.crop[1].clamp(0.0, 0.9999);
    let r = view.crop[2].clamp(l + 1e-4, 1.0);
    let b = view.crop[3].clamp(t + 1e-4, 1.0);
    [l, t, r - l, b - t]
}

fn init_wgpu() -> Result<GpuContext, String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .map_err(|e| format!("No GPU adapter: {e}"))?;

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("slashstitch-gpu"),
        ..Default::default()
    }))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let on_error = std::sync::Arc::clone(&failed);
    device.on_uncaptured_error(std::sync::Arc::new(move |error| {
        on_error.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("GPU error; selecting CPU fallback: {error}");
    }));
    let on_lost = std::sync::Arc::clone(&failed);
    device.set_device_lost_callback(move |_reason, _message| {
        on_lost.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warp_preview"),
        source: wgpu::ShaderSource::Wgsl(include_str!("warp_preview.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warp_preview_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("warp_preview_pl"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("warp_preview_pipe"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(GpuContext {
        device,
        queue,
        pipeline,
        bind_group_layout,
        adapter_name: info.name,
        backend_name: format!("{:?}", info.backend),
        failed,
        src_cache: std::sync::Mutex::new(None),
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn render_wgpu(
    gpu: &GpuContext,
    out_w: u32,
    out_h: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view_yaw_deg: f64,
    view_pitch_deg: f64,
    view_v_shift: f64,
    images: &[SourceImage],
    mip_rgba: &[&RgbaImage],
    trc: &TargetTrc,
) -> Result<RgbaImage, String> {
    render_wgpu_req(
        gpu,
        &test_request(
            out_w,
            out_h,
            projection,
            pano_hfov_deg,
            view_yaw_deg,
            view_pitch_deg,
            view_v_shift,
            images,
            mip_rgba,
            trc,
        ),
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn render_cpu(
    out_w: u32,
    out_h: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view_yaw_deg: f64,
    view_pitch_deg: f64,
    view_v_shift: f64,
    images: &[SourceImage],
    mip_rgba: &[&RgbaImage],
    trc: &TargetTrc,
) -> RgbaImage {
    render_cpu_req(&test_request(
        out_w,
        out_h,
        projection,
        pano_hfov_deg,
        view_yaw_deg,
        view_pitch_deg,
        view_v_shift,
        images,
        mip_rgba,
        trc,
    ))
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn test_request<'a>(
    out_w: u32,
    out_h: u32,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view_yaw_deg: f64,
    view_pitch_deg: f64,
    view_v_shift: f64,
    images: &'a [SourceImage],
    mip_rgba: &'a [&'a RgbaImage],
    trc: &'a TargetTrc,
) -> PreviewRequest<'a> {
    PreviewRequest {
        out_w,
        out_h,
        projection,
        pano_hfov_deg,
        view: BandViewParams {
            view_yaw_deg,
            view_pitch_deg,
            view_v_shift,
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: Some(f64::from(out_w) / f64::from(out_h.max(1))),
        },
        images,
        mips: mip_rgba,
        photometric: None,
        trc,
        source_key: None,
    }
}

fn render_wgpu_req(gpu: &GpuContext, req: &PreviewRequest<'_>) -> Result<RgbaImage, String> {
    if gpu.failed.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("GPU disconnected".into());
    }
    let (out_w, out_h) = (req.out_w, req.out_h);
    let n = req.images.len().min(req.mips.len());
    let limits = gpu.device.limits();
    let source_bytes = req
        .mips
        .iter()
        .take(n)
        .try_fold(0u64, |sum, im| {
            sum.checked_add(u64::from(im.width()) * u64::from(im.height()) * 4)
        })
        .ok_or("Source buffer size overflow")?;
    let output_bytes = u64::from(out_w) * u64::from(out_h) * 4;
    if source_bytes > limits.max_storage_buffer_binding_size
        || source_bytes > limits.max_buffer_size
        || output_bytes > limits.max_storage_buffer_binding_size
        || output_bytes > limits.max_buffer_size
        || n as u64 * std::mem::size_of::<GpuImageUniform>() as u64
            > limits.max_storage_buffer_binding_size
    {
        return Err("GPU storage budget exceeded; using CPU preview".into());
    }

    // Upload (or reuse) the packed source mips.
    let mut cache = gpu.src_cache.lock().map_err(|_| "GPU cache poisoned")?;
    let reuse = matches!((&*cache, req.source_key), (Some((key, _, offsets)), Some(k)) if *key == k && offsets.len() == n);
    if !reuse {
        let mut packed: Vec<u32> = Vec::with_capacity((source_bytes / 4) as usize);
        let mut offsets = Vec::with_capacity(n);
        for rgba in req.mips.iter().take(n) {
            offsets.push(packed.len() as u32);
            packed.extend(
                rgba.as_raw()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| u32::from_le_bytes(*p)),
            );
        }
        if packed.is_empty() {
            packed.push(0);
        }
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("src_pixels"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        *cache = Some((req.source_key.unwrap_or(u64::MAX), buffer, offsets));
        if req.source_key.is_none() {
            // Unkeyed requests must never be reused by a later keyed request.
            if let Some(entry) = cache.as_mut() {
                entry.0 = u64::MAX;
            }
        }
    }
    let (_, src_buf, offsets) = cache.as_ref().ok_or("GPU source cache missing")?;

    let photometric = req.photometric.filter(|p| p.is_valid_for(req.images.len()));
    let mut img_uniforms = Vec::with_capacity(n);
    for (i, &pixel_offset) in offsets.iter().enumerate().take(n) {
        let meta = &req.images[i];
        let rgba = req.mips[i];
        let inv_r = rotation_matrix(meta.yaw_deg, meta.pitch_deg, meta.roll_deg).transpose();
        let lens_proj = match meta.lens_projection {
            LensProjection::Rectilinear => 0,
            LensProjection::FisheyeFullframe => 1,
            LensProjection::FisheyeCircular => 2,
            LensProjection::Equirectangular => 3,
            LensProjection::FisheyeEquisolid => 4,
        };
        let (gain, vig, r2_max) = match photometric {
            Some(p) => (
                p.gains[i],
                p.vignetting[i],
                p.vignetting_r2_max.get(i).copied().unwrap_or(1.0),
            ),
            None => ([1.0; 3], [0.0; 3], 1.0),
        };
        let inv_hd2 = inv_half_diag_sq(f64::from(rgba.width()), f64::from(rgba.height())) as f32;
        let row = |r: usize| {
            [
                inv_r[(r, 0)] as f32,
                inv_r[(r, 1)] as f32,
                inv_r[(r, 2)] as f32,
                0.0,
            ]
        };
        img_uniforms.push(GpuImageUniform {
            inv_r0: row(0),
            inv_r1: row(1),
            inv_r2: row(2),
            width: rgba.width(),
            height: rgba.height(),
            lens_proj,
            pixel_offset,
            hfov_rad: meta.hfov_deg.to_radians() as f32,
            dist_a: meta.distortion[0] as f32,
            dist_b: meta.distortion[1] as f32,
            dist_c: meta.distortion[2] as f32,
            poly5: meta.distortion_poly5.unwrap_or([0.0; 2]).map(|v| v as f32),
            distortion_scale: meta.distortion_scale as f32,
            distortion_model: u32::from(meta.distortion_poly5.is_some()),
            gain: [gain[0], gain[1], gain[2], inv_hd2],
            vig: [vig[0], vig[1], vig[2], r2_max],
        });
    }

    let proj_code = match req.projection {
        PanoramaProjection::Equirectangular => 0,
        PanoramaProjection::Cylindrical => 1,
        PanoramaProjection::Mercator => 2,
        PanoramaProjection::Rectilinear => 3,
        PanoramaProjection::Vedutismo => 4,
        PanoramaProjection::Stereographic => 5,
        PanoramaProjection::StereographicDown => 6,
        PanoramaProjection::CircularFisheye => 7,
        PanoramaProjection::FullframeFisheye => 8,
        PanoramaProjection::TransverseEquirectangular => 9,
        PanoramaProjection::TransverseCylindrical => 10,
        PanoramaProjection::TransverseMercator => 11,
        PanoramaProjection::TransverseVedutismo => 12,
    };
    let region = region_of(&req.view);
    let aspect = req
        .view
        .canvas_aspect
        .unwrap_or((f64::from(out_w) / region[2]) / (f64::from(out_h.max(1)) / region[3]));
    let params = ShaderParams {
        out_w,
        out_h,
        num_images: n as u32,
        projection: proj_code,
        pano_hfov_rad: req.pano_hfov_deg.to_radians() as f32,
        view_yaw_rad: req.view.view_yaw_deg.to_radians() as f32,
        view_pitch_rad: req.view.view_pitch_deg.to_radians() as f32,
        view_v_shift: req.view.view_v_shift as f32,
        canvas_aspect: aspect as f32,
        _pad: [0; 3],
        region: region.map(|v| v as f32),
    };

    let param_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    if img_uniforms.is_empty() {
        return Err("No images".into());
    }
    let img_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("images"),
            contents: bytemuck::cast_slice(&img_uniforms),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let (e2l, thr) = req.trc.gpu_tables();
    let mut trc_vec = Vec::with_capacity(512);
    trc_vec.extend_from_slice(&e2l);
    trc_vec.extend_from_slice(&thr);
    let trc_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trc_lut"),
            contents: bytemuck::cast_slice(&trc_vec),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_pixels"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warp_preview_bg"),
        layout: &gpu.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: img_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: trc_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warp_preview_enc"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warp_preview_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&gpu.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(out_w.div_ceil(16), out_h.div_ceil(16), 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, output_bytes);
    gpu.queue.submit(Some(encoder.finish()));
    drop(cache);

    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .map_err(|e| format!("GPU wait failed: {e}"))?;
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .map_err(|e| format!("map recv failed: {e}"))?
        .map_err(|e| format!("map failed: {e}"))?;

    if gpu.failed.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("GPU failed during rendering".into());
    }
    let view = slice
        .get_mapped_range()
        .map_err(|e| format!("get_mapped_range failed: {e}"))?;
    let raw_bytes = view.to_vec();
    drop(view);
    read_buf.unmap();

    RgbaImage::from_raw(out_w, out_h, raw_bytes)
        .ok_or_else(|| "Invalid preview buffer dimensions".to_string())
}

/// CPU fallback: the shared core renderer in feather mode over the same preview mips.
fn render_cpu_req(req: &PreviewRequest<'_>) -> RgbaImage {
    let n = req.images.len().min(req.mips.len());
    let input = RenderInput {
        out_w: req.out_w,
        out_h: req.out_h,
        projection: req.projection,
        pano_hfov_deg: req.pano_hfov_deg,
        view: req.view,
        images: &req.images[..n],
        sources: req.mips[..n]
            .iter()
            .map(|m| vec![*m as &dyn Rgb16Raster])
            .collect(),
        photometric: req.photometric,
        multiband: false,
        trc: req.trc,
        interpolation: Interpolation::Bilinear,
        blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        reject_transients: false,
        seams: None,
    };
    let plan = RenderPlan::new(&input);
    let (px, valid) = plan
        .render_rows(&input, 0, req.out_h, None)
        .unwrap_or_default();
    let w = req.out_w as usize;
    let mut flat = vec![0u8; w * req.out_h as usize * 4];
    flat.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let idx = y * w + x;
            let dst = &mut row[x * 4..x * 4 + 4];
            if valid.get(idx).copied().unwrap_or(false) {
                let e = req.trc.encode_linear_to_u8(px[idx]);
                dst.copy_from_slice(&[e[0], e[1], e[2], 255]);
            } else {
                dst.copy_from_slice(&checker_rgba(x as u32, y as u32));
            }
        }
    });
    RgbaImage::from_raw(req.out_w, req.out_h, flat)
        .unwrap_or_else(|| RgbaImage::from_pixel(req.out_w, req.out_h, Rgba([22, 24, 29, 255])))
}

#[cfg(test)]
mod tests {
    use super::*;
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn renders_preview_gpu_or_cpu() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap();
        let renderer = PreviewRenderer::new();
        let mip = RgbaImage::from_pixel(160, 120, Rgba([120, 180, 240, 255]));
        let meta = SourceImage {
            id: 0,
            path: "a.png".into(),
            width: 160,
            height: 120,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 75.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let out = renderer.render_preview(
            320,
            160,
            PanoramaProjection::Equirectangular,
            120.0,
            0.0,
            0.0,
            &[meta],
            &[&mip],
        );
        println!(
            "GPU preview backend: {} ({:.2} ms)",
            out.backend_used, out.render_ms
        );
        assert_eq!(out.image.dimensions(), (320, 160));
        // Center pixel should be sampled from the blue-ish source image.
        let center = out.image.get_pixel(160, 80).0;
        assert_eq!(center, [120, 180, 240, 255]);
    }
    #[test]
    fn gpu_budget_overflow_uses_cpu_instead_of_panicking() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap();
        let Ok(gpu) = init_wgpu() else {
            eprintln!("No adapter available; GPU budget integration not exercised");
            return;
        };
        let image = RgbaImage::from_pixel(1024, 1024, Rgba([120, 180, 240, 255]));
        let source = SourceImage {
            id: 0,
            path: String::new(),
            width: 1024,
            height: 1024,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            hfov_deg: 75.0,
            lens_projection: LensProjection::Rectilinear,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let n =
            (gpu.device.limits().max_storage_buffer_binding_size as usize / (1024 * 1024 * 4)) + 1;
        let images = vec![source; n];
        let references = vec![&image; n];
        let renderer = PreviewRenderer { gpu: Some(gpu) };
        let result = renderer.render_preview(
            64,
            32,
            PanoramaProjection::Equirectangular,
            120.0,
            0.0,
            0.0,
            &images,
            &references,
        );
        assert_eq!(result.backend_used, "CPU Rayon");
        assert_eq!(result.image.dimensions(), (64, 32));
    }

    #[test]
    fn cpu_gpu_agree_across_projections_lenses_aspects_and_poly5() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap();
        let Ok(gpu) = init_wgpu() else {
            eprintln!("No adapter available; CPU/GPU parity not exercised");
            return;
        };
        let projections = [
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
        let lenses = [
            LensProjection::Rectilinear,
            LensProjection::FisheyeFullframe,
            LensProjection::FisheyeCircular,
            LensProjection::FisheyeEquisolid,
            LensProjection::Equirectangular,
        ];
        for (w, h) in [(80, 120), (120, 80)] {
            let image = RgbaImage::from_fn(w, h, |x, y| {
                Rgba([
                    (40 + x % 170) as u8,
                    (60 + y % 170) as u8,
                    (80 + (x + y) % 140) as u8,
                    255,
                ])
            });
            for lens in lenses {
                let meta = SourceImage {
                    id: 0,
                    path: String::new(),
                    width: w,
                    height: h,
                    yaw_deg: 23.0,
                    pitch_deg: -7.0,
                    roll_deg: 5.0,
                    hfov_deg: if lens == LensProjection::Equirectangular {
                        300.0
                    } else {
                        100.0
                    },
                    lens_projection: lens,
                    distortion: [0.0; 3],
                    distortion_poly5: Some([0.003, -0.0002]),
                    distortion_scale: 0.8,
                };
                let srgb_trc = TargetTrc::srgb();
                for projection in projections {
                    let cpu = render_cpu(
                        96,
                        64,
                        projection,
                        170.0,
                        12.0,
                        -8.0,
                        0.0,
                        std::slice::from_ref(&meta),
                        &[&image],
                        &srgb_trc,
                    );
                    let computed = render_wgpu(
                        &gpu,
                        96,
                        64,
                        projection,
                        170.0,
                        12.0,
                        -8.0,
                        0.0,
                        std::slice::from_ref(&meta),
                        &[&image],
                        &srgb_trc,
                    )
                    .expect("GPU rendering");
                    let mut differences = 0;
                    for (a, b) in cpu.pixels().zip(computed.pixels()) {
                        if (0..4).any(|c| a[c].abs_diff(b[c]) > 3) {
                            differences += 1;
                        }
                    }
                    assert!(differences<12,"projection={projection:?} lens={lens:?} aspect={w}/{h} differing pixels={differences}");
                }
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_with_region_aspect_and_photometric_correction() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap();
        let Ok(gpu) = init_wgpu() else {
            eprintln!("No adapter available; photometric parity not exercised");
            return;
        };
        let make = |seed: u32| {
            RgbaImage::from_fn(120, 90, |x, y| {
                Rgba([
                    (40 + (x * 3 + seed) % 150) as u8,
                    (60 + (y * 2 + seed) % 150) as u8,
                    (90 + (x + y) % 120) as u8,
                    255,
                ])
            })
        };
        let a = make(0);
        let b = make(37);
        let meta = |id: usize, yaw: f64| SourceImage {
            id,
            path: String::new(),
            width: 120,
            height: 90,
            yaw_deg: yaw,
            pitch_deg: 3.0,
            roll_deg: 0.0,
            hfov_deg: 80.0,
            lens_projection: LensProjection::FisheyeEquisolid,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let images = [meta(0, -20.0), meta(1, 25.0)];
        let mips = [&a, &b];
        let photometric = PhotometricParams {
            gains: vec![[1.1, 0.95, 0.9], [0.85, 1.05, 1.0]],
            vignetting: vec![[0.5, 0.1, 0.0], [0.5, 0.1, 0.0]],
            vignetting_r2_max: vec![0.6, 0.6],
        };
        let trc = TargetTrc::srgb();
        let req = PreviewRequest {
            out_w: 128,
            out_h: 80,
            projection: PanoramaProjection::Cylindrical,
            pano_hfov_deg: 160.0,
            view: BandViewParams {
                view_yaw_deg: 4.0,
                view_pitch_deg: 0.0,
                view_v_shift: 0.0,
                crop: [0.3, 0.2, 0.7, 0.8],
                canvas_aspect: Some(2.5),
            },
            images: &images,
            mips: &mips,
            photometric: Some(&photometric),
            trc: &trc,
            source_key: Some(42),
        };
        let cpu = render_cpu_req(&req);
        let computed = render_wgpu_req(&gpu, &req).expect("GPU rendering");
        let mut differences = 0;
        for (p, q) in cpu.pixels().zip(computed.pixels()) {
            if (0..4).any(|c| p[c].abs_diff(q[c]) > 3) {
                differences += 1;
            }
        }
        assert!(
            differences < 20,
            "photometric/region parity differing pixels={differences}"
        );
        // Second render reuses the resident source buffer and must be identical.
        let again = render_wgpu_req(&gpu, &req).expect("GPU rendering (cached sources)");
        assert_eq!(again.as_raw(), computed.as_raw());
    }

    #[test]
    fn gpu_matches_cpu_with_view_v_shift() {
        let _gpu_guard = GPU_TEST_LOCK.lock().unwrap();
        let Ok(gpu) = init_wgpu() else {
            eprintln!("No adapter available; CPU/GPU view_v_shift parity not exercised");
            return;
        };
        let image = RgbaImage::from_fn(96, 128, |x, y| {
            Rgba([
                (30 + (x * 2) % 200) as u8,
                (50 + y % 180) as u8,
                (70 + (x + y) % 160) as u8,
                255,
            ])
        });
        let meta = SourceImage {
            id: 0,
            path: String::new(),
            width: 96,
            height: 128,
            yaw_deg: 0.0,
            pitch_deg: 22.0,
            roll_deg: 0.0,
            hfov_deg: 90.0,
            lens_projection: LensProjection::FisheyeEquisolid,
            distortion: [0.0; 3],
            distortion_poly5: None,
            distortion_scale: 1.0,
        };
        let srgb_trc = TargetTrc::srgb();
        let shifted_projs = [
            PanoramaProjection::Cylindrical,
            PanoramaProjection::Mercator,
            PanoramaProjection::Rectilinear,
            PanoramaProjection::Vedutismo,
            PanoramaProjection::Stereographic,
            PanoramaProjection::TransverseCylindrical,
        ];
        for projection in shifted_projs {
            let cpu = render_cpu(
                96,
                64,
                projection,
                160.0,
                0.0,
                0.0,
                0.22,
                std::slice::from_ref(&meta),
                &[&image],
                &srgb_trc,
            );
            let gpu_img = render_wgpu(
                &gpu,
                96,
                64,
                projection,
                160.0,
                0.0,
                0.0,
                0.22,
                std::slice::from_ref(&meta),
                &[&image],
                &srgb_trc,
            )
            .expect("GPU rendering with view_v_shift");
            let mut differences = 0;
            for (a, b) in cpu.pixels().zip(gpu_img.pixels()) {
                if (0..4).any(|c| a[c].abs_diff(b[c]) > 3) {
                    differences += 1;
                }
            }
            assert!(
                differences < 12,
                "shifted projection={projection:?} differing pixels={differences}"
            );
        }
    }
}
