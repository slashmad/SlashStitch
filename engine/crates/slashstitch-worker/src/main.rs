use image::codecs::png::PngEncoder;
use image::{DynamicImage, ExtendedColorType, ImageEncoder, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
mod sources;

use slashstitch_core::exporter::{
    export_panorama_with_sources, preflight_export, validate_export_compatibility,
};
use slashstitch_core::features::{find_all_control_points_cancel, refine_star_control_points};
use slashstitch_core::files::{publish, temporary_output};
use slashstitch_core::optimizer::{
    connected_components, control_point_error_deg, optimize_poses_cancel,
};
use slashstitch_core::photometric::{estimate_photometric, PhotometricParams};
use slashstitch_core::raster::Rgb16Raster;
use slashstitch_core::raster::{
    convert_rgb16_profile, decode_source_raw_16, thumbnail_rgb16, Rgb16Image,
};
use slashstitch_core::{
    compute_seam_map, BandViewParams, Interpolation, RenderInput, RenderPlan, SeamMap,
    DEFAULT_BLEND_WIDTH_DEG,
};
use slashstitch_core::{
    inspect_image, seed_poses_from_pairs, validate_bit_depth, ControlPoint, ExportOptions,
    LensDatabase, LensProjection, OptimizeOptions, PanoramaProjection, SourceImage,
    SourceImageMeta, TargetColorProfile, TargetTrc, DEFAULT_OUTPUT_BIT_DEPTH, MAX_ICC_BYTES,
    SUPPORTED_BIT_DEPTHS, SUPPORTED_COLOR_PROFILE_IDS,
};
use slashstitch_gpu::{checker_rgba, PreviewRenderer, PreviewRequest};
use sources::{verify_stamps, SourceCache, SourceStamp};
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const MAX_IMAGES: usize = 128;
const MAX_CP: usize = 20_000;

fn default_bit_depth() -> u8 {
    DEFAULT_OUTPUT_BIT_DEPTH
}
fn default_color_profile() -> String {
    "srgb".to_string()
}
fn default_true() -> bool {
    true
}
fn default_blend_width() -> f64 {
    DEFAULT_BLEND_WIDTH_DEG
}
static MIP_GENERATION: AtomicU64 = AtomicU64::new(1);
fn next_mip_generation() -> u64 {
    MIP_GENERATION.fetch_add(1, Ordering::Relaxed)
}
#[derive(Deserialize)]
struct Request {
    id: String,
    action: String,
    #[serde(default)]
    payload: Value,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum JobState {
    Idle,
    Running,
    Finalizing,
}
impl JobState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Finalizing => "finalizing",
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectDocument {
    schema_version: u32,
    images: Vec<SourceImage>,
    #[serde(default)]
    control_points: Vec<ControlPoint>,
    projection: PanoramaProjection,
    pano_hfov_deg: f64,
    view_yaw_deg: f64,
    view_pitch_deg: f64,
    #[serde(default)]
    view_v_shift: f64,
    #[serde(default = "default_bit_depth")]
    output_bit_depth: u8,
    #[serde(default = "default_color_profile")]
    color_profile: String,
    #[serde(default)]
    custom_icc_name: Option<String>,
    #[serde(default)]
    custom_icc_base64: Option<String>,
    #[serde(default)]
    ui: Value,
    /// Canonical canvas width/height ratio. Export, crop and auto-crop geometry depend on it,
    /// so it is part of the project instead of following the window size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    canvas_aspect: Option<f64>,
    /// Apply exposure/white-balance gains and vignetting correction (preview and export).
    #[serde(default = "default_true")]
    exposure_compensation: bool,
    /// Replace transient light seen by only one of >= 3 overlapping frames.
    #[serde(default = "default_true")]
    reject_transients: bool,
    /// Route seams through areas where neighbouring frames agree (instead of fixed centre lines).
    #[serde(default = "default_true")]
    optimize_seams: bool,
    /// Correct lens vignetting (exposure/white-balance gains stay active either way).
    #[serde(default = "default_true")]
    correct_vignetting: bool,
    /// Source resampling kernel for exact previews and exports.
    #[serde(default)]
    interpolation: Interpolation,
    /// Coarsest multi-band transition width (degrees): wider = softer transitions.
    #[serde(default = "default_blend_width")]
    blend_width_deg: f64,
    /// Last estimated photometric parameters (recomputed whenever geometry changes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    photometric: Option<PhotometricParams>,
}
impl Default for ProjectDocument {
    fn default() -> Self {
        Self {
            schema_version: 1,
            images: vec![],
            control_points: vec![],
            projection: PanoramaProjection::Equirectangular,
            pano_hfov_deg: 120.0,
            view_yaw_deg: 0.0,
            view_pitch_deg: 0.0,
            view_v_shift: 0.0,
            output_bit_depth: default_bit_depth(),
            color_profile: default_color_profile(),
            custom_icc_name: None,
            custom_icc_base64: None,
            ui: Value::Null,
            canvas_aspect: None,
            exposure_compensation: true,
            reject_transients: true,
            optimize_seams: true,
            correct_vignetting: true,
            interpolation: Interpolation::Spline36,
            blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
            photometric: None,
        }
    }
}
#[derive(Clone)]
struct ProjectRuntime {
    doc: ProjectDocument,
    metas: Vec<SourceImageMeta>,
    mip_raw: Vec<(Arc<Rgb16Image>, Option<Vec<u8>>)>,
    mip_rgba: Vec<Arc<RgbaImage>>,
    /// Preview mips converted to the target profile at 16 bits (exact preview, photometry).
    mip16: Vec<Arc<Rgb16Image>>,
    /// File identity each source's metadata and mips were built from.
    stamps: Vec<SourceStamp>,
    mip_generation: u64,
    photometric_sig: u64,
    preview_width: u32,
    preview_height: u32,
}
impl Default for ProjectRuntime {
    fn default() -> Self {
        Self {
            doc: ProjectDocument::default(),
            metas: vec![],
            mip_raw: vec![],
            mip_rgba: vec![],
            mip16: vec![],
            stamps: vec![],
            mip_generation: 0,
            photometric_sig: 0,
            preview_width: 1280,
            preview_height: 640,
        }
    }
}
struct SharedState {
    job_state: Mutex<JobState>,
    cancel: AtomicBool,
    worker_thread: Mutex<Option<JoinHandle<()>>>,
    lens_db: Mutex<Arc<LensDatabase>>,
    renderer: PreviewRenderer,
    project: Mutex<ProjectRuntime>,
    preview_dir: tempfile::TempDir,
    /// Disk-backed full-resolution source pyramids (session lifetime).
    source_cache: SourceCache,
    _cache_dir: tempfile::TempDir,
    /// Exact-preview lane: newest request wins, older renders are cancelled.
    /// Cancel flag of the newest exact request per lane ("base" whole canvas, "detail" zoom),
    /// so a zoom tile never cancels the whole-canvas render or vice versa.
    exact_cancel: Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>,
    exact_lock: Mutex<()>,
    /// Seam-overlay lane (independent of exact previews; newest request wins).
    overlay_cancel: Mutex<Arc<AtomicBool>>,
    overlay_lock: Mutex<()>,
    /// Optimised seam layout keyed by everything that influences it.
    seam_cache: Mutex<Vec<(u64, Arc<SeamMap>)>>,
    preview_version: AtomicU64,
}
#[derive(Serialize, Clone)]
struct PreviewMeta {
    preview_path: String,
    width: u32,
    height: u32,
    backend: String,
    render_ms: f64,
    color_profile: String,
    color_profile_name: String,
    /// Monotonic id; the file name changes with it so viewers never show a stale image.
    version: u64,
    canvas_aspect: f64,
    /// Visible canvas region `[left, top, right, bottom]` in normalized canvas units.
    region: [f64; 4],
    exact: bool,
}
fn emit(out: &Arc<Mutex<io::Stdout>>, value: Value) {
    if let Ok(mut g) = out.lock() {
        let _ = writeln!(g, "{value}");
        let _ = g.flush();
    }
}
fn event(out: &Arc<Mutex<io::Stdout>>, name: &str, data: Value) {
    emit(out, json!({"event":name,"data":data}));
}
fn reply(out: &Arc<Mutex<io::Stdout>>, id: &str, result: Result<Value, String>) {
    emit(
        out,
        match result {
            Ok(result) => json!({"id":id,"ok":true,"result":result}),
            Err(error) => json!({"id":id,"ok":false,"error":error}),
        },
    );
}
fn state(shared: &SharedState, out: &Arc<Mutex<io::Stdout>>, next: JobState) {
    *shared.job_state.lock().unwrap() = next;
    event(out, "job-state", json!({"state":next.as_str()}));
}
fn progress(out: &Arc<Mutex<io::Stdout>>, percent: u32, stage: impl AsRef<str>) {
    event(
        out,
        "job-progress",
        json!({"percent":percent,"stage":stage.as_ref()}),
    );
}
fn check_cancel(shared: &SharedState) -> Result<(), String> {
    if shared.cancel.load(Ordering::Relaxed) {
        Err("Job cancelled; previous project and output files were preserved".into())
    } else {
        Ok(())
    }
}
fn db(shared: &SharedState) -> Arc<LensDatabase> {
    Arc::clone(&shared.lens_db.lock().unwrap())
}
fn target_profile_from_doc(doc: &ProjectDocument) -> Result<TargetColorProfile, String> {
    TargetColorProfile::from_parts(
        &doc.color_profile,
        doc.custom_icc_name.as_deref(),
        doc.custom_icc_base64.as_deref(),
    )
}
/// Convert a source-profile 16-bit thumbnail to the target profile: 16-bit for the exact
/// renderer and photometry, RGBA8 for the GPU preview and keypoint detection.
fn target_mips(
    thumb16: &Rgb16Image,
    src_icc: Option<&[u8]>,
    target: &TargetColorProfile,
) -> Result<(Arc<Rgb16Image>, Arc<RgbaImage>), String> {
    let converted = convert_rgb16_profile(thumb16, src_icc, target)?;
    let w = converted.width();
    let h = converted.height();
    let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for px in converted.as_raw().as_chunks::<3>().0 {
        // Round to nearest instead of truncating (truncation darkened the preview by ~0.5 LSB).
        rgba.push(((u32::from(px[0]) + 128) / 257).min(255) as u8);
        rgba.push(((u32::from(px[1]) + 128) / 257).min(255) as u8);
        rgba.push(((u32::from(px[2]) + 128) / 257).min(255) as u8);
        rgba.push(255);
    }
    let rgba = RgbaImage::from_raw(w, h, rgba).ok_or("Invalid preview buffer")?;
    Ok((Arc::new(converted), Arc::new(rgba)))
}
/// The target profile shared by every source: identical embedded ICC bytes, or the same
/// colorimetrically verified built-in profile. `None` if any source has no or a different one.
fn common_source_profile(raw: &[(Arc<Rgb16Image>, Option<Vec<u8>>)]) -> Option<TargetColorProfile> {
    // Developed camera RAWs carry a linear working profile, which is no export profile.
    if raw.iter().any(|(_, icc)| {
        icc.as_deref()
            .is_some_and(slashstitch_core::rawdev::is_raw_working_icc)
    }) {
        return None;
    }
    let first = raw.first()?.1.as_ref()?;
    if raw
        .iter()
        .all(|(_, icc)| icc.as_deref() == Some(first.as_slice()))
    {
        return TargetColorProfile::from_raw_icc(None, first.clone()).ok();
    }
    let preset_of = slashstitch_core::preset_match_for_icc;
    let preset = preset_of(first)?;
    raw.iter()
        .all(|(_, icc)| icc.as_deref().and_then(preset_of) == Some(preset))
        .then(|| TargetColorProfile::from_parts(preset, None, None).ok())
        .flatten()
}
fn rebuild_mips_for_target(project: &mut ProjectRuntime) -> Result<(), String> {
    let target = target_profile_from_doc(&project.doc)?;
    let rebuilt: Vec<_> = project
        .mip_raw
        .par_iter()
        .map(|(thumb16, src_icc)| target_mips(thumb16, src_icc.as_deref(), &target))
        .collect::<Result<_, _>>()?;
    let (m16, m8): (Vec<_>, Vec<_>) = rebuilt.into_iter().unzip();
    project.mip16 = m16;
    project.mip_rgba = m8;
    project.mip_generation = next_mip_generation();
    project.photometric_sig = 0;
    Ok(())
}
/// Canvas aspect used for rendering: equirectangular projections have a fixed aspect,
/// the others use the project's stored aspect (falling back to the current preview).
fn effective_aspect(doc: &ProjectDocument, fallback: f64) -> f64 {
    match doc.projection {
        PanoramaProjection::Equirectangular => 2.0,
        PanoramaProjection::TransverseEquirectangular => 0.5,
        _ => doc
            .canvas_aspect
            .filter(|a| a.is_finite() && (0.1..=20.0).contains(a))
            .unwrap_or(fallback),
    }
}
/// Fit a requested pixel box to the canvas aspect.
fn fit_to_aspect(w: u32, h: u32, aspect: f64) -> (u32, u32) {
    let (w, h) = (f64::from(w.max(16)), f64::from(h.max(16)));
    if w / h > aspect {
        (((h * aspect).round() as u32).max(16), h as u32)
    } else {
        (w as u32, ((w / aspect).round() as u32).max(16))
    }
}
/// Geometry + colour signature deciding when photometric parameters must be re-estimated.
fn photometric_signature(project: &ProjectRuntime) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    project.mip_generation.hash(&mut h);
    for im in &project.doc.images {
        im.path.hash(&mut h);
        for v in [
            im.yaw_deg,
            im.pitch_deg,
            im.roll_deg,
            im.hfov_deg,
            im.distortion[0],
            im.distortion[1],
            im.distortion[2],
            im.distortion_scale,
        ] {
            v.to_bits().hash(&mut h);
        }
        (im.lens_projection as u8).hash(&mut h);
        im.distortion_poly5
            .map(|p| (p[0].to_bits(), p[1].to_bits()))
            .hash(&mut h);
    }
    h.finish() | 1
}
fn refresh_photometric(project: &mut ProjectRuntime) -> Result<(), String> {
    let sig = photometric_signature(project);
    let n = project.doc.images.len();
    let current_ok = project
        .doc
        .photometric
        .as_ref()
        .is_some_and(|p| p.is_valid_for(n));
    if sig == project.photometric_sig && current_ok {
        return Ok(());
    }
    if n < 2 || project.mip16.len() != n {
        project.doc.photometric = (n > 0).then(|| PhotometricParams::identity(n));
    } else {
        let target = target_profile_from_doc(&project.doc)?;
        let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
        let small: Vec<&dyn Rgb16Raster> = project
            .mip16
            .iter()
            .map(|m| m.as_ref() as &dyn Rgb16Raster)
            .collect();
        project.doc.photometric = Some(estimate_photometric(&project.doc.images, &small, &trc));
    }
    project.photometric_sig = sig;
    Ok(())
}
fn active_photometric(doc: &ProjectDocument) -> Option<PhotometricParams> {
    if !doc.exposure_compensation {
        return None;
    }
    let mut p = doc
        .photometric
        .as_ref()
        .filter(|p| p.is_valid_for(doc.images.len()))?
        .clone();
    if !doc.correct_vignetting {
        p.vignetting.fill([0.0; 3]);
    }
    Some(p)
}
fn engine_info(shared: &SharedState) -> Value {
    let db = db(shared);
    let doc = shared.project.lock().unwrap().doc.clone();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu_threads": thread::available_parallelism().map(|v| v.get()).unwrap_or(1),
        "gpu_backend": shared.renderer.backend_label(),
        "lensfun_dir": db.source_dir,
        "lensfun_update_dir": Value::Null,
        "lensfun_files": db.files_loaded,
        "cameras_count": db.cameras.len(),
        "lenses_count": db.lenses.len(),
        "working_color_space": doc.color_profile,
        "output_bit_depth": doc.output_bit_depth,
        "supported_bit_depths": SUPPORTED_BIT_DEPTHS,
        "supported_color_profiles": SUPPORTED_COLOR_PROFILE_IDS,
        "max_icc_bytes": MAX_ICC_BYTES,
        "max_images": MAX_IMAGES,
        "memory_limit_bytes": slashstitch_core::memory::memory_limit(),
        "memory_reserved_bytes": slashstitch_core::memory::reserved_bytes(),
        "source_cache_bytes": shared.source_cache.held_bytes()
    })
}
fn pose_for(meta: &SourceImageMeta, id: usize, db: &LensDatabase) -> Result<SourceImage, String> {
    if let Some(kind) = meta.matched_lens_type.as_deref() {
        if !["rectilinear", "fisheye", "equisolid", "equirectangular"].contains(&kind) {
            return Err(format!("Lens projection '{kind}' is not supported; select a supported profile before importing"));
        }
    }
    let correction = db
        .find_lens(meta.lens_model.as_deref(), meta.matched_mount.as_deref())
        .and_then(|lens| {
            db.distortion_for(
                lens,
                meta.focal_length_mm
                    .or(meta.focal_length_35mm)
                    .unwrap_or(24.0),
                meta.matched_crop_factor.unwrap_or(lens.crop_factor),
            )
        });
    let (distortion, distortion_poly5, distortion_scale) =
        correction.unwrap_or(([0.0; 3], None, 1.0));
    let hfov = meta.estimated_hfov_deg.unwrap_or(65.0).clamp(
        5.0,
        if meta.estimated_lens_projection == LensProjection::Rectilinear {
            175.0
        } else {
            355.0
        },
    );
    Ok(SourceImage {
        id,
        path: meta.path.clone(),
        width: meta.width.ok_or("Missing dimensions")?,
        height: meta.height.ok_or("Missing dimensions")?,
        yaw_deg: id as f64 * hfov * 0.45,
        pitch_deg: 0.0,
        roll_deg: 0.0,
        hfov_deg: hfov,
        lens_projection: meta.estimated_lens_projection,
        distortion,
        distortion_poly5,
        distortion_scale,
    })
}
fn validate_pose(im: &SourceImage, index: usize) -> Result<(), String> {
    let finite = [
        im.yaw_deg,
        im.pitch_deg,
        im.roll_deg,
        im.hfov_deg,
        im.distortion_scale,
    ]
    .iter()
    .all(|v| v.is_finite())
        && im.distortion.iter().all(|v| v.is_finite())
        && im
            .distortion_poly5
            .is_none_or(|p| p.iter().all(|v| v.is_finite()));
    let max = if im.lens_projection == LensProjection::Rectilinear {
        175.0
    } else {
        355.0
    };
    if !finite
        || im.id != index
        || im.width < 2
        || im.height < 2
        || !(5.0..=max).contains(&im.hfov_deg)
        || !(0.01..=100.0).contains(&im.distortion_scale)
        || im.path.is_empty()
    {
        return Err(format!("Invalid geometry for source {index}"));
    }
    Ok(())
}
fn validate_document(doc: &ProjectDocument) -> Result<(), String> {
    if doc.schema_version != 1 {
        return Err("Unsupported project schema version".into());
    }
    validate_bit_depth(doc.output_bit_depth)?;
    let _ = target_profile_from_doc(doc)?;
    if doc.images.len() > MAX_IMAGES || doc.control_points.len() > MAX_CP {
        return Err("Project exceeds source/control-point budget".into());
    }
    if !doc.pano_hfov_deg.is_finite()
        || !(15.0..=360.0).contains(&doc.pano_hfov_deg)
        || !doc.view_yaw_deg.is_finite()
        || !doc.view_pitch_deg.is_finite()
        || doc.view_pitch_deg.abs() > 89.0
        || !doc.view_v_shift.is_finite()
        || doc.view_v_shift.abs() > 2.0
    {
        return Err("Invalid project view".into());
    }
    if doc
        .canvas_aspect
        .is_some_and(|a| !a.is_finite() || !(0.1..=20.0).contains(&a))
    {
        return Err("Invalid canvas aspect".into());
    }
    if !doc.blend_width_deg.is_finite() || !(0.25..=20.0).contains(&doc.blend_width_deg) {
        return Err("Blend width must be between 0.25 and 20 degrees".into());
    }
    for (i, im) in doc.images.iter().enumerate() {
        validate_pose(im, i)?;
    }
    for cp in &doc.control_points {
        let a = doc
            .images
            .get(cp.image_a)
            .ok_or("Invalid control-point image")?;
        let b = doc
            .images
            .get(cp.image_b)
            .ok_or("Invalid control-point image")?;
        if cp.image_a == cp.image_b
            || ![cp.x_a, cp.y_a, cp.x_b, cp.y_b]
                .iter()
                .all(|v| v.is_finite())
            || cp.x_a < 0.0
            || cp.y_a < 0.0
            || cp.x_b < 0.0
            || cp.y_b < 0.0
            || cp.x_a >= a.width as f64
            || cp.y_a >= a.height as f64
            || cp.x_b >= b.width as f64
            || cp.y_b >= b.height as f64
        {
            return Err("Invalid control-point coordinates".into());
        }
    }
    Ok(())
}
fn write_preview_png(
    shared: &SharedState,
    stem: &str,
    image: &RgbaImage,
    icc_bytes: Vec<u8>,
) -> Result<(String, u64), String> {
    use image::codecs::png::{CompressionType, FilterType};
    let version = shared.preview_version.fetch_add(1, Ordering::Relaxed) + 1;
    let path = shared
        .preview_dir
        .path()
        .join(format!("{stem}-{version}.png"));
    let temp = temporary_output(&path).map_err(|e| e.to_string())?;
    {
        let file = fs::File::create(temp.path()).map_err(|e| e.to_string())?;
        let mut enc = PngEncoder::new_with_quality(
            BufWriter::new(file),
            CompressionType::Fast,
            FilterType::Sub,
        );
        if !icc_bytes.is_empty() {
            let _ = enc.set_icc_profile(icc_bytes);
        }
        enc.write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgba8,
        )
        .map_err(|e| e.to_string())?;
    }
    publish(temp, &path, true).map_err(|e| e.to_string())?;
    // Keep the newest few images per stem; viewers may still be loading the previous one.
    if let Ok(entries) = fs::read_dir(shared.preview_dir.path()) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(rest) = name
                .strip_prefix(stem)
                .and_then(|r| r.strip_prefix('-'))
                .and_then(|r| r.strip_suffix(".png"))
            {
                if rest.parse::<u64>().is_ok_and(|v| v + 3 < version) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
    Ok((path.to_string_lossy().into(), version))
}
fn render_current_preview(shared: &SharedState) -> Result<PreviewMeta, String> {
    let project = shared.project.lock().map_err(|_| "Project lock poisoned")?;
    let target = target_profile_from_doc(&project.doc)?;
    let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
    let icc_bytes = target.icc_bytes()?;
    let aspect = effective_aspect(
        &project.doc,
        f64::from(project.preview_width) / f64::from(project.preview_height.max(1)),
    );
    let images: Vec<_> = project.mip_rgba.iter().map(AsRef::as_ref).collect();
    let rendered = shared.renderer.render(&PreviewRequest {
        out_w: project.preview_width,
        out_h: project.preview_height,
        projection: project.doc.projection,
        pano_hfov_deg: project.doc.pano_hfov_deg,
        view: BandViewParams {
            view_yaw_deg: project.doc.view_yaw_deg,
            view_pitch_deg: project.doc.view_pitch_deg,
            view_v_shift: project.doc.view_v_shift,
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: Some(aspect),
        },
        images: &project.doc.images,
        mips: &images,
        photometric: active_photometric(&project.doc).as_ref(),
        trc: &trc,
        source_key: Some(project.mip_generation),
    });
    let (width, height) = (project.preview_width, project.preview_height);
    drop(project);
    let (preview_path, version) = write_preview_png(shared, "preview", &rendered.image, icc_bytes)?;
    Ok(PreviewMeta {
        preview_path,
        width,
        height,
        backend: rendered.backend_used,
        render_ms: rendered.render_ms,
        color_profile: target.id().to_string(),
        color_profile_name: target.display_name(),
        version,
        canvas_aspect: aspect,
        region: [0.0, 0.0, 1.0, 1.0],
        exact: false,
    })
}
/// Render the fast preview and also publish it as an event, so the editor shows the new
/// image even when the host stopped waiting for the original request (late completion).
fn render_and_publish_preview(
    shared: &SharedState,
    out: &Arc<Mutex<io::Stdout>>,
) -> Result<PreviewMeta, String> {
    let meta = render_current_preview(shared)?;
    event(
        out,
        "preview-updated",
        serde_json::to_value(&meta).map_err(|e| e.to_string())?,
    );
    Ok(meta)
}
fn project_result(shared: &SharedState, preview: Option<PreviewMeta>) -> Value {
    let p = shared.project.lock().unwrap();
    json!({"images":p.doc.images,"metas":p.metas,"control_points":p.doc.control_points,"project":p.doc,"preview":preview})
}
fn commit(
    shared: &SharedState,
    out: &Arc<Mutex<io::Stdout>>,
    mut project: ProjectRuntime,
) -> Result<(), String> {
    check_cancel(shared)?;
    validate_document(&project.doc)?;
    let n = project.doc.images.len();
    if n != project.metas.len()
        || n != project.mip_rgba.len()
        || n != project.mip_raw.len()
        || n != project.mip16.len()
        || n != project.stamps.len()
    {
        return Err("Internal image/metadata/preview mismatch".into());
    }
    // Exposure/vignetting parameters follow geometry automatically.
    refresh_photometric(&mut project)?;
    // Exposure brackets (HDR stacks) depend on the order and set of sources.
    slashstitch_core::hdr::assign_brackets(&mut project.metas);
    let doc = project.doc.clone();
    let synced = json!({"images":doc.images,"metas":project.metas,"control_points":doc.control_points,"project":doc});
    let keep: Vec<(String, SourceStamp)> = doc
        .images
        .iter()
        .map(|im| im.path.clone())
        .zip(project.stamps.iter().copied())
        .collect();
    *shared.project.lock().map_err(|_| "Project lock poisoned")? = project;
    shared.source_cache.retain_sources(&keep);
    event(
        out,
        "project-changed",
        serde_json::to_value(doc).map_err(|e| e.to_string())?,
    );
    // Complete state for the editor, delivered even if the host stopped waiting for the
    // original request (late commit after a control timeout).
    event(out, "project-synced", synced);
    Ok(())
}
fn dimensions(payload: &Value, key: &str, default: u32, max: u32) -> Result<u32, String> {
    match payload.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0 && *v <= max)
            .ok_or_else(|| format!("Invalid {key}; maximum is {max}")),
    }
}
fn export_options(shared: &SharedState, p: &Value) -> Result<ExportOptions, String> {
    let (doc, canvas_aspect) = {
        let proj = shared.project.lock().unwrap();
        let fallback = proj.preview_width as f64 / proj.preview_height.max(1) as f64;
        (proj.doc.clone(), effective_aspect(&proj.doc, fallback))
    };
    let path = p
        .get("output_path")
        .and_then(Value::as_str)
        .filter(|p| !p.trim().is_empty())
        .ok_or("Export path is required.")?;
    let crop = match p.get("crop") {
        Some(v) => serde_json::from_value::<[f64; 4]>(v.clone()).map_err(|_| "Invalid crop")?,
        None => [0.0, 0.0, 1.0, 1.0],
    };
    let bit_depth = match p.get("bit_depth") {
        None | Some(Value::Null) => doc.output_bit_depth,
        Some(v) => v
            .as_u64()
            .and_then(|n| u8::try_from(n).ok())
            .ok_or("Invalid bit_depth")?,
    };
    let color_profile = match p.get("color_profile").and_then(Value::as_str) {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => doc.color_profile.clone(),
    };
    let custom_icc_name = match p.get("custom_icc_name") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) => None,
        _ => doc.custom_icc_name.clone(),
    };
    let custom_icc_base64 = match p.get("custom_icc_base64") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) => None,
        _ => doc.custom_icc_base64.clone(),
    };
    validate_bit_depth(bit_depth)?;
    let _ = TargetColorProfile::from_parts(
        &color_profile,
        custom_icc_name.as_deref(),
        custom_icc_base64.as_deref(),
    )?;
    let options = ExportOptions {
        output_path: path.into(),
        width: dimensions(p, "width", 4096, 200_000)?,
        height: dimensions(p, "height", 2048, 200_000)?,
        projection: doc.projection,
        pano_hfov_deg: doc.pano_hfov_deg,
        view_yaw_deg: doc.view_yaw_deg,
        view_pitch_deg: doc.view_pitch_deg,
        view_v_shift: doc.view_v_shift,
        crop,
        multiband: p.get("multiband").and_then(Value::as_bool).unwrap_or(true),
        force_bigtiff: p
            .get("force_bigtiff")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        overwrite: p.get("overwrite").and_then(Value::as_bool).unwrap_or(false),
        rows_per_strip: 128,
        canvas_aspect: Some(canvas_aspect),
        bit_depth,
        color_profile,
        custom_icc_name,
        custom_icc_base64,
        photometric: active_photometric(&doc),
        reject_transients: doc.reject_transients,
        interpolation: doc.interpolation,
        blend_width_deg: doc.blend_width_deg,
    };
    // Reject impossible outputs before any source is decoded or cached.
    preflight_export(&options, doc.images.len()).map_err(|e| e.to_string())?;
    Ok(options)
}
fn options(p: &Value) -> OptimizeOptions {
    OptimizeOptions {
        optimize_hfov: p
            .get("optimize_hfov")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        level_horizon: p
            .get("level_horizon")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        optimize_distortion: p
            .get("optimize_distortion")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        prioritize_sky: p
            .get("prioritize_sky")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    }
}

fn median_control_point_error(images: &[SourceImage], cps: &[ControlPoint]) -> Option<f64> {
    let mut errs: Vec<f64> = cps
        .iter()
        .filter_map(|cp| control_point_error_deg(images, cp))
        .collect();
    if errs.is_empty() {
        return None;
    }
    errs.sort_by(f64::total_cmp);
    Some(errs[errs.len() / 2])
}

fn prune_outlier_control_points(
    images: &[SourceImage],
    cps: &mut Vec<ControlPoint>,
    threshold_deg: f64,
    min_pair_cps: usize,
) -> bool {
    let mut pair_counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for cp in cps.iter() {
        let key = (cp.image_a.min(cp.image_b), cp.image_a.max(cp.image_b));
        *pair_counts.entry(key).or_insert(0) += 1;
    }
    let mut scored: Vec<(usize, f64)> = cps
        .iter()
        .enumerate()
        .filter_map(|(i, cp)| {
            let e = control_point_error_deg(images, cp).unwrap_or(f64::INFINITY);
            (e > threshold_deg).then_some((i, e))
        })
        .collect();
    if scored.is_empty() {
        return false;
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut remove = vec![false; cps.len()];
    let mut removed_any = false;
    for (idx, _) in scored {
        let key = (
            cps[idx].image_a.min(cps[idx].image_b),
            cps[idx].image_a.max(cps[idx].image_b),
        );
        if min_pair_cps > 0 && pair_counts.get(&key).copied().unwrap_or(0) <= min_pair_cps {
            continue;
        }
        remove[idx] = true;
        if let Some(c) = pair_counts.get_mut(&key) {
            *c -= 1;
        }
        removed_any = true;
    }
    if !removed_any {
        return false;
    }
    let candidate: Vec<ControlPoint> = cps
        .iter()
        .enumerate()
        .filter_map(|(i, cp)| (!remove[i]).then_some(cp.clone()))
        .collect();
    if candidate.len() >= 4 && connected_components(images.len(), &candidate).len() == 1 {
        *cps = candidate;
        true
    } else {
        false
    }
}

fn execute(
    shared: &SharedState,
    out: &Arc<Mutex<io::Stdout>>,
    action: &str,
    p: &Value,
) -> Result<Value, String> {
    match action {
        "load-images" => {
            let paths = p
                .get("paths")
                .and_then(Value::as_array)
                .ok_or("Image paths are required")?;
            let replace = p.get("replace").and_then(Value::as_bool).unwrap_or(false);
            let current = shared.project.lock().unwrap().clone();
            let mut next = if replace {
                ProjectRuntime {
                    preview_width: current.preview_width,
                    preview_height: current.preview_height,
                    ..ProjectRuntime::default()
                }
            } else {
                current
            };
            let started_empty = next.doc.images.is_empty();
            if paths.len() > MAX_IMAGES {
                return Err(format!(
                    "At most {MAX_IMAGES} sources are supported per project"
                ));
            }
            let target = target_profile_from_doc(&next.doc)?;
            let db = db(shared);
            let mut failures = vec![];
            // Resolve and de-duplicate first (cheap), then decode in parallel batches.
            let mut queued: Vec<String> = Vec::new();
            for path in paths {
                let Some(raw_path) = path.as_str() else {
                    return Err("Every image path must be a string".into());
                };
                match Path::new(raw_path).canonicalize() {
                    Ok(c) => {
                        let c = c.to_string_lossy().to_string();
                        if !next.doc.images.iter().any(|im| im.path == c) && !queued.contains(&c) {
                            queued.push(c);
                        }
                    }
                    Err(e) => failures.push(json!({"path":raw_path,"error":e.to_string()})),
                }
            }
            if next.doc.images.len() + queued.len() > MAX_IMAGES {
                return Err(format!(
                    "At most {MAX_IMAGES} sources are supported per project"
                ));
            }
            let total = queued.len().max(1);
            let finished = std::sync::atomic::AtomicUsize::new(0);
            let sizes: Vec<u64> = queued
                .iter()
                .map(|c| sources::header_pixels(Path::new(c)))
                .collect();
            for batch_idx in sources::budget_batches(&sizes) {
                check_cancel(shared)?;
                // Reserve the batch's decode memory on this (non-pool) thread before the
                // parallel section, so no rayon worker ever blocks on the memory ceiling.
                let _memory = slashstitch_core::memory::reserve(
                    batch_idx.iter().map(|&k| sizes[k]).sum::<u64>() * sources::DECODE_BYTES_PER_PX,
                    Some(&shared.cancel),
                )
                .ok_or("Cancelled")?;
                let batch: Vec<String> = batch_idx.iter().map(|&k| queued[k].clone()).collect();
                let decoded: Vec<Result<_, String>> = batch
                    .par_iter()
                    .map(|canonical| {
                        let path = Path::new(canonical);
                        let stamp = SourceStamp::of(path)?;
                        let mut meta = inspect_image(path, &db);
                        let raw = decode_source_raw_16(path)?;
                        meta.width = Some(raw.image_rgb16.width());
                        meta.height = Some(raw.image_rgb16.height());
                        meta.bit_depth = Some(raw.source_bit_depth);
                        meta.warnings.extend(raw.warnings.iter().cloned());
                        let thumb16 = Arc::new(thumbnail_rgb16(&raw.image_rgb16, 1024));
                        let icc = raw.icc_profile;
                        let mips = target_mips(&thumb16, icc.as_deref(), &target)?;
                        let done = finished.fetch_add(1, Ordering::Relaxed) + 1;
                        progress(
                            out,
                            ((done * 75) / total) as u32,
                            format!("Loading images ({done}/{total})"),
                        );
                        Ok((meta, (thumb16, icc), mips, stamp))
                    })
                    .collect();
                for (canonical, result) in batch.iter().zip(decoded) {
                    match result.and_then(|(meta, raw_pair, mips, stamp)| {
                        let pose = pose_for(&meta, next.doc.images.len(), &db)?;
                        Ok((pose, meta, raw_pair, mips, stamp))
                    }) {
                        Ok((pose, meta, raw_pair, (m16, m8), stamp)) => {
                            next.doc.images.push(SourceImage {
                                path: canonical.clone(),
                                ..pose
                            });
                            next.metas.push(meta);
                            next.mip_raw.push(raw_pair);
                            next.mip16.push(m16);
                            next.mip_rgba.push(m8);
                            next.stamps.push(stamp);
                        }
                        Err(error) => failures.push(json!({"path":canonical,"error":error})),
                    }
                }
            }
            next.mip_generation = next_mip_generation();
            if next.doc.images.is_empty() && !failures.is_empty() {
                return Err(format!("No images imported: {}", Value::Array(failures)));
            }
            // A new project keeps the sources' colour space when they all share one embedded
            // RGB profile (e.g. Adobe RGB from the camera/Lightroom), so wide-gamut colour is not
            // clipped to sRGB by default. Mixed or missing profiles keep sRGB.
            let mut adopted_profile = None;
            if started_empty
                && next.doc.color_profile == "srgb"
                && next.doc.custom_icc_base64.is_none()
            {
                if let Some(target) = common_source_profile(&next.mip_raw) {
                    if target.id() != "srgb" {
                        next.doc.color_profile = target.id().to_string();
                        next.doc.custom_icc_name = None;
                        next.doc.custom_icc_base64 = None;
                        if target.id() == "custom_icc" {
                            next.doc.custom_icc_name = Some(target.display_name());
                            next.doc.custom_icc_base64 = target.custom_icc_base64();
                        }
                        rebuild_mips_for_target(&mut next)?;
                        adopted_profile = Some(target.display_name());
                    }
                }
            }
            commit(shared, out, next)?;
            let preview = render_and_publish_preview(shared, out)?;
            let mut result = project_result(shared, Some(preview));
            result["failures"] = json!(failures);
            result["adopted_color_profile"] = json!(adopted_profile);
            Ok(result)
        }
        "clear-images" => {
            commit(shared, out, ProjectRuntime::default())?;
            Ok(project_result(shared, None))
        }
        "snapshot-project" => {
            let mut pstate = shared.project.lock().unwrap();
            if let Some(ui) = p.get("ui") {
                pstate.doc.ui = ui.clone();
            }
            Ok(serde_json::to_value(&pstate.doc).map_err(|e| e.to_string())?)
        }
        "restore-project" | "restore-poses" => {
            let old = shared.project.lock().unwrap().clone();
            let mut document = if action == "restore-project" {
                serde_json::from_value::<ProjectDocument>(p.clone())
                    .map_err(|e| format!("Invalid project: {e}"))?
            } else {
                let mut doc = old.doc.clone();
                doc.images =
                    serde_json::from_value(p.get("images").cloned().ok_or("Missing images")?)
                        .map_err(|e| format!("Invalid poses: {e}"))?;
                doc.control_points =
                    serde_json::from_value(p.get("control_points").cloned().unwrap_or(json!([])))
                        .map_err(|e| format!("Invalid control points: {e}"))?;
                if let Some(v) = p.get("projection") {
                    doc.projection =
                        serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
                }
                for (key, slot) in [
                    ("pano_hfov_deg", &mut doc.pano_hfov_deg),
                    ("view_yaw_deg", &mut doc.view_yaw_deg),
                    ("view_pitch_deg", &mut doc.view_pitch_deg),
                    ("view_v_shift", &mut doc.view_v_shift),
                ] {
                    if let Some(v) = p.get(key) {
                        *slot = v.as_f64().ok_or("Invalid view value")?;
                    }
                }
                if let Some(v) = p.get("output_bit_depth").and_then(Value::as_u64) {
                    doc.output_bit_depth =
                        u8::try_from(v).map_err(|_| "Invalid output_bit_depth")?;
                }
                if let Some(v) = p.get("color_profile").and_then(Value::as_str) {
                    doc.color_profile = v.to_string();
                }
                if let Some(v) = p.get("custom_icc_name") {
                    doc.custom_icc_name = v.as_str().map(str::to_string);
                }
                if let Some(v) = p.get("custom_icc_base64") {
                    doc.custom_icc_base64 = v.as_str().map(str::to_string);
                }
                doc
            };
            validate_document(&document)?;
            let old_target = target_profile_from_doc(&old.doc).ok();
            let target = target_profile_from_doc(&document)?;
            let target_changed = old_target.as_ref().map(sources::profile_signature)
                != Some(sources::profile_signature(&target));
            let db = db(shared);
            let mut next = ProjectRuntime {
                doc: document.clone(),
                preview_width: old.preview_width,
                preview_height: old.preview_height,
                ..ProjectRuntime::default()
            };
            for im in &mut document.images {
                im.path = Path::new(&im.path)
                    .canonicalize()
                    .map_err(|e| format!("Missing source {}: {e}", im.path))?
                    .to_string_lossy()
                    .into();
            }
            // Reuse cached metadata/mips only when the file on disk is byte-for-byte the same
            // source (path, dimensions, size and modification time all match).
            enum Slot {
                Reuse(usize),
                Decode,
            }
            let mut slots = Vec::with_capacity(document.images.len());
            let mut stamps = Vec::with_capacity(document.images.len());
            for im in &document.images {
                let stamp = SourceStamp::of(Path::new(&im.path))?;
                stamps.push(stamp);
                slots.push(
                    match old.doc.images.iter().enumerate().position(|(k, v)| {
                        v.path == im.path
                            && v.width == im.width
                            && v.height == im.height
                            && old.stamps.get(k) == Some(&stamp)
                    }) {
                        Some(k) => Slot::Reuse(k),
                        None => Slot::Decode,
                    },
                );
            }
            let to_decode: Vec<usize> = slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| matches!(s, Slot::Decode).then_some(i))
                .collect();
            let total = to_decode.len().max(1);
            let finished = std::sync::atomic::AtomicUsize::new(0);
            let mut decoded: std::collections::HashMap<usize, _> = std::collections::HashMap::new();
            let sizes: Vec<u64> = to_decode
                .iter()
                .map(|&i| {
                    sources::decode_cost_pixels(
                        Path::new(&document.images[i].path),
                        u64::from(document.images[i].width) * u64::from(document.images[i].height),
                    )
                })
                .collect();
            for batch_idx in sources::budget_batches(&sizes) {
                check_cancel(shared)?;
                let _memory = slashstitch_core::memory::reserve(
                    batch_idx.iter().map(|&k| sizes[k]).sum::<u64>() * sources::DECODE_BYTES_PER_PX,
                    Some(&shared.cancel),
                )
                .ok_or("Cancelled")?;
                let batch: Vec<usize> = batch_idx.iter().map(|&k| to_decode[k]).collect();
                let results: Vec<Result<_, String>> = batch
                    .par_iter()
                    .map(|&i| {
                        let im = &document.images[i];
                        let path = Path::new(&im.path);
                        let mut meta = inspect_image(path, &db);
                        let raw = decode_source_raw_16(path)?;
                        let (rw, rh) = (raw.image_rgb16.width(), raw.image_rgb16.height());
                        if (rw, rh) != (im.width, im.height) {
                            return Err(format!("Source dimensions changed: {}", im.path));
                        }
                        meta.width = Some(rw);
                        meta.height = Some(rh);
                        meta.bit_depth = Some(raw.source_bit_depth);
                        meta.warnings.extend(raw.warnings.iter().cloned());
                        let thumb16 = Arc::new(thumbnail_rgb16(&raw.image_rgb16, 1024));
                        let icc = raw.icc_profile;
                        let mips = target_mips(&thumb16, icc.as_deref(), &target)?;
                        let done = finished.fetch_add(1, Ordering::Relaxed) + 1;
                        progress(
                            out,
                            ((done * 90) / total) as u32,
                            format!("Reading sources ({done}/{total})"),
                        );
                        Ok((i, meta, (thumb16, icc), mips))
                    })
                    .collect();
                for r in results {
                    let (i, meta, raw_pair, mips) = r?;
                    decoded.insert(i, (meta, raw_pair, mips));
                }
            }
            for (i, slot) in slots.into_iter().enumerate() {
                match slot {
                    Slot::Reuse(k) => {
                        next.metas.push(old.metas[k].clone());
                        let raw_pair = old.mip_raw[k].clone();
                        let (m16, m8) = if target_changed {
                            target_mips(&raw_pair.0, raw_pair.1.as_deref(), &target)?
                        } else {
                            (Arc::clone(&old.mip16[k]), Arc::clone(&old.mip_rgba[k]))
                        };
                        next.mip_raw.push(raw_pair);
                        next.mip16.push(m16);
                        next.mip_rgba.push(m8);
                    }
                    Slot::Decode => {
                        let (meta, raw_pair, (m16, m8)) = decoded
                            .remove(&i)
                            .ok_or("Internal decode bookkeeping error")?;
                        next.metas.push(meta);
                        next.mip_raw.push(raw_pair);
                        next.mip16.push(m16);
                        next.mip_rgba.push(m8);
                    }
                }
            }
            next.stamps = stamps;
            next.mip_generation = next_mip_generation();
            // Older projects stored no canvas aspect; pin it now so later window changes
            // cannot alter the exported geometry of this project.
            if document.canvas_aspect.is_none()
                && !matches!(
                    document.projection,
                    PanoramaProjection::Equirectangular
                        | PanoramaProjection::TransverseEquirectangular
                )
            {
                document.canvas_aspect =
                    Some(f64::from(old.preview_width) / f64::from(old.preview_height.max(1)));
            }
            if action == "restore-poses" {
                // Undo/redo changes geometry: re-estimate instead of reusing stale parameters.
                document.photometric = None;
            }
            next.doc = document;
            // A saved project reopens exactly as saved: trust its photometric parameters for
            // the stored geometry (they are re-estimated as soon as the geometry changes).
            if next
                .doc
                .photometric
                .as_ref()
                .is_some_and(|ph| ph.is_valid_for(next.doc.images.len()))
            {
                next.photometric_sig = photometric_signature(&next);
            }
            commit(shared, out, next)?;
            Ok(project_result(
                shared,
                Some(render_and_publish_preview(shared, out)?),
            ))
        }
        "set-color-settings" => {
            let mut next = shared.project.lock().unwrap().clone();
            let prev_signature = (
                next.doc.color_profile.clone(),
                next.doc.custom_icc_base64.clone(),
            );
            if let Some(v) = p.get("output_bit_depth").and_then(Value::as_u64) {
                let bd = u8::try_from(v).map_err(|_| "Invalid output_bit_depth")?;
                validate_bit_depth(bd)?;
                next.doc.output_bit_depth = bd;
            }
            if let Some(idx) = p.get("adopt_source_icc_index").and_then(Value::as_u64) {
                let idx = idx as usize;
                let raw_pair = next
                    .mip_raw
                    .get(idx)
                    .ok_or_else(|| format!("Unknown source index {idx}"))?;
                let meta = next
                    .metas
                    .get(idx)
                    .ok_or_else(|| format!("Unknown source index {idx}"))?;
                let bytes = raw_pair.1.as_deref().ok_or_else(|| {
                    "Selected source image does not contain an embedded ICC profile".to_string()
                })?;
                let parsed = TargetColorProfile::from_raw_icc(
                    meta.icc_profile_name.as_deref(),
                    bytes.to_vec(),
                )?;
                next.doc.color_profile = parsed.id().to_string();
                next.doc.custom_icc_name = None;
                next.doc.custom_icc_base64 = None;
                if parsed.id() == "custom_icc" {
                    next.doc.custom_icc_name = Some(parsed.display_name());
                    next.doc.custom_icc_base64 = parsed.custom_icc_base64();
                }
            } else if let Some(icc_path) = p.get("custom_icc_path").and_then(Value::as_str) {
                let bytes = fs::read(icc_path)
                    .map_err(|e| format!("Could not read ICC profile '{icc_path}': {e}"))?;
                if bytes.len() > MAX_ICC_BYTES {
                    return Err(format!(
                        "ICC profile exceeds maximum allowed size of {MAX_ICC_BYTES} bytes"
                    ));
                }
                let file_name = Path::new(icc_path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_string);
                let parsed = TargetColorProfile::from_raw_icc(file_name.as_deref(), bytes)?;
                next.doc.color_profile = parsed.id().to_string();
                next.doc.custom_icc_name = None;
                next.doc.custom_icc_base64 = None;
                if parsed.id() == "custom_icc" {
                    next.doc.custom_icc_name = Some(parsed.display_name());
                    next.doc.custom_icc_base64 = parsed.custom_icc_base64();
                }
            } else {
                if let Some(v) = p.get("color_profile").and_then(Value::as_str) {
                    next.doc.color_profile = v.to_string();
                }
                if let Some(v) = p.get("custom_icc_name") {
                    next.doc.custom_icc_name = v.as_str().map(str::to_string);
                }
                if let Some(v) = p.get("custom_icc_base64") {
                    next.doc.custom_icc_base64 = v.as_str().map(str::to_string);
                }
            }
            validate_document(&next.doc)?;
            let next_signature = (
                next.doc.color_profile.clone(),
                next.doc.custom_icc_base64.clone(),
            );
            if prev_signature != next_signature {
                rebuild_mips_for_target(&mut next)?;
            }
            commit(shared, out, next)?;
            Ok(project_result(
                shared,
                Some(render_and_publish_preview(shared, out)?),
            ))
        }
        "update-image-pose" => {
            let mut next = shared.project.lock().unwrap().clone();
            let id = p
                .get("id")
                .and_then(Value::as_u64)
                .ok_or("Image id is required")? as usize;
            let img = next.doc.images.get_mut(id).ok_or("Unknown image id")?;
            for (key, slot) in [
                ("yaw_deg", &mut img.yaw_deg),
                ("pitch_deg", &mut img.pitch_deg),
                ("roll_deg", &mut img.roll_deg),
                ("hfov_deg", &mut img.hfov_deg),
            ] {
                if let Some(v) = p.get(key) {
                    *slot = v.as_f64().ok_or("Invalid pose value")?;
                }
            }
            if let Some(v) = p.get("lens_projection") {
                img.lens_projection =
                    serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
            }
            if let Some(arr) = p.get("distortion").and_then(Value::as_array) {
                if arr.len() != 3 {
                    return Err("distortion must contain 3 numbers".into());
                }
                for (slot, val) in img.distortion.iter_mut().zip(arr.iter()) {
                    *slot = val.as_f64().ok_or("Invalid distortion coefficient")?;
                }
            }
            validate_pose(img, id)?;
            commit(shared, out, next)?;
            Ok(project_result(
                shared,
                Some(render_and_publish_preview(shared, out)?),
            ))
        }
        "set-view" | "render-preview" => {
            let mut next = shared.project.lock().unwrap().clone();
            let previous_projection = next.doc.projection;
            if let Some(v) = p.get("projection") {
                next.doc.projection =
                    serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
            }
            for (key, slot) in [
                ("pano_hfov_deg", &mut next.doc.pano_hfov_deg),
                ("view_yaw_deg", &mut next.doc.view_yaw_deg),
                ("view_pitch_deg", &mut next.doc.view_pitch_deg),
                ("view_v_shift", &mut next.doc.view_v_shift),
            ] {
                if let Some(v) = p.get(key) {
                    *slot = v.as_f64().ok_or("Invalid view value")?;
                }
            }
            let req_w = dimensions(p, "width", next.preview_width, 4096)?.max(256);
            let req_h = dimensions(p, "height", next.preview_height, 4096)?.max(200);
            // The canvas aspect is project state: set explicitly, or pinned when the user picks
            // a projection (from the current viewport). Plain window resizes never change it.
            if action == "set-view" {
                if let Some(a) = p.get("canvas_aspect").and_then(Value::as_f64) {
                    next.doc.canvas_aspect = Some(a);
                } else if (next.doc.projection != previous_projection
                    || next.doc.canvas_aspect.is_none())
                    && !matches!(
                        next.doc.projection,
                        PanoramaProjection::Equirectangular
                            | PanoramaProjection::TransverseEquirectangular
                    )
                {
                    next.doc.canvas_aspect = Some(f64::from(req_w) / f64::from(req_h));
                }
            }
            let aspect = effective_aspect(&next.doc, f64::from(req_w) / f64::from(req_h));
            let (fw, fh) = fit_to_aspect(req_w, req_h, aspect);
            next.preview_width = fw.min(4096);
            next.preview_height = fh.min(4096);
            validate_document(&next.doc)?;
            check_cancel(shared)?;
            if action == "set-view" {
                commit(shared, out, next)?;
            } else {
                *shared.project.lock().unwrap() = next;
            }
            serde_json::to_value(render_and_publish_preview(shared, out)?)
                .map_err(|e| e.to_string())
        }
        "auto-crop" => Ok(json!({"crop":compute_auto_crop(shared)?})),
        "auto-align" | "preview-start" | "demo-start" | "reoptimize" => {
            let mut next = shared.project.lock().unwrap().clone();
            if next.doc.images.is_empty() && action != "reoptimize" {
                for step in 1..=6 {
                    check_cancel(shared)?;
                    progress(out, step * 16, "Rendering preview");
                    thread::sleep(Duration::from_millis(80));
                }
                let preview = render_and_publish_preview(shared, out)?;
                event(
                    out,
                    "preview-ready",
                    serde_json::to_value(preview).map_err(|e| e.to_string())?,
                );
                return Ok(json!({"preview":true}));
            }
            if next.doc.images.len() < 2 {
                return Err("At least two overlapping source images are required".into());
            }
            let opt = options(p);
            let mut cps = if action == "reoptimize" {
                next.doc.control_points.clone()
            } else {
                progress(
                    out,
                    15,
                    "Preparing oriented source previews for keypoint detection",
                );
                let loaded: Vec<_> = next
                    .mip_rgba
                    .iter()
                    .map(|im| DynamicImage::ImageRgba8((**im).clone()))
                    .collect();
                progress(out, 45, "Matching keypoints & spherical RANSAC");
                let (mut cps, rotations) =
                    find_all_control_points_cancel(&next.doc.images, &loaded, &shared.cancel)?;
                cps.truncate(MAX_CP);
                seed_poses_from_pairs(&mut next.doc.images, &rotations);
                check_cancel(shared)?;
                if opt.prioritize_sky
                    && cps.len() >= 4
                    && connected_components(next.doc.images.len(), &cps).len() == 1
                {
                    progress(
                        out,
                        60,
                        "Initial pose pass & refining sky star control points",
                    );
                    let init_opt = OptimizeOptions {
                        optimize_distortion: false,
                        ..opt
                    };
                    let _ = optimize_poses_cancel(
                        &mut next.doc.images,
                        &cps,
                        15,
                        init_opt,
                        &shared.cancel,
                    )?;
                    let star_cps =
                        refine_star_control_points(&next.doc.images, &loaded, &shared.cancel)?;
                    let remaining = MAX_CP.saturating_sub(cps.len());
                    cps.extend(star_cps.into_iter().take(remaining));
                }
                cps
            };
            check_cancel(shared)?;
            let components = connected_components(next.doc.images.len(), &cps);
            if cps.len() < 4 || components.len() != 1 {
                return Err(format!("Alignment needs a connected overlap graph; {} control points, disconnected image groups: {:?}",cps.len(),components));
            }
            progress(out, 75, "Robust bundle adjustment");
            let mut report =
                optimize_poses_cancel(&mut next.doc.images, &cps, 45, opt, &shared.cancel)?;
            let initial_rms = report.initial_rms_deg;
            let detected = cps.len();
            // RANSAC is pairwise; global fitting can still reveal false matches
            // or moving scene features (e.g. aurora curtains). Reject gross residuals,
            // then apply adaptive median-based pruning while protecting graph edges.
            for _ in 0..2 {
                let before = cps.len();
                cps.retain(|cp| {
                    control_point_error_deg(&next.doc.images, cp).is_some_and(|e| e <= 2.0)
                });
                if cps.len() == before {
                    break;
                }
                if cps.len() < 4 || connected_components(next.doc.images.len(), &cps).len() != 1 {
                    return Err("Global outlier rejection left disconnected source images; review overlap and lens geometry".into());
                }
                report =
                    optimize_poses_cancel(&mut next.doc.images, &cps, 45, opt, &shared.cancel)?;
            }
            for _ in 0..2 {
                let Some(med) = median_control_point_error(&next.doc.images, &cps) else {
                    break;
                };
                let thresh = (4.0 * med).clamp(0.45, 1.5);
                if !prune_outlier_control_points(&next.doc.images, &mut cps, thresh, 6) {
                    break;
                }
                report =
                    optimize_poses_cancel(&mut next.doc.images, &cps, 45, opt, &shared.cancel)?;
            }
            if report.max_error_deg > 2.05 {
                return Err(
                    "Alignment retains excessive global error; project was preserved".into(),
                );
            }
            report.initial_rms_deg = initial_rms;
            report.control_points_rejected = detected.saturating_sub(cps.len());
            next.doc.control_points = cps.clone();
            commit(shared, out, next)?;
            let mut result = project_result(shared, Some(render_and_publish_preview(shared, out)?));
            result["report"] = serde_json::to_value(report).map_err(|e| e.to_string())?;
            result["controlPointsCount"] = json!(cps.len());
            if action != "reoptimize" {
                event(out, "align-complete", result.clone());
            }
            Ok(result)
        }
        "export-panorama" => {
            let mut opts = export_options(shared, p)?;
            let (_ext, target) = validate_export_compatibility(
                &opts.output_path,
                opts.bit_depth,
                &opts.color_profile,
                opts.custom_icc_name.as_deref(),
                opts.custom_icc_base64.as_deref(),
            )?;
            let (images, stamps, project_mips, doc, mip_generation, mip_raw) = {
                let proj = shared.project.lock().unwrap();
                (
                    proj.doc.images.clone(),
                    proj.stamps.clone(),
                    proj.mip16.clone(),
                    proj.doc.clone(),
                    proj.mip_generation,
                    proj.mip_raw.clone(),
                )
            };
            // Preview mips and photometry live in the project's profile. An export to another
            // profile gets mips converted to that profile and photometry re-estimated there,
            // so no colour space is ever mixed under the embedded ICC label.
            let same_space = target_profile_from_doc(&doc)
                .map(|p| sources::profile_signature(&p) == sources::profile_signature(&target))
                .unwrap_or(false);
            let mip16: Vec<Arc<Rgb16Image>> = if same_space {
                project_mips.clone()
            } else {
                mip_raw
                    .par_iter()
                    .map(|(thumb, icc)| target_mips(thumb, icc.as_deref(), &target).map(|m| m.0))
                    .collect::<Result<_, _>>()?
            };
            if !same_space && opts.photometric.is_some() {
                let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
                let small: Vec<&dyn Rgb16Raster> = mip16
                    .iter()
                    .map(|m| m.as_ref() as &dyn Rgb16Raster)
                    .collect();
                let mut p = estimate_photometric(&images, &small, &trc);
                if !doc.correct_vignetting {
                    p.vignetting.fill([0.0; 3]);
                }
                opts.photometric = Some(p);
            }
            if images.is_empty() {
                return Err("No source images to export".into());
            }
            verify_stamps(&images, &stamps)?;
            // Full-resolution, colour-converted source pyramids: built once per session (in
            // parallel) and shared with the exact preview; later exports start rendering at once.
            let pyramids = shared.source_cache.get_or_build(
                &images,
                &stamps,
                &target,
                &shared.cancel,
                &|done, total| {
                    progress(
                        out,
                        ((done * 25) / total.max(1)) as u32,
                        format!(
                            "Decoding & colour-converting full-resolution sources ({done}/{total})"
                        ),
                    );
                },
            )?;
            check_cancel(shared)?;
            let sources: Vec<Vec<&dyn Rgb16Raster>> = pyramids
                .iter()
                .zip(mip16.iter())
                .map(|(pyr, mip)| {
                    let mut levels: Vec<&dyn Rgb16Raster> =
                        pyr.levels.iter().map(|l| l as &dyn Rgb16Raster).collect();
                    levels.push(mip.as_ref());
                    levels
                })
                .collect();
            let seams = if opts.multiband {
                progress(out, 25, "Optimising seams");
                // Seams come from the project's own mips: identical to the preview's layout.
                seam_map_for(
                    shared,
                    &doc,
                    &project_mips,
                    mip_generation,
                    opts.canvas_aspect.unwrap_or(2.0),
                    Some(&shared.cancel),
                )?
            } else {
                None
            };
            check_cancel(shared)?;
            let summary = export_panorama_with_sources(
                &opts,
                &images,
                sources,
                seams.as_deref(),
                Some(&shared.cancel),
                |rows, total| {
                    progress(
                        out,
                        25 + (rows as f64 / total.max(1) as f64 * 70.0) as u32,
                        format!("Rendering & blending strips ({rows}/{total} rows)"),
                    );
                    !shared.cancel.load(Ordering::Relaxed)
                },
            )
            .map_err(|e| e.to_string())?;
            let result = serde_json::to_value(summary).map_err(|e| e.to_string())?;
            event(out, "export-complete", result.clone());
            Ok(result)
        }
        "validate-lensfun" => {
            let dir = p
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Database path required")?;
            let loaded = LensDatabase::load_from_dir(Path::new(dir)).map_err(|e| e.to_string())?;
            Ok(
                json!({"files":loaded.files_loaded,"cameras":loaded.cameras.len(),"lenses":loaded.lenses.len()}),
            )
        }
        "reload-lensfun" => {
            let fresh = if let Some(path) = p.get("path").and_then(Value::as_str) {
                LensDatabase::load_from_dir(Path::new(path))
                    .map_err(|e| e.to_string())?
                    .with_builtin()
            } else {
                LensDatabase::load_default()
            };
            *shared.lens_db.lock().unwrap() = Arc::new(fresh);
            // Existing project geometry stays stable until the user explicitly
            // requests profile reapplication; a DB reload must not move images.
            Ok(engine_info(shared))
        }
        "reapply-lensfun" => {
            let mut next = shared.project.lock().unwrap().clone();
            let db = db(shared);
            for (im, meta) in next.doc.images.iter_mut().zip(next.metas.iter_mut()) {
                check_cancel(shared)?;
                *meta = inspect_image(Path::new(&im.path), &db);
                let updated = pose_for(meta, im.id, &db)?;
                im.distortion = updated.distortion;
                im.distortion_poly5 = updated.distortion_poly5;
                im.distortion_scale = updated.distortion_scale;
            }
            next.doc.control_points.clear();
            commit(shared, out, next)?;
            Ok(project_result(
                shared,
                Some(render_and_publish_preview(shared, out)?),
            ))
        }
        "set-render-settings" => {
            let mut next = shared.project.lock().unwrap().clone();
            if let Some(v) = p.get("exposure_compensation") {
                next.doc.exposure_compensation = v
                    .as_bool()
                    .ok_or("exposure_compensation must be a boolean")?;
            }
            if let Some(v) = p.get("correct_vignetting") {
                next.doc.correct_vignetting =
                    v.as_bool().ok_or("correct_vignetting must be a boolean")?;
            }
            if let Some(v) = p.get("interpolation") {
                next.doc.interpolation = serde_json::from_value(v.clone())
                    .map_err(|_| "interpolation must be bilinear, spline36 or lanczos3")?;
            }
            if let Some(v) = p.get("blend_width_deg") {
                next.doc.blend_width_deg = v.as_f64().ok_or("blend_width_deg must be a number")?;
            }
            if let Some(v) = p.get("optimize_seams") {
                next.doc.optimize_seams = v.as_bool().ok_or("optimize_seams must be a boolean")?;
            }
            if let Some(v) = p.get("reject_transients") {
                next.doc.reject_transients =
                    v.as_bool().ok_or("reject_transients must be a boolean")?;
            }
            commit(shared, out, next)?;
            Ok(project_result(
                shared,
                Some(render_and_publish_preview(shared, out)?),
            ))
        }
        "inspect-images" => {
            let paths = p
                .get("paths")
                .and_then(Value::as_array)
                .ok_or("Paths required")?;
            let db = db(shared);
            let mut result = vec![];
            for path in paths {
                check_cancel(shared)?;
                result.push(inspect_image(
                    Path::new(path.as_str().ok_or("Invalid path")?),
                    &db,
                ));
            }
            serde_json::to_value(result).map_err(|e| e.to_string())
        }
        _ => Err(format!("Unknown worker action: {action}")),
    }
}

fn start_job(shared: &Arc<SharedState>, out: &Arc<Mutex<io::Stdout>>, req: Request) {
    let asynchronous = matches!(
        req.action.as_str(),
        "auto-align" | "preview-start" | "demo-start" | "export-panorama"
    );
    if *shared.job_state.lock().unwrap() != JobState::Idle {
        reply(
            out,
            &req.id,
            Err("A job is already active; wait for finalization".into()),
        );
        return;
    }
    // Validation failures cannot emit a fictitious running state.
    if req.action == "export-panorama" {
        if let Err(e) = export_options(shared, &req.payload) {
            reply(out, &req.id, Err(e));
            return;
        }
    }
    {
        let mut busy = shared.job_state.lock().unwrap();
        if *busy != JobState::Idle {
            reply(
                out,
                &req.id,
                Err("A job is already active; wait for finalization".into()),
            );
            return;
        }
        *busy = JobState::Running;
    }
    shared.cancel.store(false, Ordering::Relaxed);
    event(out, "job-state", json!({"state":"running"}));
    if let Some(previous) = shared.worker_thread.lock().unwrap().take() {
        let _ = previous.join();
    }
    if asynchronous {
        reply(out, &req.id, Ok(json!({"started":true})));
    }
    let shared_bg = Arc::clone(shared);
    let out_bg = Arc::clone(out);
    let handle = thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(&shared_bg, &out_bg, &req.action, &req.payload)
        }))
        .unwrap_or_else(|_| {
            Err("Backend operation failed; restart the engine before continuing".into())
        });
        state(&shared_bg, &out_bg, JobState::Finalizing);
        if asynchronous {
            if let Err(error) = result {
                event(
                    &out_bg,
                    if shared_bg.cancel.load(Ordering::Relaxed) {
                        "job-cancelled"
                    } else {
                        "worker-error"
                    },
                    json!({"text":error}),
                );
            }
        } else {
            // A synchronous reply means both the operation and finalization are
            // complete. Callers may issue their next command immediately.
            state(&shared_bg, &out_bg, JobState::Idle);
            reply(&out_bg, &req.id, result);
            return;
        }
        state(&shared_bg, &out_bg, JobState::Idle);
    });
    *shared.worker_thread.lock().unwrap() = Some(handle);
}

/// Exact preview: the export renderer (photometry, multi-band, level selection) for the
/// requested canvas region at the requested pixel size. Runs off the job system; a newer
/// request cancels an older one.
fn spawn_exact_preview(
    shared: &Arc<SharedState>,
    out: &Arc<Mutex<io::Stdout>>,
    id: &str,
    payload: Value,
) {
    let token = payload.get("token").and_then(Value::as_u64).unwrap_or(0);
    let lane = payload
        .get("lane")
        .and_then(Value::as_str)
        .unwrap_or("base")
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(16)
        .collect::<String>();
    let flag = Arc::new(AtomicBool::new(false));
    {
        let mut lanes = shared.exact_cancel.lock().unwrap();
        if let Some(previous) = lanes.insert(lane, Arc::clone(&flag)) {
            previous.store(true, Ordering::Relaxed);
        }
    }
    reply(out, id, Ok(json!({"started":true,"token":token})));
    let shared = Arc::clone(shared);
    let out = Arc::clone(out);
    thread::spawn(move || {
        let _lane = shared
            .exact_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if flag.load(Ordering::Relaxed) {
            return;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render_exact_preview(&shared, &out, &payload, &flag, token)
        }))
        .unwrap_or_else(|_| Err("Exact preview renderer failed".into()));
        if flag.load(Ordering::Relaxed) {
            return;
        }
        match result {
            Ok(meta) => {
                let mut v = serde_json::to_value(meta).unwrap_or(Value::Null);
                v["token"] = json!(token);
                event(&out, "exact-preview-ready", v);
            }
            Err(text) => event(
                &out,
                "exact-preview-failed",
                json!({"token":token,"text":text}),
            ),
        }
    });
}

/// Optimised seams for the current project (cached; computed from the preview mips).
fn seam_map_for(
    shared: &SharedState,
    doc: &ProjectDocument,
    mip16: &[Arc<Rgb16Image>],
    mip_generation: u64,
    aspect: f64,
    cancel: Option<&AtomicBool>,
) -> Result<Option<Arc<SeamMap>>, String> {
    use std::hash::{Hash, Hasher};
    if !doc.optimize_seams || doc.images.len() < 2 || mip16.len() != doc.images.len() {
        return Ok(None);
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(&(
        &doc.images,
        doc.projection,
        doc.pano_hfov_deg,
        doc.view_yaw_deg,
        doc.view_pitch_deg,
        doc.view_v_shift,
        aspect,
        doc.exposure_compensation,
        doc.correct_vignetting,
        &doc.photometric,
        &doc.color_profile,
    ))
    .map_err(|e| e.to_string())?
    .hash(&mut h);
    mip_generation.hash(&mut h);
    let key = h.finish();
    if let Some((_, map)) = shared
        .seam_cache
        .lock()
        .unwrap()
        .iter()
        .find(|(k, _)| *k == key)
    {
        return Ok(Some(Arc::clone(map)));
    }
    let target = target_profile_from_doc(doc)?;
    let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
    let photometric = active_photometric(doc);
    let template = RenderInput {
        out_w: 16,
        out_h: 16,
        projection: doc.projection,
        pano_hfov_deg: doc.pano_hfov_deg,
        view: BandViewParams {
            view_yaw_deg: doc.view_yaw_deg,
            view_pitch_deg: doc.view_pitch_deg,
            view_v_shift: doc.view_v_shift,
            crop: [0.0, 0.0, 1.0, 1.0],
            canvas_aspect: Some(aspect),
        },
        images: &doc.images,
        sources: mip16
            .iter()
            .map(|m| vec![m.as_ref() as &dyn Rgb16Raster])
            .collect(),
        photometric: photometric.as_ref(),
        multiband: true,
        trc: &trc,
        interpolation: Interpolation::Bilinear,
        blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        reject_transients: false,
        seams: None,
    };
    let Some(map) = compute_seam_map(&template, SEAM_GRID_W, SEAM_CORRIDOR_DEG, cancel) else {
        return Ok(None);
    };
    let map = Arc::new(map);
    {
        // Keep the most recent layouts so switching back to a previous view is instant.
        let mut cache = shared.seam_cache.lock().unwrap();
        cache.retain(|(k, _)| *k != key);
        cache.insert(0, (key, Arc::clone(&map)));
        cache.truncate(SEAM_CACHE_ENTRIES);
    }
    Ok(Some(map))
}
const SEAM_GRID_W: usize = 4096;
const SEAM_CORRIDOR_DEG: f64 = 8.0;
const SEAM_CACHE_ENTRIES: usize = 6;

/// Distinct, stable frame colour per source (golden-ratio hues).
fn source_colors(n: usize) -> Vec<[u8; 3]> {
    (0..n)
        .map(|i| {
            let h = (i as f64 * 0.618_033_988_75 + 0.08).fract() * 6.0;
            let (s, v) = (0.75, 1.0);
            let c = v * s;
            let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
            let (r, g, b) = match h as u32 {
                0 => (c, x, 0.0),
                1 => (x, c, 0.0),
                2 => (0.0, c, x),
                3 => (0.0, x, c),
                4 => (x, 0.0, c),
                _ => (c, 0.0, x),
            };
            let m = v - c;
            [
                ((r + m) * 255.0) as u8,
                ((g + m) * 255.0) as u8,
                ((b + m) * 255.0) as u8,
            ]
        })
        .collect()
}

fn spawn_seam_overlay(
    shared: &Arc<SharedState>,
    out: &Arc<Mutex<io::Stdout>>,
    id: &str,
    payload: Value,
) {
    let token = payload.get("token").and_then(Value::as_u64).unwrap_or(0);
    let flag = Arc::new(AtomicBool::new(false));
    {
        let mut current = shared.overlay_cancel.lock().unwrap();
        current.store(true, Ordering::Relaxed);
        *current = Arc::clone(&flag);
    }
    reply(out, id, Ok(json!({"started":true,"token":token})));
    let shared = Arc::clone(shared);
    let out = Arc::clone(out);
    thread::spawn(move || {
        let _lane = shared
            .overlay_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if flag.load(Ordering::Relaxed) {
            return;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render_seam_overlay(&shared, &payload, &flag)
        }))
        .unwrap_or_else(|_| Err("Seam overlay renderer failed".into()));
        if flag.load(Ordering::Relaxed) {
            return;
        }
        match result {
            Ok(mut v) => {
                v["token"] = json!(token);
                event(&out, "seam-overlay-ready", v);
            }
            Err(text) => event(
                &out,
                "seam-overlay-failed",
                json!({"token":token,"text":text}),
            ),
        }
    });
}

/// Frames and seams for a canvas region, from the preview mips (geometry and seam labels are
/// resolution independent; the transient rule uses the same sources as the exact preview).
fn render_seam_overlay(
    shared: &SharedState,
    p: &Value,
    cancel: &AtomicBool,
) -> Result<Value, String> {
    let (doc, mip16, preview_aspect, mip_generation) = {
        let proj = shared.project.lock().map_err(|_| "Project lock poisoned")?;
        (
            proj.doc.clone(),
            proj.mip16.clone(),
            f64::from(proj.preview_width) / f64::from(proj.preview_height.max(1)),
            proj.mip_generation,
        )
    };
    if doc.images.is_empty() || mip16.len() != doc.images.len() {
        return Err("No sources".into());
    }
    let region = match p.get("region") {
        Some(v) => serde_json::from_value::<[f64; 4]>(v.clone()).map_err(|_| "Invalid region")?,
        None => [0.0, 0.0, 1.0, 1.0],
    };
    if region.iter().any(|v| !v.is_finite())
        || region[2] - region[0] < 1e-4
        || region[3] - region[1] < 1e-4
    {
        return Err("Invalid region".into());
    }
    let mut width = dimensions(p, "width", 1024, 4096)?.max(16);
    let mut height = dimensions(p, "height", 512, 4096)?.max(16);
    let selected = p
        .get("selected")
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let target = target_profile_from_doc(&doc)?;
    let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
    let aspect = effective_aspect(&doc, preview_aspect);
    let seams = seam_map_for(shared, &doc, &mip16, mip_generation, aspect, Some(cancel))?;
    let photometric = active_photometric(&doc);
    let input = RenderInput {
        out_w: width,
        out_h: height,
        projection: doc.projection,
        pano_hfov_deg: doc.pano_hfov_deg,
        view: BandViewParams {
            view_yaw_deg: doc.view_yaw_deg,
            view_pitch_deg: doc.view_pitch_deg,
            view_v_shift: doc.view_v_shift,
            crop: region,
            canvas_aspect: Some(aspect),
        },
        images: &doc.images,
        sources: mip16
            .iter()
            .map(|m| vec![m.as_ref() as &dyn Rgb16Raster])
            .collect(),
        photometric: photometric.as_ref(),
        multiband: true,
        trc: &trc,
        interpolation: Interpolation::Bilinear,
        blend_width_deg: DEFAULT_BLEND_WIDTH_DEG,
        reject_transients: doc.reject_transients,
        seams: seams.as_deref(),
    };
    let colors = source_colors(doc.images.len());
    // The overlay is a single band: lower its resolution until pass A fits the budget (the
    // UI scales the image; seams and frames stay at their positions).
    let budget = slashstitch_core::memory::band_budget();
    let mut input = input;
    let mut plan = RenderPlan::new(&input);
    while width > 256 && height > 128 && plan.label_band_bytes(&input, 0, height as usize) > budget
    {
        width = width * 7 / 10;
        height = height * 7 / 10;
        input.out_w = width;
        input.out_h = height;
        plan = RenderPlan::new(&input);
    }
    let rgba = plan
        .seam_overlay(&input, selected, &colors, Some(cancel))
        .ok_or("Cancelled")?;
    let image = RgbaImage::from_raw(width, height, rgba).ok_or("Invalid overlay buffer")?;
    let (preview_path, version) = write_preview_png(shared, "seams", &image, Vec::new())?;
    let hex: Vec<String> = colors
        .iter()
        .map(|c| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]))
        .collect();
    Ok(
        json!({"preview_path":preview_path,"version":version,"width":width,"height":height,"region":region,"colors":hex}),
    )
}

fn render_exact_preview(
    shared: &SharedState,
    out: &Arc<Mutex<io::Stdout>>,
    p: &Value,
    cancel: &AtomicBool,
    token: u64,
) -> Result<PreviewMeta, String> {
    let started = std::time::Instant::now();
    let (doc, mip16, stamps, preview_aspect, mip_generation) = {
        let proj = shared.project.lock().map_err(|_| "Project lock poisoned")?;
        (
            proj.doc.clone(),
            proj.mip16.clone(),
            proj.stamps.clone(),
            f64::from(proj.preview_width) / f64::from(proj.preview_height.max(1)),
            proj.mip_generation,
        )
    };
    if doc.images.is_empty() || mip16.len() != doc.images.len() {
        return Err("No sources to preview".into());
    }
    let region = match p.get("region") {
        Some(v) => serde_json::from_value::<[f64; 4]>(v.clone()).map_err(|_| "Invalid region")?,
        None => [0.0, 0.0, 1.0, 1.0],
    };
    if region.iter().any(|v| !v.is_finite())
        || region[0] < -1e-9
        || region[1] < -1e-9
        || region[2] > 1.0 + 1e-9
        || region[3] > 1.0 + 1e-9
        || region[2] - region[0] < 1e-4
        || region[3] - region[1] < 1e-4
    {
        return Err("Invalid region".into());
    }
    let width = dimensions(p, "width", 1024, 4096)?.max(16);
    let height = dimensions(p, "height", 512, 4096)?.max(16);
    let multiband = p.get("multiband").and_then(Value::as_bool).unwrap_or(true);
    let target = target_profile_from_doc(&doc)?;
    let trc = TargetTrc::try_from_profile(&target.to_moxcms_profile()?)?;
    let aspect = effective_aspect(&doc, preview_aspect);
    let view = BandViewParams {
        view_yaw_deg: doc.view_yaw_deg,
        view_pitch_deg: doc.view_pitch_deg,
        view_v_shift: doc.view_v_shift,
        crop: region,
        canvas_aspect: Some(aspect),
    };
    let photometric = active_photometric(&doc);
    let seams = if multiband {
        seam_map_for(shared, &doc, &mip16, mip_generation, aspect, Some(cancel))?
    } else {
        None
    };
    let mip_sources = || -> Vec<Vec<&dyn Rgb16Raster>> {
        mip16
            .iter()
            .map(|m| vec![m.as_ref() as &dyn Rgb16Raster])
            .collect()
    };
    let deficit = RenderPlan::new(&RenderInput {
        out_w: width,
        out_h: height,
        projection: doc.projection,
        pano_hfov_deg: doc.pano_hfov_deg,
        view,
        images: &doc.images,
        sources: mip_sources(),
        photometric: photometric.as_ref(),
        multiband,
        trc: &trc,
        interpolation: doc.interpolation,
        blend_width_deg: doc.blend_width_deg,
        reject_transients: doc.reject_transients,
        seams: seams.as_deref(),
    })
    .resolution_deficit;
    // Zoomed-in views need detail beyond the preview mips: use (and if necessary build) the
    // full-resolution source pyramids shared with the exporter.
    let pyramids = if deficit > 1.15 {
        verify_stamps(&doc.images, &stamps)?;
        Some(shared.source_cache.get_or_build(
            &doc.images,
            &stamps,
            &target,
            cancel,
            &|done, total| {
                event(
                    out,
                    "exact-preview-progress",
                    json!({"token":token,"done":done,"total":total,"stage":"Loading full-resolution sources"}),
                );
            },
        )?)
    } else {
        None
    };
    let sources: Vec<Vec<&dyn Rgb16Raster>> = match &pyramids {
        Some(pyrs) => pyrs
            .iter()
            .zip(mip16.iter())
            .map(|(pyr, mip)| {
                let mut v: Vec<&dyn Rgb16Raster> =
                    pyr.levels.iter().map(|l| l as &dyn Rgb16Raster).collect();
                v.push(mip.as_ref());
                v
            })
            .collect(),
        None => mip_sources(),
    };
    let input = RenderInput {
        out_w: width,
        out_h: height,
        projection: doc.projection,
        pano_hfov_deg: doc.pano_hfov_deg,
        view,
        images: &doc.images,
        sources,
        photometric: photometric.as_ref(),
        multiband,
        trc: &trc,
        interpolation: doc.interpolation,
        blend_width_deg: doc.blend_width_deg,
        reject_transients: doc.reject_transients,
        seams: seams.as_deref(),
    };
    let mut plan = RenderPlan::new(&input);
    // Same coarse-level model as the export, so "exact" means identical for the same
    // region and size (deep pyramids only).
    plan.prepare_coarse(
        &input,
        slashstitch_core::exporter::band_budget(),
        Some(cancel),
    )
    .ok_or("Cancelled")?;
    // Large views (zoom detail on high-DPI screens) render in budgeted bands like exports.
    let (px, valid) = plan
        .render_all_within(
            &input,
            slashstitch_core::memory::band_budget(),
            Some(cancel),
        )
        .ok_or("Cancelled")?;
    let w = width as usize;
    let mut rgba = vec![0u8; w * height as usize * 4];
    rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let idx = y * w + x;
            let dst = &mut row[x * 4..x * 4 + 4];
            if valid[idx] {
                let e = trc.encode_linear_to_u8(px[idx]);
                dst.copy_from_slice(&[e[0], e[1], e[2], 255]);
            } else {
                dst.copy_from_slice(&checker_rgba(x as u32, y as u32));
            }
        }
    });
    let image = RgbaImage::from_raw(width, height, rgba).ok_or("Invalid exact preview buffer")?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }
    let (preview_path, version) = write_preview_png(shared, "exact", &image, target.icc_bytes()?)?;
    Ok(PreviewMeta {
        preview_path,
        width,
        height,
        backend: format!(
            "Exact {} · {} levels{}",
            if multiband { "multi-band" } else { "feather" },
            plan.levels,
            if pyramids.is_some() {
                " · full-res"
            } else {
                ""
            }
        ),
        render_ms: started.elapsed().as_secs_f64() * 1000.0,
        color_profile: target.id().to_string(),
        color_profile_name: target.display_name(),
        version,
        canvas_aspect: aspect,
        region,
        exact: true,
    })
}

fn main() {
    let mut builder = tempfile::Builder::new();
    builder.prefix("slashstitch-preview-");
    let preview_dir = match std::env::var_os("SLASHSTITCH_PREVIEW_ROOT") {
        Some(root) => builder.tempdir_in(root),
        None => builder.tempdir(),
    }
    .expect("Cannot create private preview directory");
    // Full-resolution source caches are large; keep them on a disk-backed location when the
    // host provides one (the preview root may be a RAM-backed tmpfs).
    let cache_dir = {
        let mut b = tempfile::Builder::new();
        b.prefix("slashstitch-sources-");
        std::env::var_os("SLASHSTITCH_CACHE_ROOT")
            .and_then(|root| {
                let _ = fs::create_dir_all(&root);
                b.tempdir_in(root).ok()
            })
            .unwrap_or_else(|| {
                b.tempdir_in(preview_dir.path())
                    .expect("Cannot create source cache directory")
            })
    };
    let shared = Arc::new(SharedState {
        job_state: Mutex::new(JobState::Idle),
        cancel: AtomicBool::new(false),
        worker_thread: Mutex::new(None),
        lens_db: Mutex::new(Arc::new(LensDatabase::load_default())),
        renderer: PreviewRenderer::new(),
        project: Mutex::new(ProjectRuntime::default()),
        source_cache: SourceCache::new(cache_dir.path().to_path_buf()),
        _cache_dir: cache_dir,
        preview_dir,
        exact_cancel: Mutex::new(std::collections::HashMap::new()),
        exact_lock: Mutex::new(()),
        overlay_cancel: Mutex::new(Arc::new(AtomicBool::new(false))),
        overlay_lock: Mutex::new(()),
        seam_cache: Mutex::new(Vec::new()),
        preview_version: AtomicU64::new(0),
    });
    let out = Arc::new(Mutex::new(io::stdout()));
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else {
            break;
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                event(
                    &out,
                    "protocol-error",
                    json!({"text":format!("Invalid request: {e}")}),
                );
                continue;
            }
        };
        match req.action.as_str() {
            "ping" | "echo" => reply(&out, &req.id, Ok(req.payload)),
            "engine-info" => reply(&out, &req.id, Ok(engine_info(&shared))),
            "job-status" => reply(
                &out,
                &req.id,
                Ok(json!({"state":shared.job_state.lock().unwrap().as_str()})),
            ),
            "lensfun-search" => {
                let db = db(&shared);
                reply(
                    &out,
                    &req.id,
                    Ok(json!(db.search(
                        req.payload
                            .get("query")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        req.payload
                            .get("limit")
                            .and_then(Value::as_u64)
                            .unwrap_or(40)
                            .min(200) as usize
                    ))),
                );
            }
            "render-exact" => spawn_exact_preview(&shared, &out, &req.id, req.payload),
            "render-seams" => spawn_seam_overlay(&shared, &out, &req.id, req.payload),
            "cancel-exact" => {
                for flag in shared.exact_cancel.lock().unwrap().values() {
                    flag.store(true, Ordering::Relaxed);
                }
                reply(&out, &req.id, Ok(json!({"requested":true})));
            }
            "job-cancel" | "demo-stop" => {
                shared.cancel.store(true, Ordering::Relaxed);
                reply(&out, &req.id, Ok(json!({"requested":true})));
            }
            "shutdown" => {
                if *shared.job_state.lock().unwrap() != JobState::Idle {
                    reply(
                        &out,
                        &req.id,
                        Err("Stop the active job before closing.".into()),
                    );
                    continue;
                }
                if let Some(h) = shared.worker_thread.lock().unwrap().take() {
                    let _ = h.join();
                }
                for flag in shared.exact_cancel.lock().unwrap().values() {
                    flag.store(true, Ordering::Relaxed);
                }
                let _lane = shared.exact_lock.lock();
                reply(&out, &req.id, Ok(json!({"bye":true})));
                return;
            }
            _ => start_job(&shared, &out, req),
        }
    }
    // EOF means the parent went away. Cancel and join before dropping caches.
    shared.cancel.store(true, Ordering::Relaxed);
    if let Some(h) = shared.worker_thread.lock().unwrap().take() {
        let _ = h.join();
    };
}
fn compute_auto_crop(shared: &SharedState) -> Result<[f64; 4], String> {
    use slashstitch_core::{pano_uv_to_ray_shifted, ray_to_image_px, rotate_view_ray};

    let proj = shared.project.lock().map_err(|_| "Project lock poisoned")?;
    if proj.doc.images.is_empty() {
        return Ok([0.0, 0.0, 1.0, 1.0]);
    }
    let aspect = effective_aspect(
        &proj.doc,
        (proj.preview_width as f64) / (proj.preview_height.max(1) as f64),
    );
    let mask_h: usize = 200;
    let mask_w: usize = ((mask_h as f64 * aspect).round() as usize).clamp(120, 420);
    let view_yaw = proj.doc.view_yaw_deg;
    let view_pitch = proj.doc.view_pitch_deg;
    let view_v_shift = proj.doc.view_v_shift;
    let projection = proj.doc.projection;
    let pano_hfov = proj.doc.pano_hfov_deg;
    let images = proj.doc.images.clone();
    drop(proj);

    let rows: Vec<Vec<bool>> = (0..mask_h)
        .into_par_iter()
        .map(|y| {
            let v = (y as f64 + 0.5) / mask_h as f64;
            let mut row = vec![false; mask_w];
            for (x, slot) in row.iter_mut().enumerate() {
                let u = (x as f64 + 0.5) / mask_w as f64;
                if let Some(raw_ray) =
                    pano_uv_to_ray_shifted(projection, u, v, pano_hfov, aspect, view_v_shift)
                {
                    let ray = rotate_view_ray(raw_ray, view_yaw, view_pitch);
                    if images.iter().any(|im| ray_to_image_px(ray, im).is_some()) {
                        *slot = true;
                    }
                }
            }
            row
        })
        .collect();

    let mut heights = vec![0usize; mask_w];
    let mut best_area = 0usize;
    let mut best_rect = (0usize, 0usize, mask_w, mask_h);

    for (y, row) in rows.iter().enumerate() {
        for x in 0..mask_w {
            heights[x] = if row[x] { heights[x] + 1 } else { 0 };
        }
        let mut stack: Vec<usize> = Vec::with_capacity(mask_w);
        for x in 0..=mask_w {
            let cur_h = if x < mask_w { heights[x] } else { 0 };
            while let Some(&top) = stack.last() {
                if heights[top] <= cur_h {
                    break;
                }
                stack.pop();
                let rect_h = heights[top];
                let left_x = stack.last().map(|&idx| idx + 1).unwrap_or(0);
                let rect_w = x - left_x;
                let area = rect_w * rect_h;
                if area > best_area && rect_w >= 4 && rect_h >= 4 {
                    best_area = area;
                    best_rect = (left_x, y + 1 - rect_h, x, y + 1);
                }
            }
            stack.push(x);
        }
    }

    if best_area == 0 {
        return Ok([0.0, 0.0, 1.0, 1.0]);
    }
    let (x0, y0, x1, y1) = best_rect;
    let pad_x = 1.0 / (mask_w as f64);
    let pad_y = 1.0 / (mask_h as f64);
    let l = ((x0 as f64 / mask_w as f64) + pad_x).clamp(0.0, 0.9);
    let t = ((y0 as f64 / mask_h as f64) + pad_y).clamp(0.0, 0.9);
    let r = ((x1 as f64 / mask_w as f64) - pad_x).clamp(l + 0.05, 1.0);
    let b = ((y1 as f64 / mask_h as f64) - pad_y).clamp(t + 0.05, 1.0);
    Ok([l, t, r, b])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(icc: Option<Vec<u8>>) -> (Arc<Rgb16Image>, Option<Vec<u8>>) {
        (Arc::new(Rgb16Image::new(4, 4)), icc)
    }

    #[test]
    fn new_projects_adopt_only_a_shared_source_profile() {
        let adobe = TargetColorProfile::AdobeRgb.icc_bytes().unwrap();
        let p3 = TargetColorProfile::DisplayP3.icc_bytes().unwrap();
        let same = [raw(Some(adobe.clone())), raw(Some(adobe.clone()))];
        assert_eq!(common_source_profile(&same).unwrap().id(), "adobe_rgb");
        let mixed = [raw(Some(adobe.clone())), raw(Some(p3))];
        assert!(common_source_profile(&mixed).is_none());
        let missing = [raw(Some(adobe)), raw(None)];
        assert!(common_source_profile(&missing).is_none());
        assert!(common_source_profile(&[]).is_none());
    }
}
