//! Core panorama geometry, EXIF metadata inspection, feature matching,
//! Levenberg-Marquardt bundle adjustment, and strip-based BigTIFF export.

pub mod blend;
pub mod color;
pub mod exporter;
pub mod features;
pub mod files;
pub mod image_meta;
pub mod lensfun;
pub mod memory;
pub mod optics;
pub mod optimizer;
pub mod photometric;
pub mod project;
pub mod raster;
pub mod render;

pub use blend::{
    render_band, render_band_10bit_with_view, render_band_16bit_with_view,
    render_band_8bit_with_view, render_band_u16_with_view, render_band_with_view, BandViewParams,
};
pub use color::{
    detect_preset_match, extract_icc_description, preset_match_for_icc, quantize_10bit_to_u16,
    u16_to_10bit_level, validate_bit_depth, ColorProfileConfig, TargetColorProfile, TargetTrc,
    DEFAULT_OUTPUT_BIT_DEPTH, MAX_ICC_BYTES, SUPPORTED_BIT_DEPTHS, SUPPORTED_COLOR_PROFILE_IDS,
};
pub use exporter::{export_panorama, ExportError, ExportOptions, ExportSummary};
pub use features::{
    detect_keypoints, find_all_control_points, match_image_pair, refine_star_control_points,
    Keypoint,
};
pub use image_meta::{inspect_image, inspect_images, SourceImageMeta};
pub use lensfun::{CameraProfile, DistortionCalibration, LensDatabase, LensProfile, LensSearchHit};
pub use optics::{
    apply_ptlens_distortion, image_px_to_world_ray, invert_ptlens_distortion, pano_uv_to_ray,
    pano_uv_to_ray_shifted, ray_to_image_px, rotate_view_ray, rotation_matrix, shift_pano_v,
    ImageProjector,
};
pub use optimizer::{
    level_horizon_poses, matrix_to_ypr_deg, optimize_poses, optimize_poses_with_options,
    seed_poses_from_pairs, OptimizationReport, OptimizeOptions,
};
pub use photometric::{estimate_photometric, PhotometricParams};
pub use project::{ControlPoint, LensProjection, PanoramaProjection, SourceImage, StitchProject};
pub use render::{
    compute_seam_map, compute_seam_map_within, Interpolation, RenderInput, RenderPlan, SeamMap,
    SourceRef, DEFAULT_BLEND_WIDTH_DEG,
};
