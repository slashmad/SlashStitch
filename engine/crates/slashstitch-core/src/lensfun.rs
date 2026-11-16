use roxmltree::{Document, ParsingOptions};

fn parse_document(xml: &str) -> Result<Document<'_>, roxmltree::Error> {
    // Upstream Lensfun files contain a DTD. Never resolve external entities;
    // bound the parsed tree as well as the downloader's byte budget.
    Document::parse_with_options(
        xml,
        ParsingOptions {
            allow_dtd: true,
            nodes_limit: 500_000,
            entity_resolver: None,
        },
    )
}
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

const BUILTIN_LENSES_XML: &str = include_str!("builtin_lenses.xml");

#[derive(Debug, Error)]
pub enum LensDbError {
    #[error("Invalid Lensfun XML: {0}")]
    InvalidXml(String),
    #[error("Failed to read Lensfun directory {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CameraProfile {
    pub maker: String,
    pub model: String,
    pub variant: Option<String>,
    pub mount: String,
    pub crop_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "model", rename_all = "lowercase")]
pub enum DistortionModel {
    Ptlens { a: f64, b: f64, c: f64 },
    Poly3 { k1: f64 },
    Poly5 { k1: f64, k2: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistortionCalibration {
    pub focal_mm: f64,
    pub distortion: DistortionModel,
    pub crop_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LensProfile {
    pub maker: String,
    pub model: String,
    pub mounts: Vec<String>,
    pub crop_factor: f64,
    pub lens_type: String,
    pub distortions: Vec<DistortionCalibration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensSearchHit {
    pub kind: &'static str,
    pub maker: String,
    pub model: String,
    pub mount: String,
    pub crop_factor: f64,
    pub lens_type: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct LensDatabase {
    pub source_dir: Option<PathBuf>,
    pub cameras: Vec<CameraProfile>,
    pub lenses: Vec<LensProfile>,
    pub files_loaded: usize,
}

impl LensDatabase {
    /// Interpolate within a compatible model and crop calibration; never drop Poly5 k2.
    pub fn distortion_for(
        &self,
        lens: &LensProfile,
        focal: f64,
        crop: f64,
    ) -> Option<([f64; 3], Option<[f64; 2]>, f64)> {
        let best_crop = lens
            .distortions
            .iter()
            .min_by(|a, b| {
                (a.crop_factor - crop)
                    .abs()
                    .total_cmp(&(b.crop_factor - crop).abs())
            })?
            .crop_factor;
        let mut points: Vec<_> = lens
            .distortions
            .iter()
            .filter(|d| {
                (d.crop_factor - best_crop).abs() < 1e-6
                    && d.focal_mm.is_finite()
                    && d.focal_mm > 0.0
            })
            .collect();
        points.sort_by(|a, b| a.focal_mm.total_cmp(&b.focal_mm));
        let lo = points
            .iter()
            .rev()
            .find(|d| d.focal_mm <= focal)
            .copied()
            .unwrap_or(points[0]);
        let hi = points
            .iter()
            .find(|d| d.focal_mm >= focal)
            .copied()
            .unwrap_or(*points.last()?);
        let t = if hi.focal_mm > lo.focal_mm {
            ((focal - lo.focal_mm) / (hi.focal_mm - lo.focal_mm)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mix = |a: f64, b: f64| a + (b - a) * t;
        let scale = best_crop / crop.max(0.01);
        let convert = |d: &DistortionModel| match d {
            DistortionModel::Ptlens { a, b, c } => ([*a, *b, *c], None, scale),
            DistortionModel::Poly3 { k1 } => ([0.0, *k1, 0.0], None, scale),
            DistortionModel::Poly5 { k1, k2 } => ([0.0; 3], Some([*k1, *k2]), scale),
        };
        let a = convert(&lo.distortion);
        let b = convert(&hi.distortion);
        if a.1.is_some() != b.1.is_some() {
            return Some(if t < 0.5 { a } else { b });
        }
        Some((
            [
                mix(a.0[0], b.0[0]),
                mix(a.0[1], b.0[1]),
                mix(a.0[2], b.0[2]),
            ],
            a.1.zip(b.1)
                .map(|(a, b)| [mix(a[0], b[0]), mix(a[1], b[1])]),
            scale,
        ))
    }
    /// Preferred writable user Lensfun directory for updates across Linux, Windows, and macOS.
    pub fn user_update_dir() -> Option<PathBuf> {
        #[cfg(target_os = "linux")]
        {
            env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local/share/lensfun/version_1"))
        }
        #[cfg(target_os = "windows")]
        {
            env::var("LOCALAPPDATA")
                .ok()
                .map(|l| PathBuf::from(l).join("lensfun/version_1"))
        }
        #[cfg(target_os = "macos")]
        {
            env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join("Library/Application Support/lensfun/version_1"))
        }
    }

    /// Discover the best available Lensfun XML directory across Linux, Windows, and macOS.
    pub fn discover_dir() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(explicit) = env::var("SLASHSTITCH_LENSFUN_DIR") {
            let p = PathBuf::from(explicit);
            candidates.push(p.clone());
            candidates.push(p.join("version_2"));
            candidates.push(p.join("version_1"));
        }

        if let Ok(exe) = env::current_exe() {
            if let Some(dir) = exe.parent() {
                for base in [dir.join("lensfun"), dir.join("resources").join("lensfun")] {
                    candidates.push(base.join("version_2"));
                    candidates.push(base.join("version_1"));
                    candidates.push(base);
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(home) = env::var("HOME") {
                let user_base = PathBuf::from(home).join(".local/share/lensfun");
                candidates.push(user_base.join("version_2"));
                candidates.push(user_base.join("version_1"));
                candidates.push(user_base);
            }
            for sys in ["/usr/local/share/lensfun", "/usr/share/lensfun"] {
                let base = PathBuf::from(sys);
                candidates.push(base.join("version_2"));
                candidates.push(base.join("version_1"));
                candidates.push(base);
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(local) = env::var("LOCALAPPDATA") {
                let base = PathBuf::from(local).join("lensfun");
                candidates.push(base.join("version_2"));
                candidates.push(base.join("version_1"));
                candidates.push(base);
            }
            if let Ok(prog) = env::var("ProgramData") {
                let base = PathBuf::from(prog).join("lensfun");
                candidates.push(base.join("version_2"));
                candidates.push(base.join("version_1"));
                candidates.push(base);
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(home) = env::var("HOME") {
                let base = PathBuf::from(home).join("Library/Application Support/lensfun");
                candidates.push(base.join("version_2"));
                candidates.push(base.join("version_1"));
                candidates.push(base);
            }
            let sys = PathBuf::from("/Library/Application Support/lensfun");
            candidates.push(sys.join("version_2"));
            candidates.push(sys.join("version_1"));
            candidates.push(sys);
        }

        candidates.into_iter().find(|path| has_xml_files(path))
    }

    /// Load the discovered Lensfun database plus built-in supplemental profiles.
    pub fn load_default() -> Self {
        let db = match Self::discover_dir() {
            Some(dir) => Self::load_from_dir(&dir).unwrap_or_default(),
            None => Self::default(),
        };
        db.with_builtin()
    }

    pub fn with_builtin(mut self) -> Self {
        let mut supplemental = Self::default();
        supplemental.parse_xml(BUILTIN_LENSES_XML);
        for lens in supplemental.lenses {
            if !self.lenses.iter().any(|existing| {
                normalize(&existing.maker) == normalize(&lens.maker)
                    && normalize(&existing.model) == normalize(&lens.model)
            }) {
                self.lenses.push(lens);
            }
        }
        self
    }

    pub fn load_from_dir(dir: &Path) -> Result<Self, LensDbError> {
        let mut db = Self {
            source_dir: Some(dir.to_path_buf()),
            ..Self::default()
        };
        let entries = fs::read_dir(dir).map_err(|source| LensDbError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let mut xml_paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|v| v.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("xml"))
            .collect();
        xml_paths.sort();

        for path in xml_paths {
            let content = fs::read_to_string(&path).map_err(|source| LensDbError::Io {
                path: path.clone(),
                source,
            })?;
            let doc = parse_document(&content)
                .map_err(|e| LensDbError::InvalidXml(format!("{}: {e}", path.display())))?;
            if doc.root_element().tag_name().name() != "lensdatabase" {
                return Err(LensDbError::InvalidXml(format!(
                    "{}: expected lensdatabase",
                    path.display()
                )));
            }
            db.parse_xml(&content);
            db.files_loaded += 1;
        }
        if db.files_loaded == 0 || db.cameras.is_empty() && db.lenses.is_empty() {
            return Err(LensDbError::InvalidXml("No usable profiles".into()));
        }
        Ok(db)
    }

    pub fn parse_xml(&mut self, xml: &str) {
        let Ok(doc) = parse_document(xml) else {
            return;
        };
        for node in doc.root_element().children().filter(|n| n.is_element()) {
            match node.tag_name().name() {
                "camera" => {
                    if let Some(cam) = parse_camera(node) {
                        self.cameras.push(cam);
                    }
                }
                "lens" => {
                    if let Some(lens) = parse_lens(node) {
                        self.lenses.push(lens);
                    }
                }
                _ => {}
            }
        }
    }

    pub fn find_camera(&self, make: Option<&str>, model: Option<&str>) -> Option<&CameraProfile> {
        let model_norm = normalize(model?);
        if model_norm.is_empty() {
            return None;
        }
        let make_norm = make.map(normalize).unwrap_or_default();

        self.cameras
            .iter()
            .filter_map(|cam| {
                let cam_model = normalize(&cam.model);
                let cam_maker = normalize(&cam.maker);
                let make_matches = make_norm.is_empty()
                    || cam_maker == make_norm
                    || make_norm.contains(&cam_maker)
                    || cam_maker.contains(&make_norm);

                let score = if cam_model == model_norm && make_matches {
                    100
                } else if model_norm.ends_with(&cam_model) && make_matches {
                    85
                } else if (cam_model.contains(&model_norm) || model_norm.contains(&cam_model))
                    && make_matches
                {
                    60
                } else {
                    0
                };
                (score > 0).then_some((score, cam))
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, cam)| cam)
    }

    pub fn find_lens(
        &self,
        lens_model: Option<&str>,
        preferred_mount: Option<&str>,
    ) -> Option<&LensProfile> {
        let raw_query = lens_model?;
        let query = normalize(raw_query);
        if query.is_empty() {
            return None;
        }
        let query_focals = extract_focal_mm(raw_query);
        let query_fisheye = raw_query.to_ascii_lowercase().contains("fisheye");

        self.lenses
            .iter()
            .filter_map(|lens| {
                let full_raw = format!("{} {}", lens.maker, lens.model);
                let full = normalize(&full_raw);
                let model_only = normalize(&lens.model);

                // Never match a lens with a mismatched focal length (e.g. 15mm vs 85mm).
                let cand_focals = extract_focal_mm(&full_raw);
                if !query_focals.is_empty()
                    && !cand_focals.is_empty()
                    && query_focals != cand_focals
                {
                    return None;
                }

                // Never match fisheye EXIF to non-fisheye Lensfun entry or vice versa.
                let cand_fisheye = full_raw.to_ascii_lowercase().contains("fisheye")
                    || !lens.lens_type.eq_ignore_ascii_case("rectilinear");
                if query_fisheye && !cand_fisheye
                    || !query_fisheye && cand_fisheye && full != query && model_only != query
                {
                    return None;
                }

                let mount_bonus = if preferred_mount
                    .map(|m| lens.mounts.iter().any(|lm| lm.eq_ignore_ascii_case(m)))
                    .unwrap_or(false)
                {
                    15
                } else {
                    0
                };

                let base_score = if full == query || model_only == query {
                    120
                } else if full.contains(&query) || query.contains(&model_only) {
                    85
                } else {
                    token_overlap_score(&query, &full)
                };
                (base_score >= 55).then_some((base_score + mount_bonus, lens))
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, lens)| lens)
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<LensSearchHit> {
        let q = query.trim().to_lowercase();
        let terms: Vec<&str> = q.split_whitespace().collect();
        let mut hits = Vec::new();
        let cap = limit.clamp(1, 200);

        for cam in &self.cameras {
            let hay = format!("{} {} {}", cam.maker, cam.model, cam.mount).to_lowercase();
            if terms.is_empty() || terms.iter().all(|t| hay.contains(t)) {
                hits.push(LensSearchHit {
                    kind: "camera",
                    maker: cam.maker.clone(),
                    model: cam.model.clone(),
                    mount: cam.mount.clone(),
                    crop_factor: cam.crop_factor,
                    lens_type: None,
                });
                if hits.len() >= cap {
                    return hits;
                }
            }
        }

        for lens in &self.lenses {
            let mounts = lens.mounts.join(", ");
            let hay = format!(
                "{} {} {} {}",
                lens.maker, lens.model, mounts, lens.lens_type
            )
            .to_lowercase();
            if terms.is_empty() || terms.iter().all(|t| hay.contains(t)) {
                hits.push(LensSearchHit {
                    kind: "lens",
                    maker: lens.maker.clone(),
                    model: lens.model.clone(),
                    mount: mounts,
                    crop_factor: lens.crop_factor,
                    lens_type: Some(lens.lens_type.clone()),
                });
                if hits.len() >= cap {
                    return hits;
                }
            }
        }

        hits
    }
}

fn extract_focal_mm(text: &str) -> Vec<u32> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut vals = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            let num_str = &lower[start..i];
            // Accept numbers followed by "mm" or preceded by '-' (e.g. "24-70mm").
            let after = &lower[i..];
            let followed_by_mm = after.starts_with("mm");
            let range_before_mm = after.starts_with('-')
                && after[1..]
                    .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.')
                    .starts_with("mm");
            if followed_by_mm || range_before_mm {
                if let Ok(v) = num_str.parse::<f64>() {
                    vals.push(v.round() as u32);
                }
            }
        } else {
            i += 1;
        }
    }
    vals
}

fn has_xml_files(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries
        .filter_map(|e| e.ok().map(|v| v.path()))
        .any(|p| p.extension().and_then(|s| s.to_str()) == Some("xml"))
}

fn child_text(node: roxmltree::Node<'_, '_>, name: &str) -> Option<String> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == name)
        .and_then(|n| n.text())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_camera(node: roxmltree::Node<'_, '_>) -> Option<CameraProfile> {
    let maker = child_text(node, "maker")?;
    let model = child_text(node, "model")?;
    let mount = child_text(node, "mount").unwrap_or_default();
    let crop_factor = child_text(node, "cropfactor")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let variant = child_text(node, "variant");
    Some(CameraProfile {
        maker,
        model,
        variant,
        mount,
        crop_factor,
    })
}

fn parse_lens(node: roxmltree::Node<'_, '_>) -> Option<LensProfile> {
    let maker = child_text(node, "maker")?;
    let model = child_text(node, "model")?;
    let mounts: Vec<String> = node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "mount")
        .filter_map(|n| n.text().map(|t| t.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect();
    let crop_factor = child_text(node, "cropfactor")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let lens_type = child_text(node, "type").unwrap_or_else(|| "rectilinear".to_string());

    let mut distortions = Vec::new();
    for cal in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "calibration")
    {
        for dist in cal
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "distortion")
        {
            if let Some(mut parsed) = parse_distortion(dist) {
                parsed.crop_factor = cal
                    .attribute("cropfactor")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(crop_factor);
                distortions.push(parsed);
            }
        }
    }

    Some(LensProfile {
        maker,
        model,
        mounts,
        crop_factor,
        lens_type,
        distortions,
    })
}

fn parse_distortion(node: roxmltree::Node<'_, '_>) -> Option<DistortionCalibration> {
    let focal_mm = node.attribute("focal")?.parse::<f64>().ok()?;
    let model = node.attribute("model")?;
    let distortion = match model {
        "ptlens" => DistortionModel::Ptlens {
            a: node
                .attribute("a")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            b: node
                .attribute("b")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            c: node
                .attribute("c")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        },
        "poly3" => DistortionModel::Poly3 {
            k1: node
                .attribute("k1")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        },
        "poly5" => DistortionModel::Poly5 {
            k1: node
                .attribute("k1")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            k2: node
                .attribute("k2")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        },
        _ => return None,
    };
    Some(DistortionCalibration {
        crop_factor: 1.0,
        focal_mm,
        distortion,
    })
}

fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_overlap_score(a: &str, b: &str) -> usize {
    let ta: Vec<&str> = a.split_whitespace().collect();
    let tb: Vec<&str> = b.split_whitespace().collect();
    if ta.is_empty() || tb.is_empty() {
        return 0;
    }
    let matched = ta.iter().filter(|tok| tb.contains(tok)).count();
    (matched * 80) / ta.len().max(tb.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_poly5_and_uses_all_crop_calibrations() {
        let mut db = LensDatabase::default();
        db.parse_xml(r#"<lensdatabase><lens><maker>Test</maker><model>24-50mm</model><cropfactor>1</cropfactor><calibration cropfactor="1"><distortion model="poly5" focal="24" k1="0.01" k2="0.02"/><distortion model="poly5" focal="50" k1="0.03" k2="0.04"/></calibration><calibration cropfactor="1.5"><distortion model="poly5" focal="24" k1="0.1" k2="0.2"/></calibration></lens></lensdatabase>"#);
        let (abc, poly, scale) = db.distortion_for(&db.lenses[0], 37.0, 1.0).unwrap();
        assert_eq!(abc, [0.0; 3]);
        let poly = poly.unwrap();
        assert!((poly[0] - 0.02).abs() < 1e-10 && (poly[1] - 0.03).abs() < 1e-10);
        assert_eq!(scale, 1.0);
        let (_, poly, scale) = db.distortion_for(&db.lenses[0], 24.0, 1.5).unwrap();
        assert_eq!(poly, Some([0.1, 0.2]));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn parses_lensfun_camera_and_lens() {
        let xml = r#"
        <lensdatabase version="1">
            <camera>
                <maker>Sony</maker>
                <model>ILCE-7M4</model>
                <mount>Sony E</mount>
                <cropfactor>1.0</cropfactor>
            </camera>
            <lens>
                <maker>Samyang</maker>
                <model>AF 12mm f/2.0 E</model>
                <mount>Sony E</mount>
                <cropfactor>1.534</cropfactor>
                <type>rectilinear</type>
                <calibration>
                    <distortion model="ptlens" focal="12" a="0.01" b="-0.02" c="0.005"/>
                </calibration>
            </lens>
        </lensdatabase>
        "#;
        let mut db = LensDatabase::default();
        db.parse_xml(xml);
        assert_eq!(db.cameras.len(), 1);
        assert_eq!(db.lenses.len(), 1);

        let cam = db.find_camera(Some("SONY"), Some("ILCE-7M4")).unwrap();
        assert_eq!(cam.mount, "Sony E");
        assert!((cam.crop_factor - 1.0).abs() < 1e-6);

        let lens = db
            .find_lens(Some("Samyang AF 12mm f/2.0 E"), Some("Sony E"))
            .unwrap();
        assert_eq!(lens.distortions.len(), 1);
    }

    #[test]
    fn accepts_lensfun_dtd_without_resolving_external_entities() {
        let xml = r#"<!DOCTYPE lensdatabase SYSTEM "lensfun-database.dtd">
            <lensdatabase><lens><maker>Test</maker><model>50mm</model></lens></lensdatabase>"#;
        let mut db = LensDatabase::default();
        db.parse_xml(xml);
        assert_eq!(db.lenses.len(), 1);
        assert!(parse_document(r#"<!DOCTYPE lensdatabase [<!ENTITY external SYSTEM "file:///nonexistent">]><lensdatabase>&external;</lensdatabase>"#).is_err());
    }

    #[test]
    fn never_mismatches_15mm_fisheye_to_85mm_rectilinear() {
        let mut db = LensDatabase::default();
        db.parse_xml(
            r#"<lensdatabase version="1">
                <lens>
                    <maker>Sigma</maker>
                    <model>Sigma 85mm F1.4 DG DN | Art 020</model>
                    <mount>Sony E</mount>
                    <cropfactor>1.0</cropfactor>
                    <type>rectilinear</type>
                </lens>
            </lensdatabase>"#,
        );
        assert!(db
            .find_lens(
                Some("15mm F1.4 DG DN DIAGONAL FISHEYE | Art 024"),
                Some("Sony E")
            )
            .is_none());

        db.parse_xml(BUILTIN_LENSES_XML);
        let matched = db
            .find_lens(
                Some("15mm F1.4 DG DN DIAGONAL FISHEYE | Art 024"),
                Some("Sony E"),
            )
            .unwrap();
        assert_eq!(matched.lens_type, "equisolid");
    }
}
