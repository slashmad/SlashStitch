import { useCallback, useEffect, useRef, useState, type DragEvent, type PointerEvent, type WheelEvent } from "react";
import { ProjectionPicker } from "./ProjectionPicker";

type Interp = "bilinear" | "spline36" | "lanczos3";
interface StitchSettings { multiband: boolean; blendWidth: number; interpolation: Interp; optimizeSeams: boolean; exposure: boolean; vignetting: boolean; transients: boolean }
/** Stitch modes: presets over the advanced settings (all saved with the project). */
const STITCH_PRESETS: Array<{ id: string; label: string; hint: string; s: StitchSettings }> = [
  { id: "balanced", label: "Balanced", hint: "Multi-band (3°), Spline36, optimised seams, exposure + vignetting, transient light removal.", s: { multiband: true, blendWidth: 3, interpolation: "spline36", optimizeSeams: true, exposure: true, vignetting: true, transients: true } },
  { id: "night", label: "Night sky & aurora", hint: "Wide 6° transitions for moving aurora and sky gradients; otherwise like Balanced.", s: { multiband: true, blendWidth: 6, interpolation: "spline36", optimizeSeams: true, exposure: true, vignetting: true, transients: true } },
  { id: "detail", label: "Sharp detail", hint: "Narrow 1.5° transitions and Lanczos3 resampling for architecture and landscapes with fine structure.", s: { multiband: true, blendWidth: 1.5, interpolation: "lanczos3", optimizeSeams: true, exposure: true, vignetting: true, transients: true } },
  { id: "draft", label: "Fast draft", hint: "Feather blending, bilinear sampling, centre-line seams, gains only — fastest, lower quality.", s: { multiband: false, blendWidth: 3, interpolation: "bilinear", optimizeSeams: false, exposure: true, vignetting: false, transients: false } },
];

interface EngineInfo {
  version: string;
  os: string;
  arch: string;
  cpu_threads: number;
  gpu_backend: string;
  lensfun_dir: string | null;
  lensfun_update_dir: string | null;
  lensfun_files: number;
  cameras_count: number;
  lenses_count: number;
  working_color_space?: string;
  output_bit_depth?: number;
  supported_bit_depths?: number[];
  supported_color_profiles?: string[];
}

interface SourceImageMeta {
  path: string;
  file_name: string;
  width: number | null;
  height: number | null;
  bit_depth?: number | null;
  icc_profile_name?: string | null;
  icc_preset_match?: string | null;
  icc_status?: string;
  camera_make: string | null;
  camera_model: string | null;
  lens_model: string | null;
  focal_length_mm: number | null;
  focal_length_35mm: number | null;
  aperture: number | null;
  iso: number | null;
  matched_crop_factor: number | null;
  matched_mount: string | null;
  matched_lens: string | null;
  matched_lens_type: string | null;
  estimated_lens_projection: string;
  estimated_hfov_deg: number | null;
  warnings: string[];
}

interface SourceImagePose {
  id: number;
  path: string;
  width: number;
  height: number;
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
  hfov_deg: number;
  lens_projection: string;
  distortion: [number, number, number];
  distortion_poly5?: [number, number] | null;
  distortion_scale?: number;
}

interface ControlPoint { image_a: number; x_a: number; y_a: number; image_b: number; x_b: number; y_b: number }

interface PreviewMeta {
  width: number;
  height: number;
  backend: string;
  render_ms: number;
  color_profile?: string;
  color_profile_name?: string;
  preview_url?: string;
  version?: number;
  canvas_aspect?: number;
  region?: [number, number, number, number];
  exact?: boolean;
}

interface AlignReport {
  initial_rms_deg: number;
  final_rms_deg: number;
  max_error_deg: number;
  iterations: number;
  control_points_used: number;
  control_points_rejected?: number;
  optimized_hfov_deg?: number | null;
  optimized_distortion?: [number, number, number] | null;
}

interface ExportSummary {
  output_path: string;
  width: number;
  height: number;
  bigtiff: boolean;
  bytes_written: number;
  bit_depth?: number;
  container_bit_depth?: number;
  color_profile?: string;
  color_profile_name?: string;
  icc_embedded?: boolean;
}

interface LensSearchHit {
  kind: "camera" | "lens";
  maker: string;
  model: string;
  mount: string;
  crop_factor: number;
  lens_type: string | null;
}

interface CropRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const FULL_CROP: CropRect = { left: 0, top: 0, right: 1, bottom: 1 };

const HORIZON_SHIFT_PROJECTIONS = new Set([
  "cylindrical",
  "mercator",
  "rectilinear",
  "vedutismo",
  "stereographic",
  "transverse_cylindrical",
  "transverse_mercator",
  "transverse_vedutismo",
]);

const COLOR_PROFILE_LABELS: Record<string, string> = {
  srgb: "sRGB IEC61966-2.1",
  adobe_rgb: "Adobe RGB (1998)",
  display_p3: "Display P3",
  prophoto_rgb: "ProPhoto RGB",
  bt2020: "Rec.2020 (BT.2020)",
  custom_icc: "Custom ICC",
};

interface HistorySnapshot {
  poses: SourceImagePose[];
  controlPoints: ControlPoint[];
  projection: string;
  panoHfov: number;
  viewYaw: number;
  viewPitch: number;
  viewVShift?: number;
  crop: CropRect;
  outputBitDepth?: number;
  colorProfile?: string;
  customIccName?: string | null;
  customIccBase64?: string | null;
}

type InspectorTab = "lens" | "optimizer" | "lensdb" | "export";

function useSplitter(initial: number, min: number, max: number, side: "left" | "right") {
  const [w, setW] = useState(initial);
  const start = (e: PointerEvent) => {
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = w;
    const move = (ev: globalThis.PointerEvent) => {
      const next = side === "left" ? w0 + (ev.clientX - x0) : w0 - (ev.clientX - x0);
      setW(Math.min(max, Math.max(min, next)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };
  return [w, start, (delta: number) => setW(current => Math.max(min, Math.min(max, current + delta)))] as const;
}

export function App() {
  const [leftW, leftDrag, leftKey] = useSplitter(290, 200, 560, "left");
  const [rightW, rightDrag, rightKey] = useSplitter(390, 270, 720, "right");

  const layout = useRef<HTMLDivElement>(null);
  const [state, setState] = useState("idle");
  const [busyText, setBusyText] = useState("");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<{ percent: number; stage: string } | null>(null);
  const [engine, setEngine] = useState<EngineInfo | null>(null);
  const [metas, setMetas] = useState<SourceImageMeta[]>([]);
  const [poses, setPoses] = useState<SourceImagePose[]>([]);
  const [controlPoints, setControlPoints] = useState<ControlPoint[]>([]);
  const controlPointsRef = useRef(controlPoints);
  controlPointsRef.current = controlPoints;
  const [projectPath, setProjectPath] = useState<string | null>(null);
  const [projectDirty, setProjectDirty] = useState(false);
  const [engineAvailable, setEngineAvailable] = useState(true);
  const [recoveryAvailable, setRecoveryAvailable] = useState(false);
  const stateRef = useRef(state);
  stateRef.current = state;
  const viewQueue = useRef<Promise<void>>(Promise.resolve());
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [dragOver, setDragOver] = useState(false);

  const [projection, setProjection] = useState("equirectangular");
  const [panoHfov, setPanoHfov] = useState(120);
  const [viewYaw, setViewYaw] = useState(0);
  const [viewPitch, setViewPitch] = useState(0);
  const [viewVShift, setViewVShift] = useState(0);
  const [viewZoom, setViewZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragMode, setDragMode] = useState<"view" | "pan">("view");
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [crop, setCrop] = useState<CropRect>(FULL_CROP);
  const [cropMode, setCropMode] = useState(false);

  const viewportRef = useRef<HTMLDivElement>(null);
  const [stageSize, setStageSize] = useState({ w: 960, h: 540 });

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewInfo, setPreviewInfoRaw] = useState<PreviewMeta | null>(null);
  const previewMetaRef = useRef<PreviewMeta | null>(null);
  // Fast (GPU) previews arrive as metadata with a URL served by the host's preview protocol.
  const setPreviewInfo = useCallback((meta: PreviewMeta | null) => {
    // The same image may arrive twice (event + reply); re-applying it would restart renders.
    const cur = previewMetaRef.current;
    if (meta && cur && meta.version !== undefined && meta.version === cur.version && meta.preview_url === cur.preview_url) return;
    previewMetaRef.current = meta;
    setPreviewInfoRaw(meta);
    if (meta?.preview_url) setPreviewUrl(meta.preview_url);
  }, []);
  const [exposureCompensation, setExposureCompensation] = useState(true);
  const [rejectTransients, setRejectTransients] = useState(true);
  const [optimizeSeams, setOptimizeSeams] = useState(true);
  const [correctVignetting, setCorrectVignetting] = useState(true);
  const [interpolation, setInterpolation] = useState<Interp>("spline36");
  const [blendWidth, setBlendWidth] = useState(3);
  // Exact previews use the export renderer: one for the whole canvas, one 1:1 detail tile
  // for the visible region when zoomed in. Tokens discard results of superseded requests.
  const [exactBase, setExactBase] = useState<{ url: string; label: string; ms: number } | null>(null);
  const [detail, setDetail] = useState<{ url: string; region: [number, number, number, number] } | null>(null);
  const [exactStatus, setExactStatus] = useState("");
  const [notice, setNotice] = useState("");
  // Stitch structure overlay: coloured frame per source and the actual blend seams.
  const [showSeams, setShowSeams] = useState(false);
  const [seamOverlay, setSeamOverlay] = useState<{ url: string; region: [number, number, number, number]; colors: string[] } | null>(null);
  const seamTokenRef = useRef(0);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 9000);
    return () => clearTimeout(timer);
  }, [notice]);
  const exactTokenRef = useRef({ next: 1, base: 0, detail: 0 });
  const [alignReport, setAlignReport] = useState<AlignReport | null>(null);
  const [optimizeHfov, setOptimizeHfov] = useState(true);
  const [levelHorizon, setLevelHorizon] = useState(true);
  const [optimizeDistortion, setOptimizeDistortion] = useState(true);
  const [prioritizeSky, setPrioritizeSky] = useState(true);

  const [undoStack, setUndoStack] = useState<HistorySnapshot[]>([]);
  const [redoStack, setRedoStack] = useState<HistorySnapshot[]>([]);

  const [tab, setTab] = useState<InspectorTab>("lens");
  const [dbQuery, setDbQuery] = useState("Sigma 15mm");
  const [dbHits, setDbHits] = useState<LensSearchHit[]>([]);
  const [dbUpdateMsg, setDbUpdateMsg] = useState("");

  const [exportPath, setExportPath] = useState("");
  const [exportWidth, setExportWidth] = useState(4096);
  const [exportHeight, setExportHeight] = useState(2048);
  const [outputBitDepth, setOutputBitDepth] = useState<number>(8);
  const [colorProfile, setColorProfile] = useState<string>("srgb");
  const [customIccName, setCustomIccName] = useState<string | null>(null);
  const [customIccBase64, setCustomIccBase64] = useState<string | null>(null);
  const [multiband, setMultiband] = useState(true);
  const [forceBigTiff, setForceBigTiff] = useState(false);
  const [lockExportAspect,setLockExportAspect]=useState(true);
  const [exportSummary, setExportSummary] = useState<ExportSummary | null>(null);
  const stitchNow: StitchSettings = { multiband, blendWidth, interpolation, optimizeSeams, exposure: exposureCompensation, vignetting: correctVignetting, transients: rejectTransients };
  const currentStitchPreset = STITCH_PRESETS.find(p => (Object.keys(p.s) as Array<keyof StitchSettings>).every(k => p.s[k] === stitchNow[k]))?.id ?? "custom";
  // Apply (part of) a stitch configuration: multi-band is an editor/export setting, the
  // rest is saved in the project by the engine and re-renders the preview.
  const applyStitch = async (patch: Partial<StitchSettings>) => {
    if (patch.multiband !== undefined) setMultiband(patch.multiband);
    const payload: Record<string, unknown> = {};
    if (patch.blendWidth !== undefined) payload.blend_width_deg = patch.blendWidth;
    if (patch.interpolation !== undefined) payload.interpolation = patch.interpolation;
    if (patch.optimizeSeams !== undefined) payload.optimize_seams = patch.optimizeSeams;
    if (patch.exposure !== undefined) payload.exposure_compensation = patch.exposure;
    if (patch.vignetting !== undefined) payload.correct_vignetting = patch.vignetting;
    if (patch.transients !== undefined) payload.reject_transients = patch.transients;
    if (Object.keys(payload).length === 0) return;
    const reply = await run("set-render-settings", payload);
    if (reply.ok) {
      const data = JSON.parse(reply.text);
      syncProjectColor(data.project);
      if (data.preview) setPreviewInfo(data.preview);
    }
  };


  const posesRef = useRef(poses);
  const projRef = useRef({ projection, panoHfov, viewYaw, viewPitch, viewVShift, crop });
  const colorRef = useRef({ outputBitDepth, colorProfile, customIccName, customIccBase64 });
  const stageRef = useRef(stageSize);
  posesRef.current = poses;
  projRef.current = { projection, panoHfov, viewYaw, viewPitch, viewVShift, crop };
  colorRef.current = { outputBitDepth, colorProfile, customIccName, customIccBase64 };
  stageRef.current = stageSize;

  const syncProjectColor = useCallback((doc?: Record<string, unknown> | null) => {
    if (!doc) return;
    if (typeof doc.output_bit_depth === "number") setOutputBitDepth(doc.output_bit_depth);
    if (typeof doc.color_profile === "string") setColorProfile(doc.color_profile);
    if (typeof doc.exposure_compensation === "boolean") setExposureCompensation(doc.exposure_compensation);
    if (typeof doc.reject_transients === "boolean") setRejectTransients(doc.reject_transients);
    if (typeof doc.optimize_seams === "boolean") setOptimizeSeams(doc.optimize_seams);
    if (typeof doc.correct_vignetting === "boolean") setCorrectVignetting(doc.correct_vignetting);
    if (doc.interpolation === "bilinear" || doc.interpolation === "spline36" || doc.interpolation === "lanczos3") setInterpolation(doc.interpolation);
    if (typeof doc.blend_width_deg === "number") setBlendWidth(doc.blend_width_deg);
    if ("custom_icc_name" in doc) setCustomIccName(typeof doc.custom_icc_name === "string" ? doc.custom_icc_name : null);
    if ("custom_icc_base64" in doc) setCustomIccBase64(typeof doc.custom_icc_base64 === "string" ? doc.custom_icc_base64 : null);
  }, []);

  const computeRenderSize = useCallback((projName: string, sw: number, sh: number) => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    // The worker fits this box to the project's canvas aspect.
    const maxW = Math.max(640, Math.min(4096, Math.round(sw * dpr)));
    const maxH = Math.max(360, Math.min(4096, Math.round(sh * dpr)));
    if (projName === "equirectangular") {
      const rw = Math.max(960, Math.min(4096, maxW, maxH * 2));
      return { width: rw, height: Math.round(rw / 2) };
    }
    return { width: maxW, height: maxH };
  }, []);

  const pushHistory = useCallback((override?: Partial<HistorySnapshot>) => {
    const snap: HistorySnapshot = {
      poses: posesRef.current.map((p) => ({ ...p })),
      controlPoints: controlPointsRef.current.map(p => ({...p})),
      ...projRef.current,
      crop: { ...projRef.current.crop },
      ...colorRef.current,
      ...override,
    };
    setUndoStack((prev) => [...prev.slice(-29), snap]);
    setRedoStack([]);
  }, []);

  const refreshPreviewImage = useCallback(async () => {
    if (previewMetaRef.current?.preview_url) return;
    const res = await window.Host.call("read-preview");
    if (res.ok) {
      const parsed = JSON.parse(res.text) as { dataUrl: string | null };
      setPreviewUrl(parsed.dataUrl);
    }
  }, []);

  useEffect(() => {
    const element=layout.current;
    if (!element) return;
    const fit=()=>{
      const ratio=Math.min(1,Math.max(470,element.clientWidth-328)/(leftW+rightW));
      element.style.setProperty("grid-template-columns", `${Math.max(180,Math.round(leftW*ratio))}px 4px minmax(300px, 1fr) 4px ${Math.max(250,Math.round(rightW*ratio))}px`);
    };
    fit(); const observer=new ResizeObserver(fit);observer.observe(element);
    return ()=>observer.disconnect();
  }, [leftW, rightW]);

  useEffect(() => {
    const off = window.Host.onEvent((name, data) => {
      if (!data || typeof data !== "object") return;
      if (name === "job-state") {
        const nextState = (data as { state: string }).state;
        setState(nextState);
        if (nextState === "idle") {
          setProgress(null);
          setBusyText("");
        }
      }
      if(name==="request-ui-snapshot"){
        const id=(data as {id:string}).id;
        void viewQueue.current.then(()=>window.Host.call("ui-snapshot",{id,ui:uiSettingsRef.current()}));
      }
      if (name === "project-changed") {
        const doc=data as {images?:SourceImagePose[];control_points?:ControlPoint[];output_bit_depth?:number;color_profile?:string;custom_icc_name?:string|null;custom_icc_base64?:string|null};
        if (doc.control_points) {controlPointsRef.current=doc.control_points;setControlPoints(doc.control_points);}
        syncProjectColor(doc);
      }
      if (name === "preview-updated") {
        // Published by the engine for every new fast preview (also after a late completion).
        setPreviewInfo(data as PreviewMeta);
      }
      if (name === "project-synced") {
        // Complete project state after every backend commit (also after a host timeout).
        const result = data as { images: SourceImagePose[]; metas: SourceImageMeta[]; control_points?: ControlPoint[] };
        posesRef.current = result.images; setPoses(result.images); setMetas(result.metas);
        controlPointsRef.current = result.control_points ?? []; setControlPoints(result.control_points ?? []);
        setSelectedIdx(current => Math.min(current, Math.max(0, result.images.length - 1)));
      }
      if (name === "exact-preview-ready") {
        const d = data as PreviewMeta & { token: number };
        const t = exactTokenRef.current;
        if (d.preview_url && d.token === t.base) { setExactBase({ url: d.preview_url, label: d.backend, ms: d.render_ms }); setExactStatus(""); }
        else if (d.preview_url && d.region && d.token === t.detail) { setDetail({ url: d.preview_url, region: d.region }); setExactStatus(""); }
      }
      if (name === "seam-overlay-ready") {
        const d = data as { token: number; preview_url?: string; region: [number, number, number, number]; colors: string[] };
        if (d.token === seamTokenRef.current && d.preview_url) setSeamOverlay({ url: d.preview_url, region: d.region, colors: d.colors });
      }
      if (name === "exact-preview-progress") {
        const d = data as { token: number; done: number; total: number; stage: string };
        setExactStatus(`${d.stage} (${d.done}/${d.total})`);
      }
      if (name === "exact-preview-failed") {
        const d = data as { token: number; text: string };
        const t = exactTokenRef.current;
        if (d.token === t.base || d.token === t.detail) setExactStatus(/cancel/i.test(d.text) ? "" : `Exact preview unavailable: ${d.text}`);
      }
      if(name==="project-updated"){
        const result=data as {images:SourceImagePose[];metas:SourceImageMeta[];control_points?:ControlPoint[];preview?:PreviewMeta|null};
        posesRef.current=result.images;setPoses(result.images);setMetas(result.metas);setControlPoints(result.control_points??[]);
        setSelectedIdx(current=>Math.min(current,Math.max(0,result.images.length-1)));
        setPreviewInfo(result.preview??null);
        if(!result.images.length)setPreviewUrl(null);else void refreshPreviewImage();
      }
      if (name === "project-file") {
        const info=data as {path:string|null;dirty:boolean;engineAvailable:boolean;recoveryAvailable:boolean};
        setProjectPath(info.path);setProjectDirty(info.dirty);setEngineAvailable(info.engineAvailable);setRecoveryAvailable(info.recoveryAvailable);
      }
      if (name === "engine-disconnected") {
        setState("disconnected");setEngineAvailable(false);setError((data as {text:string}).text);setBusyText("");setProgress(null);
      }
      if (name === "job-progress") {
        setProgress(data as { percent: number; stage: string });
      }
      if (name === "preview-ready") {
        setPreviewInfo(data as PreviewMeta);
        void refreshPreviewImage();
      }
      if (name === "align-complete") {
        const payload = data as {
          images?: SourceImagePose[];
          report?: AlignReport;
          preview?: PreviewMeta;
        };
        if (payload.images) setPoses(payload.images);
        if (payload.report) setAlignReport(payload.report);
        if (payload.preview) setPreviewInfo(payload.preview);
        void refreshPreviewImage();
      }
      if (name === "export-complete") {
        setExportSummary(data as ExportSummary);
      }
      if (name === "worker-error" || name === "protocol-error" || name === "close-blocked") {
        setError((data as { text: string }).text);
        setBusyText("");
      }
    });

    void window.Host.call("shell-info").then((reply) => {
      if (reply.ok) {
        const info=JSON.parse(reply.text);setState(info.jobState);setProjectPath(info.projectFile);setProjectDirty(info.dirty);setEngineAvailable(info.engineAvailable);setRecoveryAvailable(info.recoveryAvailable);
      }
    });
    void window.Host.call("engine-info").then((reply) => {
      if (reply.ok) setEngine(JSON.parse(reply.text) as EngineInfo);
    });
    void window.Host.call("lensfun-search", { query: "Sigma 15mm", limit: 15 }).then((reply) => {
      if (reply.ok) setDbHits(JSON.parse(reply.text) as LensSearchHit[]);
    });
    return off;
  }, [refreshPreviewImage]);

  const run = useCallback(async (action: string, payload: unknown = null) => {
    try {
      if (!["read-preview", "engine-info", "lensfun-search", "shell-info", "job-status", "render-preview", "set-view", "snapshot-project"].includes(action)) setError("");
      const reply = await window.Host.call(action, payload);
      if (!reply.ok && !/cancelled/i.test(reply.text) && !(action==="render-preview" && /job is active|job is already active|wait for/i.test(reply.text))) setError(reply.text);

      return reply;
    } catch (failure) {
      setError(String(failure));
      return { ok: false, text: String(failure) };
    }
  }, []);

  const loadPaths = useCallback(async (paths: string[]) => {
    if (paths.length === 0 || stateRef.current !== "idle") return;
    const before: HistorySnapshot = {poses:posesRef.current.map(p=>({...p})),controlPoints:controlPointsRef.current.map(p=>({...p})),...projRef.current,crop:{...projRef.current.crop},...colorRef.current};
    setBusyText(`Loading ${paths.length} item(s)...`);
    setProgress({ percent: 5, stage: `Loading ${paths.length} item(s)...` });
    const loaded = await run("load-images", { paths, replace: false });
    setBusyText("");
    setProgress(null);
    if (!loaded.ok) return;
    setUndoStack(prev=>[...prev.slice(-29),before]);setRedoStack([]);
    const parsed = JSON.parse(loaded.text) as {
      images: SourceImagePose[];
      metas: SourceImageMeta[];
      project?: Record<string, unknown>;
      preview?: PreviewMeta | null;
      control_points?: ControlPoint[];
      failures?: Array<{path?:string;error:string}>;
      adopted_color_profile?: string | null;
    };
    setPoses(parsed.images);
    setMetas(parsed.metas);
    setControlPoints(parsed.control_points ?? []);
    syncProjectColor(parsed.project);
    if (parsed.adopted_color_profile) setNotice(`Export colour profile set to the sources' ${parsed.adopted_color_profile} (change it in Export)`);
    if (parsed.failures?.length) setError(parsed.failures.map(f=>`${f.path ?? "Import"}: ${f.error}`).join("; "));
    if (parsed.preview) setPreviewInfo(parsed.preview);
    const view=projRef.current;
    const size=computeRenderSize(view.projection,stageRef.current.w,stageRef.current.h);
    const resized=await run("set-view",{projection:view.projection,pano_hfov_deg:view.panoHfov,view_yaw_deg:view.viewYaw,view_pitch_deg:view.viewPitch,view_v_shift:view.viewVShift,...size});
    if(resized.ok)setPreviewInfo(JSON.parse(resized.text));
    await refreshPreviewImage();
  }, [computeRenderSize, refreshPreviewImage, run, syncProjectColor]);

  const addImages = useCallback(async () => {
    const picked = await run("pick-images");
    if (!picked.ok) return;
    const data = JSON.parse(picked.text) as { canceled: boolean; paths: string[] };
    if (data.canceled || data.paths.length === 0) return;
    await loadPaths(data.paths);
  }, [loadPaths, run]);

  const clearAll = async () => {
    if (stateRef.current !== "idle") return;
    pushHistory();
    const cleared=await run("clear-images");
    if (!cleared.ok) return;
    setPoses([]);
    setMetas([]);
    setSelectedIdx(0);
    setPreviewUrl(null);
    setPreviewInfo(null);
    setAlignReport(null);
    setControlPoints([]);
    setProjection("equirectangular");setPanoHfov(120);setViewYaw(0);setViewPitch(0);setViewVShift(0);
    setOutputBitDepth(8);setColorProfile("srgb");setCustomIccName(null);setCustomIccBase64(null);
    setCrop(FULL_CROP);
    setCropMode(false);
    setViewZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const applySnapshot = useCallback(async (snap: HistorySnapshot) => {
    const reply = await run("restore-poses", {
      images: snap.poses,
      control_points: snap.controlPoints,
      projection: snap.projection,
      pano_hfov_deg: snap.panoHfov,
      view_yaw_deg: snap.viewYaw,
      view_pitch_deg: snap.viewPitch,
      view_v_shift: snap.viewVShift ?? 0,
      output_bit_depth: snap.outputBitDepth,
      color_profile: snap.colorProfile,
      custom_icc_name: snap.customIccName,
      custom_icc_base64: snap.customIccBase64,
    });
    if (!reply.ok) return false;
    const data=JSON.parse(reply.text);
    setPoses(data.images);setMetas(data.metas);setControlPoints(data.control_points ?? []);
    syncProjectColor(data.project);
    setSelectedIdx(current=>Math.min(current,Math.max(0,data.images.length-1)));
    setProjection(snap.projection);setPanoHfov(snap.panoHfov);setViewYaw(snap.viewYaw);setViewPitch(snap.viewPitch);setViewVShift(snap.viewVShift ?? 0);setCrop(snap.crop ?? FULL_CROP);
    setAlignReport(null);
    if (data.preview) setPreviewInfo(data.preview);
    await refreshPreviewImage();return true;
  }, [refreshPreviewImage, run, syncProjectColor]);

  // Short background jobs (fast preview after an import or view change) finish within a
  // moment: Undo/Redo wait for them instead of silently dropping the click.
  const waitForIdle = useCallback(async (timeoutMs = 8000) => {
    const end = Date.now() + timeoutMs;
    while (stateRef.current !== "idle" && Date.now() < end) await new Promise(r => setTimeout(r, 50));
    return stateRef.current === "idle";
  }, []);
  const undo = useCallback(async () => {
    if (undoStack.length === 0 || !(await waitForIdle())) return;
    const target = undoStack[undoStack.length - 1];
    const current: HistorySnapshot = {
      poses: posesRef.current.map((p) => ({ ...p })),
      controlPoints: controlPointsRef.current.map(p => ({...p})),
      ...projRef.current,
      crop: { ...projRef.current.crop },
      ...colorRef.current,
    };
    if (await applySnapshot(target)) {setUndoStack(prev=>prev.slice(0,-1));setRedoStack(prev=>[...prev,current]);}
  }, [applySnapshot, state, undoStack]);

  const redo = useCallback(async () => {
    if (redoStack.length === 0 || !(await waitForIdle())) return;
    const target = redoStack[redoStack.length - 1];
    const current: HistorySnapshot = {
      poses: posesRef.current.map((p) => ({ ...p })),
      controlPoints: controlPointsRef.current.map(p => ({...p})),
      ...projRef.current,
      crop: { ...projRef.current.crop },
      ...colorRef.current,
    };
    if (await applySnapshot(target)) {setRedoStack(prev=>prev.slice(0,-1));setUndoStack(prev=>[...prev,current]);}
  }, [applySnapshot, redoStack, state]);

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    // Last requested render box: only a real change of the render size re-renders, so
    // layout jitter (e.g. status text, scrollbars) can never start a render loop.
    let lastBox = "";
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect || rect.width < 40 || rect.height < 40) return;
      const w = Math.round(rect.width);
      const h = Math.round(rect.height);
      setStageSize({ w, h });
      if (posesRef.current.length === 0) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(async () => {
        if (stateRef.current !== "idle") return;
        const { width, height } = computeRenderSize(projRef.current.projection, w, h);
        const box = `${projRef.current.projection}:${width}x${height}`;
        if (box === lastBox) return;
        lastBox = box;
        const reply = await run("render-preview", { width, height });
        if (reply.ok) {
          setPreviewInfo(JSON.parse(reply.text) as PreviewMeta);
          await refreshPreviewImage();
        }
      }, 150);
    });
    ro.observe(el);
    return () => {
      if (timer) clearTimeout(timer);
      ro.disconnect();
    };
  }, [computeRenderSize, refreshPreviewImage, run]);

  const runAutoAlign = useCallback(async () => {
    if (state !== "idle") return;
    if (posesRef.current.length > 0) pushHistory();
    await run("auto-align", {
      optimize_hfov: optimizeHfov,
      level_horizon: levelHorizon,
      optimize_distortion: optimizeDistortion,
      prioritize_sky: prioritizeSky,
    });
  }, [levelHorizon, optimizeDistortion, optimizeHfov, prioritizeSky, pushHistory, run, state]);

  const runReoptimize = async () => {
    if (state !== "idle" || poses.length < 2) return;
    pushHistory();
    setBusyText("Re-optimizing...");
    const reply = await run("reoptimize", {
      optimize_hfov: optimizeHfov,
      level_horizon: levelHorizon,
      optimize_distortion: optimizeDistortion,
      prioritize_sky: prioritizeSky,
    });
    setBusyText("");
    if (reply.ok) {
      const data = JSON.parse(reply.text) as {
        images: SourceImagePose[];
        report: AlignReport;
        preview?: PreviewMeta;
      };
      setPoses(data.images);
      setAlignReport(data.report);
      if (data.preview) setPreviewInfo(data.preview);
      await refreshPreviewImage();
    }
  };

  const uiSettings=useCallback(()=>({crop,exportPath,exportWidth,exportHeight,multiband,forceBigTiff,optimizeHfov,levelHorizon,optimizeDistortion,prioritizeSky,lockExportAspect}),[crop,exportPath,exportWidth,exportHeight,multiband,forceBigTiff,optimizeHfov,levelHorizon,optimizeDistortion,prioritizeSky,lockExportAspect]);
  const suppressUiDirty=useRef(false);
  const uiSettingsRef=useRef(uiSettings);uiSettingsRef.current=uiSettings;
  useEffect(()=>{
    if(suppressUiDirty.current){suppressUiDirty.current=false;return;}
    if (!poses.length) return;
    const timer=setTimeout(()=>{void window.Host.call("mark-ui-dirty",uiSettingsRef.current());},200);
    return ()=>clearTimeout(timer);
  },[crop,exportPath,exportWidth,exportHeight,multiband,forceBigTiff,optimizeHfov,levelHorizon,optimizeDistortion,prioritizeSky,lockExportAspect]);
  const saveProject=useCallback(async(saveAs=false)=>{if(stateRef.current!=="idle" && engineAvailable)return;await run("save-project",{ui:uiSettingsRef.current(),saveAs});},[engineAvailable,run]);
  const acceptProject=useCallback(async(data:Record<string,any>)=>{
    if (data.canceled || !data.project) return;
    const doc=data.project;
    suppressUiDirty.current=true;
    setPoses(data.images);setMetas(data.metas);setControlPoints(data.control_points??[]);setSelectedIdx(0);
    setProjection(doc.projection);setPanoHfov(doc.pano_hfov_deg);setViewYaw(doc.view_yaw_deg);setViewPitch(doc.view_pitch_deg);setViewVShift(typeof doc.view_v_shift==="number"?doc.view_v_shift:0);
    syncProjectColor(doc);
    const ui=doc.ui??{};
    const savedCrop=ui.crop;
    if(savedCrop && [savedCrop.left,savedCrop.top,savedCrop.right,savedCrop.bottom].every(Number.isFinite) && savedCrop.left>=0 && savedCrop.top>=0 && savedCrop.right<=1 && savedCrop.bottom<=1 && savedCrop.right>savedCrop.left && savedCrop.bottom>savedCrop.top)setCrop(savedCrop);else setCrop(FULL_CROP);
    if(Number.isInteger(ui.exportWidth)&&ui.exportWidth>0&&ui.exportWidth<=200000)setExportWidth(ui.exportWidth);
    if(Number.isInteger(ui.exportHeight)&&ui.exportHeight>0&&ui.exportHeight<=200000)setExportHeight(ui.exportHeight);
    if(typeof ui.exportPath==="string")setExportPath(ui.exportPath);
    if(typeof ui.multiband==="boolean")setMultiband(ui.multiband);
    if(typeof ui.forceBigTiff==="boolean")setForceBigTiff(ui.forceBigTiff);
    if(typeof ui.optimizeHfov==="boolean")setOptimizeHfov(ui.optimizeHfov);
    if(typeof ui.levelHorizon==="boolean")setLevelHorizon(ui.levelHorizon);
    if(typeof ui.optimizeDistortion==="boolean")setOptimizeDistortion(ui.optimizeDistortion);
    if(typeof ui.prioritizeSky==="boolean")setPrioritizeSky(ui.prioritizeSky);
    if(typeof ui.lockExportAspect==="boolean")setLockExportAspect(ui.lockExportAspect);
    setUndoStack([]);setRedoStack([]);setAlignReport(null);setPreviewInfo(data.preview??null);setViewZoom(1);setPan({x:0,y:0});
    const resized=await run("render-preview",computeRenderSize(doc.projection,stageRef.current.w,stageRef.current.h));
    if(resized.ok)setPreviewInfo(JSON.parse(resized.text));
    await refreshPreviewImage();
  },[computeRenderSize,run,refreshPreviewImage,syncProjectColor]);
  const openProject=useCallback(async(action="open-project")=>{if(stateRef.current!=="idle" && action!=="restart-engine")return;const result=await run(action);if(result.ok) {await acceptProject(JSON.parse(result.text));if(action==="restart-engine"){setEngineAvailable(true);setState("idle");}}},[acceptProject,run]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const target = e.target as HTMLElement | null;
      const inText = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT");

      const key = e.key.toLowerCase();
      if (key === "z" && !e.shiftKey && !inText) {
        e.preventDefault();
        void undo();
      } else if (((key === "z" && e.shiftKey) || key === "y") && !inText) {
        e.preventDefault();
        void redo();
      } else if (key === "s") {
        e.preventDefault();void saveProject(e.shiftKey);
      } else if (key === "o" && e.shiftKey) {
        e.preventDefault();void openProject();
      } else if (key === "o") {
        e.preventDefault();
        void addImages();
      } else if (key === "a" && e.shiftKey) {
        e.preventDefault();
        void runAutoAlign();
      } else if (key === "e") {
        e.preventDefault();
        setTab("export");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [addImages, redo, runAutoAlign, undo, saveProject, openProject]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const interactive =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.tagName === "BUTTON" ||
          target.getAttribute("role") === "option");
      if (e.code === "Space" && !interactive) {
        e.preventDefault();
        setSpaceHeld(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceHeld(false);
    };
    const blur = () => setSpaceHeld(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  const updateView = useCallback(async (
    nextProj = projection,
    nextHfov = panoHfov,
    nextYaw = viewYaw,
    nextPitch = viewPitch,
    recordHistory = true,
    nextVShift = projRef.current.viewVShift,
  ) => {
    if (recordHistory) pushHistory();
    setProjection(nextProj);
    setPanoHfov(nextHfov);
    setViewYaw(nextYaw);
    setViewPitch(nextPitch);
    setViewVShift(nextVShift);
    if (posesRef.current.length === 0) return;
    const { width, height } = computeRenderSize(nextProj, stageRef.current.w, stageRef.current.h);
    viewQueue.current=viewQueue.current.then(async()=>{
      const reply = await run("set-view", {projection:nextProj,pano_hfov_deg:nextHfov,view_yaw_deg:nextYaw,view_pitch_deg:nextPitch,view_v_shift:nextVShift,width,height});
      if(reply.ok){setPreviewInfo(JSON.parse(reply.text));await refreshPreviewImage();}
    }).catch(failure=>setError(String(failure)));
    await viewQueue.current;
  }, [computeRenderSize, panoHfov, projection, pushHistory, refreshPreviewImage, run, viewPitch, viewYaw]);

  const runAutoCrop = async () => {
    if (poses.length === 0 || stateRef.current !== "idle") return;
    pushHistory();
    const reply = await run("auto-crop");
    if (reply.ok) {
      const data = JSON.parse(reply.text) as { crop: [number, number, number, number] };
      const [left, top, right, bottom] = data.crop;
      setCrop({ left, top, right, bottom });
      setCropMode(true);
    }
  };

  const resetCrop = () => {
    pushHistory();
    setCrop(FULL_CROP);
  };

  const resetViewportTransform = () => {
    setViewZoom(1);
    setPan({ x: 0, y: 0 });
    setDragMode("view");
    void updateView(
      projection,
      panoHfov,
      0,
      0,
      viewYaw !== 0 || viewPitch !== 0 || Math.abs(viewVShift) > 1e-4,
      0,
    );
  };

  const updatePose = async (patch: Partial<SourceImagePose>) => {
    if (stateRef.current !== "idle") return;
    const cur = poses[selectedIdx];
    if (!cur) return;
    pushHistory();
    const updated = { ...cur, ...patch };

    const reply = await run("update-image-pose", updated);
    if (reply.ok) {
      const data = JSON.parse(reply.text) as { images: SourceImagePose[]; preview?: PreviewMeta };
      setPoses(data.images);
      if (data.preview) setPreviewInfo(data.preview);
      await refreshPreviewImage();
    }
  };

  const searchDb = async (query: string) => {
    setDbQuery(query);
    const reply = await run("lensfun-search", { query, limit: 25 });
    if (reply.ok) setDbHits(JSON.parse(reply.text) as LensSearchHit[]);
  };

  const updateLensfunDb = async () => {
    setBusyText("Updating Lensfun DB...");
    setDbUpdateMsg("Downloading latest XML profiles from Lensfun GitHub...");
    const reply = await run("update-lensfun");
    setBusyText("");
    if (reply.ok) {
      const res = JSON.parse(reply.text) as { updatedFiles: number; targetDir: string; engine: EngineInfo };
      setEngine(res.engine);
      setDbUpdateMsg(`Updated ${res.updatedFiles} XML files (${res.engine.cameras_count} cameras, ${res.engine.lenses_count} lenses).`);
      void searchDb(dbQuery);
    } else {
      setDbUpdateMsg(reply.text);
    }
  };

  const chooseExport = async () => {
    const reply = await run("pick-export-file");
    if (!reply.ok) return;
    const data = JSON.parse(reply.text) as { canceled: boolean; path: string | null };
    if (!data.canceled && data.path) setExportPath(data.path);
  };

  const updateColorSettings = async (patch: {
    output_bit_depth?: number;
    color_profile?: string;
    custom_icc_path?: string;
    adopt_source_icc_index?: number;
  }) => {
    if (stateRef.current !== "idle") return;
    pushHistory();
    const reply = await run("set-color-settings", patch);
    if (!reply.ok) return;
    const data = JSON.parse(reply.text) as {
      project?: Record<string, unknown>;
      preview?: PreviewMeta | null;
    };
    syncProjectColor(data.project);
    if (data.preview) setPreviewInfo(data.preview);
    if (posesRef.current.length > 0) await refreshPreviewImage();
  };

  const chooseIccProfile = async () => {
    if (stateRef.current !== "idle") return;
    const reply = await run("pick-icc-profile");
    if (!reply.ok) return;
    const data = JSON.parse(reply.text) as { canceled: boolean; path: string | null };
    if (!data.canceled && data.path) {
      await updateColorSettings({ custom_icc_path: data.path });
    }
  };

  const startExport = async () => {
    if(stateRef.current!=="idle" || !Number.isInteger(exportWidth) || !Number.isInteger(exportHeight) || exportWidth<1 || exportHeight<1 || exportWidth>200000 || exportHeight>200000){setError("Export dimensions must be whole numbers between 1 and 200000.");return;}
    setExportSummary(null);
    await run("export-panorama", {
      output_path: exportPath,
      width: exportWidth,
      height: exportHeight,
      crop: [crop.left, crop.top, crop.right, crop.bottom],
      multiband,
      force_bigtiff: forceBigTiff,
      bit_depth: outputBitDepth,
      color_profile: colorProfile,
      custom_icc_name: customIccName,
      custom_icc_base64: customIccBase64,
    });
  };

  const handleWheelZoom = (e: WheelEvent<HTMLDivElement>) => {
    if (poses.length === 0 || stateRef.current !== "idle") return;
    e.preventDefault();
    const rect = viewportRef.current?.getBoundingClientRect();
    const factor = e.deltaY < 0 ? 1.14 : 1 / 1.14;
    const nextZoom = Math.min(8, Math.max(0.35, viewZoom * factor));
    if (rect && Math.abs(nextZoom - viewZoom) > 1e-4) {
      const dx = e.clientX - (rect.left + rect.width / 2);
      const dy = e.clientY - (rect.top + rect.height / 2);
      const ratio = nextZoom / viewZoom;
      setViewZoom(nextZoom);
      setPan({
        x: dx - (dx - pan.x) * ratio,
        y: dy - (dy - pan.y) * ratio,
      });
    }
  };

  const dragPreview = (e: PointerEvent<HTMLDivElement>) => {
    if (poses.length === 0 || stateRef.current !== "idle") return;
    e.preventDefault();
    const x0 = e.clientX;
    const y0 = e.clientY;
    // Pan whole frame inside workspace in Pan mode, Space+drag, Alt+drag, or Middle/Right drag.
    if (e.button === 1 || e.button === 2 || e.altKey || dragMode === "pan" || spaceHeld) {
      const pan0 = { ...pan };
      const movePan = (ev: globalThis.PointerEvent) => {
        setPan({ x: pan0.x + (ev.clientX - x0), y: pan0.y + (ev.clientY - y0) });
      };
      const upPan = () => {
        window.removeEventListener("pointermove", movePan);
        window.removeEventListener("pointerup", upPan);
        window.removeEventListener("pointercancel", upPan);
      };
      window.addEventListener("pointermove", movePan);
      window.addEventListener("pointerup", upPan);
      window.addEventListener("pointercancel", upPan);
      return;
    }

    // Left-drag in Move mode shifts the panorama within the projection frame smoothly.
    // On horizon-level projections (Cylindrical/Mercator/Rectilinear/etc.), vertical drag
    // shifts the canvas vertically (viewVShift) without bending the level horizon.
    const yaw0 = viewYaw;
    const pitch0 = viewPitch;
    const vShift0 = viewVShift;
    const useVShift = HORIZON_SHIFT_PROJECTIONS.has(projection);
    const sens = 0.3 / Math.max(1, viewZoom * 0.7);
    let lastYaw = yaw0;
    let lastPitch = pitch0;
    let lastVShift = vShift0;
    let inFlight = false;
    let lastPush = 0;

    const sendLiveView = async (yw: number, pt: number, vs: number) => {
      inFlight = true;
      lastPush = performance.now();
      await updateView(projection, panoHfov, yw, pt, false, vs);
      inFlight = false;
    };

    const move = (ev: globalThis.PointerEvent) => {
      lastYaw = yaw0 - (ev.clientX - x0) * sens;
      if (useVShift) {
        const pxSpan = Math.max(120, frameH);
        lastVShift = Math.max(-0.75, Math.min(0.75, vShift0 + (ev.clientY - y0) / pxSpan));
        setViewVShift(lastVShift);
      } else {
        lastPitch = Math.max(-89, Math.min(89, pitch0 + (ev.clientY - y0) * sens));
        setViewPitch(lastPitch);
      }
      setViewYaw(lastYaw);
      if (!inFlight && performance.now() - lastPush > 45) {
        void sendLiveView(lastYaw, lastPitch, lastVShift);
      }
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      if (
        Math.abs(lastYaw - yaw0) > 0.1 ||
        Math.abs(lastPitch - pitch0) > 0.1 ||
        Math.abs(lastVShift - vShift0) > 0.002
      ) {
        pushHistory({ viewYaw: yaw0, viewPitch: pitch0, viewVShift: vShift0 });
        void updateView(projection, panoHfov, lastYaw, lastPitch, false, lastVShift);
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };

  type CropHandle = "n" | "s" | "e" | "w" | "nw" | "ne" | "sw" | "se" | "move";
  const startCropDrag = (handle: CropHandle, e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const frameEl = e.currentTarget.closest(".preview-frame") as HTMLElement | null;
    if (!frameEl || stateRef.current!=="idle") return;
    const rect = frameEl.getBoundingClientRect();
    const x0 = e.clientX;
    const y0 = e.clientY;
    const c0 = { ...crop };
    const minSpan = 0.04;

    const move = (ev: globalThis.PointerEvent) => {
      const dx = (ev.clientX - x0) / Math.max(1, rect.width);
      const dy = (ev.clientY - y0) / Math.max(1, rect.height);
      const next = { ...c0 };
      if (handle === "move") {
        const w = c0.right - c0.left;
        const h = c0.bottom - c0.top;
        next.left = Math.min(1 - w, Math.max(0, c0.left + dx));
        next.right = next.left + w;
        next.top = Math.min(1 - h, Math.max(0, c0.top + dy));
        next.bottom = next.top + h;
      } else {
        if (handle.includes("w")) next.left = Math.min(c0.right - minSpan, Math.max(0, c0.left + dx));
        if (handle.includes("e")) next.right = Math.max(c0.left + minSpan, Math.min(1, c0.right + dx));
        if (handle.includes("n")) next.top = Math.min(c0.bottom - minSpan, Math.max(0, c0.top + dy));
        if (handle.includes("s")) next.bottom = Math.max(c0.top + minSpan, Math.min(1, c0.bottom + dy));
      }
      setCrop(next);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      pushHistory({ crop: c0 });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };

  const cropKeyboard=(handle:CropHandle,e:React.KeyboardEvent<HTMLDivElement>)=>{
    if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)||stateRef.current!=="idle")return;
    e.preventDefault();e.stopPropagation();pushHistory();
    const step=e.shiftKey?0.001:0.01;
    const dx=e.key==="ArrowLeft"?-step:e.key==="ArrowRight"?step:0;
    const dy=e.key==="ArrowUp"?-step:e.key==="ArrowDown"?step:0;
    setCrop(c=>{
      const n={...c};
      if(handle==="move"){
        const w=c.right-c.left,h=c.bottom-c.top;n.left=Math.min(1-w,Math.max(0,c.left+dx));n.right=n.left+w;n.top=Math.min(1-h,Math.max(0,c.top+dy));n.bottom=n.top+h;
      }else{
        if(handle.includes("w"))n.left=Math.min(c.right-.04,Math.max(0,c.left+dx));
        if(handle.includes("e"))n.right=Math.max(c.left+.04,Math.min(1,c.right+dx));
        if(handle.includes("n"))n.top=Math.min(c.bottom-.04,Math.max(0,c.top+dy));
        if(handle.includes("s"))n.bottom=Math.max(c.top+.04,Math.min(1,c.bottom+dy));
      }return n;
    });
  };
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (stateRef.current !== "idle") return;
    const files = Array.from(e.dataTransfer.files);
    const paths = files.map((f) => window.Host.getPathForFile(f)).filter(Boolean);
    if (paths.length > 0) void loadPaths(paths);
  };

  const selectedMeta = metas[selectedIdx] ?? null;
  const selectedPose = poses[selectedIdx] ?? null;
  const statusDisplay = error || busyText || state;
  const hasCrop = crop.left > 0.002 || crop.top > 0.002 || crop.right < 0.998 || crop.bottom < 0.998;
  const previewAspect = previewInfo && previewInfo.height > 0 ? previewInfo.width / previewInfo.height : 2;
  const baseFrameH = Math.max(80, Math.min(stageSize.h, stageSize.w / previewAspect));
  const baseFrameW = baseFrameH * previewAspect;
  const frameW = baseFrameW * viewZoom;
  const frameH = baseFrameH * viewZoom;
  const cropAspect = ((crop.right - crop.left) * previewAspect) / Math.max(0.01, crop.bottom - crop.top);

  // Whole-canvas exact preview: re-rendered whenever the fast preview changes (geometry, view,
  // colour, exposure settings) once interaction has paused.
  useEffect(() => {
    // Any result still in flight belongs to the previous view: invalidate it immediately.
    exactTokenRef.current.base = 0;
    exactTokenRef.current.detail = 0;
    setExactBase(null);
    setDetail(null);
    setExactStatus("");
    if (!previewInfo || previewInfo.exact || posesRef.current.length === 0) return;
    const meta = previewInfo;
    let timer: ReturnType<typeof setTimeout>;
    const request = () => {
      // A short fast-preview job may be running (sliders, projection changes): retry shortly
      // instead of dropping the request.
      if (stateRef.current !== "idle") { timer = setTimeout(request, 250); return; }
      const t = exactTokenRef.current;
      const token = t.next++;
      t.base = token;
      setExactStatus("Rendering exact preview…");
      void window.Host.call("render-exact", { token, lane: "base", width: meta.width, height: meta.height, region: [0, 0, 1, 1], multiband });
    };
    timer = setTimeout(request, 220);
    return () => clearTimeout(timer);
  }, [previewInfo, multiband]);

  // A new view/geometry invalidates the overlay at once (it is re-requested below).
  useEffect(() => {
    seamTokenRef.current += 1;
    setSeamOverlay(null);
  }, [previewInfo]);

  // Seam overlay for the visible part of the canvas (whole canvas unless zoomed in).
  useEffect(() => {
    if (!showSeams || !previewInfo || posesRef.current.length === 0) {
      setSeamOverlay(null);
      return;
    }
    const timer = setTimeout(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const fx0 = (stageSize.w - frameW) / 2 + pan.x;
      const fy0 = (stageSize.h - frameH) / 2 + pan.y;
      const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
      const zoomed = frameW * dpr > previewInfo.width * 1.05;
      const region: [number, number, number, number] = zoomed
        ? [clamp01(-fx0 / frameW), clamp01(-fy0 / frameH), clamp01((stageSize.w - fx0) / frameW), clamp01((stageSize.h - fy0) / frameH)]
        : [0, 0, 1, 1];
      if (region[2] - region[0] < 1e-3 || region[3] - region[1] < 1e-3) return;
      const width = Math.max(16, Math.min(4096, Math.round((region[2] - region[0]) * frameW * dpr)));
      const height = Math.max(16, Math.min(4096, Math.round((region[3] - region[1]) * frameH * dpr)));
      const token = ++seamTokenRef.current;
      void window.Host.call("render-seams", { token, width, height, region, selected: selectedIdx });
    }, 200);
    return () => clearTimeout(timer);
  }, [showSeams, previewInfo, selectedIdx, frameW, frameH, pan.x, pan.y, stageSize.w, stageSize.h]);

  // Zoomed in beyond the canvas preview resolution: render the visible region 1:1 at device
  // pixels from full-resolution sources. The tile stays aligned while panning/zooming.
  useEffect(() => {
    if (!previewInfo || posesRef.current.length === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (frameW * dpr <= previewInfo.width * 1.05) {
      exactTokenRef.current.detail = 0;
      setDetail(null);
      return;
    }
    exactTokenRef.current.detail = 0;
    let timer: ReturnType<typeof setTimeout>;
    const request = () => {
      if (stateRef.current !== "idle") { timer = setTimeout(request, 250); return; }
      const fx0 = (stageSize.w - frameW) / 2 + pan.x;
      const fy0 = (stageSize.h - frameH) / 2 + pan.y;
      const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
      const u0 = clamp01(-fx0 / frameW), u1 = clamp01((stageSize.w - fx0) / frameW);
      const v0 = clamp01(-fy0 / frameH), v1 = clamp01((stageSize.h - fy0) / frameH);
      if (u1 - u0 < 1e-3 || v1 - v0 < 1e-3) return;
      const width = Math.max(16, Math.min(4096, Math.round((u1 - u0) * frameW * dpr)));
      const height = Math.max(16, Math.min(4096, Math.round((v1 - v0) * frameH * dpr)));
      const t = exactTokenRef.current;
      const token = t.next++;
      t.detail = token;
      setExactStatus("Rendering 1:1 detail…");
      void window.Host.call("render-exact", { token, lane: "detail", width, height, region: [u0, v0, u1, v1], multiband });
    };
    timer = setTimeout(request, 280);
    return () => clearTimeout(timer);
  }, [previewInfo, multiband, frameW, frameH, pan.x, pan.y, stageSize.w, stageSize.h]);

  const activeProfileLabel =
    colorProfile === "custom_icc"
      ? customIccName || "Custom ICC"
      : COLOR_PROFILE_LABELS[colorProfile] ?? colorProfile;
  const isJpegExport = /\.(jpe?g)$/i.test(exportPath.trim());
  const jpegBlocksHighBit = isJpegExport && outputBitDepth > 8;

  const sourceBitDepths = Array.from(
    new Set(metas.map((m) => m.bit_depth).filter((v): v is number => typeof v === "number")),
  );
  const allSources8Bit = sourceBitDepths.length > 0 && sourceBitDepths.every((d) => d <= 8);
  const sourceProfiles = Array.from(
    new Set(
      metas.map((m) =>
        m.icc_profile_name
          ? m.icc_profile_name
          : m.icc_status === "invalid"
            ? "Unreadable ICC (assumed sRGB)"
            : m.icc_status === "unsupported_colorspace"
              ? "Non-RGB ICC"
              : "No embedded ICC (assumed sRGB)",
      ),
    ),
  );
  const missingIccCount = metas.filter((m) => !m.icc_profile_name).length;
  const adoptCandidateIdx =
    selectedMeta?.icc_status === "embedded"
      ? selectedIdx
      : metas.findIndex((m) => m.icc_status === "embedded");
  const adoptCandidateMeta = adoptCandidateIdx >= 0 ? metas[adoptCandidateIdx] : null;

  useEffect(()=>{
    if(lockExportAspect && poses.length) setExportHeight(Math.min(200000,Math.max(1,Math.round(exportWidth/Math.max(0.01,cropAspect)))));
  },[lockExportAspect,exportWidth,cropAspect,poses.length]);

  return (
    <div
      className="app"
      ref={layout}
      onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragOver={(e) => { e.preventDefault(); if (!dragOver) setDragOver(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false); }}
      onDrop={handleDrop}
    >
      {dragOver && (
        <div className="drop-overlay" onDragLeave={() => setDragOver(false)}>
          <div className="drop-box">Drop images or folders to add to SlashStitch</div>
        </div>
      )}

      <header className="topbar">
        <span className="brand">SlashStitch</span>
        <span className="proj" title={projectPath??"Unsaved project"}>{projectPath?.split(/[\\/]/).at(-1)??"untitled.ssp"}{projectDirty?" *":""}</span>
        <button disabled={state!=="idle"} onClick={()=>void openProject()} title="Open Project (Ctrl+Shift+O)">Open</button>
        <button disabled={state!=="idle" && engineAvailable} onClick={()=>void saveProject()} title="Save Project (Ctrl+S; Shift for Save As)">Save</button>
        {recoveryAvailable&&<button disabled={state!=="idle"} onClick={()=>void openProject("recover-project")}>Recover</button>}
        {!engineAvailable&&<button onClick={()=>void openProject("restart-engine")}>Restart Engine</button>}
        <button title="Add Images (Ctrl+O)" disabled={state !== "idle" || Boolean(busyText)} onClick={() => void addImages()}>+ Add Images</button>
        <button id="start-job" title="Detect control points, optimize & level horizon (Ctrl+Shift+A)" disabled={state !== "idle" || Boolean(busyText)} onClick={() => void runAutoAlign()}>
          {poses.length >= 2 ? "Auto Align" : "Align & Preview"}
        </button>
        <button id="stop-job" disabled={state !== "running"} onClick={() => void run("job-cancel")}>
          Stop
        </button>
        <button title="Undo (Ctrl+Z)" disabled={undoStack.length === 0 || state !== "idle"} onClick={() => void undo()}>
          Undo
        </button>
        <button title="Redo (Ctrl+Shift+Z / Ctrl+Y)" disabled={redoStack.length === 0 || state !== "idle"} onClick={() => void redo()}>
          Redo
        </button>
        <span className="grow" />
        <span className="meta-pill">
          {engine
            ? `${engine.gpu_backend} · ${engine.cpu_threads}T · Lensfun ${engine.cameras_count}/${engine.lenses_count}`
            : "Connecting engine..."}
        </span>
        <span id="job-status" role="status" className={error ? "status-bad" : "status-pill"}>
          {statusDisplay}
        </span>
      </header>

      {error&&<div className="error-banner" role="alert"><span>{error}</span><button onClick={()=>setError("")} aria-label="Dismiss error">Dismiss</button></div>}
      <aside className="left-panel">
        <div className="panel-head">
          <strong>Source Images ({metas.length})</strong>
          {metas.length > 0 && (
            <button className="mini" disabled={state!=="idle"} onClick={() => void clearAll()}>
              Clear
            </button>
          )}
        </div>
        {metas.length === 0 ? (
          <p className="muted">Drag &amp; drop images or folders here, or click “+ Add Images” (Ctrl+O).</p>
        ) : (
          <ul className="image-list" role="listbox" aria-label="Source images">
            {metas.map((img, idx) => {
              const pose = poses[idx];
              return (
                <li
                  key={img.path}
                  className={idx === selectedIdx ? "image-item active" : "image-item"}
                  role="option" aria-selected={idx===selectedIdx} tabIndex={idx===selectedIdx?0:-1}
                  onKeyDown={e=>{if(e.key==="ArrowDown"||e.key==="ArrowUp"){e.preventDefault();setSelectedIdx(Math.max(0,Math.min(metas.length-1,idx+(e.key==="ArrowDown"?1:-1))));}if(e.key==="Enter"||e.key===" "){e.preventDefault();setSelectedIdx(idx);}}}
                  onClick={() => setSelectedIdx(idx)}
                  style={showSeams && seamOverlay?.colors[idx] ? { boxShadow: `inset 4px 0 0 ${seamOverlay.colors[idx]}` } : undefined}
                >
                  <div className="image-name">{img.file_name}</div>
                  <div className="image-sub">
                    {img.width && img.height ? `${img.width}×${img.height}` : "—"}
                    {img.bit_depth ? ` · ${img.bit_depth}-bit` : ""}
                    {pose ? ` · Y:${pose.yaw_deg.toFixed(1)}° P:${pose.pitch_deg.toFixed(1)}° R:${pose.roll_deg.toFixed(1)}°` : ""}
                  </div>
                  {(img.matched_lens || img.lens_model || img.icc_profile_name || img.icc_status === "missing") && (
                    <div className="image-lens">
                      {[img.matched_lens ?? img.lens_model, img.icc_profile_name ?? (img.icc_status === "missing" ? "No ICC (sRGB)" : null)].filter(Boolean).join(" · ")}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </aside>

      <div className="splitter" role="separator" tabIndex={0} aria-label="Resize source panel" aria-orientation="vertical" aria-valuenow={leftW} aria-valuemin={200} aria-valuemax={560} onKeyDown={e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();leftKey(e.key==="ArrowRight"?20:-20);}}} onPointerDown={leftDrag} />

      <main className="center">
        <div className="viewport-toolbar">
          <span>Projection:</span>
          <ProjectionPicker
            disabled={state!=="idle"}
            value={projection}
            onChange={(nextProj) => {
              const defaultFov =
                nextProj === "rectilinear"
                  ? Math.min(panoHfov, 120)
                  : nextProj === "stereographic_down" || nextProj === "circular_fisheye" || nextProj === "fullframe_fisheye"
                    ? Math.max(panoHfov, 210)
                    : nextProj === "vedutismo" || nextProj === "transverse_vedutismo"
                      ? Math.min(Math.max(panoHfov, 140), 280)
                      : panoHfov;
              void updateView(nextProj, defaultFov, viewYaw, viewPitch);
            }}
          />
          {(projection === "rectilinear" ||
            projection === "vedutismo" ||
            projection === "transverse_vedutismo" ||
            projection === "stereographic" ||
            projection === "stereographic_down" ||
            projection === "circular_fisheye" ||
            projection === "fullframe_fisheye") && (
            <label className="inline-field">
              FOV {panoHfov.toFixed(0)}°
              <input
                type="range"
                min={30}
                max={
                  projection === "rectilinear"
                    ? 145
                    : projection === "vedutismo" || projection === "transverse_vedutismo"
                      ? 300
                      : 360
                }
                value={panoHfov}
                onChange={(e) => void updateView(projection, Number(e.target.value), viewYaw, viewPitch)}
              />
            </label>
          )}
          {poses.length > 0 && HORIZON_SHIFT_PROJECTIONS.has(projection) && (
            <label className="inline-field" title="Vertical framing shift (keeps horizon straight)">
              Shift {(viewVShift * 100).toFixed(0)}%
              <input
                type="range"
                min={-0.6}
                max={0.6}
                step={0.01}
                value={viewVShift}
                onChange={(e) => void updateView(projection, panoHfov, viewYaw, viewPitch, false, Number(e.target.value))}
              />
            </label>
          )}
          {poses.length > 0 && (
            <>
              <span className="seg-group" role="group" aria-label="Drag mode">
                <button
                  disabled={state !== "idle"}
                  className={dragMode === "view" && !spaceHeld ? "mini tab active" : "mini"}
                  title="Left-drag shifts the stitched panorama (Yaw / V-Shift)"
                  onClick={() => setDragMode("view")}
                >
                  Move Pano
                </button>
                <button
                  disabled={state !== "idle"}
                  className={dragMode === "pan" || spaceHeld ? "mini tab active" : "mini"}
                  title="Left-drag (or Space/Right/Alt+drag) moves the whole canvas inside the workspace"
                  onClick={() => setDragMode("pan")}
                >
                  Pan Canvas
                </button>
              </span>
              <button disabled={state!=="idle"} className={cropMode ? "mini tab active" : "mini"} title="Toggle crop handles (drag edges/corners)" onClick={() => setCropMode((v) => !v)}>
                Crop
              </button>
              <button
                className={showSeams ? "mini tab active" : "mini"}
                aria-pressed={showSeams}
                title="Show a coloured frame around every source image and the exact blend seams used by the export"
                onClick={() => setShowSeams(v => !v)}
              >
                Seams
              </button>
              <button disabled={state!=="idle"} className="mini" title="Automatically crop largest rectangular area without black borders" onClick={() => void runAutoCrop()}>
                Auto Crop
              </button>
              {hasCrop && (
                <button disabled={state!=="idle"} className="mini" onClick={resetCrop}>
                  Reset Crop
                </button>
              )}
              <span className="meta-pill">{Math.round(viewZoom * 100)}%</span>
              <button className="mini" title="Reset zoom, pan, vertical shift & panorama center (or double-click canvas)" onClick={resetViewportTransform}>
                Reset View
              </button>
            </>
          )}
          <span className="muted toolbar-status" title={typeof progress === "object" && progress ? progress.stage : undefined}>
            {progress
              ? `${progress.stage} (${progress.percent}%)`
              : notice
                ? notice
                : exactStatus
                ? exactStatus
                : previewInfo
                  ? `${detail ? "Exact 1:1 detail" : exactBase ? `${exactBase.label} · ${exactBase.ms.toFixed(0)} ms` : `Fast preview · ${previewInfo.backend} · ${previewInfo.render_ms.toFixed(1)} ms`} · ${previewInfo.color_profile_name ?? activeProfileLabel} (Export ${outputBitDepth}-bit)`
                  : "Ready"}
          </span>
        </div>

        <div
          ref={viewportRef}
          className="viewport-canvas"
          style={{ cursor: dragMode === "pan" || spaceHeld ? "all-scroll" : undefined }}
          onPointerDown={dragPreview}
          onWheel={handleWheelZoom}
          onDoubleClick={resetViewportTransform}
          onContextMenu={(e) => e.preventDefault()}
          title="Drag: Move Pano (or Pan Canvas) · Scroll: zoom · Space/Right/Alt+drag: always pan canvas"
        >
          <div className="equirect-grid" />
          {previewUrl ? (
            <div
              className="preview-frame"
              style={{
                width: `${frameW}px`,
                height: `${frameH}px`,
                transform: `translate(${pan.x}px, ${pan.y}px)`,
              }}
            >
              <img className="preview-img" src={exactBase?.url ?? previewUrl} alt="Panorama preview" draggable={false} />
              {showSeams && seamOverlay && (
                <img
                  className="preview-detail seam-overlay"
                  src={seamOverlay.url}
                  alt=""
                  draggable={false}
                  style={{
                    left: `${seamOverlay.region[0] * 100}%`,
                    top: `${seamOverlay.region[1] * 100}%`,
                    width: `${(seamOverlay.region[2] - seamOverlay.region[0]) * 100}%`,
                    height: `${(seamOverlay.region[3] - seamOverlay.region[1]) * 100}%`,
                  }}
                />
              )}
              {detail && (
                <img
                  className="preview-detail"
                  src={detail.url}
                  alt=""
                  draggable={false}
                  style={{
                    left: `${detail.region[0] * 100}%`,
                    top: `${detail.region[1] * 100}%`,
                    width: `${(detail.region[2] - detail.region[0]) * 100}%`,
                    height: `${(detail.region[3] - detail.region[1]) * 100}%`,
                  }}
                />
              )}
              {(cropMode || hasCrop) && (
                <div className="crop-overlay">
                  <div className="crop-shade" style={{ left: 0, top: 0, right: 0, height: `${crop.top * 100}%` }} />
                  <div className="crop-shade" style={{ left: 0, bottom: 0, right: 0, height: `${(1 - crop.bottom) * 100}%` }} />
                  <div className="crop-shade" style={{ left: 0, top: `${crop.top * 100}%`, height: `${(crop.bottom - crop.top) * 100}%`, width: `${crop.left * 100}%` }} />
                  <div className="crop-shade" style={{ right: 0, top: `${crop.top * 100}%`, height: `${(crop.bottom - crop.top) * 100}%`, width: `${(1 - crop.right) * 100}%` }} />
                  <div
                    className={cropMode ? "crop-box editable" : "crop-box"}
                    style={{
                      left: `${crop.left * 100}%`,
                      top: `${crop.top * 100}%`,
                      width: `${(crop.right - crop.left) * 100}%`,
                      height: `${(crop.bottom - crop.top) * 100}%`,
                    }}
                    role="slider" tabIndex={cropMode?0:-1} aria-label="Move crop region" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} onKeyDown={e=>cropKeyboard("move",e)}
                    onPointerDown={cropMode ? (e) => startCropDrag("move", e) : undefined}
                  >
                    {cropMode && (
                      <>
                        <div className="crop-handle h-n" role="slider" tabIndex={0} aria-label="Crop n edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("n",e)} onPointerDown={(e) => startCropDrag("n", e)} />
                        <div className="crop-handle h-s" role="slider" tabIndex={0} aria-label="Crop s edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("s",e)} onPointerDown={(e) => startCropDrag("s", e)} />
                        <div className="crop-handle h-w" role="slider" tabIndex={0} aria-label="Crop w edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("w",e)} onPointerDown={(e) => startCropDrag("w", e)} />
                        <div className="crop-handle h-e" role="slider" tabIndex={0} aria-label="Crop e edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("e",e)} onPointerDown={(e) => startCropDrag("e", e)} />
                        <div className="crop-handle h-nw" role="slider" tabIndex={0} aria-label="Crop nw edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("nw",e)} onPointerDown={(e) => startCropDrag("nw", e)} />
                        <div className="crop-handle h-ne" role="slider" tabIndex={0} aria-label="Crop ne edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("ne",e)} onPointerDown={(e) => startCropDrag("ne", e)} />
                        <div className="crop-handle h-sw" role="slider" tabIndex={0} aria-label="Crop sw edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("sw",e)} onPointerDown={(e) => startCropDrag("sw", e)} />
                        <div className="crop-handle h-se" role="slider" tabIndex={0} aria-label="Crop se edge" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(crop.left*100)} aria-valuetext={`Left ${Math.round(crop.left*100)}%, top ${Math.round(crop.top*100)}%, right ${Math.round(crop.right*100)}%, bottom ${Math.round(crop.bottom*100)}%`} onKeyDown={e=>cropKeyboard("se",e)} onPointerDown={(e) => startCropDrag("se", e)} />
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="viewport-overlay">
              <h2>Panorama Viewport</h2>
              <p className="muted">
                Drag &amp; drop overlapping images or folders here to render a live GPU preview, run Auto Align, and export via strip-based multi-band BigTIFF.
              </p>
            </div>
          )}
          {progress && (
            <div className="progress-banner">
              <span>{progress.stage} ({progress.percent}%)</span>
              <div className="progress-wrap">
                <div className="progress-bar" style={{ width: `${progress.percent}%` }} />
              </div>
            </div>
          )}
        </div>
      </main>

      <div className="splitter" role="separator" tabIndex={0} aria-label="Resize inspector panel" aria-orientation="vertical" aria-valuenow={rightW} aria-valuemin={270} aria-valuemax={720} onKeyDown={e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();rightKey(e.key==="ArrowLeft"?20:-20);}}} onPointerDown={rightDrag} />

      <aside className="right-panel">
        <div className="tabs">
          <button className={tab === "lens" ? "tab active" : "tab"} onClick={() => setTab("lens")}>Lens &amp; Pose</button>
          <button className={tab === "optimizer" ? "tab active" : "tab"} onClick={() => setTab("optimizer")}>Optimizer</button>
          <button className={tab === "lensdb" ? "tab active" : "tab"} onClick={() => setTab("lensdb")}>Lensfun DB</button>
          <button className={tab === "export" ? "tab active" : "tab"} onClick={() => setTab("export")}>Export</button>
        </div>

        {tab === "lens" && (
          <fieldset className="inspector-body" disabled={state!=="idle"}>
            <h3>Selected Frame EXIF &amp; Geometry</h3>
            {!selectedMeta || !selectedPose ? (
              <p className="muted">Select a source image to adjust yaw/pitch/roll/HFOV or inspect Lensfun distortion.</p>
            ) : (
              <>
                <dl className="kv">
                  <dt>File</dt><dd>{selectedMeta.file_name}</dd>
                  <dt>Camera</dt><dd>{[selectedMeta.camera_make, selectedMeta.camera_model].filter(Boolean).join(" ") || "—"}</dd>
                  <dt>EXIF Lens</dt><dd>{selectedMeta.lens_model ?? "—"}</dd>
                  <dt>Lensfun Match</dt><dd>{selectedMeta.matched_lens ?? "Inferred from EXIF"}</dd>
                  <dt>Bit Depth</dt><dd>{selectedMeta.bit_depth ? `${selectedMeta.bit_depth}-bit/ch` : "Unknown"}</dd>
                  <dt>Color Profile</dt><dd>{selectedMeta.icc_profile_name ?? (selectedMeta.icc_status === "invalid" ? "Unreadable ICC (assumed sRGB)" : selectedMeta.icc_status === "unsupported_colorspace" ? "Non-RGB ICC" : "No embedded ICC (assumed sRGB)")}</dd>
                  <dt>Distortion</dt><dd className="mono">{selectedPose.distortion_poly5?`Poly5 ${selectedPose.distortion_poly5.join(", ")}`:`PTLens ${selectedPose.distortion.map(v=>v.toFixed(4)).join(", ")}`} · scale {(selectedPose.distortion_scale??1).toFixed(3)}</dd>
                </dl>
                {selectedMeta.warnings.length>0&&<ul className="muted" aria-label="Source warnings">{selectedMeta.warnings.map((warning,index)=><li key={index}>{warning}</li>)}</ul>}
                <div className="pose-grid">
                  <label>Yaw (°)<input type="number" step={0.5} value={Number(selectedPose.yaw_deg.toFixed(2))} onChange={(e) => void updatePose({ yaw_deg: Number(e.target.value) })} /></label>
                  <label>Pitch (°)<input type="number" step={0.5} value={Number(selectedPose.pitch_deg.toFixed(2))} onChange={(e) => void updatePose({ pitch_deg: Number(e.target.value) })} /></label>
                  <label>Roll (°)<input type="number" step={0.5} value={Number(selectedPose.roll_deg.toFixed(2))} onChange={(e) => void updatePose({ roll_deg: Number(e.target.value) })} /></label>
                  <label>HFOV (°)<input type="number" step={0.5} min={5} max={selectedPose.lens_projection==="rectilinear"?175:355} value={Number(selectedPose.hfov_deg.toFixed(2))} onChange={(e) => void updatePose({ hfov_deg: Number(e.target.value) })} /></label>
                </div>
                <label className="stack-field">
                  Lens Projection
                  <select
                    value={selectedPose.lens_projection}
                    onChange={(e) => void updatePose({ lens_projection: e.target.value })}
                  >
                    <option value="rectilinear">Rectilinear</option>
                    <option value="fisheye_equisolid">Fisheye (Diagonal / Equisolid)</option>
                    <option value="fisheye_fullframe">Fisheye (Equidistant Full-frame)</option>
                    <option value="fisheye_circular">Fisheye (Circular)</option>
                    <option value="equirectangular">Equirectangular</option>
                  </select>
                </label>
              </>
            )}
          </fieldset>
        )}

        {tab === "optimizer" && (
          <fieldset className="inspector-body" disabled={state!=="idle"}>
            <h3>Bundle Adjustment &amp; Horizon Options</h3>
            <p className="muted">Topbar “Auto Align” detects new control points + optimizes. Here you can configure solver options or re-optimize existing control points.</p>
            <label className="check-row">
              <input type="checkbox" checked={optimizeHfov} onChange={(e) => setOptimizeHfov(e.target.checked)} />
              Optimize shared HFOV (lens calibration)
            </label>
            <label className="check-row">
              <input type="checkbox" checked={optimizeDistortion} onChange={(e) => setOptimizeDistortion(e.target.checked)} />
              Optimize lens distortion (a / b / c)
            </label>
            <label className="check-row">
              <input type="checkbox" checked={prioritizeSky} onChange={(e) => setPrioritizeSky(e.target.checked)} />
              Prioritize sky &amp; stars (subpixel star alignment)
            </label>
            <label className="check-row">
              <input type="checkbox" checked={levelHorizon} onChange={(e) => setLevelHorizon(e.target.checked)} />
              Level camera sweep automatically
            </label>
            <div className="export-row">
              <button disabled={state !== "idle" || poses.length < 2} onClick={() => void runAutoAlign()}>
                Detect &amp; Align
              </button>
              <button disabled={state !== "idle" || controlPoints.length < 4 || poses.length < 2} title="Re-run bundle adjustment on the current control points (also after opening a project)" onClick={() => void runReoptimize()}>
                Re-optimize CPs
              </button>
            </div>
            {alignReport && (
              <dl className="kv">
                <dt>Control Points</dt><dd>{alignReport.control_points_used} used · {alignReport.control_points_rejected??0} rejected</dd>
                <dt>Iterations</dt><dd>{alignReport.iterations}</dd>
                <dt>Initial RMS</dt><dd>{(Number.isFinite(alignReport.initial_rms_deg)?alignReport.initial_rms_deg.toFixed(3):"not measured")}°</dd>
                <dt>Final RMS</dt><dd>{(Number.isFinite(alignReport.final_rms_deg)?alignReport.final_rms_deg.toFixed(3):"not measured")}°</dd>
                <dt>Max Error</dt><dd>{(Number.isFinite(alignReport.max_error_deg)?alignReport.max_error_deg.toFixed(3):"not measured")}°</dd>
                {alignReport.optimized_hfov_deg && (
                  <>
                    <dt>Optimized HFOV</dt><dd>{alignReport.optimized_hfov_deg.toFixed(2)}°</dd>
                  </>
                )}
                {alignReport.optimized_distortion && (
                  <>
                    <dt>Distortion (a/b/c)</dt><dd className="mono">{alignReport.optimized_distortion.map((v) => v.toFixed(4)).join(", ")}</dd>
                  </>
                )}
              </dl>
            )}
          </fieldset>
        )}

        {tab === "lensdb" && (
          <fieldset className="inspector-body" disabled={state!=="idle"}>
            <div className="panel-head">
              <h3>Open Lensfun Database</h3>
              <button className="mini" disabled={state !== "idle" || Boolean(busyText)} onClick={() => void updateLensfunDb()}>
                Update DB
              </button>
            </div>
            <button disabled={state!=="idle"||poses.length===0} onClick={async()=>{pushHistory();const result=await run("reapply-lensfun");if(result.ok){const data=JSON.parse(result.text);setPoses(data.images);setMetas(data.metas);setControlPoints([]);setAlignReport(null);await refreshPreviewImage();}}}>Reapply profiles to source images</button>
            <p className="muted">Updating the database keeps current project geometry. Reapply explicitly, then run Auto Align again.</p>
            {dbUpdateMsg && <p className="muted">{dbUpdateMsg}</p>}
            <input
              className="search-input"
              value={dbQuery}
              onChange={(e) => void searchDb(e.target.value)}
              placeholder="Search camera or lens (e.g. Sigma 15mm, Sony, Canon)"
            />
            <ul className="db-list">
              {dbHits.map((hit, i) => (
                <li key={`${hit.kind}-${hit.maker}-${hit.model}-${i}`} className="db-item">
                  <span className="db-kind">{hit.kind}</span>
                  <div>
                    <div>{hit.maker} {hit.model}</div>
                    <div className="muted">{hit.mount || "Generic"} · crop {hit.crop_factor.toFixed(2)}× {hit.lens_type ? `· ${hit.lens_type}` : ""}</div>
                  </div>
                </li>
              ))}
            </ul>
          </fieldset>
        )}

        {tab === "export" && (
          <fieldset className="inspector-body" disabled={state!=="idle"}>
            <h3>Panorama Export · {outputBitDepth}-bit {activeProfileLabel}</h3>
            <p className="muted">
              TIFF / BigTIFF uses bounded strips and a 16-bit disk-backed source cache (8-bit, 10-bit in RGB16, or full 16-bit). PNG supports 8, 10, and 16-bit (max 512 MiB buffer); JPEG supports 8-bit only.
            </p>

            {metas.length > 0 && (
              <div className="stack-field">
                <strong>Source Color &amp; Bit Depth</strong>
                <dl className="kv">
                  <dt>Sources</dt>
                  <dd>
                    {sourceBitDepths.length > 1
                      ? `Mixed (${sourceBitDepths.join("-bit, ")}-bit)`
                      : sourceBitDepths.length === 1
                        ? `${sourceBitDepths[0]}-bit/ch`
                        : "Unknown"}
                  </dd>
                  <dt>Profiles</dt>
                  <dd>
                    {sourceProfiles.length > 1
                      ? `Mixed (${sourceProfiles.join(", ")})`
                      : sourceProfiles[0] ?? "Unknown"}
                  </dd>
                </dl>
                {missingIccCount > 0 && (
                  <p className="muted">
                    {missingIccCount} of {metas.length} source(s) have no embedded RGB ICC profile and are interpreted as sRGB.
                  </p>
                )}
                {adoptCandidateMeta && (
                  <button
                    className="mini"
                    disabled={state !== "idle"}
                    onClick={() => void updateColorSettings({ adopt_source_icc_index: adoptCandidateIdx })}
                  >
                    Adopt ICC from {adoptCandidateMeta.file_name} ({adoptCandidateMeta.icc_profile_name ?? "Embedded"} · {adoptCandidateMeta.bit_depth ?? 8}-bit source)
                  </button>
                )}
              </div>
            )}

            <label className="stack-field">
              Output Bit Depth
              <select
                value={outputBitDepth}
                onChange={(e) => void updateColorSettings({ output_bit_depth: Number(e.target.value) })}
              >
                <option value={8}>8 bits/channel · 256 levels (TIFF, PNG, JPEG)</option>
                <option value={10}>10 bits/channel · 1024 levels in RGB16 (TIFF/BigTIFF, PNG)</option>
                <option value={16}>16 bits/channel · 65,536 levels, full precision (TIFF/BigTIFF, PNG)</option>
              </select>
            </label>

            <div className="stack-field">
              <label htmlFor="target-color-profile">Target Color Profile ({outputBitDepth}-bit/ch Export &amp; Preview)</label>
              <div className="export-row">
                <select
                  id="target-color-profile"
                  className="search-input"
                  value={colorProfile}
                  onChange={(e) => void updateColorSettings({ color_profile: e.target.value })}
                >
                  <option value="srgb">sRGB IEC61966-2.1 ({outputBitDepth}-bit/ch)</option>
                  <option value="adobe_rgb">Adobe RGB (1998) ({outputBitDepth}-bit/ch)</option>
                  <option value="display_p3">Display P3 ({outputBitDepth}-bit/ch)</option>
                  <option value="prophoto_rgb">ProPhoto RGB ({outputBitDepth}-bit/ch)</option>
                  <option value="bt2020">Rec.2020 / BT.2020 ({outputBitDepth}-bit/ch)</option>
                  {(customIccBase64 || colorProfile === "custom_icc") && (
                    <option value="custom_icc">Custom ICC: {customIccName ?? "Loaded Profile"} ({outputBitDepth}-bit/ch)</option>
                  )}
                </select>
                <button disabled={state !== "idle"} onClick={() => void chooseIccProfile()}>
                  Load ICC…
                </button>
              </div>
              <p className="muted">
                All profiles (sRGB, Adobe RGB, Display P3, ProPhoto RGB, Rec.2020, Custom ICC) support 8, 10, and 16 bits/channel.
                {(colorProfile === "prophoto_rgb" || colorProfile === "bt2020") && outputBitDepth === 8
                  ? " Note: wide-gamut profiles like ProPhoto RGB and Rec.2020 benefit from 16-bit (or 10-bit) TIFF/PNG to prevent banding."
                  : ""}
              </p>
            </div>

            {allSources8Bit && outputBitDepth > 8 && (
              <p className="muted">
                Note: all loaded sources are 8-bit/ch. Exporting {outputBitDepth}-bit converts cleanly into {activeProfileLabel} inside a 16-bit container and avoids new blending banding, but cannot recover tonal detail absent from the 8-bit originals.
              </p>
            )}

            <div className="export-row">
              <input className="search-input" value={exportPath} onChange={(e) => setExportPath(e.target.value)} placeholder="Choose output path..." />
              <button onClick={() => void chooseExport()}>Browse…</button>
            </div>
            {jpegBlocksHighBit && (
              <div className="error-banner" role="alert">
                <span>JPEG only supports 8 bits per channel. Switch Output Bit Depth to 8 bits/ch, or export to TIFF/BigTIFF or PNG for {outputBitDepth}-bit precision.</span>
              </div>
            )}
            <div className="pose-grid">
              <label>Width (px)<input type="number" min={64} step={64} value={exportWidth} onChange={(e) => setExportWidth(Number(e.target.value))} /></label>
              <label>Height (px)<input type="number" min={64} step={64} value={exportHeight} onChange={(e) => {const value=Number(e.target.value);setExportHeight(value);if(lockExportAspect)setExportWidth(Math.max(1,Math.min(200000,Math.round(value*cropAspect))));}} /></label>
            </div>
            <label className="check-row"><input type="checkbox" checked={lockExportAspect} onChange={e=>setLockExportAspect(e.target.checked)}/>Keep crop aspect ratio</label>
            <div className="export-row">
              <button className="mini" onClick={() => setExportHeight(Math.max(64, Math.round(exportWidth / Math.max(0.1, cropAspect))))}>
                Match Crop Aspect
              </button>
              <button className="mini" disabled={poses.length === 0} onClick={() => void runAutoCrop()}>
                Auto Crop
              </button>
              {hasCrop && (
                <button disabled={state!=="idle"} className="mini" onClick={resetCrop}>
                  Reset Crop
                </button>
              )}
            </div>
            <div className="muted">
              Crop region: L {(crop.left * 100).toFixed(1)}% · T {(crop.top * 100).toFixed(1)}% · R {(crop.right * 100).toFixed(1)}% · B {(crop.bottom * 100).toFixed(1)}%
            </div>
            <div className="stitch-mode">
              <label className="field-label" htmlFor="stitch-mode">Stitch mode</label>
              <select
                id="stitch-mode"
                disabled={state !== "idle" || poses.length === 0}
                value={currentStitchPreset}
                title={STITCH_PRESETS.find(p => p.id === currentStitchPreset)?.hint ?? "Custom combination of the advanced settings below."}
                onChange={(e) => { const preset = STITCH_PRESETS.find(p => p.id === e.target.value); if (preset) void applyStitch(preset.s); }}
              >
                {STITCH_PRESETS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                {currentStitchPreset === "custom" && <option value="custom">Custom</option>}
              </select>
              <div className="muted small">{STITCH_PRESETS.find(p => p.id === currentStitchPreset)?.hint ?? "Custom combination of the advanced settings below."}</div>
            </div>
            <details className="advanced-stitch">
              <summary>Advanced stitching</summary>
              <label className="inline-select">
                Blending
                <select value={multiband ? "multiband" : "feather"} disabled={state !== "idle"} onChange={(e) => void applyStitch({ multiband: e.target.value === "multiband" })}>
                  <option value="multiband">Multi-band (Laplacian pyramid)</option>
                  <option value="feather">Feather (soft weighted average)</option>
                </select>
              </label>
              <label className="inline-select" title="Angular width of the coarsest blend band: wider hides exposure and colour differences (aurora, sky gradients), narrower keeps more local contrast near seams.">
                Transition width
                <select value={String(blendWidth)} disabled={state !== "idle" || !multiband} onChange={(e) => void applyStitch({ blendWidth: Number(e.target.value) })}>
                  {[1.5, 3, 6, 10].map(w => <option key={w} value={String(w)}>{w === 1.5 ? "Narrow" : w === 3 ? "Normal" : w === 6 ? "Wide" : "Extra wide"} ({w}°)</option>)}
                  {![1.5, 3, 6, 10].includes(blendWidth) && <option value={String(blendWidth)}>{blendWidth}°</option>}
                </select>
              </label>
              <label className="inline-select" title="Resampling kernel for the exact preview and the export.">
                Interpolation
                <select value={interpolation} disabled={state !== "idle"} onChange={(e) => void applyStitch({ interpolation: e.target.value as Interp })}>
                  <option value="spline36">Spline36 (sharp, clean stars)</option>
                  <option value="lanczos3">Lanczos3 (sharpest)</option>
                  <option value="bilinear">Bilinear (fast, soft)</option>
                </select>
              </label>
              <label className="check-row" title="Correct lens vignetting estimated from the overlaps. Exposure/white-balance gains stay active.">
                <input type="checkbox" checked={correctVignetting} disabled={state !== "idle" || !exposureCompensation} onChange={(e) => void applyStitch({ vignetting: e.target.checked })} />
                Vignetting correction
              </label>
            </details>
            <label className="check-row" title="Matches exposure and white balance between overlapping frames and removes lens vignetting. Applied identically to the preview and the export.">
              <input
                type="checkbox"
                checked={exposureCompensation}
                disabled={state !== "idle" || poses.length === 0}
                onChange={async (e) => {
                  const value = e.target.checked;
                  setExposureCompensation(value);
                  const reply = await run("set-render-settings", { exposure_compensation: value });
                  if (reply.ok) { const data = JSON.parse(reply.text); if (data.preview) setPreviewInfo(data.preview); }
                  else setExposureCompensation(!value);
                }}
              />
              Exposure, white balance &amp; vignetting compensation
            </label>
            <label className="check-row" title="Where three or more frames overlap and one of them shows a brief light (a torch on a rock, a passing car) that the others do not, that area is taken from the frames that agree. Applied to the exact preview and the export.">
              <input
                type="checkbox"
                checked={rejectTransients}
                disabled={state !== "idle" || poses.length === 0}
                onChange={async (e) => {
                  const value = e.target.checked;
                  setRejectTransients(value);
                  const reply = await run("set-render-settings", { reject_transients: value });
                  if (reply.ok) { const data = JSON.parse(reply.text); if (data.preview) setPreviewInfo(data.preview); }
                  else setRejectTransients(!value);
                }}
              />
              Remove transient light (majority of overlapping frames)
            </label>
            <label className="check-row" title="Route each seam through the part of the overlap where the neighbouring frames agree best (moving aurora, clouds, parallax), instead of a fixed line between the image centres. Shown by the Seams overlay; applied to the exact preview and the export.">
              <input
                type="checkbox"
                checked={optimizeSeams}
                disabled={state !== "idle" || poses.length === 0}
                onChange={async (e) => {
                  const value = e.target.checked;
                  setOptimizeSeams(value);
                  const reply = await run("set-render-settings", { optimize_seams: value });
                  if (reply.ok) { const data = JSON.parse(reply.text); if (data.preview) setPreviewInfo(data.preview); }
                  else setOptimizeSeams(!value);
                }}
              />
              Optimise seams (follow areas where frames agree)
            </label>
            <label className="check-row">
              <input type="checkbox" checked={forceBigTiff} onChange={(e) => setForceBigTiff(e.target.checked)} />
              Force BigTIFF (64-bit offsets, auto above 3.5 GB)
            </label>
            <button disabled={state !== "idle" || poses.length === 0 || !exportPath || jpegBlocksHighBit} onClick={() => void startExport()}>
              Export Panorama
            </button>
            {exportSummary && (
              <dl className="kv">
                <dt>Output</dt><dd className="mono">{exportSummary.output_path}</dd>
                <dt>Size</dt><dd>{exportSummary.width} × {exportSummary.height}</dd>
                <dt>Precision</dt><dd>{exportSummary.bit_depth ?? outputBitDepth}-bit/ch ({exportSummary.container_bit_depth ?? (outputBitDepth > 8 ? 16 : 8)}-bit container)</dd>
                <dt>Profile</dt><dd>{exportSummary.color_profile_name ?? activeProfileLabel}{exportSummary.icc_embedded ? " · ICC embedded" : ""}</dd>
                <dt>Format</dt><dd>{exportSummary.bigtiff ? "BigTIFF" : "Standard"} · {(exportSummary.bytes_written / (1024 * 1024)).toFixed(2)} MB</dd>
              </dl>
            )}
          </fieldset>
        )}
      </aside>
    </div>
  );
}
