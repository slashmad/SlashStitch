import { randomUUID } from "node:crypto";
import { app, BrowserWindow, dialog, ipcMain, net, protocol } from "electron";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, statSync } from "node:fs";
import { readdir, realpath, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { WorkerBridge, type WorkerReply } from "./worker";
import { ShutdownGate } from "./lifecycle";
import { protectSources, writeJsonAtomic } from "./storage";
import { downloadLensfunSnapshot } from "./lensfun-update";

app.setName("SlashStitch");
const userData = app.getPath("userData");
const settingsFile = join(userData, "settings.json");
const recoveryFile = join(userData, "recovery.ssp");
const lensfunPointer = join(userData, "lensfun-active.json");
let settings = { width: 1440, height: 900, displayBackend: "auto" };
try {
  const saved = JSON.parse(readFileSync(settingsFile, "utf8"));
  if (["auto", "wayland", "x11"].includes(saved.displayBackend)) settings.displayBackend = saved.displayBackend;
  for (const key of ["width", "height"] as const) if (Number.isFinite(saved[key])) settings[key] = Math.round(Math.min(10000, Math.max(600, saved[key])));
} catch { /* First launch or invalid settings. */ }
if (process.platform === "linux") {
  const requested = process.env.SLASHSTITCH_DISPLAY || app.commandLine.getSwitchValue("ozone-platform") || settings.displayBackend;
  const backend = requested === "auto" ? (process.env.WAYLAND_DISPLAY ? "wayland" : "x11") : requested;
  if (backend !== "wayland" && backend !== "x11") throw new Error("SLASHSTITCH_DISPLAY must be auto, wayland or x11");
  app.commandLine.appendSwitch("ozone-platform", backend);
  if (backend === "wayland") app.commandLine.appendSwitch("disable-features", "Vulkan");
}

// Set the display backend before requesting the lock: Electron may initialize Ozone
// while opening its single-instance channel, before app.whenReady().
// A second instance must exit before it can clean up the first instance's cache.
if (!app.requestSingleInstanceLock()) {
  app.exit(0);
  process.exit(0);
}

const previewRoot = mkdtempSync(join(tmpdir(), "slashstitch-session-"));
// Full-resolution source pyramids are large: keep them on disk under userData rather than in
// the (often RAM-backed) temp directory. Stale caches from crashed sessions are removed here;
// only one instance per profile runs (single-instance lock above), so nothing here is in use.
const sourceCacheRoot = join(userData, "source-cache");
try {
  mkdirSync(sourceCacheRoot, { recursive: true });
  for (const entry of readdirSync(sourceCacheRoot)) if (entry.startsWith("slashstitch-sources-")) rmSync(join(sourceCacheRoot, entry), { recursive: true, force: true });
} catch (error) { console.error("Source cache cleanup failed", error); }
const workerEnv: Record<string, string> = { SLASHSTITCH_PREVIEW_ROOT: previewRoot, SLASHSTITCH_CACHE_ROOT: sourceCacheRoot };
// Preview images are served by URL instead of base64 data URLs over IPC (no 33 % inflation,
// no main-process encoding, browser-side caching per version).
const PREVIEW_SCHEME = "slashstitch-preview";
protocol.registerSchemesAsPrivileged([{ scheme: PREVIEW_SCHEME, privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } }]);
function previewUrlFor(path: string): string | null {
  const absolute = resolve(path);
  const rel = relative(previewRoot, absolute);
  if (!rel || rel.startsWith("..") || isAbsolute(rel) || extname(absolute).toLowerCase() !== ".png") return null;
  return `${PREVIEW_SCHEME}://session/${rel.split(sep).map(encodeURIComponent).join("/")}`;
}
function decoratePreview(value: unknown): void {
  if (!value || typeof value !== "object") return;
  const v = value as Record<string, unknown>;
  if (typeof v.preview_path === "string") { const url = previewUrlFor(v.preview_path); if (url) v.preview_url = url; }
  if (v.preview) decoratePreview(v.preview);
}
try {
  const pointer = JSON.parse(readFileSync(lensfunPointer, "utf8"));
  if (typeof pointer.path === "string" && existsSync(pointer.path)) workerEnv.SLASHSTITCH_LENSFUN_DIR = pointer.path;
} catch { /* Use system/built-in profiles until an app snapshot is activated. */ }
function resolveWorker() {
  if (process.env.SLASHSTITCH_USE_DEMO_WORKER === "1") return { command: process.execPath, args: [join(__dirname, "demo-worker.js")], env: { ...workerEnv, ELECTRON_RUN_AS_NODE: "1" } };
  const exe = process.platform === "win32" ? "slashstitch-worker.exe" : "slashstitch-worker";
  const base = app.isPackaged ? join(process.resourcesPath, "bin") : join(__dirname, "../../engine/target", process.env.SLASHSTITCH_DEBUG_WORKER === "1" ? "debug" : "release");
  return { command: join(base, exe), args: [], env: workerEnv };
}
const spec = resolveWorker();
const worker = new WorkerBridge(spec.command, spec.args, spec.env,
  process.platform === "win32" && process.env.SLASHSTITCH_USE_DEMO_WORKER !== "1" ? {
    probeAction: "engine-info", timeoutMs: 15_000,
    fallbackEnv: process.env.SLASHSTITCH_FORCE_CPU === undefined ? {SLASHSTITCH_FORCE_CPU:"1"} : undefined,
    retryExitCodes: [0xc0000005,0xc000001d,0xc0000409],
  } : undefined);
let win: BrowserWindow | null = null;
let jobState = "idle";
let pendingMutation = false;
let databaseAbort: AbortController | null = null;
let engineAvailable = true;
let previewPath: string | null = null;
let projectFile: string | null = null;
let dirty = false;
let recovery: Record<string, unknown> | null = null;
let savedRecovery: Record<string, unknown> | null = null;
try { savedRecovery = JSON.parse(readFileSync(recoveryFile, "utf8")); } catch { /* No valid recovery snapshot. */ }
const ok = (result: unknown): WorkerReply => ({ ok: true, text: JSON.stringify(result) });
const fail = (message: string): WorkerReply => ({ ok: false, text: message });
const busy = () => pendingMutation || jobState === "running" || jobState === "finalizing";
const send = (name: string, data: unknown) => win?.webContents.send("host:event", name, data);
const notifyProject = () => send("project-file", { path: projectFile, dirty, recoveryAvailable: Boolean(savedRecovery), engineAvailable });
function rememberPreview(value: unknown): void {
  if (!value || typeof value !== "object") return;
  const v = value as Record<string, unknown>;
  if (typeof v.preview_path === "string") {
    const path = resolve(v.preview_path);
    const rel = relative(previewRoot, path);
    if (rel && !rel.startsWith("..") && !isAbsolute(rel)) previewPath = path;
  }
  if (v.preview) rememberPreview(v.preview);
}
function rememberDocument(document: unknown): void {
  if (!document || typeof document !== "object") return;
  const firstDocument=!recovery;
  const doc=document as Record<string,unknown>;
  recovery = {...doc,ui:doc.ui??recovery?.ui??null};
  dirty = true;
  try {
    mkdirSync(userData, { recursive: true });
    if(savedRecovery && firstDocument)writeJsonAtomic(join(userData,`recovery-previous-${Date.now()}.ssp`),savedRecovery);
    writeJsonAtomic(recoveryFile, recovery);
  }
  catch (error) { send("worker-error", { text: `Recovery could not be saved: ${(error as Error).message}` }); }
  notifyProject();
}
async function mutate(action: string, payload: unknown = null, timeout = 90_000): Promise<WorkerReply> {
  if (gate.stopping || gate.allowed) return fail("Application is closing");
  if (busy()) return fail("A job is active. Stop it and wait for finalization before changing the project.");
  if (!engineAvailable) return fail("Engine disconnected. Restart it to restore the current project.");
  pendingMutation = true;
  try {
    const reply = await worker.request(action, payload, timeout);
    if (reply.ok) {
      const result=JSON.parse(reply.text);rememberPreview(result);decoratePreview(result);
      reply.text=JSON.stringify(result);
      if(action==="clear-images")previewPath=null;
      if(result && Array.isArray(result.images) && Array.isArray(result.metas))send("project-updated",result);
    }
    return reply;
  } finally { pendingMutation = false; }
}
// Camera RAW/DNG extensions, kept in sync with RAW_EXTENSIONS in the engine (rawdev.rs).
const RAW_EXTS = ["3fr", "ari", "arw", "cr2", "cr3", "crw", "dcr", "dcs", "dng", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "raf", "rw2", "sr2", "srf", "srw", "x3f"];
const RASTER_EXTS = ["jpg", "jpeg", "png", "tif", "tiff", "webp"];
const IMAGE_EXTS = new Set([...RASTER_EXTS, ...RAW_EXTS].map(ext => `.${ext}`));
async function expandImagePaths(paths: string[]): Promise<{paths:string[];failures:string[]}> {
  if(paths.length>128)throw new Error("At most 128 sources can be imported at once.");
  const out=new Set<string>();const failures:string[]=[];
  const add=async(path:string)=>{
    try{if(IMAGE_EXTS.has(extname(path).toLowerCase()))out.add(await realpath(path));else failures.push(`Unsupported image: ${path}`);}
    catch(error){failures.push(`${path}: ${(error as Error).message}`);}
    if(out.size>128)throw new Error("Folder contains more than the supported 128 source images.");
  };
  for(const path of paths){
    let directory=false;
    try{directory=(await stat(path)).isDirectory();}catch(error){failures.push(`${path}: ${(error as Error).message}`);continue;}
    if(directory){for(const entry of (await readdir(path,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))){if((entry.isFile()||entry.isSymbolicLink())&&IMAGE_EXTS.has(extname(entry.name).toLowerCase()))await add(join(path,entry.name));}}
    else await add(path);
  }
  return{paths:[...out],failures:failures.slice(0,20)};
}

async function saveProject(ui: unknown, saveAs = false): Promise<WorkerReply> {
  if (!win || busy()) return fail("Wait for the active job before saving.");
  let path = projectFile;
  if (!path || saveAs) {
    const picked = await dialog.showSaveDialog(win, { title: "Save SlashStitch Project", defaultPath: path || "untitled.ssp", filters: [{ name: "SlashStitch Project", extensions: ["ssp"] }] });
    if (picked.canceled || !picked.filePath) return ok({ canceled: true });
    path = picked.filePath;
  }
  if (extname(path).toLowerCase() !== ".ssp") return fail("Project files must use the .ssp extension.");
  let document = recovery;
  if (engineAvailable) {
    const snapshot = await mutate("snapshot-project", { ui });
    if (!snapshot.ok) return snapshot;
    document = JSON.parse(snapshot.text);
  } else if (document) document = { ...document, ui };
  if (!document) return fail("No project snapshot is available to save.");
  protectSources(path, ((document.images ?? []) as Array<{ path: string }>).map(im => im.path));
  mkdirSync(dirname(path), { recursive: true });
  const diskDocument=structuredClone(document);
  // Resolve directory aliases too (/var -> /private/var on macOS, junctions on Windows).
  // Both sides must use the same canonical base for portable source-relative paths.
  const documentDir=realpathSync.native(dirname(resolve(path)));
  for(const image of (diskDocument.images??[]) as Array<{path:string}>) {
    const source=existsSync(image.path)?realpathSync.native(image.path):resolve(image.path);
    image.path=relative(documentDir,source)||source;
  }
  writeJsonAtomic(path, diskDocument);
  recovery = document; projectFile = path; dirty = false;
  rmSync(recoveryFile, { force: true }); savedRecovery = null;
  notifyProject(); return ok({ path, canceled: false });
}
const uiSnapshots=new Map<string,(ui:unknown)=>void>();
async function currentEditorSettings():Promise<unknown>{
  if(!win)throw new Error("Editor window is unavailable");
  const id=randomUUID();
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{uiSnapshots.delete(id);reject(new Error("Editor did not return current settings"));},3000);
    uiSnapshots.set(id,ui=>{clearTimeout(timer);resolve(ui);});
    send("request-ui-snapshot",{id});
  });
}
let closing: Promise<void> | null = null;
let discardOnQuit=false;
const gate = new ShutdownGate(busy, () => worker.stop(), error => dialog.showErrorBox("Could not close safely", error.message));
async function confirmDiscard(forQuit=false): Promise<boolean> {
  if (!win || !dirty || !((recovery?.images as unknown[] | undefined)?.length)) return true;
  const picked = await dialog.showMessageBox(win, { type: "question", title: "Unsaved project", message: "Save your panorama project before closing?", buttons: ["Save", "Discard", "Cancel"], defaultId: 0, cancelId: 2 });
  if (picked.response === 2) return false;
  if (picked.response === 0) {
    let ui:unknown;
    try {ui=await currentEditorSettings();}
    catch {
      const fallback=await dialog.showMessageBox(win,{type:"warning",message:"The editor is unavailable. Save the last recovery snapshot instead?",buttons:["Save recovery snapshot","Cancel"],defaultId:1,cancelId:1});
      if(fallback.response!==0)return false;ui=recovery?.ui;
    }
    const result=await saveProject(ui);if(!result.ok){dialog.showErrorBox("Project was not saved",result.text);return false;}
    return !JSON.parse(result.text).canceled;
  }
  if(forQuit)discardOnQuit=true;
  return true;
}
function quit(): void {
  if (closing) return;
  if (busy()) { send("close-blocked", { text: "A job is active. Stop it and wait for finalization before closing." }); return; }
  closing = (async () => { if (await confirmDiscard(true) && await gate.request()) app.quit(); })().catch(error=>dialog.showErrorBox("Could not close safely",(error as Error).message)).finally(() => { closing = null; });
}
function createWindow(): void {
  win = new BrowserWindow({ ...settings, minWidth: 1100, minHeight: 680, title: app.getName(), backgroundColor: "#16181d", autoHideMenuBar: true,
    icon: join(__dirname, "../renderer/icon-256.png"), webPreferences: { preload: join(__dirname, "../preload/index.js"), contextIsolation: true, sandbox: true, nodeIntegration: false, webSecurity: true } });
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", event => event.preventDefault());
  win.on("close", event => { if (!gate.allowed) { event.preventDefault(); quit(); } });
  win.on("resize", () => {
    if (!win || win.isMaximized()) return;
    const [width, height] = win.getSize(); settings = { ...settings, width, height };
    try { mkdirSync(userData, { recursive: true }); writeJsonAtomic(settingsFile, settings); } catch (error) { console.error("Settings save failed", error); }
  });
  win.on("closed", () => { win = null; });
  void win.loadFile(join(__dirname, "../renderer/index.html"));
}
app.on("second-instance", () => {
  if (!win) return;
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
});
app.whenReady().then(() => {
  protocol.handle(PREVIEW_SCHEME, async (request) => {
    try {
      const url = new URL(request.url);
      const rel = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
      const target = resolve(previewRoot, ...rel);
      const inside = relative(realpathSync.native(previewRoot), realpathSync.native(target));
      if (url.host !== "session" || !inside || inside.startsWith("..") || isAbsolute(inside) || extname(target).toLowerCase() !== ".png") return new Response("Not found", { status: 404 });
      return await net.fetch(pathToFileURL(target).toString());
    } catch { return new Response("Not found", { status: 404 }); }
  });
  worker.on("event", (name: string, data: unknown) => {
    if (name === "job-state" && data && typeof data === "object" && "state" in data) {
      const next = (data as { state: unknown }).state;
      if (["idle", "running", "finalizing"].includes(String(next))) jobState = String(next);
    }
    if (name === "project-changed") rememberDocument(data);
    if (name === "worker-exit" && !gate.stopping && !gate.allowed) {
      engineAvailable = false; jobState = "disconnected"; previewPath = null;
      try {for(const entry of readdirSync(previewRoot)) rmSync(join(previewRoot,entry),{recursive:true,force:true,maxRetries:3,retryDelay:50});}
      catch(error){console.error("Could not remove disconnected worker cache",error);}
      send("engine-disconnected", { text: "The engine stopped. Restart it to restore the last complete project snapshot." }); notifyProject();
    }
    if (name !== "exact-preview-ready" && name !== "seam-overlay-ready") rememberPreview(data);
    decoratePreview(data); send(name, data);
  });
  ipcMain.handle("host:call", async (event, action: unknown, payload: unknown): Promise<WorkerReply> => {
    if (!win || event.sender !== win.webContents || event.senderFrame !== win.webContents.mainFrame) return fail("Unknown sender");
    if (gate.stopping || gate.allowed) return fail("Application is closing");
    const p = payload && typeof payload === "object" ? payload as Record<string, unknown> : {};
    try {
      switch (action) {
        case "shell-info": return ok({ platform: process.platform, version: app.getVersion(), jobState, dirty, projectFile, engineAvailable, recoveryAvailable: Boolean(savedRecovery) });
        case "close-window": setTimeout(() => win?.close(), 50); return ok(null);
        case "quit-app": setTimeout(quit, 50); return ok(null);
        case "ui-snapshot": {
          const id=typeof p.id==="string"?p.id:"";const accept=uiSnapshots.get(id);
          if(!accept)return fail("Unknown or expired editor snapshot request");uiSnapshots.delete(id);accept(p.ui);return ok(null);
        }
        case "mark-ui-dirty":
          if (recovery) rememberDocument({ ...recovery, ui: payload });
          return ok(null);
        case "save-project": return await saveProject(p.ui, Boolean(p.saveAs));
        case "open-project": {
          if (busy() || !await confirmDiscard()) return ok({ canceled: true });
          const picked = await dialog.showOpenDialog(win, { title: "Open SlashStitch Project", properties: ["openFile"], filters: [{ name: "SlashStitch Project", extensions: ["ssp"] }] });
          if (picked.canceled || !picked.filePaths[0]) return ok({ canceled: true });
          const path = picked.filePaths[0]; if (statSync(path).size > 16 * 1024 * 1024) return fail("Project file exceeds the 16 MiB budget.");
          const document = JSON.parse(readFileSync(path, "utf8"));
          for (const image of document.images ?? []) if (typeof image.path === "string" && !isAbsolute(image.path)) image.path = resolve(dirname(path), image.path);
          const result = await mutate("restore-project", document);
          if (result.ok) { projectFile = path; dirty = false; savedRecovery = null; rmSync(recoveryFile, { force: true }); notifyProject(); }
          return result;
        }
        case "recover-project": {
          if(busy() || !await confirmDiscard()) return ok({canceled:true});
          if (!savedRecovery) return fail("No recovery snapshot is available.");
          const result = await mutate("restore-project", savedRecovery);
          if (result.ok) { savedRecovery = null; projectFile = null; notifyProject(); }
          return result;
        }
        case "restart-engine": case "restart-engine-cpu": {
          if (engineAvailable) return fail("The engine is already connected.");
          worker.restart(action === "restart-engine-cpu" ? {SLASHSTITCH_FORCE_CPU:"1"} : {}); engineAvailable = true; jobState = "idle";
          const result = recovery ? await mutate("restore-project", recovery) : await worker.request("engine-info");
          if (!result.ok) { send("worker-error", { text: result.text }); }
          notifyProject(); return result;
        }
        case "pick-directory": {
          const result = await dialog.showOpenDialog(win, { properties: ["openDirectory"] }); return ok({ canceled: result.canceled, path: result.filePaths[0] ?? null });
        }
        case "pick-images": {
          if (busy()) return fail("Wait for the active job before importing.");
          const result = await dialog.showOpenDialog(win, { title: "Add Source Images", properties: ["openFile", "multiSelections"], filters: [{ name: "Images and Camera RAW", extensions: [...RASTER_EXTS, ...RAW_EXTS, ...RAW_EXTS.map(ext => ext.toUpperCase())] }, { name: "Camera RAW / DNG", extensions: [...RAW_EXTS, ...RAW_EXTS.map(ext => ext.toUpperCase())] }, { name: "Images", extensions: RASTER_EXTS }] }); return ok({ canceled: result.canceled, paths: result.filePaths });
        }
        case "pick-export-file": {
          const result = await dialog.showSaveDialog(win, { title: "Export Panorama", defaultPath: "panorama.tif", filters: [{ name: "TIFF / BigTIFF", extensions: ["tif", "tiff"] }, { name: "JPEG", extensions: ["jpg", "jpeg"] }, { name: "PNG", extensions: ["png"] }] }); return ok({ canceled: result.canceled, path: result.filePath ?? null });
        }
        case "pick-icc-profile": {
          const result = await dialog.showOpenDialog(win, { title: "Select ICC Color Profile", properties: ["openFile"], filters: [{ name: "ICC Profiles", extensions: ["icc", "icm"] }] }); return ok({ canceled: result.canceled, path: result.filePaths[0] ?? null });
        }
        case "read-preview": {
          if (!previewPath || !existsSync(previewPath)) return ok({ dataUrl: null });
          if (statSync(previewPath).size > 32 * 1024 * 1024) return fail("Preview exceeds the file budget.");
          return ok({ dataUrl: `data:image/png;base64,${readFileSync(previewPath).toString("base64")}` });
        }
        case "load-images": {
          if (busy()) return fail("Wait for finalization before importing.");
          const expanded = await expandImagePaths(Array.isArray(p.paths) ? p.paths.filter((v): v is string => typeof v === "string") : []);
          if (!expanded.paths.length) return fail(expanded.failures.join("\n") || "No supported images found.");
          const result = await mutate("load-images", { paths: expanded.paths, replace: Boolean(p.replace) });
          if (result.ok && expanded.failures.length) { const data = JSON.parse(result.text); data.failures = [...(data.failures ?? []), ...expanded.failures.map(error => ({ error }))]; return ok(data); }
          return result;
        }
        case "export-panorama": {
          if (busy()) return fail("A job is already active.");
          const path = typeof p.output_path === "string" ? p.output_path : "";
          if (!path.trim()) return fail("Export path is required.");
          protectSources(path, ((recovery?.images ?? []) as Array<{ path: string }>).map(im => im.path));
          let overwrite = false;
          if (existsSync(path)) {
            const picked = await dialog.showMessageBox(win, { type: "question", message: `Replace ${path}?`, detail: "The previous file will be kept if rendering fails or is cancelled.", buttons: ["Replace", "Cancel"], defaultId: 1, cancelId: 1 });
            if (picked.response !== 0) return ok({ canceled: true }); overwrite = true;
          }
          return await mutate("export-panorama", { ...p, overwrite });
        }
        case "update-lensfun": {
          if (busy()) return fail("Wait for the active job before updating profiles.");
          pendingMutation = true; databaseAbort=new AbortController();jobState="running";send("job-state",{state:jobState});
          try {
            const snapshot = await downloadLensfunSnapshot(join(userData, "lensfun-snapshots"), async path => { const validation = await worker.request("validate-lensfun", { path }, 90_000); if (!validation.ok) throw new Error(validation.text); },fetch,databaseAbort.signal);
            databaseAbort.signal.throwIfAborted();
            const previous = workerEnv.SLASHSTITCH_LENSFUN_DIR;
            const loaded = await worker.request("reload-lensfun", { path: snapshot.path }, 90_000); if (!loaded.ok) throw new Error(loaded.text);
            try { mkdirSync(userData, { recursive: true }); writeJsonAtomic(lensfunPointer, snapshot); }
            catch (error) { await worker.request("reload-lensfun", previous ? { path: previous } : null); throw error; }
            workerEnv.SLASHSTITCH_LENSFUN_DIR = snapshot.path;
            return ok({ updatedFiles: snapshot.files, targetDir: snapshot.path, sha: snapshot.sha, engine: JSON.parse(loaded.text) });
          } finally { pendingMutation = false;databaseAbort=null;jobState="idle";send("job-state",{state:jobState}); }
        }
        case "render-exact": case "cancel-exact": case "render-seams": return worker.request(action, payload, 15_000);
        case "job-cancel": case "demo-stop": databaseAbort?.abort(new Error("Database update cancelled"));return worker.request("job-cancel");
        case "echo": case "engine-info": case "lensfun-search": case "job-status": return worker.request(action, payload, 15_000);
        case "inspect-images": case "clear-images": case "update-image-pose": case "restore-poses": case "restore-project": case "snapshot-project": case "reoptimize": case "set-view": case "render-preview": case "auto-crop": case "reapply-lensfun": case "auto-align": case "preview-start": case "demo-start": case "set-color-settings": case "set-render-settings":
          return await mutate(action, payload);
        default: return fail("Unknown action");
      }
    } catch (error) { return fail((error as Error).message); }
  });
  createWindow();
  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on("before-quit", event => { if (!gate.allowed) { event.preventDefault(); quit(); } });
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
app.on("will-quit", () => { if(discardOnQuit)rmSync(recoveryFile,{force:true}); rmSync(previewRoot, { recursive: true, force: true }); });
let gpuCrashes = 0;
app.on("child-process-gone", (_event, details) => {
  if (process.platform !== "linux" || details.type !== "GPU" || details.reason === "clean-exit") return;
  if (++gpuCrashes < 2 || app.commandLine.getSwitchValue("ozone-platform") !== "wayland") return;
  settings.displayBackend = "x11";
  try { mkdirSync(userData, { recursive: true }); writeJsonAtomic(settingsFile, settings); } catch (error) { console.error("Display fallback could not be saved", error); }
  send("worker-error", { text: "Repeated Wayland GPU failures: X11 has been selected for the next launch. Save the project before restarting." });
});
