# SlashStitch

An open-source panorama workbench using Electron, React and a native Rust worker. The application includes feature matching, spherical rotation RANSAC, robust bundle adjustment, interactive projections and cropping, GPU preview with CPU fallback, and atomic panorama export.

## Working with panoramas

Add JPEG, PNG, TIFF, WebP or camera RAW/DNG images with **Add Images** or drag and drop. Import failures are listed explicitly. EXIF orientation is normalized for preview, matching and export. Auto Align requires a connected set of overlapping images; failed alignment preserves the previous geometry. Automatic leveling fits the camera sweep: review the horizon before exporting.

Use **Open**, **Save** and **Save As** for versioned `.ssp` projects. Projects include sources, geometry, control points, projection, crop and export settings; source images remain separate. Relative paths allow moving a project together with its source folders. Missing or changed sources produce an error without partially replacing the open project. Closing unsaved work offers Save, Discard or Cancel. A recovery snapshot is retained after unexpected worker/application failure, and a disconnected engine can be restarted explicitly.

Shortcuts: Ctrl/Cmd+O adds images, Ctrl/Cmd+Shift+O opens a project, Ctrl/Cmd+S saves, Ctrl/Cmd+Shift+S saves as, Ctrl/Cmd+Z/Shift+Z undo/redo, Ctrl/Cmd+Shift+A aligns, and Ctrl/Cmd+E opens Export. Source selection, panel splitters and crop handles support keyboard controls. Use arrow keys on a focused crop handle; Shift makes smaller adjustments.

### Preview and blending

- The viewport first shows a fast GPU preview (feathered blend of preview-size sources) while you interact. When interaction pauses, it is replaced by an **exact preview** rendered by the same engine as the export: exposure/vignetting correction, multi-band blending and the same colour pipeline. When you zoom in past the preview resolution, the visible region is rendered 1:1 from full-resolution sources. The status line states which preview is shown.
- **Exposure, white balance & vignetting compensation** (Export tab, on by default) estimates per-image, per-channel gains and a radial vignetting model per lens from the overlaps, and applies them identically in preview and export. The parameters are saved in the project and re-estimated when the geometry changes.
- Multi-band blending chooses its number of pyramid levels from the output's angular resolution, so preview and export blend over the same angular width. Seams follow the highest edge-distance weight; each image's pyramid is extrapolated beyond its coverage so no black borders bleed into the blend. Strip-wise export is verified to match a single-band render.
- Exact previews and exports resample with the Panorama Tools **Spline36** kernel (6×6, interpolating, minimal ringing around stars); the fast interactive preview uses bilinear sampling.
- Deep multi-band pyramids (high-resolution exports) blend their coarse levels once for the whole canvas (anti-aliased, 4× oversampled) and only the fine levels per strip, keeping strip padding small.
- Vignetting correction follows the measured curve only up to the largest radius where overlaps provide evidence and is held constant beyond it (no extrapolation into the extreme corners).
- **Seams** (viewport toolbar) overlays a coloured frame around the area each source covers (thicker for the selected source; the source list shows the same colours) and white lines where the blend switches source. It uses the same seam labels as the export. Seams follow the source that sees each direction closest to its optical centre (sharpest, least vignetted), faded out near frame borders.
- **Stitch mode** (Export tab): presets over the advanced settings, saved with the project — *Balanced* (multi-band 3°, Spline36, optimised seams, exposure + vignetting, transient removal), *Night sky & aurora* (wide 6° transitions), *Sharp detail* (narrow 1.5°, Lanczos3) and *Fast draft* (feather, bilinear, centre-line seams, gains only). **Advanced stitching** exposes blending (multi-band/feather), transition width (1.5–10°), interpolation (Spline36, Lanczos3, bilinear) and vignetting correction; any change shows the mode as *Custom*.
- The projection menu shows a small shape icon per projection (transverse variants rotated).
- **Optimise seams** (Export tab, on by default): instead of a fixed line between image centres, each seam between two neighbouring frames is routed through the part of their overlap where they agree best (dynamic programming over a ±8° corridor on locally averaged, exposure-corrected colour differences, with a small pull towards the centre line and a small price per sideways step). The layout is computed once per geometry for the whole canvas and shared by the exact preview, zoom, the Seams overlay and the export.
- **Remove transient light** (Export tab, on by default): where three or more frames overlap and one of them is markedly brighter *and* shows higher relative contrast than the others (a torch hitting a rock, a passing car), that area is taken from the best-covering frame that agrees with the majority. With only two overlapping frames nothing is replaced.
- Each sample picks its source resolution from the local footprint (trilinear between pyramid levels), so downscaled exports average noise instead of aliasing it; at 1:1 or larger the full-resolution source is used.

### Image and resource contract

- Sources up to 16 bits per channel keep their precision through blending. Output can be **8, 10 or 16 bits per channel** (10-bit is stored in RGB16 with 1024 levels; JPEG supports 8-bit only) in sRGB, Adobe RGB, Display P3, ProPhoto RGB, Rec.2020 or a custom/source RGB ICC profile. Pixels are converted to the target profile and that profile is embedded. A custom or adopted ICC profile is embedded byte-for-byte; it is only replaced by a built-in profile when primaries, white point and tone curves are colorimetrically identical. LUT-only ICC profiles without RGB tone curves are rejected as targets. 8-bit sources exported at 10/16 bits do not regain lost precision. Non-RGB ICC sources and transparent sources must be converted/flattened first.
- A new project adopts the sources' colour space when every source embeds the same RGB profile (for example Adobe RGB), so wide-gamut colour is not clipped to sRGB by default; mixed or missing profiles keep sRGB. The status line reports the adoption and the Export tab can change it.
- The canvas aspect ratio is part of the project, so resizing the window never changes exported geometry or a saved crop. Equirectangular projections use a fixed 2:1 (transverse 1:2) canvas.
- Full-resolution sources are decoded in parallel within a memory budget (half of the available memory, at most 32 GiB; `SLASHSTITCH_DECODE_BUDGET_MB` overrides; the process-wide memory ceiling below still applies), colour-converted once and kept as a disk-backed resolution pyramid under the application data directory for the session. Exact previews and every export reuse it. Plan disk space of roughly 8 bytes per source pixel. A source changed on disk after loading is detected (size and modification time) and must be reloaded; cached data is never mixed with a changed file.
- Export sizes are checked before any source is decoded or any large buffer is allocated. The band memory budget adapts to the machine (at most 1.5 GiB, and at most 40 % of available memory and of the process address-space limit; hard ceiling 80 %), so a constrained machine refuses an impossible export with a clear message instead of aborting. The global coarse blending layer of deep exports is part of that budget and can be cancelled.
- Camera RAW/DNG files (ARW, CR2, CR3, CRW, NEF, NRW, RAF, ORF, RW2, PEF, SRW, DNG, 3FR, IIQ, ERF, MEF, MOS, MRW, KDC, DCR, X3F and more, via the rawler decoder) are developed in floating point: black/white levels, white balance on the sensor mosaic with a common clip point (blown highlights stay neutral), RCD demosaicing for Bayer sensors (the RawTherapee/darktable default; X-Trans uses the decoder's bilinear method), the camera colour matrix — interpolated between the file's two calibration illuminants for the white point, as the DNG specification describes — to linear Rec.2020 (no gamut clipping), the DNG baseline exposure, and a smooth highlight shoulder when exposure is raised. The result is a 16-bit linear Rec.2020 source with its own ICC profile, converted to the project profile like any other source. A RAW-only project keeps sRGB as its export default (choose Adobe RGB, ProPhoto RGB or Rec.2020 in Export to keep the wider gamut). The development is neutral: there is no camera "look", tone curve or noise reduction.
- Lightroom/Adobe Camera Raw settings are read from an XMP sidecar (`name.xmp` or `name.ext.xmp`) or from XMP embedded in a DNG. White balance (temperature/tint, converted with the DNG colour-temperature method and dual-illuminant matrices; an approximation where a file has only one colour matrix), **Exposure** and the orientation (including rotations made in Lightroom) are applied. Other non-default settings — tone and presence sliders, tone curves, HSL, colour grading, camera profiles, crops, lens vignetting and local adjustments — are listed in the image info instead of being approximated. Editing the sidecar marks the source as changed; reload it to use the new settings.
- A DNG with an enhanced image (NewSubFileType 16, e.g. Lightroom Denoise, usually JPEG XL compressed) uses that image when it can be decoded. Some Lightroom JPEG XL streams are not yet supported by the decoder; the original sensor data is then developed and the image info shows a note.
- Exposure brackets are detected from EXIF (same camera/lens/focal length, clearly different exposures, fired as one burst or marked "Auto bracket") and shown as **HDR Stack** in the image info. Merging stacks into an HDR panorama is not implemented yet; bracketed frames are aligned and blended as individual sources, with exposure matching.
- Upper limits keep large projects from taking the machine down. All large working buffers (decoding, render bands, the coarse layer, seam layouts, exact previews) draw from one process-wide memory ceiling: 90 % of physical memory by default (`SLASHSTITCH_MEMORY_LIMIT_MB` overrides). Each reservation also checks the live free memory and waits while granting it would leave less than 5 % of physical memory (at least 1 GiB) for the OS and other programs; work that does not fit waits instead of pushing the system into swap. The source cache may use up to 90 % of the usable disk space on its filesystem, always leaving at least 2 GiB free (`SLASHSTITCH_CACHE_LIMIT_GB` overrides); least recently used pyramids that no render holds are evicted first, and a project that cannot fit is refused with a clear message. Limits only queue or refuse work — they never change the rendered result.
- One SlashStitch instance runs per data profile; starting it again focuses the running window (the session's source cache and recovery file are never shared or removed under a running instance).
- TIFF/BigTIFF output uses bounded horizontal strips sized to a band memory budget. PNG/JPEG additionally require a complete output buffer, limited to 512 MiB.
- At most 128 sources and 20,000 retained control points per project. Source dimensions have a decode budget; export dimensions are checked before integer conversion and allocation. Oversized GPU storage buffers use the CPU preview instead of crashing.
- Export protects source file identities, including links. An existing destination requires confirmation. Rendering, cancellation and encoding errors preserve the previous output; publication uses an atomic replacement. Keep Crop Aspect is enabled by default; preview geometry is retained even when output dimensions differ.
- Lensfun geometric corrections support PTLens, Poly3 and Poly5 with focal interpolation and calibration crop scaling. Supported source projections are rectilinear, equidistant/circular fisheye, equisolid and equirectangular. Unsupported source profile projections are reported rather than silently interpreted as rectilinear.

Lensfun discovery reads installed XML profiles and a supplemental built-in profile. **Update DB** downloads an immutable upstream commit, checks Git blob identities and validates all XML before activating an app-private snapshot. Existing user/system Lensfun directories are never overwritten. Existing project geometry stays fixed until **Reapply profiles** is chosen; rerun Auto Align afterwards. Downloading can be stopped.

## Build and verify

Requires Node.js 22+, npm, Rust/Cargo and a supported native OS toolchain. Linux **release** builds require `cargo-zigbuild` and Zig; they target GLIBC 2.28 and check the resulting ELF version requirements. A debug build is available for local development without Zig.

```sh
npm ci
npm run typecheck
npm test
npm run build
npm start
```

One way to prepare isolated Linux build tools in the checkout:

```sh
cargo install cargo-zigbuild --version 0.24.1 --locked --root "$PWD/.build-tools"
python3 -m venv .build-tools/python
.build-tools/python/bin/pip install ziglang==0.17.1
```

The build script discovers these local tools automatically; alternatively put `cargo-zigbuild` and `zig` on PATH. `objdump` is required for the Linux ABI check. On Linux with Weston installed, run `npm run smoke` and `npm run smoke:regressions` for isolated headless UI tests. They use a private display/profile and disable Chromium's sandbox only for the throwaway test process. The normal app keeps renderer sandboxing enabled.

```sh
npm run build:engine:debug
SLASHSTITCH_DEBUG_WORKER=1 npm start
```

`npm test` includes Rust geometry/GPU tests, the worker bridge, real worker integration, project persistence/recovery and atomic file/database-update tests. GPU parity covers all panorama projections, source lens types and portrait/landscape aspects when an adapter is available. GPU tests serialize device initialization for driver stability. A GPU-less runner cannot qualify GPU execution. `npm run verify:images` optionally validates the local sample-image folders; those images are not distributed.

## Package

```sh
npm run pack
npm run dist:linux
npm run dist:win
npm run dist:mac
```

Build each installer on its corresponding OS; the scripts reject cross-packaging a worker for the wrong OS. Only verified release workers can be staged. Packaging includes the application license, Rust dependency license texts, and Electron/Chromium notices. Linux workbench execution and an unpacked Linux package have been exercised locally. Windows/macOS CI is prepared and the engine passes strict clippy for the Windows (x86_64-pc-windows-gnu) and macOS (aarch64-apple-darwin) targets, with native memory and disk-space queries on both; native runtime verification, installers and signing still require those platforms. A GLIBC ABI check alone does not qualify every Linux distribution or GPU driver.

Source entry points are `src/main/`, `src/preload/`, `src/renderer/` and the three crates under `engine/` (`slashstitch-core/src/render.rs` is the shared preview/export renderer, `photometric.rs` the exposure/vignetting model). The worker uses JSON-lines requests and events over stdio. Project mutations run behind one worker job gate; cancellation is cooperative and finalization completes before synchronous replies. Control timeouts do not cancel backend jobs, and every completed commit publishes the full project state so the editor stays in sync even after a timeout. Exact previews run in a separate lane where a newer request cancels an older one. Preview images are served to the renderer through a private `slashstitch-preview:` protocol restricted to the session directory.
