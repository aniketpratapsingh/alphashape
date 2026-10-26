# Changelog

All notable changes to alphashape are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Build Depth Maps keeps the cameras it finished when cancelled.** A cancel
  (including one during filtering) stores the finished maps before it stops,
  and `reuse_depth` (now on by default) skips their search on the next run and
  filters them. A cancel commits only depth maps: point clouds, models, DEMs,
  orthomosaics and earlier depth-map sets are left untouched. Repeated cancels
  merge into one set marked "partial" instead of piling up a set per cancel, and
  the run that finally completes replaces it in place.
- **Depth-map reuse is keyed on the whole neighbourhood.** Each map records a
  fingerprint of the reference camera's pose and calibration, the downscale,
  `max_neighbors`, the neighbour cameras' ids, poses and calibrations, the depth
  range, the reference image file and the PatchMatch settings, so moving one
  camera rebuilds every map that had it as a neighbour, not only its own map,
  and changing `max_neighbors` rebuilds every map. A map also records the filter
  mode it has been through, so changing `filter_mode` filters rather than
  returns the weaker old result.
- **Build Depth Maps reports progress inside one camera.** The search reports
  after every refinement sweep on both the CPU and the GPU path, so the
  progress bar moves during a long image and Cancel is honoured within one
  sweep instead of after the current camera. Each finished camera logs its
  index, resolution, device (GPU/CPU) and wall-clock time.

### Changed

- **Project format version 3.** The depth-map sidecar moved from ADM1 to ADM2,
  so the project version is bumped: an older 0.1.0 build refuses such a project
  with its clear "newer than this build supports" message instead of failing
  deep inside the sidecar decoder with "depth map file: bad header". Version 2
  projects still open, their ADM1 maps decoding as unfiltered-unknown (never
  reused, "mild" filter).
- **BuildPointCloud and unfiltered depth maps.** A map with no confidence (one
  a cancelled Build Depth Maps checkpointed) now logs a warning and is fused
  rather than silently dropped when `min_confidence > 0`: the absence of a
  measurement is not a failing measurement.

### Security

- **Windows: no DLL planting on the Vulkan loader.** A bare `vulkan-1.dll` is now loaded with `LOAD_LIBRARY_SEARCH_SYSTEM32`, so it only ever comes from `System32`. Plain `LoadLibraryW` searched the application directory and the current working directory first, so a planted copy beside the executable (or in whatever directory the app was started from) would have been loaded instead of the driver's loader. `ALPHASHAPE_VULKAN_LIBRARY` obeys the same rule: name an alternative loader by absolute path.

### Fixed

- **Build Model no longer freezes a 16 GB M1.** Long GPU iteration loops (TV-L1 fusion, screened Poisson, PatchMatch) are now split into submissions sized to about 50–100 ms, and a fence timeout or device loss falls back to the CPU for the rest of the session instead of leaving the display starved.

## [0.1.0] - Unreleased

The first release. Photos in, 3-D models and maps out, from a desktop app, the command
line or Python.

### Added

**Desktop application (Qt 6)**

- One window with a 3-D view, a workspace tree, a properties pane, a photo list and a
  log. Light and dark themes, saveable dock layout.
- Start screen with New project, Open..., Add photos..., a recent-project list, and
  drag-and-drop of a project file, photos or a folder of photos.
- Workflow menu in pipeline order (Add Photos, Match Photos, Align Cameras, Optimize
  Cameras, Build Depth Maps, Build Point Cloud, Build Model, Build UV, Build Texture,
  Build DEM, Build Orthomosaic). Each entry opens a parameter dialog generated from the
  task's own schema and greys itself out with a reason when the project is missing what
  the task needs.
- Long-running tasks run on a worker thread with a progress bar, an estimated time and a
  Cancel button; the window stays responsive and the project is locked while a task runs.
- 3-D view of camera positions, sparse and dense point clouds, and the textured mesh,
  with orbit, pan, zoom, fit-to-scene, perspective/orthographic toggle and a ground grid.
- File > Export for the model (OBJ with its material and texture page, or PLY), the point
  cloud, the DEM and the orthomosaic (GeoTIFF).
- Undo and redo for project edits, and unsaved-changes protection on New, Open and Close.

**Pipeline**

- Structure from motion: difference-of-Gaussians keypoints with binary M-LDB
  descriptors, exhaustive, sequential, GPS-reference and generic preselection, match
  ratio test, RANSAC, incremental alignment and Ceres bundle adjustment.
- Dense stereo: PatchMatch multi-view matching, depth-map filtering, and voxel fusion
  into a dense point cloud with normals, colours and confidence.
- Meshing: TV-L1 volumetric fusion or screened Poisson, with quadric decimation.
- Texturing: chart-based UV unwrapping, single-page atlas packing, and a baked texture
  page. `blending_mode` is `mosaic` (the default: each face takes its single best view,
  with optional gain correction between views), `average` (weighted across the views that
  see the texel), or `disabled` (the same best view with gain correction off). `BuildUV`
  clears the model's texture pages and `BuildTexture` replaces page 0, so a texture is
  rebuilt rather than kept beside an older one.
- Rasters: DEM and orthomosaic on a north-up grid in a PROJ coordinate system, exported
  as GeoTIFF.
- Multi-chunk projects: duplicate, merge, append and align chunks. Models, point clouds,
  DEMs and orthomosaics are labelled assets, so building one again adds a new asset and
  keeps the previous one. Textures are the exception: there is one texture set per model,
  replaced in place by the runs above.
- GPU acceleration through Vulkan compute (PatchMatch and the Poisson solve) where a
  suitable device and driver are present, with an automatic CPU fallback.

**Front ends and formats**

- `alphashape-cli` for scripted runs: `--list-tasks`, `--list-params`, `--gpu-info`,
  `--proj-info`, `--new`, and `--project ... --task ... --set key=value`.
- A Python module (`import alphashape`) with the same tasks, parameters and error types,
  so a pipeline script ports between the two front ends unchanged.
- Projects are a `.psx` file plus a `.files/` folder with a versioned format; the reader
  refuses project files from a newer format version instead of guessing.
- A sample dataset of twelve synthetic aerial photographs with true camera poses, released
  under CC0, and a generator script that regenerates it.

**Packaging**

- macOS (Apple silicon) `.dmg`, Ubuntu 24.04 `.deb` and AppImage, and a Windows
  x86-64 `.exe` installer, with `SHA256SUMS` beside every release asset.
- The command line and the Python module ship inside the macOS app and the Windows
  installer; the AppImage runs the command line through `alphashape-*.AppImage cli`.

**Windows GPU acceleration**

- The Windows build now compiles the Vulkan GPU layer in, like the Linux and macOS
  builds. The Vulkan headers and the shader tools (`glslc`, `spirv-val`) come from
  vcpkg, and the configuration fails loudly rather than quietly dropping back to a
  CPU-only build.
- Nothing links a Vulkan loader on any platform: the backend opens
  `vulkan-1.dll` / `libvulkan.so.1` / `libvulkan.1.dylib` at run time. On a Windows
  machine without a Vulkan driver (a VM, a server, a driver that predates Vulkan)
  the application starts normally, reports why GPU acceleration is unavailable,
  and runs every task on the CPU. The installer deliberately does not bundle a
  loader; a private `vulkan-1.dll` would shadow the driver's own loader and could
  not see the driver's ICDs.
- The Windows test lanes run the GPU suites against Mesa's lavapipe software
  driver, the same arrangement Linux uses with llvmpipe. Correctness assertions
  run; the timing and speed-up assertions skip themselves on a software device,
  and the TV-L1 fusion port keeps declining software devices (its fp16 kernel
  cannot be compiled by the gallium LLVM backend).

### Known limitations

See [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md). The short version for this release:
the installers are unsigned, the macOS build is Apple silicon only, Windows GPU
acceleration needs a Vulkan-capable graphics driver (the app falls back to the CPU
without one), and there is no georeferencing, marker editing or tiled-model output yet.

[0.1.0]: https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.1.0
