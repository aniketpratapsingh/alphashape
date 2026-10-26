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
- **The orthomosaic's default cell is the ground sample distance the photographs actually resolve.** `BuildOrthomosaic` measures each camera's footprint on the DEM (median of slant range / (focal × cos incidence) over the enabled cameras) and uses that for the output grid instead of a fixed fraction of the DEM cell, so an oblique flight correctly comes out at about twice the nadir footprint. The measurement assumes level ground at the DEM median height: the incidence term is the vertical, not the local surface normal, so it does not follow a slope. A Geographic (EPSG:4326) grid gets the result converted to degrees at the surface's latitude.
- **`BuildDem`'s automatic cell applies the same incidence correction**, so an oblique flight's DEM is no longer over-sampled by 1/cos(incidence).
- Orthomosaic memory is bounded before allocation: per-camera tiles are cropped to their footprints, photographs are decoded one at a time, and a run whose estimated peak exceeds 60 % of physical RAM (`ALPHASHAPE_MEMORY_BUDGET_MB` overrides for tests) is refused with a message naming a coarser resolution.
- **On a low-memory machine the orthomosaic's default resolution is coarsened automatically.** With `resolution` left at 0, a run whose estimate does not fit the memory budget is built at the finest cell that does fit (three significant figures), and a Warning reports the measured ground sample distance, the estimate, the budget and the chosen cell. An explicitly set `resolution` is never changed: it is still refused with an actionable message.

### Changed

- **Project format version 3.** The depth-map sidecar moved from ADM1 to ADM2,
  so the project version is bumped: an older 0.1.0 build refuses such a project
  with its clear "newer than this build supports" message instead of failing
  deep inside the sidecar decoder with "depth map file: bad header". Version 2
  projects still open: their ADM1 maps read with fingerprint 0 ("unknown", so
  never reused) and the "mild" filter a pre-ADM2 project could only have run.
- **BuildPointCloud and unfiltered depth maps.** A map with no confidence (one
  a cancelled Build Depth Maps checkpointed) now logs a warning and is fused
  rather than silently dropped when `min_confidence > 0`: the absence of a
  measurement is not a failing measurement.
- The occlusion ray march steps at half a **DEM** cell rather than half the larger of the ortho and DEM cells. Its stop is now metric and exact — the ray ends when it rises above the tallest DEM sample — instead of a fixed 4096-sample cap, which could drop a genuine occluder on a long slope viewed at a grazing angle. Asking for an ortho 4× finer than the DEM no longer multiplies the march cost per unit area. Occluders thinner than about one DEM cell can still be missed.
- The orthomosaic memory estimate is now per stage (rectify, seams — including the alpha-expansion push-relabel graph — blend and finalize), reporting the largest stage rather than a sum of unrelated ones, and the guard runs **before** any grid-sized allocation, including the plane→local node map that used to be built first.
- `BuildDem`'s robust extent clipping now applies from 200 samples instead of 1000, so a small block is protected too, and it logs an Info line naming the axis, the pre-clip and clipped extents in metres, and how many far samples were ignored.
- `BuildDem` and `BuildOrthomosaic` no longer clamp an automatic resolution to a quarter of the DEM cell. The measured GSD stands, with only an eighth-of-a-cell sanity floor.

### Security

- **Windows: no DLL planting on the Vulkan loader.** A bare `vulkan-1.dll` is now loaded with `LOAD_LIBRARY_SEARCH_SYSTEM32`, so it only ever comes from `System32`. Plain `LoadLibraryW` searched the application directory and the current working directory first, so a planted copy beside the executable (or in whatever directory the app was started from) would have been loaded instead of the driver's loader. `ALPHASHAPE_VULKAN_LIBRARY` obeys the same rule: name an alternative loader by absolute path.

### Fixed

- **Build Orthomosaic no longer overflows its grid dimensions.** A resolution fine enough to need more than 2^31−1 output cells (or a non-finite one) is refused in double precision with the resolution, the extent, the cell count and "choose a coarser resolution", instead of casting an unbounded double to `int` and then sizing buffers from a negative length.
- **Build Orthomosaic reports an exhausted allocation honestly.** A `std::length_error` (a buffer whose size exceeds the maximum the process can represent) is treated the same as `std::bad_alloc` and converted to the actionable memory message, instead of escaping as an internal error.
- **Build Model no longer freezes a 16 GB M1.** Long GPU iteration loops (TV-L1 fusion, screened Poisson, PatchMatch) are now split into submissions sized to about 50–100 ms, and a fence timeout or device loss falls back to the CPU for the rest of the session instead of leaving the display starved.
- **The GPU PatchMatch shaders no longer rely on undefined behaviour.** Two
  latent defects in the Vulkan kernels were fixed. A never-taken branch in
  `keep_bindings_alive()` indexed `views[]`, `view_ints[]` and `offsets[]` by the
  lane's pixel, but those buffers hold 24×, 4× and 2× the neighbour/window counts
  — out of range for almost every image, and an out-of-range storage-buffer
  access is undefined behaviour in Vulkan even in a branch the optimiser may
  speculate over. It now reads element 0 of each. Search state was also threaded
  through aggregate `inout` parameters, which glslang lowers to a Function-storage
  pointer with a copy-back that inlining is least likely to preserve: the RNG is
  now a stateless keyed stream (draw *n* = `splitmix64^(n+1)` of a key of seed,
  pixel, iteration and stream, with a literal draw index at every call site), and
  `Best` is passed and returned by value. `neighbour_cost()` also gained the
  CPU's quorum early exit, so a neighbour that cannot reach 75% of the window
  stops after a few samples instead of walking all 49.
  **Speed is unchanged: both search kernels stay unoptimised.** `pm_init` and
  `pm_refine` are still built without the SPIR-V optimiser, which is what they
  were before, because an optimised module of either changes results on NVIDIA
  hardware: on a Tesla T4 the optimised `pm_init` agrees with the CPU on 100.00%
  of depths but 0.92% of normals, and optimised `pm_refine` on 93.4% of the dense
  fixture, where the unoptimised modules are at 100%. The fix that would restore
  that agreement — marking the whole chain from a candidate plane to the
  projected pixel coordinate `precise`, because `sample_image()` floors that
  coordinate and a fused multiply-add moves the sample a texel — makes `glslc`
  need more memory than a CI runner has: it is OOM-killed at `-O`, at `-Os`, and
  at `-O0`, so this toolchain cannot compile it at all. Two tests now hold the
  line for a future attempt: `PatchMatchGpu.RefineSweepsMatchCpu` gates the
  search kernel itself (≥99% against the CPU on hardware, which fails the 93.4%
  regression; the existing `iterations = 0` test never reaches it), and
  `TvFusionGpu.BuildModelMeetsCpuBounds` asserts the fused TV-L1 mesh's
  boundary-edge count is *equal* to the CPU's again — a discontinuous function of
  that agreement — instead of tolerating a difference. See `LIMITATIONS.md` for
  the measurements and what has been ruled out.

### Limitations

- A full-resolution orthomosaic of a very large site on a low-memory machine is not possible yet: the output grid is still held in one piece, so the automatic cell is coarsened (with a Warning) or an explicit cell is refused. Block-wise processing — building the ortho in overlapping memory-bounded blocks at full resolution — is planned. See `engine/LIMITATIONS.md`.

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

### Fixed

- The default orthomosaic resolution now follows the true ground sample distance instead of
  falling back to a 0.25 m cell. On a 36-photo oblique building flight (Sony NEX-5,
  16 mm, true GSD 1.9 cm) the automatic ortho was ~25 cm per pixel, 13x coarser than the
  imagery, because the measured footprint was dropped whenever the DEM's own cell was
  coarse (a decimated mesh's median edge, or a sparse point cloud's spacing) and the
  hard-coded 0.25 m literal took over. The rule now measures each camera's footprint on
  the surface at its image centre (slant range / focal / cos incidence), so an oblique
  flight is no longer under-estimated by up to 2x, and the cell falls back to the DEM's
  own cell rather than a metric literal.
- Automatic DEM and ortho cells are expressed in the output CRS's units. On an EPSG:4326
  (geographic) grid the depth-map ground sample distance and the TGV height scaling were
  applied as if the pixel size were metres, and the occlusion march mixed degree steps with
  metre heights; all three now convert through the frame's metres-per-degree at the model's
  latitude. An orthomosaic built in degrees also used to step the occlusion test by ~1e-6
  per iteration.
- BuildOrthomosaic's memory is now proportional to the imagery, not to cameras x grid. The
  ortho default following the ground sample distance made a 2 cm ortho of a 36-photo block
  28 million cells instead of ~0.2 million, and at that size the old layout needed ~12 GB:
  every camera's rectified tile was allocated on the whole output grid (8 bytes per cell)
  and all of them stayed resident through seamline optimisation and blending. Each tile is
  now cropped to that camera's own footprint box, photographs are decoded one at a time
  instead of all up front, the projection node map stores float offsets, and the hole-fill
  accumulator runs in bounded row bands. The result is byte-identical to before.
- BuildOrthomosaic refuses a run it cannot fit instead of dying with "bad allocation": it
  estimates the peak before allocating anything and raises a ValueError naming the memory
  needed, the machine's RAM and a coarser resolution to try. The budget is 60% of physical
  memory (override with `ALPHASHAPE_MEMORY_BUDGET_MB`); an allocation that still fails is
  reported the same way.
- The DEM's extent is robust against far samples. It used the raw min/max over every
  depth-map point, so a handful of far mismatches could turn a 100 m block into a
  kilometres-wide grid -- the actual cause of the "bad allocation" an 8 GB laptop hit at the
  old 0.25 m ortho default, before any ortho existed. An axis whose extent exceeds twice the
  0.5th-99.5th percentile core is now clipped to that core plus 3%; an ordinary,
  outlier-free sample set keeps exactly the grid it always had.

[0.1.0]: https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.1.0
