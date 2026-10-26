# Changelog

All notable changes to alphashape are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Build Model no longer freezes a 16 GB M1.** Long GPU iteration loops (TV-L1 fusion, screened Poisson, PatchMatch) are now split into submissions sized to about 50–100 ms, and a fence timeout or device loss falls back to the CPU for the rest of the session instead of leaving the display starved.
- **Depth maps are several times faster.** The two PatchMatch search kernels were compiled without the SPIR-V optimiser as a workaround for a wrong random-plane normal on NVIDIA hardware. The real cause was undefined behaviour in the shaders, not the driver: the random-number generator carried its state in an aggregate `inout` parameter (the copy-back that the optimiser did not preserve), a never-taken branch indexed three buffers far out of range, and the `a + b * u` expressions whose rounding the match depends on were free to be contracted into fused multiply-adds. The generator is now a stateless keyed stream, the dead-branch index is in range, and those few scalar expressions are marked `precise`. Every GPU kernel is built optimised again on every backend.

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

### Known limitations

See [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md). The short version for this release:
the installers are unsigned, the macOS build is Apple silicon only, GPU acceleration is
off in the Windows build, and there is no georeferencing, marker editing or tiled-model
output yet.

[0.1.0]: https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.1.0
