# Changelog

All notable changes to alphashape are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The orthomosaic's default cell is the ground sample distance the photographs actually resolve.** `BuildOrthomosaic` measures each camera's footprint on the DEM (median of slant range / (focal × cos incidence) over the enabled cameras) and uses that for the output grid instead of a fixed fraction of the DEM cell, so an oblique flight correctly comes out at about twice the nadir footprint. The measurement assumes level ground at the DEM median height: the incidence term is the vertical, not the local surface normal, so it does not follow a slope. A Geographic (EPSG:4326) grid gets the result converted to degrees at the surface's latitude.
- **`BuildDem`'s automatic cell applies the same incidence correction**, so an oblique flight's DEM is no longer over-sampled by 1/cos(incidence).
- Orthomosaic memory is bounded before allocation: per-camera tiles are cropped to their footprints, photographs are decoded one at a time, and a run whose estimated peak exceeds 60 % of physical RAM (`ALPHASHAPE_MEMORY_BUDGET_MB` overrides for tests) is refused with a message naming a coarser resolution.

### Changed

- The occlusion ray march steps at half a **DEM** cell rather than half the larger of the ortho and DEM cells, and each ray is capped at 4096 samples. Asking for an ortho 4× finer than the DEM no longer multiplies the march cost per unit area by 16. Occluders thinner than about one DEM cell can still be missed.
- `BuildDem` and `BuildOrthomosaic` no longer clamp an automatic resolution to a quarter of the DEM cell. The measured GSD stands, with only an eighth-of-a-cell sanity floor.

### Fixed

- **Build Orthomosaic no longer overflows its grid dimensions.** A resolution fine enough to need more than 2^31−1 output cells (or a non-finite one) is refused in double precision with the resolution, the extent, the cell count and "choose a coarser resolution", instead of casting an unbounded double to `int` and then sizing buffers from a negative length.
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

### Known limitations

See [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md). The short version for this release:
the installers are unsigned, the macOS build is Apple silicon only, GPU acceleration is
off in the Windows build, and there is no georeferencing, marker editing or tiled-model
output yet.

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
