# alphashape

**An open-source photogrammetry engine: photos in, 3D models and maps out.**

[![CI](https://github.com/aniketpratapsingh/alphashape/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aniketpratapsingh/alphashape/actions/workflows/ci.yml)
[![Nightly](https://github.com/aniketpratapsingh/alphashape/actions/workflows/nightly.yml/badge.svg)](https://github.com/aniketpratapsingh/alphashape/actions/workflows/nightly.yml)
[![CodeQL](https://github.com/aniketpratapsingh/alphashape/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/aniketpratapsingh/alphashape/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/aniketpratapsingh/alphashape/badge)](https://securityscorecards.dev/viewer/?uri=github.com/aniketpratapsingh/alphashape)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15791/badge)](https://www.bestpractices.dev/projects/15791)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)

alphashape turns overlapping photographs into camera poses, dense point clouds, textured
meshes, digital elevation models (DEMs) and orthomosaics. It has a desktop application, a
command-line tool and a Python module, all driving the same processing tasks, and it uses
Vulkan compute on GPUs that support it. The engine is C++20; the desktop application is
Qt 6.

> **Status: early development (0.1.x).** The whole pipeline runs end to end, but formats,
> APIs and results still change between versions. Not yet suitable for production
> surveys. Known gaps are listed in [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md).
>
> **Coordinate systems (M12): done.** CRS definitions from EPSG, WKT or a PROJ
> string, datum transformations with offline grids, geoids and vertical datums,
> local frames, GeoTIFF GeoKeys, and raster reprojection are implemented. See
> [`engine/docs/crs-guide.md`](engine/docs/crs-guide.md).

## What it does

| Stage | Task | Notes |
|---|---|---|
| Import | `AddPhotos` | JPEG/PNG/TIFF, EXIF (focal length, GPS) |
| Features | `DetectFeatures` | Difference-of-Gaussians detector, binary M-LDB descriptors |
| Matching | `MatchPhotos` | exhaustive / sequential / GPS-reference / generic preselection, ratio test, RANSAC |
| Alignment | `AlignCameras`, `OptimizeCameras` | incremental structure-from-motion, bundle adjustment (Ceres) |
| Dense | `BuildDepthMaps`, `BuildPointCloud` | PatchMatch multi-view stereo, depth fusion and filtering (GPU optional) |
| Mesh | `BuildModel` | TV-L1 volumetric fusion or screened Poisson, decimation (GPU optional) |
| Texture | `BuildUV`, `BuildTexture` | chart UV layout, atlas packing, mosaic/average blending |
| Raster | `BuildDem`, `BuildOrthomosaic`, `ExportDem`, `ExportOrthomosaic` | GeoTIFF output, PROJ coordinate systems |
| Export | `ExportModel`, `ExportPointCloud` | OBJ (with its texture) or PLY |
| Chunks | `DuplicateChunk`, `MergeChunks`, `AppendProject`, `AlignChunks` | multi-chunk projects |

Projects are saved as a `.psx` file plus a `.files/` folder. Every task declares its
parameters once and the CLI, the Python module, the desktop dialogs and the project file
all bind against that one schema, so the four cannot drift apart. Edits that change the
project are undoable.

Task and parameter names follow the naming of Agisoft Metashape's public Python API, so
that existing processing scripts are easy to port. alphashape is an independent project.
It is not affiliated with, endorsed by or derived from Agisoft or Metashape.

## Install

Download the package for your platform from the
[latest release](https://github.com/aniketpratapsingh/alphashape/releases/latest):

| Platform | File | What it installs |
|---|---|---|
| macOS (Apple silicon) | `alphashape-<version>-macos-arm64.dmg` | `alphashape.app` with the GUI, `alphashape-cli` and the Python module inside it |
| Ubuntu 24.04 / x86-64 | `alphashape-<version>-linux-amd64.deb` | GUI, CLI and the Python module for the system `python3` |
| Linux, any distribution | `alphashape-<version>-linux-x86_64.AppImage` | GUI and CLI with every non-glibc dependency bundled (no Python module) |
| Windows 10/11 / x86-64 | `alphashape-<version>-windows-x86_64.exe` | GUI, CLI and the Python module |

Every release publishes `SHA256SUMS` next to the packages. It lists every platform's
assets, so check the file you downloaded rather than the whole list:

```sh
# Linux
sha256sum --ignore-missing -c SHA256SUMS

# macOS (no --ignore-missing in shasum; check the one line)
grep 'macos-arm64' SHA256SUMS | shasum -a 256 -c -
```

```powershell
# Windows: compare the hash of the downloaded file with its SHA256SUMS line
$file = Get-ChildItem .\alphashape-*-windows-x86_64.exe
$actual = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
$expected = (Select-String -Path .\SHA256SUMS -Pattern 'windows-x86_64\.exe').Line.Split()[0]
if ($actual -eq $expected) { "OK" } else { "MISMATCH" }
```

**The builds are unsigned, so both operating systems warn you.**

- **macOS.** The app is ad-hoc signed and not notarized. macOS 15 removed the
  right-click **Open** shortcut, so approve it in **System Settings → Privacy & Security →
  Open Anyway** after the first blocked launch. Or clear the quarantine flag once:

  ```sh
  xattr -dr com.apple.quarantine /Applications/alphashape.app
  ```

- **Windows.** Defender SmartScreen shows "Windows protected your PC". Choose **More
  info**, check that the app is **alphashape**, then **Run anyway** — see
  [why this happens and how to verify the download instead](packaging/windows/README-smartscreen.md).

The macOS app carries the command line and the Python module
([details](packaging/macos/README.md)); on Linux and Windows they are installed alongside
the GUI. Where each one ends up:

- **Linux .deb.** `alphashape-cli` and `alphashape-gui` go to `/usr/bin`, so both are on
  `PATH`, and the Python module goes to `/usr/lib/python3/dist-packages`.
- **AppImage.** The command line is reached through the AppImage itself:
  `./alphashape-*.AppImage cli ...` (see below).
- **macOS.** The binaries live inside the bundle:
  `/Applications/alphashape.app/Contents/MacOS/alphashape-cli` (and `alphashape-gui`).
  Add that directory to `PATH` or call the full path.
- **Windows.** The installer defaults to a per-user install in
  `%LOCALAPPDATA%\Programs\alphashape`, with **bin** *not* added to `PATH` unless you tick
  the optional **Add to PATH** entry on the components page. Without it, call
  `%LOCALAPPDATA%\Programs\alphashape\bin\alphashape-cli.exe`, or the same path under
  `%ProgramFiles%\alphashape` for a per-machine install. The Python module needs
  `PYTHONPATH=%LOCALAPPDATA%\Programs\alphashape\python` (`%ProgramFiles%\alphashape\python`
  for a per-machine install).

## Quick start: the desktop app

1. **Launch alphashape.** The start screen offers **New project**, **Open...** and
   **Add photos...**, and lists your recent projects. You can also drag photographs (or a
   folder of them) onto it.
2. **Add photos.** Choose the images of one site. Every part of the subject should appear
   in several photographs; 60–80% overlap between neighbours is a good starting point.
3. **Run the Workflow menu top to bottom.** Each entry asks for its parameters in a dialog
   built from the task's own schema, then runs the task on a worker thread with progress,
   an estimated time and a Cancel button:

   | Menu entry | What it produces |
   |---|---|
   | Add Photos | imports the photographs (the start screen's **Add photos...** does the same) |
   | Match Photos | keypoints, matches and tracks |
   | Align Cameras | camera positions and orientations, and the sparse point cloud |
   | Optimize Cameras | refines the alignment; optional, re-runnable |
   | Build Depth Maps | a depth map per photograph |
   | Build Point Cloud | the dense point cloud |
   | Build Model | a triangle mesh |
   | Build UV | texture coordinates on the mesh |
   | Build Texture | a baked texture page |
   | Build DEM | an elevation model over the site |
   | Build Orthomosaic | a georeferenced mosaic of the photographs |

   Align Cameras needs the matches from Match Photos, so run them in this order. While a
   stage's input is missing, its entry is greyed out with the reason ("Build depth maps
   first").
4. **Inspect the result** in the 3-D view: drag to orbit, drag with the right button to
   pan, scroll to zoom, **F** to frame the whole scene and **5** to switch between
   perspective and orthographic.
5. **Export** from **File > Export**: the model as OBJ (with its material and texture
   page) or PLY, the point cloud as PLY, and the DEM and orthomosaic as GeoTIFF. The DEM
   and orthomosaic are *built* from the Workflow menu and *exported* from this menu.
6. **Save** the project with **File > Save** (`.psx` plus a `.files/` folder). Reopen it
   later and every product is still there.

A sample site to practise on is produced by the
[`sample` workflow](.github/workflows/sample.yml) and published as a downloadable Actions
artifact named `alphashape-sample` holding `alphashape-sample-<version>.zip`: twelve
synthetic aerial photographs with GPS, plus a manifest of the true camera poses. Download
it from the [Actions page](https://github.com/aniketpratapsingh/alphashape/actions/workflows/sample.yml)
(the latest run's **Artifacts** section), or with the GitHub CLI:

```sh
gh run download --repo aniketpratapsingh/alphashape \
  --workflow sample.yml --name alphashape-sample
```

Attaching the zip to the release itself is planned but is not done yet. See
[`tools/sample/SOURCE.md`](tools/sample/SOURCE.md) for how it is made and its licence.

## Quick start: the command line

`alphashape-cli` runs one task per invocation and saves the project after each, so a
script can stop between stages and resume. It is on `PATH` for the `.deb`; on macOS,
Windows and the AppImage, substitute the full path from the list above (for example
`/Applications/alphashape.app/Contents/MacOS/alphashape-cli`, or
`.\alphashape-<version>.AppImage cli`).

```sh
alphashape-cli --list-tasks                      # every task this build ships
alphashape-cli --list-params BuildDepthMaps      # one task's parameters, types and defaults

# Create a project from photographs, then run the pipeline task by task.
alphashape-cli --new site.psx --add-images photos/*.jpg
alphashape-cli --project site.psx --task MatchPhotos --set keypoint_limit_per_mpx=10000
alphashape-cli --project site.psx --task AlignCameras
alphashape-cli --project site.psx --task BuildDepthMaps --set downscale=4
alphashape-cli --project site.psx --task BuildPointCloud
alphashape-cli --project site.psx --task BuildModel --set face_count=medium
alphashape-cli --project site.psx --task BuildUV --set texture_size=4096
alphashape-cli --project site.psx --task BuildTexture
alphashape-cli --project site.psx --task ExportModel --set path=site.obj --set format=obj
```

`--set key=value` is parsed against the parameter's declared type, so a mistyped name or
an out-of-range value fails before anything is computed. Progress goes to stderr. Exit
codes: `0` success, `1` engine error, `2` unknown task, `64` bad usage.

Two diagnostics help with setup: `--gpu-info` lists the Vulkan devices the engine can see,
and `--proj-info` names the PROJ database it resolved.

The AppImage runs the command line when `cli` is its first argument:

```sh
./alphashape-*.AppImage cli --list-tasks
```

## Quick start: Python

The module takes the same tasks and parameter names as the CLI:

```python
import alphashape

doc = alphashape.Document()
chunk = doc.add_chunk("Site")
doc.active_chunk = chunk

doc.run_task("AddPhotos", filenames=["photos/IMG_0001.jpg", "photos/IMG_0002.jpg"])
doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
doc.run_task("AlignCameras")
doc.run_task("BuildDepthMaps", downscale=4)
doc.run_task("BuildPointCloud")
doc.run_task("BuildModel", face_count="medium")
doc.run_task("BuildUV")
doc.run_task("BuildTexture")

print(chunk.aligned_camera_count, "of", chunk.camera_count, "cameras aligned")
doc.save("site.psx")

# Later:
doc = alphashape.load_project("site.psx")
print(alphashape.list_tasks())
```

Error types follow the failure: an unknown keyword raises `InvalidArgumentError`, an
argument of the wrong Python type raises `TypeError`, and an out-of-range or unaccepted
value raises `ValueError`, so a mistake surfaces at the call rather than in the middle of
a run.

On macOS the module lives inside the app bundle:

```sh
PYTHONPATH=/Applications/alphashape.app/Contents/Resources/python python3 -c "import alphashape"
```

A compiled module is loadable only by the CPython minor version it was built against. The
packages are built against 3.12 (Ubuntu 24.04's `python3` for the `.deb`, and the pinned
3.12 interpreter in the Windows build). The macOS module is built against whichever
`python3` the build machine had; the module's file name names it, for example
`alphashape.cpython-313-darwin.so`. [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md) has
the per-platform detail.

## Building from source

Requirements: CMake ≥ 3.24, a C++20 compiler (Clang, GCC 13+, or clang-cl on Windows),
Eigen 3, Ceres Solver, PROJ ≥ 9.3, zlib, libjpeg, libpng, libtiff, GoogleTest, pybind11
and Python 3. The GPU backend also needs the Vulkan headers and loader, `glslc`,
`spirv-val` and the VulkanMemoryAllocator header; without them it is switched off and
every task runs on the CPU. The desktop app needs Qt 6.7 or newer (it is built and tested
against Qt 6.8).

```sh
# Ubuntu 24.04
sudo apt install build-essential cmake ninja-build libeigen3-dev libceres-dev libproj-dev proj-data \
  zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libgtest-dev pybind11-dev python3-dev \
  libvulkan-dev glslc spirv-tools mesa-vulkan-drivers

# macOS (Homebrew)
brew install cmake ninja eigen ceres-solver proj libtiff libpng jpeg-turbo googletest pybind11 \
  libomp molten-vk vulkan-loader vulkan-headers shaderc spirv-tools

# Windows (vcpkg)
vcpkg install eigen3 ceres[eigensparse] proj[core,tiff] libjpeg-turbo libpng tiff gtest zlib \
  vulkan-headers vulkan-memory-allocator shaderc spirv-tools[tools]

cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows the Vulkan headers and the shader tools come from vcpkg, which keeps
the GPU layer on by default; `vulkan-loader` is not needed to *build* (nothing
links a loader — the backend opens `vulkan-1.dll` at run time), only to run the
GPU tests against a software driver. A machine without a Vulkan driver (any
Windows VM, or a host whose GPU driver predates Vulkan) still runs the app: the
GPU layer reports why it is unavailable and every task falls back to the CPU.

The desktop app is opt-in: add `-DALPHASHAPE_BUILD_GUI=ON` and put a Qt 6.8 kit on
`CMAKE_PREFIX_PATH`. Windows (clang-cl + vcpkg) is described in
[`engine/TESTING.md`](engine/TESTING.md), and [CONTRIBUTING.md](CONTRIBUTING.md) has the
rules every change is held to.

## Releases

| Channel | What | Where |
|---|---|---|
| Stable | Tagged `vX.Y.Z`, full test suite, packages for Linux, macOS and Windows | [Releases](https://github.com/aniketpratapsingh/alphashape/releases/latest) |
| Pre-release | `vX.Y.Z-rc.N` / `-beta.N` tags; never marked "latest" | [Releases](https://github.com/aniketpratapsingh/alphashape/releases) |
| Nightly | Built from `main` every night when it changed | [`nightly` release](https://github.com/aniketpratapsingh/alphashape/releases/tag/nightly) |

What changed between versions is in [`CHANGELOG.md`](CHANGELOG.md).

## Repository layout

```
engine/
  src/base      foundation: errors, logging, a portable RNG, the OpenMP helper, zip,
                the platform layer (paths, env, temp files, processes, consoles,
                dynamically loaded libraries) and a voxel grid
  src/image     image types and codecs
  src/model     geometry: cameras, meshes, rasters, coordinate systems
  src/common    data model, task registry, commands (undo), project I/O, algorithms
  src/gpu       Vulkan compute backend and shaders
  src/apps/cli  the command-line tool
  src/apps/gui  the Qt desktop application
  python/       Python bindings
  tests/        C++ tests (GoogleTest) and synthetic fixtures
  data/         the synthetic scenes the tests and the sample dataset are rendered from
packaging/      per-OS bundling, installers and their smoke tests
tools/sample/   the sample dataset generator and its provenance
design/         design system and GUI mockups
```

## Documentation

- [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md) — what is missing or approximate in this
  release, and what would fix it.
- [`engine/docs/crs-guide.md`](engine/docs/crs-guide.md) — the coordinate-system layer:
  CRS and geoid registration, axis and heading conventions, PROJ data discovery, GeoTIFF
  GeoKeys and raster reprojection.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each release.
- [`engine/TESTING.md`](engine/TESTING.md) — how to build and test, and what CI runs.
- [`SECURITY.md`](SECURITY.md) — supported versions and how to report a vulnerability.
- [`packaging/`](packaging/) — how each platform's package is assembled.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in
[SECURITY.md](SECURITY.md), not in a public issue.

## License

alphashape is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

Third-party libraries keep their own licences (PROJ: MIT, Ceres: BSD, Eigen: MPL-2.0, Qt:
LGPL-3, and others). They are linked dynamically, and their notices ship with the
installers.

The sample dataset is dedicated to the public domain under CC0-1.0; see
[`tools/sample/SOURCE.md`](tools/sample/SOURCE.md).
