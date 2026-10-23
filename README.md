# alphashape

**An open-source photogrammetry engine: photos in, 3D models and maps out.**

[![CI](https://github.com/aniketpratapsingh/alphashape/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aniketpratapsingh/alphashape/actions/workflows/ci.yml)
[![Nightly](https://github.com/aniketpratapsingh/alphashape/actions/workflows/nightly.yml/badge.svg)](https://github.com/aniketpratapsingh/alphashape/actions/workflows/nightly.yml)
[![CodeQL](https://github.com/aniketpratapsingh/alphashape/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/aniketpratapsingh/alphashape/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/aniketpratapsingh/alphashape/badge)](https://securityscorecards.dev/viewer/?uri=github.com/aniketpratapsingh/alphashape)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15791/badge)](https://www.bestpractices.dev/projects/15791)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)

alphashape turns overlapping photographs into camera poses, dense point clouds, textured meshes,
digital elevation models (DEMs) and orthomosaics. It is written in C++20, scriptable from Python and
a command-line tool, and accelerated with Vulkan compute on GPUs that support it. A Qt desktop
application is in development.

> **Status: early development (0.1.x).** The full pipeline runs from the CLI and Python, but formats,
> APIs and results still change between versions. Not yet suitable for production surveys.

## What it does today

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
| Export | `ExportModel`, `ExportPointCloud` | OBJ/PLY and more |
| Chunks | `DuplicateChunk`, `MergeChunks`, `AppendProject`, `AlignChunks` | multi-chunk projects |

Projects are saved as a `.psx` file plus a `.files/` folder. Every task has one parameter schema shared
by the CLI, Python and the project file, so the three cannot drift apart. Edits that change the project
are undoable.

Task and parameter names follow the naming of Agisoft Metashape's public Python API, so that existing
processing scripts are easy to port. alphashape is an independent project. It is not affiliated with,
endorsed by or derived from Agisoft or Metashape.

Known gaps are listed honestly in [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md).

## Quick start

```sh
# create a project from photos, then run the pipeline task by task
alphashape-cli --new site.psx --add-images photos/*.jpg
alphashape-cli --project site.psx --task MatchPhotos --set keypoint_limit=40000
alphashape-cli --project site.psx --task AlignCameras
alphashape-cli --project site.psx --task BuildDepthMaps
alphashape-cli --project site.psx --task BuildPointCloud
alphashape-cli --project site.psx --task BuildModel
alphashape-cli --list-tasks          # every task and its parameters
```

```python
import alphashape

doc = alphashape.Document()
doc.active_chunk = doc.add_chunk("Chunk 1")
# the same tasks and parameter names as the CLI
doc.run_task("AddPhotos", filenames=["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"])
doc.run_task("MatchPhotos", keypoint_limit=40000)
doc.run_task("AlignCameras")
doc.run_task("BuildDepthMaps", downscale=4)
doc.run_task("BuildPointCloud")
doc.save("site.psx")
```

## Building from source

Requirements: CMake ≥ 3.24, a C++20 compiler (Clang, GCC 13+, or clang-cl on Windows), Eigen 3, Ceres
Solver, PROJ ≥ 9.3, zlib, libjpeg, libpng, libtiff, GoogleTest, pybind11 and Python 3. The GPU backend
also needs the Vulkan headers and loader, `glslc`, `spirv-val` and the VulkanMemoryAllocator header;
without them it is switched off and every task runs on the CPU.

```sh
# Ubuntu 24.04
sudo apt install build-essential cmake ninja-build libeigen3-dev libceres-dev libproj-dev proj-data \
  zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libgtest-dev pybind11-dev python3-dev \
  libvulkan-dev glslc spirv-tools mesa-vulkan-drivers

# macOS (Homebrew)
brew install cmake ninja eigen ceres-solver proj libtiff libpng jpeg-turbo googletest pybind11 \
  libomp molten-vk vulkan-loader vulkan-headers shaderc spirv-tools

cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows (clang-cl + vcpkg) is being brought up; see the CI workflow for the current recipe. More detail
is in [`engine/TESTING.md`](engine/TESTING.md).

## Releases

| Channel | What | Where |
|---|---|---|
| Stable | Tagged `vX.Y.Z`, full test suite, installers for Linux, macOS and Windows | [Releases](https://github.com/aniketpratapsingh/alphashape/releases/latest) |
| Pre-release | `vX.Y.Z-rc.N` / `-beta.N` tags; never marked "latest" | [Releases](https://github.com/aniketpratapsingh/alphashape/releases) |
| Nightly | Built from `main` every night when it changed | [`nightly` release](https://github.com/aniketpratapsingh/alphashape/releases/tag/nightly) |

macOS builds are ad-hoc signed, not notarized: on first launch, right-click the app and choose
**Open**.

## Repository layout

```
engine/
  src/base      foundation: math, zip, platform layer, logging
  src/image     image types and codecs
  src/model     geometry: cameras, meshes, rasters, coordinate systems
  src/common    data model, task registry, commands (undo), project I/O, algorithms
  src/gpu       Vulkan compute backend and shaders
  src/apps/cli  the command-line tool
  python/       Python bindings
  tests/        C++ tests (GoogleTest) and synthetic fixtures
design/         design system and GUI mockups
```

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in
[SECURITY.md](SECURITY.md).

## License

alphashape is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. See [LICENSE](LICENSE).

Third-party libraries keep their own licences (PROJ: MIT, Ceres: BSD, Eigen: MPL-2.0, and others).
They are linked dynamically, and their notices ship with the installers.
