# Testing

Normal is the configuration whose 684 tests (C++, Python and CLI e2e) must stay green.

## Normal (Release, day-to-day)

```
cmake -S . -B build-norm -DALPHASHAPE_WARNINGS_AS_ERRORS=ON
cmake --build build-norm -j && ctest --test-dir build-norm --output-on-failure
```

## Linux build machine (Ubuntu 24.04, x86-64)

The second supported platform. A remote 4-core box builds and runs the full
suite against distribution packages (g++ 13.3, CMake 3.28, Ninja, Eigen, Ceres,
libtiff/png/jpeg-turbo, GTest, pybind11, Vulkan loader + `glslc`) and Mesa's
llvmpipe software Vulkan driver — no hardware GPU.

M12 adds the CRS engine: **PROJ ≥ 9.3** (MIT, dynamically linked) and its data
package. Install it before configuring:

```
# macOS (Homebrew; proj-data comes in as a dependency)
brew install proj

# Ubuntu 24.04 (libproj-dev 9.4, proj-data carries proj.db and the grids)
sudo apt install libproj-dev proj-data
```

CMake finds PROJ either through the Homebrew CMake config package
(`find_package(PROJ 9.3 CONFIG)`) or through `proj.pc` on the distro packages;
if neither is found the configure fails with both install lines. The data
directory is resolved at runtime in this order — `ALPHASHAPE_PROJ_DATA`,
`<exe_dir>/share/proj`, `<project>/../tools/proj-data`, then PROJ's own search
(`proj_context_get_database_path()`). Network access is forced off
(`PROJ_NETWORK=OFF`, `network = off` in `proj.ini`), so a missing grid is an
error rather than a download. `crs::describe_environment()` prints the resolved
paths. `-DALPHASHAPE_USE_PROJ=OFF` builds the legacy three-CRS stub instead and
the `Crs.*` tests are not compiled.

```
# sync the working tree to the build machine (build trees stay remote)
rsync -az --delete --exclude 'build*/' --exclude '__pycache__/' \
    engine/ <host>:Projects/alphashape/engine/

# configure, build, test (tools/guard.sh enforces a wall clock and a memory floor)
tools/guard.sh 900  cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DALPHASHAPE_WARNINGS_AS_ERRORS=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=$HOME/Projects/alphashape/tools/ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=$HOME/Projects/alphashape/tools/ccache
tools/guard.sh 2400 cmake --build build-linux -j2
OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive tools/guard.sh 5400 ctest --test-dir build-linux --output-on-failure -j2
```

- **`-j2`, not `-j4`.** `glslc` peaks above 5 GB per shader on the larger
  kernels, so the shader steps run in a one-slot CMake job pool (see
  `alphashape_vulkan_job_pools`); two concurrent C++ jobs are what a 4-core /
  32 GB machine handles.
- **llvmpipe.** The Vulkan layer loads the system loader (`libvulkan.so.1`) and
  the system ICDs; nothing is baked in at configure time. `gpu::is_software_device`
  identifies the CPU driver: device-correctness tests (buffers, dispatches,
  barriers, PatchMatch, Poisson, validation layer) all run and must pass there,
  while speed-up assertions skip themselves. The GPU TV-L1 fusion path declines
  software devices: its fp16 packing pattern makes llvmpipe's LLVM backend abort
  the process (`Cannot select: ... anyext from i16`).
- **Numeric consistency.** The warning set carries `-ffp-contract=off` on every
  platform, and every seeded scene draws through `base/utils/portable_rng.hpp`
  instead of `<random>` distributions (whose algorithms are implementation-defined,
  so one seed gave different samples under libc++ and libstdc++). Two further
  rules keep one seed selecting the same scene on both platforms:
  - **Never draw twice in one expression.** Argument and initialiser-list
    evaluation order is unspecified (clang left to right, g++ right to left), so
    `Vec3(f(rng), f(rng), f(rng))` consumes the stream differently on each.
    Draw into named locals, or use `draw2`/`draw3`/`draw4` from
    `tests/test_rng_helpers.hpp`, which sequence the draws in g++'s
    right-to-left order, so a seed also keeps the scenes the existing bounds
    were calibrated on.
  - Both rules are enforced by `rng_order_check`, a ctest that runs
    `tools/check_rng_order.py` over `src/` and `tests/`. A statement that
    legitimately makes several draws carries a `// rng-order: ok` marker. Run it
    alone with `ctest --test-dir build-linux -R rng_order_check`.

## Apple Silicon test machine (M5)

The hardware-GPU host: a MacBook Air M5 (16 GB, macOS 26) driven over SSH, no
sudo, Homebrew owned by the login user. It is the only machine on which the
hardware-only paths (TV-L1 fp16 dual kernel, speed-up assertions, the real
MoltenVK device) actually run.

Dependencies (user-owned, so no sudo):

```
export PATH=/opt/homebrew/bin:$PATH
export HOMEBREW_NO_AUTO_UPDATE=1
brew install cmake ninja eigen ceres-solver libtiff libpng jpeg-turbo \
    googletest pybind11 libomp molten-vk vulkan-loader vulkan-headers \
    shaderc spirv-tools vulkan-validationlayers
```

`spirv-tools` supplies `spirv-val` (CMake requires it beside `glslc`), and the
validation layers are needed by `Gpu.ValidationModeReportsNoErrors`. Homebrew
has **no formula for VulkanMemoryAllocator** and the project does not vendor it:
copy `include/vma/vk_mem_alloc.h` from a Vulkan SDK (or the Khronos header of
the same name) into `/opt/homebrew/include/vma/`. Without it a fresh configure
reports `Vulkan SDK or shader tools not found; GPU layer disabled` and every GPU
test silently skips — check `build-mac/src/gpu/generated/gpu/gpu_config.hpp`
shows `ALPHASHAPE_GPU_HAS_VULKAN 1` after configuring.

Configure, build and test (build trees stay on the test machine; sync the source
tree from the development Mac):

```
# from the tree above engine/
rsync -az --delete --exclude 'build*/' --exclude '__pycache__/' \
    engine/ <host>:Projects/alphashape/engine/

export PATH=/opt/homebrew/bin:$PATH
cd ~/Projects/alphashape/engine
../tools/guard.sh 900 cmake -S . -B build-mac -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DALPHASHAPE_WARNINGS_AS_ERRORS=ON \
    -DALPHASHAPE_GPU=ON
../tools/guard.sh 5400 cmake --build build-mac -j4
ctest --test-dir build-mac -N | tail -2      # discovery ran during the link
```

- **No Vulkan SDK is needed on this host.** The system headers, the Homebrew
  loader (`libvulkan.1.dylib` on the default search path) and the Homebrew ICD
  (`/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`, found by the loader's
  own search path) are used; nothing is baked into the binary. Do not set
  `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` — that would hide the installed driver.
- **GPU tests, one suite at a time** (the guard is a wall-clock watchdog: 124 on
  timeout, 125 on low memory). Never run two suites concurrently — a second
  process competes for the single compute queue and turns a 15 s suite into a
  guard timeout.
  ```
  OMP_NUM_THREADS=4 ../tools/guard.sh 480 ./tests/alphashape_tests --gtest_filter='Gpu.*'
  OMP_NUM_THREADS=4 ../tools/guard.sh 480 ./tests/alphashape_tests --gtest_filter='PoissonGpu.*'
  OMP_NUM_THREADS=4 ../tools/guard.sh 480 ./tests/alphashape_tests --gtest_filter='PatchMatchGpu.*'
  OMP_NUM_THREADS=4 ../tools/guard.sh 480 ./tests/alphashape_tests --gtest_filter='TvFusionGpu.*'
  ```
- **Driver diagnostics.** `MVK_CONFIG_LOG_LEVEL=2` makes MoltenVK print its own
  errors; a `vkQueueSubmit ... Caused GPU Address Fault Error` line there means
  a kernel indexed outside a buffer (the `VK_ERROR_DEVICE_LOST` the layer
  reports afterwards is a symptom, and it poisons the context for every later
  dispatch, so the following `with_fallback` messages are noise). Set
  `VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d`
  with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` for validation output.
- **Run the binary, not just ctest, when investigating discovery.** CMake's
  POST_BUILD step runs `alphashape_tests --gtest_list_tests --gtest_output=json:...`
  with a 5 s timeout; running the same command by hand reproduces exactly what
  the build step saw.

## Coordinate-system tests (M12)

The C++ suites live in `engine/tests/` and are listed in the
`if(ALPHASHAPE_USE_PROJ)` block of `engine/tests/CMakeLists.txt`:

- `test_crs_context.cpp` — discovery order, data publishing, offline grids.
- `test_crs.cpp` — the value type, built-ins, WKT/PROJ-string round trips.
- `test_crs_transform.cpp` — axis normalisation, batch validity, the 20 000-point
  `cs2cs` comparison.
- `test_datum_transform.cpp` — candidate listing, pinning, ballpark refusal,
  Helmert and grid shifts.
- `test_geoid.cpp` — grid evaluation, the store, NGS benchmarks.
- `test_local_frame.cpp`, `test_euler_angles.cpp`,
  `test_reference_convert.cpp`, `test_surface_frame.cpp` — frames, angles and
  reference/CRS migration.
- `test_reproject.cpp` and `test_geotiff_crs.cpp` — raster reprojection and
  GeoKey encoding.

Python coverage is `engine/python/tests/test_crs.py`.

These sources are **compiled only when PROJ is enabled**. With
`-DALPHASHAPE_USE_PROJ=OFF` they are absent from the test binary rather than
skipped, so the test count is smaller by design. They need the PROJ library and
`proj.db`, not the PROJ command-line tools; the `cs2cs` reference values are
checked in. Cases that need a grid the host does not have call
`GTEST_SKIP()`, naming the file: the NTv2/NADCON/OSTN15 datum cases, the
EGM96/EGM2008 benchmark when the full grid is absent, and a vertical/compound or
greater-than-65535 GeoKey case when the local `proj.db` has no such CRS. CI runs
the grid-free Helmert cases everywhere.

### Fixtures and regeneration

All fixtures are under `engine/tests/fixtures/crs/` and are checked in; the
tests read them without network access.

- `egm96_subset.tif` — a 1-degree EGM96 subset (181 × 361 float32 samples),
  made by keeping every fourth node of NGA's `us_nga_egm96_15.tif`. Regenerate
  with `python3 make_egm96_subset.py [--full /path/to/us_nga_egm96_15.tif]`;
  without `--full` the script downloads the source from `cdn.proj.org`, and it
  needs `numpy`, `tifffile` and `imagecodecs`. Only the generator touches the
  network; the engine never does.
- `ngs_geoid_benchmarks.csv` — 50 seeded CONUS positions with NGS's USGG2012
  geoid heights. Regenerate with `python3 make_ngs_benchmarks.py` (network to
  the NGS service); `--check` verifies the recorded values without rewriting.
- `builtin_wkt_skeletons.txt` — the normalised WKT2:2019 skeletons for the
  48-code built-in list. `make_skeletons.cpp` compiles against the local PROJ
  and writes one `<epsg> <skeleton>` line per code; regenerate it whenever the
  built-in list or the PROJ data release changes.
- `cs2cs_reference.csv` — 20 000 reference points for EPSG:4326 → EPSG:32633,
  generated from an explicit 32-bit LCG by `tests/make_cs2cs_reference.sh
  [count] [output]`. That script invokes `cs2cs -f %.9f`; the checked-in file
  pins the PROJ data version it was made with, so regenerating it is a
  deliberate acknowledgement of a numerical change.

### Regenerating everything: `alphashape_crs_fixtures`

The CMake target is opt-in: it is defined only when `ALPHASHAPE_USE_PROJ` is on,
is not part of `all` (the helper executable is `EXCLUDE_FROM_ALL` and the custom
target is not declared `ALL`), and is never a ctest. Run it explicitly:

```sh
cmake --build build --target alphashape_crs_fixtures
```

It builds and runs `make_crs_skeletons`, writing
`builtin_wkt_skeletons.txt` as above. When CMake found a Python 3 interpreter
(`find_package(Python3 COMPONENTS Interpreter)`), it also runs
`make_egm96_subset.py` and `make_ngs_benchmarks.py`; when no interpreter is
found those two steps are skipped and configure still succeeds. On a Unix host
it runs `make_cs2cs_reference.sh`, which needs `cs2cs` on `PATH`. The target
performs network access for the EGM96 source and the NGS service; the test
suite itself does not. Review the generated diffs before committing them,
because the fixtures intentionally record a PROJ data version.

## Bug-hunting (Debug + ASan + UBSan + Eigen NaN init)

```
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DALPHASHAPE_SANITIZE=ON \
    -DALPHASHAPE_EIGEN_NAN_INIT=ON -DALPHASHAPE_BUILD_PYTHON=OFF \
    -DALPHASHAPE_WARNINGS_AS_ERRORS=ON
cmake --build build-san -j
ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 UBSAN_OPTIONS=print_stacktrace=1 \
    ctest --test-dir build-san --output-on-failure -j4
```

- `ALPHASHAPE_SANITIZE` instruments our targets (not Homebrew libraries) and
  defines `ALPHASHAPE_SANITIZED_BUILD`, which may widen a *timing* budget only.
- `ALPHASHAPE_EIGEN_NAN_INIT` defines `EIGEN_INITIALIZE_MATRICES_BY_NAN` for our
  TUs only: any NaN failure is a real bug until proven otherwise.
- Python is off (its interpreter is not ASan-instrumented), so `cli_e2e` does
  not run; LeakSanitizer is unsupported on macOS arm64, hence `detect_leaks=0`.
- `cmake/sanitizers/asan.supp` suppresses one third-party false positive: the
  GoogleTest JSON printer tripping ASan's container-overflow check during
  CMake's build-time discovery (single-quoted: the tree path has a space).
