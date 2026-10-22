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
