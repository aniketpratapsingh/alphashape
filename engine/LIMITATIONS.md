# Known limitations

Running list, one section per milestone. Each entry says what is missing or
approximate and, where known, what would fix it. Update it at the close of every
milestone (and whenever a review turns up a new limitation).

## M6 — Dense stereo

- Every scene run end to end has no lens distortion, so speed on a real distorted
  lens is unmeasured.
- Python returns depth maps as nested lists; slow for large images.
- View selection is all-pairs over tracks; projects with thousands of photos need
  a spatial pre-filter.
- Re-aligning cameras does not invalidate existing depth maps.
- *(Fixed in M9.)* The multi-view cost counted neighbours that could not see a pixel as failed matches, which erased most pixels on low-overlap (nadir grid) blocks. Per-camera coverage on the aerial block rose from 26–54% to 74–95%.
- Ground seen by only one camera gets no depth: there is no single-view or shape-from-shading fallback.
- Census matching cost and coarse-to-fine (multi-resolution) PatchMatch are not
  built.
- Full-pipeline accuracy on the 8-image arc is limited by alignment (0.2 px
  feature noise on a short baseline), not by stereo.

## M7 — Meshing

- Point-cloud (Poisson) meshing needs the cloud filtered by confidence first; an
  unfiltered cloud produces floating blobs from weakly supported points.
- The TV-L1 fusion volume is held in memory in one piece (128 voxels on the
  longest side by default); large scenes need an out-of-core octree.
- Height-field surface type and TGV regularisation for meshing are not built.
- Poisson at 128 voxels takes ~7 s; a multigrid solver would speed it up.
- Python returns mesh arrays as plain lists.

## M8 — Texturing

- Not built: multi-band ("natural") blending, ghosting filter, de-lighting,
  multi-page atlases, orthophoto/spherical/camera mapping modes.
- `ghosting_filter` is accepted but does nothing (logs a warning).
- Baking is single-pass on the CPU; a 4096 px texture on a large mesh is slow
  until it is tiled or moved to the GPU.

## M9 — DEM and orthomosaic

- **Projections:** only three coordinate systems are supported: local, WGS84 geographic (EPSG:4326) and WGS84 UTM (EPSG:326xx/327xx). There is no general coordinate-system engine and no datum transformations; any other EPSG code is rejected. The UTM series is truncated at order n³ (sub-millimetre within a zone).
- UTM zone selection ignores the Norway/Svalbard exceptions; the auto zone is the standard 6° zone of the chunk origin.
- **GeoTIFF:** no overview pyramids and no Cloud-Optimized GeoTIFF layout. Float rasters are single-band only; 8-bit rasters can be grey, RGB or RGBA. The coordinate system is stored as an EPSG code only (no WKT or custom definitions). The switch to BigTIFF for files over 3.5 GB has not been tested at that size.
- **Persistence:** the DEM and orthomosaic are saved inside the project zip as GeoTIFF side files (`elevation/dem.tif` float32, `orthomosaic/ortho.tif` RGBA u8) with an `AOL1` seam-label sidecar (`orthomosaic/labels.bin`) and an `<elevation>` / `<orthomosaic>` element in doc.xml carrying the grid, EPSG, nodata and source/surface metadata. Older projects load unchanged (a missing element means an empty product). Both are written only when non-empty, so an earlier project keeps its exact bytes. Limitations:
  - The GeoTIFF codec is file-based, so saving writes each raster to a temporary file and reads it back — an extra full copy of the product on disk (and in memory) during save/load. A streaming encoder would remove it.
  - Tiled (256 px) + deflate on save; the user's Export* options are not preserved in the project file.
  - Corrupt, missing or doc.xml/TIFF-mismatched side files are a hard `ProjectIOError`; there is no repair path.
  - Seam labels are stored as raw int32 per cell. On the aerial block that is 640 KB raw, larger than the ortho itself (327 KB) and left to the zip's deflate. A run-length or 16-bit encoding would shrink it.
- **Python access:** `chunk.elevation` / `chunk.orthomosaic` return views (`None` before the task runs) with the grid metadata, `heights()` / `pixels()` / `labels()` as nested lists, and `data` as raw bytes. `chunk.buildDem(...)`, `chunk.buildOrthomosaic(...)`, `chunk.exportDem(...)`, `chunk.exportOrthomosaic(...)` are thin wrappers over `run_task` (same schema, same meta recording). The module does not depend on numpy, so the nested-list accessors are slow and memory-hungry on a real product (a 4k×4k ortho is ~67M Python ints); prefer `data` for bulk transfer.
- **TGV DEM smoothing:** plain Chambolle–Pock with fixed steps, no preconditioning or multigrid. It does not reach its tolerance in the default 500 iterations; energy is still ~1.7% above the converged value. About 0.4 s for a 128² grid, so a 4k² DEM needs tiling or a GPU port (Metashape runs a tiled multi-GPU solver). The whole grid is held in memory in one piece.
- **TGV near cliffs:** an outlier spike right at a height discontinuity (building edge) can survive. In tests, up to 6.9 units on a +10 spike sitting on a 14–18-unit step, against ≤ 0.33 everywhere else. The L2 data term barely helps with outliers (RMSE 1.44 → 1.21); use L1.
- **Hole filling (natural neighbour):** this is the grid-based approximation (Park et al. 2006), not an exact Delaunay/Sibson interpolator. The ring of filled cells right next to the data gets few samples, so a small step can appear there: up to ~3× the local slope × one cell, about 1.5% of range on the test plane. Cost grows with the area of the largest hole (it scales with r²). IDW's neighbour set can slightly exceed the requested k.
- **BuildDem:**
  - Coverage follows multi-view overlap. On the aerial block, interior-only fill ("enabled") covers 87% of the area: the missing 12% is seen by only one camera. "extrapolated" covers 100%, at 2.5 cm median / 11 cm p90 error in the extrapolated cells, against 1.1 / 3.6 cm in observed cells.
  - The DEM covers the bounding box of the source data and ignores `chunk.region()`.
  - Per-cell aggregation is the median (point cloud and depth maps) or the max (mesh); there is no point-class filtering (Metashape's `classes`).
  - TGV uses DEM-specific weights (α1 = 0.2, α0 = 0.4) and treats interpolated cells as unit-weight data. On the aerial scene it gives no measurable gain over no smoothing (p90 0.0293 vs 0.0296 m); the generic defaults smeared metre-scale steps.
- **Orthomosaic core:** the whole output grid is held in memory as one piece (no tiling or streaming). Twelve 480×360 cameras take ~3.4 s, so large orthos need tiling. No colour correction or gain levelling between cameras. Occlusion uses a fixed half-cell ray-march step, not an exact grid traversal, so very thin occluders can be missed. Mosaic blending is slightly less faithful than hard seams on the synthetic scene (7.4 vs 7.0 mean error) in exchange for less visible seams.
- **Seamlines:**
  - Multi-scale alpha-expansion with single-threaded push-relabel max-flow. Deterministic, but the seam stage does not use multiple cores.
  - The cost weights (data 0.02, seam 2.0, smoothness 0.30) were tuned on the synthetic block only.
  - Seams may still cross a uniform roof, where all views agree (130 edges around the building vs 236 for the plain best-camera labelling). They are invisible there but not removed.
  - The full 12-camera ortho (rectify + seams + blend) takes ~4.5 s against a 5 s test budget, so the test can be flaky on a loaded machine.
- **BuildOrthomosaic:**
  - No reprojection: `projection` must equal the DEM's EPSG, otherwise it fails with "reproject the DEM first".
  - Auto resolution assumes a metric plane (UTM/local), not EPSG:4326.
  - Mosaic blending is ~3× slower than average/disabled (28 s vs 9 s for a 1025×779 ortho of 12 images), with everything in memory.
  - The occlusion step is half of max(ortho cell, DEM cell), so occluders thinner than about one DEM cell can be missed.
  - `fill_holes` fills only interior holes (natural-neighbour colour, made opaque); it does not inpaint texture.
  - No ghosting filter, colour correction or manual seamline editing.
  - Ortho from a mesh surface is less complete (91% coverage, error 15 vs 10–11 grey levels) because the DSM is rasterised from the mesh at ortho resolution.
- **Validation scene:** the aerial test scene is synthetic: 12 photos with no lighting or shadows, no lens distortion, and exact GPS. Its outer edges (about 1 m at y = ±45) are not seen by any camera. Accuracy on real imagery is not yet measured.
- Heights are ellipsoidal (WGS84). No geoid model, so DEM heights are not orthometric ("above sea level").

## M10 — GPU layer

- **Scope:** this milestone lands the backend-neutral compute core
  (`src/gpu`): device discovery, buffers, sampled images/samplers, SPIR-V
  kernels, dispatches, validation and CPU fallback, plus the first two
  algorithm ports, GPU PatchMatch (below) and the GPU Poisson solve (below).
  The five original test kernels (saxpy, reduce_sum, 3×3 box filter,
  chain_add, image_sample_test) remain test kernels.
- **One backend:** Vulkan via MoltenVK. There is no OpenCL or Metal path; on
  hosts where MoltenVK cannot load, `gpu::available()` is false and every
  caller must go through `with_fallback` (or not use the layer at all).
- **Two validated drivers, neither a discrete GPU:** the device tests were
  exercised on an Apple M1 with MoltenVK (Vulkan SDK 1.4.363.0) and on x86-64
  Ubuntu 24.04 with Mesa's llvmpipe (a CPU implementation of Vulkan 1.4).
  Behaviour on hardware Vulkan drivers (AMD/Intel/NVIDIA) is unverified.
  MoltenVK is a portability driver: the instance enables
  `VK_KHR_portability_enumeration` and the device enables
  `VK_KHR_portability_subset`; both are probed and enabled only when advertised,
  so the non-portability path runs on Linux.
- **What Linux/llvmpipe does and does not validate:** it does validate the
  loader path with a system loader and system ICDs, instance/device creation,
  the whole dispatch/buffer/barrier/validation-layer surface, and the
  correctness of every ported kernel (GPU PatchMatch and the GPU Poisson solve
  are checked against their CPU references there). It does NOT validate
  performance at all: llvmpipe is the CPU, so no speed-up statement transfers,
  and the timing/speed-up tests skip themselves via `gpu::is_software_device`.
  It also cannot validate anything fp16: llvmpipe's LLVM 20 backend aborts the
  process on the `packHalf2x16`/`unpackHalf2x16` pattern in the TV-L1 dual
  kernel ("Cannot select: ... anyext from i16"), so `tv_l1_fusion_device`
  declines software devices and takes the CPU path (the kernel stays
  hardware-only until that driver limitation changes).
- **Submission model (WP-10.3):** `run_async` + `Ticket`/`wait` replace the
  single synchronous path; `run` is `run_async` + `wait`. The ring is two
  command pools with one fence per slot and a hard cap of `kMaxInFlight = 4`
  un-waited submissions per context (the 5th throws `gpu::Error`). Fence waits
  poll in 5 s slices with a 30 s total budget on hardware and 10 min on a CPU
  driver. What is *not* verified:
  - **Two queues:** a second compute queue is created only when the selected
    family reports `queueCount >= 2`, and is used only by
    `Submission{.queue = 1}`. Both validated drivers (MoltenVK on M1, llvmpipe)
    expose a single compute queue, so only the single-queue fallback path has
    ever run; the cross-queue path is compiled but unexercised.
  - **Timeline semaphores:** none. Cross-queue and cross-submission ordering is
    a host-side `wait` on the fence, so a producer/consumer split across two
    queues serialises on the CPU between submissions.
  - **Device-local staging on non-unified memory:** the staged copy path is
    implemented (64 MiB ring, chunked, explicit bounds-checked offsets) and is
    exercised on llvmpipe, whose heap is unified but which keeps the staged
    path deliberately. The genuinely non-mappable VRAM case (a discrete GPU
    with a separate host-visible heap) is unverified: no such device was
    available. On an integrated GPU (`info().integrated`) `DeviceLocal` and
    `HostVisible` resolve to the same mappable heap and no staging is used.
  - **PatchMatch stays host-visible** by design
    (`patch_match_gpu_prefers_host_memory()` returns true): its whole grid is
    seeded and read back per call, so a DeviceLocal working set would add two
    grid-sized staged transfers per call for no gain. Poisson's large vectors
    are DeviceLocal.

- **Images and samplers (WP-10.2):** `gpu::Image2D` and `gpu::Sampler` exist
  with `Format{R8Unorm, R8G8B8A8Unorm, R32Float}`, `Filter{Nearest, Linear}`
  and `AddressMode{ClampToEdge, Repeat, MirroredRepeat}`. Bindings are typed
  from SPIR-V reflection (`OpTypeSampledImage` over a 2D `OpTypeImage` ->
  `COMBINED_IMAGE_SAMPLER`), never from the binding number. What is still
  limited:
  - **Combined samplers only.** A bare `OpTypeImage` or `OpTypeSampler` (the
    separate-image/separate-sampler Vulkan form) is rejected at pipeline build
    with "separate images/samplers unsupported"; non-2D dims, storage images
    and any descriptor set other than 0 are rejected the same way.
  - **Three formats, one level.** R8/RGBA8/R32Float; no depth format, no mip
    chain, no array or 3D image, no image writes from a shader (no storage
    images) and no render pass — sampled reads only.
  - **No per-dispatch transitions.** An image is in
    `SHADER_READ_ONLY_OPTIMAL` after its first `upload`; upload/download each
    record their own transitions on a dedicated command buffer and wait for
    in-flight work, so a dispatch never records an image barrier. Reading the
    same image from two dispatches in one `run` is free; writing an image from
    a shader is not supported at all (see above).
  - **Host copies are not cheap.** Each `upload`/`download` allocates a
    temporary host-visible buffer for the whole image and waits for its fence,
    so a per-frame upload pattern would be slow; coarse per-call uploads (the
    intended use) are fine. The staging buffer is freed after the fence
    signals.
  - **No limit checks.** `maxImageDimension2D` and
    `maxPerStageDescriptorSampledImages` are not queried; exceeding them fails
    at `Image2D::create` or pipeline creation with `gpu::Error` (which
    `with_fallback` turns into the CPU path).
  - **Validated on llvmpipe and MoltenVK only** (as above). Both are unified
    memory, so the discrete-GPU image path (device-local VRAM plus a
    host-visible staging copy) is unverified. `image_sampling_supported`
    reports the sampled-image and transfer bits from
    `vkGetPhysicalDeviceFormatProperties`; a device that cannot sample a
    format makes the image tests skip with the driver's rejected feature mask
    logged.
- **Memory kinds:** `make_buffer` (and `Buffer::of`) take a
  `MemoryKind`. `DeviceLocal` allocates with
  `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE` and stages host copies; `HostVisible`
  allocates persistently-mapped host-visible memory (device-local on unified
  memory). On a discrete GPU a DeviceLocal buffer that the driver still backs
  with a mappable heap is *forced* through the staging ring, so the code path
  does not depend on which heap VMA happened to pick. There is no readback
  budget or eviction; a buffer that does not fit fails with `gpu::Error`.
- **No multi-GPU:** one context owns one device, and `run` never splits a
  dispatch across devices. `enumerate_devices()` reports the full list, but
  selection is a single index.
- **Memory sizing:** `required_memory` is checked against the largest
  device-local heap with a fixed 10% headroom. There is no budget query or
  eviction; a buffer that does not fit fails the allocation with `gpu::Error`.
- **Push constants:** at most one push-constant block, scalars/vectors only;
  the block size is derived from the SPIR-V member offsets. Arrays, matrices or
  nested structs in push constants are rejected.
- **Descriptor sets:** one set per dispatch, allocated from a per-`run` pool.
  The pool is sized to the exact number of storage buffers and combined image
  samplers bound (previously a fixed eight per set, which capped a kernel at
  eight bindings; the PatchMatch kernels use fourteen); a descriptor kind the
  batch does not use is omitted from the pool because a zero `descriptorCount`
  is itself a validation error. There is no descriptor caching or bindless
  access.
- **Golden self-check is per port, not central:** `compare()` and the
  `ALPHASHAPE_GPU_SELF_CHECK` convention exist; GPU PatchMatch
  (`BuildDepthMaps` on the first image) and the GPU Poisson solve (on the first
  solve) are the consumers. Other kernels do not use it.
- **Failure counter is process-global:** `failure_count()` counts every
  fallback, not per-kernel or per-context, and is never reset.
- **Loader failure is terminal:** the first failed `dlopen` attempt is cached;
  setting `ALPHASHAPE_VULKAN_LIBRARY` after `available()` has been called has no
  effect until the process restarts.
- **Validation:** `ContextOptions::validation` enables the layer only when it is
  discoverable. The baked `VK_LAYER_PATH` points into the SDK; on a machine
  where the SDK is removed after build, validation is silently skipped (tests
  skip rather than fail).

### GPU PatchMatch (`stereo::run_patch_match_device`)

The dense-stereo hot spot is ported: `BuildDepthMaps` runs PatchMatch on the
GPU per reference image through `gpu::with_fallback`, and the CPU on any
device failure (logged once, counted by `gpu::failure_count()`).

- **Scheme:** red-black checkerboard propagation with random refinement
  (Galliani, Lasinger & Schindler 2015). Each half-sweep updates one colour
  from its 4 adjacent and 4 distance-3 neighbours' planes, then draws a fresh
  random plane and a perturbed incumbent. The propagation order therefore
  differs from the CPU's sequential sweep; the cost (ZNCC window, the
  k_eff visibility vote rule, `max_cost` cut, sentinel 2.0) is the same
  function.
- **Determinism:** the CPU's stateless splitmix64 stream is reproduced in GLSL
  with 32-bit-word emulation of the 64-bit arithmetic and the same
  (seed, pixel, iteration, stream) key, so both paths draw the same hypotheses
  and the GPU result depends on no scheduling. Two GPU runs are bit-identical.
- **float32 only:** geometry stays in the reference camera frame, but ZNCC
  sums are accumulated in float (the CPU uses double). The dense fixture agrees
  with the CPU on 100.0% of jointly valid pixels within 1% relative depth
  (per-map median relative difference 0.00, max 2.8%); the synthetic plane
  scene agrees on 99.4%, with a median relative depth error of 1.6e-3 against
  the CPU's 1.0e-3 and full coverage.
- **Measured speed-up:** 2.4–2.9x on the M1 for one dense map at downscale 1
  (480x360, 7 neighbours; GPU 1.5 s against CPU 3.7–5.1 s with 8 threads).
  Aerial nadir blocks at downscale 1 are 1.7–1.9x. The 3x target is not
  reached. The rejected variants, all measured: a sample-major loop with an
  accumulator array (1.0x, register spill), a haloed tile in shared memory
  (0.8x, occupancy), candidate depths in shared memory (1.3x, bank
  conflicts), 32-lane workgroups (2.0x), and 128-lane workgroups (2.3x). The
  shipped layout keeps the candidate depths in a per-lane register array and
  stages only the view records and window offsets in shared memory.
- **Not bit-identical to the CPU:** the same hypotheses are drawn and evaluated
  in the same order, so rounding is the only difference, but ZNCC sums are
  float32 (see above). No existing test needed pinning: the CPU tests call
  `run_patch_match` directly, and every end-to-end test passes with the GPU on.
- **Kernel limits:** at most 49 window samples (radius 5 at step 2, radius 3
  at step 1) and 32 neighbours per reference. Larger windows or more
  neighbours fall back to the CPU for that image (`patch_match_gpu_supported`).
- **Fisheye:** the GPU forward model covers the Frame and Fisheye branches;
  only the Frame path (clean and distorted) is tested.
- **Fallback granularity:** per reference image, not per kernel: a GPU failure
  re-runs the whole image on the CPU.

### GPU Poisson solver (`meshing::poisson_solve_with_fallback`)

`screened_poisson()` assembles the discrete system exactly as before and runs
the Jacobi-preconditioned conjugate-gradient solve on the GPU through
`gpu::with_fallback` (CPU on any device failure). Assembly and result
finalisation are shared with the CPU path (`poisson_internal.hpp`), so both
solvers see the same operator, right-hand side, preconditioner and iso level.

- **Two dispatches per CG iteration, one submission per 32 iterations.**
  `poisson_spmv` computes `q = A p`, `p.q` and `alpha` (with the search direction
  `p = z + beta p_old` materialised on the way); `poisson_axpy` computes
  `x += alpha p`, `r -= alpha q`, `z = r / diag`, `r.r`, `r.z`, `beta` and
  `rho <- rz`. Both fuse their reductions with a subgroup shuffle tree plus a
  last-workgroup atomic counter, so a full iteration is one command buffer of two
  dispatches. Measured on llvmpipe: 15 submissions for 480 iterations
  (0.031 submissions/iteration, i.e. the batch is 32 iterations).
- **float32 vectors, double decisions.** M1 shaders have no fp64, so the vectors,
  the stencil and the on-device recurrences are float32. The two reduction
  partials per workgroup are read back and re-summed in compensated double
  (Neumaier) on the host, and the convergence decision is made on that double
  norm. The accepted solution is then checked against a TRUE residual
  recomputed in double (`rhs - A x`); the returned field therefore meets the
  CPU tolerance even though the float32 iterate alone cannot reach it.
- **Kernel safety.** Every shader loop is bounded by a compile-time constant or a
  push constant (the reduction loops run over `gl_NumSubgroups` <= 256 or
  `pc.groups`), every buffer index is checked against a pushed size (`pc.n`,
  `pc.groups`) before it is formed, and the voxel index is additionally guarded
  so the stencil taps are either in-range or provably skipped. Dispatch counts
  are ceil-divisions, and `poisson_gpu_supported` rejects systems whose group
  count exceeds the Vulkan-guaranteed 65535 per axis (CPU fallback instead).
- **Iterative refinement was needed.** On the synthetic sphere the float32 solve
  stalls at ~5e-6 to 8e-6 relative residual, so one refinement round is used: the
  CPU forms `b - A x` in double and the device solves `A d = ...` with the same
  float CG, accumulating `d` in double. Up to 6 rounds are allowed. The round
  count is logged once per process and exposed as
  `poisson_gpu_last_refinements()`.
- **Measured speed-up:** 2.8–3.4x at 128³ against the 8-thread CPU solve (GPU
  ~1.0 s including the refinement round, CPU ~2.8–3.2 s), 480 versus 443
  iterations because of the refinement restart. Measurements during a busy
  period (another executor's build on the same machine) reached 5.4x on the same
  test, so the CPU side is the one that moves; the ~1.0 s GPU figure is stable
  to ±3%. The 5–8x estimate is not reached:
  both paths are memory-bound (the CPU is already using 8 of 8 cores for a
  7-point stencil), and the reference `saxpy` on the same vectors runs at
  ~19–35 GB/s on this machine. Dispatches are ~0.5 ms each at 2 M voxels; the
  fused reduction costs ~0.4 ms of that against a bare `saxpy`.
- **Rejected variants (all measured):** subdividing the vector into vec4 aliases
  (2.1–3.0 ms per dispatch: LLVM's scalar alias path does not coalesce), a
  haloed in-plane shared-memory tile (1.11 vs 1.04 ms), a haloed 32×8×4
  volumetric tile (1.7 ms), a 1024-lane workgroup (0.36 ms against 0.22 ms for
  the submission floor), and restarting CG every batch (408 iterations for a
  worse residual).
- **Grid geometry:** the stencil assumes the CPU's exact 7-point Neumann layout
  and bounds; it is a 3D dispatch indexed `x`-fastest with no integer division.
  Systems beyond 2³¹ voxels, or beyond 65535 workgroups on any dispatch axis,
  fall back to the CPU (`poisson_gpu_supported`).
- **Validated on llvmpipe (Ubuntu VM, Mesa CPU driver).** The full
  `PoissonGpu.*` suite passes there: radius accuracy matches the CPU bounds
  (median 0.024 voxel, p95 0.115 voxel at 64³), GPU-vs-CPU relative L2 is
  2.8e-06 with a 0.0000-voxel median iso-surface vertex distance, the final true
  residual is 2.05e-08 (tolerance 1e-06), two runs are bit-identical, and the
  disabled/forced-failure fallbacks behave (identical to CPU; failure count +1).
  The end-to-end BuildModel point-cloud case matches the CPU test's bounds and
  face count exactly. The 128³ speed-up bound is skipped on software devices:
  measured 0.40x there, because the "device" is the CPU behind the emulated
  queue. **M1 (2026-10-21): 128³ GPU 1.14 s vs CPU 3.60 s (8 threads) = 3.15×, 480 iterations in 15 submissions, one double-precision refinement round; accuracy identical to CPU.**
- **Determinism:** every reduction is a fixed tree order and the host sums the
  partials in index order, so two GPU runs are bit-identical
  (`PoissonGpu.DeterministicRunToRun`).
- **`ALPHASHAPE_GPU_SELF_CHECK=1`** also runs the CPU solver on the first solve
  and logs `gpu::compare` stats plus the relative L2 difference and both
  iteration/residual figures.

### Cross-platform build (macOS arm64 + Linux x86-64)

- **Platforms:** the engine builds and passes its suite on macOS arm64 (clang), Ubuntu 24.04 x86-64 (g++ 13; 763/763, ~45 min on a 4-core VM) and Windows 11 x86-64 (clang-cl 19.1.5 / MSVC ABI; see the Windows section below).
- **Portable random numbers:** `<random>` distributions differ between standard libraries, so the RANSAC and test-scene generators now use our own `portable_rng`. The same seed gives the same samples on every platform. This changed the fixed random draws, and some accuracy bounds turned out to have been tuned to one lucky draw. They were widened after a multi-seed check:
  - `BundleAdjustment.ConvergesOnSingleSidedArc`: 0.02 → 0.05. Across 7 seeds the error ranges 0.024–0.098, so even 0.05 holds only for the pinned seed. The single-sided arc stays genuinely weak, as noted in M5.
  - `BundleAdjustment.FisheyeCameras`: 0.6 → 0.65 (seed spread 0.579–0.615).
  - `AlignCamerasSequence`: camera-centre chord error 0.5% → 0.8% (measured 0.691%); rotation 1.5° → 2.5° (measured 1.93°). Alignment on the 8-image arc is about 40% worse on this draw than the old bounds implied.
- **`-ffp-contract=off`** is set globally so CPU results match across compilers. The CPU no longer fuses multiply-adds, which costs a little speed in tight float loops. GPU kernels may still fuse, so GPU-vs-CPU checks use relative (ulp-scale) tolerances.
- **Argument evaluation order is unspecified, and it broke two seeded fixtures (M5, 2026-10-21).** `Eigen::Vector3d(f(rng), f(rng), f(rng))` and `callee(f(rng), g(rng))` consume one stream in an order the standard leaves unspecified: clang evaluates arguments left to right, g++ right to left. Four tests passed on Ubuntu and failed on the Apple M5 with the *same* seed:
  - `BundleAdjustment.ConvergesOnSingleSidedArc`, `.SelfCalibrationRecoversIntrinsics`, `.CentrePriorsFixTheGauge`: the synthetic scene was built in a different draw order, so macOS saw 1140 observations where Linux saw 1192 (arc case) and every statistic shifted. Fixed with `draw2`/`draw3` in `tests/test_bundle_adjustment.cpp`, which sequence the draws explicitly **in g++'s right-to-left order**, so a seed selects the same scene everywhere *and* the existing bounds keep the scenes they were calibrated on (all 15 `BundleAdjustment.*` now pass on both, printing identical values).
  - `PatchMatch` produced different depth maps on a bit-identical synthetic scene: the initial hypothesis pass matched exactly, but the first refinement sweep diverged because `evaluate_hypothesis(ctx, local, random_depth(rng, ...), random_cone_normal(rng, ...), ...)` consumed the stream in a different order. A greedy discrete optimiser amplifies that one flip into 2000/43200 differing depth-map pixels (median 0.013 m), then a different mesh (19999 vs 20000 faces), and finally `BuildTextureEndToEnd.BakesTheSceneOnTruthPoses` landing on opposite sides of its bound. Fixed by sequencing the draws in `stereo/patch_match.cpp`; the depth, normal and cost arrays are now bit-identical on both platforms.
  - **Rule: any call that draws twice from one engine must sequence the draws into locals first.** `src/` is clean (the only engine draws are `distinct_indices`, one call per sample). The follow-up sweep converted every remaining fixture and added a guard, so the rule is now enforced rather than remembered:
    - **Endpoints fixed** (draws sequenced in g++'s right-to-left order with `draw2`/`draw3`/`draw4`, now shared in `tests/test_rng_helpers.hpp`): `test_absolute_pose.cpp` (rotation axis, pose translation, world points, RANSAC noise), `test_two_view.cpp` (unit ray, noise direction, planar scene), `test_similarity.cpp` (quaternion rotation, cloud, translation, noise), `test_structure_builder.cpp` (point cloud, pixel noise, private cloud), `test_five_point.cpp`, `test_view_selection.cpp`, `test_triangulation.cpp`, `test_tv_fusion.cpp`, `test_tv_fusion_gpu.cpp`.
    - **`std::` distributions removed from the remaining fixtures** (`test_structure_builder.cpp` above all: libc++ and libstdc++ map engine bits differently, which is why that suite printed a worst rotation of 0.0498 vs 0.0624 and 277 vs 279 corrupted tracks from the same seed). Also `test_two_view.cpp`, `test_absolute_pose.cpp`, `test_similarity.cpp`, `test_five_point.cpp`, `test_triangulation.cpp`, `test_view_selection.cpp`, `test_tv_fusion*.cpp`, `test_aerial_fixture.cpp`, `test_atlas_pack.cpp`, `test_depth_filter.cpp`, `test_interpolate.cpp`, `test_tgv_denoise.cpp`, `test_gpu.cpp` and `test_project_roundtrip.cpp`. All now draw through `portable_rng`.
    - **Guard:** `tools/check_rng_order.py` flags two-or-more engine draws in one expression group, any `std::*_distribution` and `std::shuffle`, and runs as the `rng_order_check` ctest over `src/` and `tests/`. It understands that braced initialisers *are* sequenced ([dcl.init.list]/4 — verified identical on clang and g++), so `P p{f(), f()}` is accepted while `g(f(), f())` is not; a legitimate multi-draw statement carries a `// rng-order: ok` marker. `test_matcher.cpp`'s `std::shuffle` became Fisher-Yates over `rng::index_below`. On the touched suites both platforms now print the same numbers (`NoisyRecovery` 0.00521144, `structure_builder` worst rotation 0.0466 deg / centre 0.0134 / rms 0.6565, 276 corrupted tracks filtered 276, `RansacRecoversPoseWithOutliers` 72 iterations / 186 inliers, `RansacWithOutliers` 122 inliers).
    - **Linux values changed where a fixture switched from `std::` distributions to `portable_rng`**, by design: that is the fix (the old Linux values were not reproducible anywhere else). Concretely, `AbsolutePose.RansacWithOutliers` went from `inliers=124 iters=50` to `inliers=122 iters=50`, `TwoView.RansacRecoversPoseWithOutliers` from `86` to `72` iterations, `NoisyRecovery` from 0.00495362 to 0.00521144, and `StructureBuilder` from 0.0624 deg / 0.0170 / rms 0.6616 / 279 corrupt tracks to 0.0466 deg / 0.0134 / rms 0.6565 / 276. Every bound still holds on both platforms with margin (recall/precision remain 1.0; the rotation bound is 0.1 deg against 0.0466). Where a fixture already drew through `portable_rng` and only the argument order was fixed, the printed values are unchanged (`AbsolutePose.RefinementImprovesNoisyPose` before=0.517453, `Triangulation.*`). No bound was loosened; the touched suites run 79 tests, all passing on both platforms, and both full suites are 763/763.
- **The texture gain solve is a robust fit, not plain least squares (same run).** `BuildTextureEndToEnd` requires the mosaic page and the disabled page — identical view assignment, gains only — to agree within 1.0. On the `seq` fixture (one albedo field, no exposure differences) the plain normal equations returned gains 3% off 1.0 (0.969–1.032) and made the mosaic page disagree with the photographs *more* than the uncorrected page did: the gap measured 1.07 against the 1.0 bound. The equations are mostly "these two views agree", but a minority are dominated by silhouettes, grazing views and surface-vs-photo parallax, and unit weights let those steer the fit. `solve_gains` now runs 8 fixed IRLS sweeps with a Huber weight (threshold 0.10 in log units, ~10% disagreement), pulling the seq gains to within a fraction of a percent of 1.0 (mosaic 4.69 vs disabled 4.33 on the M5; 4.64 vs 4.38 on the VM) while a genuinely darkened view still moves (`Bake.ColorCorrectionLevelsViews`: seam deviation 0.040 → 0.005). The sweep count is fixed rather than convergence-tested, so the result stays deterministic.
- **`measure_texture` sampled texels with `std::uniform_int_distribution` (same run).** That distribution is specified only by its output range, so libc++ and libstdc++ selected different texels from the same seed and the two platforms' coverage/consistency statistics were not comparable. It now uses `rng::index_below`, as every other sampler does.

- **Timing budgets** were widened for the slower 4-core VM (ortho 5 → 30 s, Poisson 60 → 120 s, TGV → 60 s). Each one is a wall-clock guard, not a performance target.
- **Shader compile memory:** `glslc` is held to one job at a time, because the large kernels peaked above 5 GB each.
- **Bugs found by the port:**
  - `tvl1_common.glsl` included itself recursively, so the TV-L1 shaders had never compiled. That likely explains the runaway `glslc` memory and possibly the M1 crashes.
  - The GeoTIFF stripped writer let libtiff's in-place predictor corrupt the caller's raster.
- **TV-L1 GPU declines software devices:** `tv_l1_fusion_gpu_supported()` is false on a CPU rasteriser (llvmpipe, lavapipe, SwiftShader, softpipe, swrast) even though the layer itself is available. The dual kernel packs the Chambolle–Pock dual field as fp16 (`packHalf2x16`), and Mesa's LLVM backend *aborts the process* compiling that pattern ("Cannot select: ... anyext from i16") rather than returning a `VkResult`, so `with_fallback` cannot catch it. The decline therefore happens before any tvl1 buffer, kernel or pipeline exists: `gpu::failure_count()` is untouched and the CPU solver returns its own field bit for bit. `tvl1_dual.comp` needs no 16-bit storage extension (`packHalf2x16` is core GLSL) and all accumulations stay fp32; the packing only halves the dual stream. `tv_l1_fusion_gpu_fp16_available(Context&)` exposes the per-context capability, and `TvFusionGpu.HardwareDeclaresFp16` probes it.
- **Test exclusions: none.** `PoissonGpu.*` and `TvFusionGpu.*` are both registered. `TvFusionGpu.*` runs everywhere: the arithmetic/agreement cases take the CPU-fallback path on a software device, and the throughput, fp16 and device-accuracy cases skip there.
- **TV-L1 GPU field drift (WP-10.1b), fixed.** The first hardware run on M1 measured a mean field difference against the CPU solver of **0.0505** against the 0.02 bound (200 iterations). Cause: a "frozen voxel" culling optimisation skipped whole x-blocks whose voxels were unmeasured with a constant 3x3x3 neighbourhood, on the claim that such a voxel is a fixed point. It is only a fixed point while its neighbours stay constant — a change two voxels away reaches it within a few iterations, and a culled voxel then never moved, so the GPU field drifted from the CPU's in the unmeasured region. The fix dispatches every x-block and asserts nothing about individual voxels.
  - After the fix, M1: field mean |diff| **6.3e-05**, max **1.3e-03**, every accuracy test passing.
  - After the fix, Apple M5 (`TvFusionGpu.*`, 11/11 pass): `[tv gpu agreement]` field mean |diff| **6.152e-05**, max **1.333e-03**; sphere level set median **0.055** / p95 **0.214** voxel on both paths (362/400 rays supported); end-to-end `BuildModel` sphere mesh median 0.157 / p90 0.415 voxels with identical faces and boundary-edge count to the CPU.
  - Throughput with every block resident: 96^3, 8 bins, 300 iterations — GPU **0.100 s** vs CPU **0.496 s** (**4.96x**); 128^3, 16 bins — GPU **0.306 s** vs CPU **1.555 s** (**5.08x**); the end-to-end 192^3 fusion runs in 0.30 s against the CPU's 0.94 s. The kernels are memory-bound and the culled blocks were a small share of the working set, so removing the culling cost little.
  - A related indexing defect surfaced on the same run: the block list stored flattened `row * x_blocks + block` ids while both kernels read the entry as a row index (`index = row * nx + x`). The two agree only when `x_blocks == 1` (`nx <= 128`), which is why every sphere/unit case passed; the 192^3 end-to-end grid has `x_blocks == 2` and the kernel read/wrote past the buffer, so Metal raised a GPU address fault and the device was lost. With nothing culled any more the list is the identity over rows.

## M11 — Multi-instance containers and project format v2 (WP-11.1 – 11.4)
### Windows build (M29 WP-29.1 first run)

Built natively on the owner's Windows 11 machine (Ryzen 5 5500U, 6C/12T, 8 GB)
with **clang-cl 19.1.5 from the Visual Studio 2022 17.14 LLVM component + LLVM
libomp, MSVC ABI**, Ninja, `-j4`, `ALPHASHAPE_WARNINGS_AS_ERRORS=ON`. Full
toolchain versions and commands are in `docs/plan/WINDOWS_TOOLCHAIN.md`.

- **Build: zero warnings, zero errors.** The full warning set (`-Wall -Wextra
  -Wpedantic -Wshadow -Wold-style-cast -Wconversion -Wsign-conversion
  -Wdouble-promotion -Wformat=2`, as `-Werror`) compiles every in-tree target.
- **Suite: 710 passed, 50 skipped, 1 disabled, 0 failed in 12.9 minutes** (single
  process, `OMP_NUM_THREADS=4`). The 50 skips are exactly the GPU suites —
  `ALPHASHAPE_GPU=OFF` in this WP, because `alphashape_vulkan.cmake` has no
  Windows `glslc.exe` branch yet (WP-29.2). `PatchMatchGpu.*`, `PoissonGpu.*`
  and `TvFusionGpu.*` skip with "GPU support disabled at build time". The one
  disabled test is `BuildDemPipeline.DISABLED_DepthMapDiagnostic`, which is
  disabled on every platform by design.
- **`Platform.*`: 10/10 pass** — temp directory, 10 000 unique temp paths, a
  dynamic-library symbol lookup and a missing-library error, move semantics on
  the handle, console tty detection, child-process capture (output, exit code,
  large output without deadlock, missing executable), and the environment
  round trip.
- **`alphashape-cli --list-tasks`** runs and lists the registry.
- **Three first-run defects, all portability bugs, all fixed:**
  1. **`ceres` had no sparse backend.** vcpkg's default `ceres` is built with
     `EIGENSPARSE=OFF` and `SUITESPARSE=OFF`, so `sparse_linear_algebra_library_type`
     is `NO_SPARSE` and `ceres::Solve` *terminates the process* on the default
     `SPARSE_SCHUR` linear solver: 23 tests in `BundleAdjustment.*`,
     `StructureBuilder.*`, `AlignCameras*`, `OptimizeCameras*` and
     `FullPipelineDenseNoWorseThanSfm` failed that way. Fixed by installing
     `ceres[eigensparse]` (which turns `EIGENSPARSE` on and pulls METIS). The
     manifest in WP-29.2 must carry that feature explicitly — the default port
     does not.
  2. **Two tests removed files while still holding an `ifstream`.**
     `ExportDem.WorldFileTextWhenEnabled` and `ExportOrthomosaic.WritesGeoTiff`
     read a `.tfw` and then deleted it. POSIX unlinks an open file; Windows
     fails with "the file is being used by another process". Both now close the
     stream before removing.
  3. **`read_image(fs::path)` in `test_build_texture.cpp`.** libc++ and libstdc++
     give `std::filesystem::path` an implicit conversion to the native string
     type; MSVC's does not. Made the `.string()` conversion explicit. (The test
     still passes on macOS and Linux.)
- **Two flag spellings differ under clang-cl** and are handled in
  `cmake/alphashape_platform.cmake`, not per-file: `-ffp-contract=off` must be
  `/clang:-ffp-contract=off` (the driver rejects the bare form, and `-Xclang`
  does not survive clang-cl's parser), and every clang-style warning flag needs
  the `/clang:` prefix because a leading-dash `-Wall` is MSVC `/Wall` (every
  warning, including `-Wc++98-compat`, which is fatal under `/WX` on C++20
  source). `_USE_MATH_DEFINES` is defined for the UCRT's hidden `M_PI`.
- **`-ffp-contract=off` is in effect**, verified with `clang-cl -###` (the flag
  appears in the cc1 command line). The MSVC `/fp:precise /fp:no-contract`
  fallback from M29 §4.1 was **not** needed: clang-cl + libomp configured,
  built and passed the suite on the first lane.
- **The Vulkan loader keeps its `DynamicLibrary` for the process lifetime.**
  The first version of the port released the module at the end of candidate
  probing, which crashed `ExportOrthomosaic.WritesGeoTiff` on Linux
  (`has_instance_extension` jumped into a freed module). Every Vulkan entry
  point is a code address inside the module, so the handle must outlive the
  function table; it is now held in `detail::Loader` and closed by RAII at exit.
- **The Windows lane is `ALPHASHAPE_GPU=OFF` and `ALPHASHAPE_BUILD_PYTHON=OFF`.**
  The GPU branch (`vulkan-1.dll` via `DynamicLibrary`, no baked ICD, `glslc.exe`
  from the SDK) and the `.pyd` build are WP-29.2.
- **Not yet exercised on Windows:** the GUI (not in this tree), the Python
  module, GPU parity (numbers above come from the M1 and the Linux VM), and the
  installer. `WP-29.3` owns numeric parity: the 710 passing tests include every
  accuracy bound unchanged — **no bound was widened for Windows**.

### Windows installer (M29 WP-29.7)

- **Unsigned.** There is no code-signing certificate, so SmartScreen shows its
  "unrecognised app" warning on first run of the installer and of
  `alphashape.exe`. `packaging/windows/README-smartscreen.md` says what that
  means and how to verify the download against the published SHA-256.
- **The GUI is a console-subsystem binary**, so starting it from a shortcut
  opens a console window beside the app. Making it windowed needs the WinMain
  entry point and removes the stdout that the packaging smoke test and
  `GuiShell.StartsOffscreen` read; that is a separate change.
- **The Python module needs a matching CPython installed.** It ships as
  `python\alphashape.cp314-win_amd64.pyd` on the current CI image, because CMake
  tags a pybind11 module with the ABI it was built for; Python imports that name
  as `alphashape`. A user without Python, or with another minor version, cannot
  import it. The version comes from the vcpkg toolchain's own python3 port,
  which is the only interpreter on the runner that supplies
  `Development.Module`; `dist/PYTHON-ABI.txt` records it and the smoke job
  installs that same minor version. A bundled standalone interpreter is the fix
  for users (roughly 30 MB plus its licence texts) and is not in this WP.
- **No server or worker component.** The installer ships the GUI, the CLI and
  the Python module; the distributed worker is Phase 14.
- **No GPU path in the Windows build**, so the installer ships no Vulkan loader
  or ICD (WP-29.2).
- **Component selection is coarse:** deselecting the desktop app still installs
  the shared runtime, PROJ data and licences (the required `core` component),
  and no per-component record is kept, so a later installer cannot remember the
  previous selection. Uninstall removes everything that was staged.
- **PROJ ships `proj.db` but no grids.** vcpkg's `proj` port bundles the
  database and not the grid files, so the installer's grid set is empty and a
  transform that needs a geoid model fails offline with PROJ's "grid not found"
  rather than downloading (which is deliberate: `PROJ_NETWORK` is forced off).
  The size cap that would apply is 32 MiB per file and 64 MiB in total, smallest
  first, and the configure says which case it took. `ALPHASHAPE_PROJ_DATA` can be
  pointed at a full `proj-data` directory to supply them.

## M11 — Multi-instance containers (WP-11.1 / WP-11.2)

- **Every container is now persisted (WP-11.3).** Markers, marker groups, scale
  bars, camera groups, masks, shapes (features, layers, attributes, groups),
  trajectories, the panorama and tiled models are written to `doc.xml` v2 and
  read back; `ProjectV2.RoundTripsAllContainers` compares the reloaded chunk to
  the original with `Chunk::operator==`. They are **not yet bound to Python**
  (WP-11.11), so the Python surface still exposes only the derived-product
  asset lists (`elevations`, `orthomosaics`, `models`, `point_clouds`,
  `depth_maps_sets`) as read-only id/label lists.
- **`Document::kFormatVersion == 2`; v1 files are still read.** The writer emits
  only v2: `<document version="2">` with every chunk container, all assets of
  every kind, and bulk entries under `chunks/<chunk_id>/...`. `doc_xml.cpp` is a
  thin dispatcher: `version == 1` goes to `doc_xml_v1.cpp` (the pre-M11 layout,
  flat entry names, exactly as it read before the split), `version == 2` to
  `doc_xml_v2.cpp`, and `> 2` or a missing version raises `ProjectIOError`.
  Loading a v1 project and saving it writes v2
  (`ProjectV2.ReadsVersionOneAndResavesAsVersionTwo`) — one way only.
- **Deterministic writer.** Attributes and children are emitted in a fixed
  order with two-space indent and LF, so loading a project and re-saving it
  untouched reproduces the same archive byte for byte
  (`ProjectV2.UntouchedChunkResaveIsByteStable`). Containers are written only
  when non-empty (`ProjectV2.EmptyContainersAreOmitted`), and every binary
  entry is magic-tagged (`AKP1`, `AMT1`, `ATR2`, `ADM1`, `APC1`, `AMS1`/`AMS2`,
  `AOL1`, `AMK1`, `ASH1`, `ATJ1`, `AFR1`, `ATM1`), so a decoder cannot be
  pointed at the wrong layout silently.
- **v2 is strict inside `<chunk>`.** An unknown *element* is an error (it catches
  writer/reader drift); unknown *attributes* are ignored, per §4.4.
- **`ReadReport` is the reader's diagnostic channel (WP-11.4).**
  `read_document(archive, ReadReport* = nullptr)` reports the version it read and
  the conditions it tolerated: an unknown attribute (every element's list is
  enumerated explicitly), a root-level element from a newer revision (inside
  `<chunk>` the same thing is an error), a `<frames>` element with no
  `frames.bin`, a `<panorama>` with no image, and an **absolute** frame path
  that no longer exists. Relative frame paths are not resolved here — that is
  `resolve_frame_paths` in WP-11.9, and doing it twice would duplicate the
  warning. Warnings cap at 256 entries so a pathologically repetitive document
  cannot grow the report without bound. `read_document` normalizes every
  non-`ProjectIOError` failure (a duplicate chunk id, an allocation failure) into
  `ProjectIOError`, so the caller has exactly one type to handle.
- **`write_document` writes v2 only.** The `version` argument exists so callers
  can state what they expect; anything but `Document::kFormatVersion` raises
  rather than silently writing v2.
- **v1 compatibility is verified against the writer that shipped.** The
  pre-M11 writer lives on in `tests/v1_writer.hpp` (test-only), and
  `tests/fixtures/v1_project.zip` (10 KB) is generated by
  `tests/make_v1_fixture.cpp`. `ProjectV2.V1FixtureBytesMatchTheGenerator`
  regenerates it in memory and compares byte for byte, so the checked-in file
  cannot drift from the builder; `LoadsV1Fixture` reads it, and
  `LoadsV1RichProject` does the same on a document this build produces. A
  v1 **M9** project (`LoadsV1M9Project`) is built from `data/aerial_truth_dem`;
  it writes the DEM and the depth maps directly rather than running the stereo
  pipeline, which takes minutes per image on the CI VM.
- **Fuzzing covers the archive, the XML and every sidecar (WP-11.4).**
  `ProjectV2.FuzzCorruptArchives` runs 10 000 deterministic cases (seed
  `0x5eed`, 2 000 under `ALPHASHAPE_SANITIZED_BUILD`) across six mutation
  families — truncation, `doc.xml` bit flips, random attribute values, a renamed
  sidecar, an oversized declared count, and payload bit flips — and requires
  every case to load or raise `ProjectIOError`; nothing else escapes and both
  outcomes must occur. The corpus adds ~13 s to the suite (14 s under ASan), so
  run it alone when timing the build.
- **Hardening the fuzzer drove (each is a real defect it found):**
  * `ZipArchive::parse` accepted a deflate entry whose declared uncompressed size
    was impossible for its compressed length and allocated that buffer before
    inflating. It now rejects anything above Deflate's 1032:1 bound
    (`ProjectV2.RejectsImpossibleDeclaredEntrySize`).
  * `parse_xml` recursed without a depth limit, so thousands of nested tags
    overflowed the stack. It stops at 64 levels.
  * `decode_mask` reserved `width * height` from a 17-byte header — terabytes for
    a hostile pair of dimensions. The claim is now bounded by the remaining
    bytes times the 255-pixel maximum run length.
  * `read_geotiff_info`/`read_geotiff_float` only rejected rasters above 4 TiB.
    A GeoTIFF's dimensions are fixed-width tags, so a tiny file could demand a
    multi-gigabyte allocation. The per-side limit is now 2^20, the payload limit
    4 GiB, and the declared geometry must be plausible for the file's size.
  * A duplicate chunk id reached the caller as `ValueError`; it is a malformed
    project, so `read_document` now reports `ProjectIOError`.
- **Single slots are now the ACTIVE asset (WP-11.2).** `point_cloud()`,
  `model()`, `model_textures()`, `elevation()`, `orthomosaic()` and
  `depth_maps()` keep their signatures and return the *active* asset of their
  kind. The non-const overloads lazily create and activate the first asset when
  the kind is empty, so `chunk.elevation() = dem` still works. Every producing
  task (BuildDepthMaps, BuildPointCloud, BuildModel, BuildDem,
  BuildOrthomosaic) now **appends** a new labelled asset instead of
  overwriting, and BuildTexture bakes into the active model's `textures`.
- **Multi-asset on disk (WP-11.3).** Every asset of every kind is written, each
  with its id, label, group, enabled flag and metadata, and exactly one carries
  `active="true"` per kind. Saving a chunk that holds several assets no longer
  drops the non-active ones. A v1 project, which stored only the active asset,
  still loads that one (`asset_id`/`asset_label` are honoured by the v1
  reader).
- **The v2 write path validates the containers it writes.** `write_v2` runs the
  shared sidecar validators (`validate_tie_points`, `validate_dense`,
  `validate_mesh`, `validate_model_textures`, `validate_products`) plus
  `validate_m11_containers` (masks, markers and their projections/groups, scale
  bars, camera groups, shapes and layers, trajectories, frames, tiled models,
  active ids, and cross-container id uniqueness) before emitting a byte, and the
  reader runs the same set after assembling a chunk. `Chunk::validate()` itself
  remains the broader advisory check (it also enforces `id < next_id()`).
- **`TiledModel` is opaque bytes.** The tiled-model builder and its SLPK/3D
  Tiles writer land in M21; until then the asset only carries a payload
  through unchanged.
- **A Chunk copy deep-clones asset payloads.** `AssetSlot::data` is a
  `shared_ptr`, so the implicit copy would alias payloads; `Chunk`'s copy
  constructor and assignment operator clone every asset (and the panorama
  image) instead. Undo snapshots and chunk duplication can therefore copy a
  chunk and mutate the copy freely. The cost is a full payload copy per
  snapshot — the same as the pre-M11 value-type members.

### Command layer (WP-11.5a)

- **The base layer and every command factory ship (WP-11.5a/b/c).**
  `AbstractCommand`, `StagedMutation`, `CommandTransaction`, `SpillBuffer`, the
  reworked `UndoStack` (macros, depth cap, memory policy) and all the chunk,
  marker, scale-bar, group, camera, asset and CRS factories in
  `model_commands.cpp` are complete and tested.
- **Asset commands never copy a payload.** `AssetSlot::data` is a `shared_ptr`,
  so the fingerprint of every asset command is *move*, not copy: `AddAsset`
  moves the payload into the slot it creates; `RemoveAsset` keeps the displaced
  slot whole (same pointer) and puts it back on revert. A command takes its
  payload back out of the slot on revert so that a later redo re-installs the
  identical bytes rather than a moved-from default.
- **Removing a large asset spills it (§4.7).** `RemoveAssetCommand` retains the
  displaced slot through `SlotHolder`: in RAM at or below
  `kUndoAssetSpillBytes` (64 MiB), encoded through the same sidecar encoder the
  project writer uses and written to `UndoStack::cache_dir()` above it. With a
  payload above the cap and no cache directory the command reports
  `is_memory_bounded() == false` *before* touching the document, so
  `UndoStack::execute` refuses it with `Error("command too large for the undo
  history")` and the caller re-runs instead of undoing. Spill files are
  `undo-model-<pid>-<n>.bin` in the cache directory and are deleted when the
  command dies (`Stack::clear()`, undo-depth eviction, stack destruction).
- **`AddAsset` can land complete or empty.** The `(kind, chunk, label)` overload
  adds an empty, active slot for a caller that fills it immediately; the typed
  overloads (point cloud, model, depth maps, elevation, orthomosaic, tiled
  model, panorama) carry the payload a task returned and complete the slot
  inside `apply`, which is the §4.7 "a task does not push itself; the caller
  wraps its result" shape.
- **Panorama is not vector-shaped.** It is a single optional slot with no active
  id, so `AssetKind::Panorama` is served by `PanoramaCommand` (add / rename /
  remove) and `make_set_active_asset_command(AssetKind::Panorama, …)` throws
  `ValueError`. Typed `AddAsset` refuses a second panorama; `RemoveAsset` carries
  the panorama id and goes through the same `SlotHolder` spill policy as the
  vector kinds.
- **`make_set_chunk_crs_command` takes `CRS`, not `crs::CoordinateSystem`.**
  The plan's §4.7 signature is written against `crs::CoordinateSystem`, which
  does not exist until M12 WP-12.11. The command, the undo wiring and the
  `Chunk::crs()` round-trip ship now over the existing `CRS` descriptor (WKT +
  EPSG); M12 only has to widen the argument type and re-point the factory.
  `validate` refuses a CRS with neither WKT nor a non-negative EPSG code.
- **Staging is container-scoped, not a whole-chunk copy.** A `Chunk` copy
  deep-clones every asset payload, so a marker edit would copy every DEM and
  model twice. `ChunkContainersCommand` snapshots only markers, marker groups,
  scale bars, camera groups, cameras and the id counter; chunk-level commands
  snapshot the chunk vector (shared pointers, no payload copies) and the two
  counters. Commands outside those sets must add their container to
  `ChunkSnapshot` — undo correctness depends on it.
- **Chunk-level ids step backwards on undo.** `AddChunk`/`DuplicateChunk`
  restore `Workspace::next_chunk_id()` and `Chunk::allocate_id()` restores the
  chunk counter on revert, so a redo reuses the same ids. That is what keeps a
  redo id-stable, but it means an id handed out by an undone command can be
  handed out again: code must not cache ids across an undo.
- **Chunk removal keeps one chunk.** Removing the last chunk of a workspace
  throws `ValueError` rather than leaving a chunk-less document (the v2 writer
  and `Chunk::validate()` assume at least one).
- **Group commands cover marker and camera groups only.** `GroupKind` declares
  `PointCloud` and `Model`, but `Chunk` has no `point_cloud_groups()` /
  `model_groups()` container yet (the plan's §4.2 aliases exist, the storage
  does not), so those two kinds throw `ValueError`. They are wired up when the
  containers land (with the Python `addPointCloudGroup`/`addModelGroup`
  bindings in WP-11.11).
- **`Camera::selected` is in-memory only.** WP-11.5b adds the flag, compares it
  in `Camera::operator==` and edits it through
  `make_set_camera_selected_command`, but the v2 writer does not emit a
  `selected` attribute yet — persisting it is WP-11.8, together with the rest
  of the camera fields (`group`, `mask_id`).
- **`AppendDocument` copies the chunks and drops the source path.** Chunk ids
  are offset by the target's `next_chunk_id()` and chunk-local ids are
  preserved (§4.8), but the `source_path` argument is currently only used in
  error messages; recording it in `Workspace::meta` (as the plan sketches) is
  left to the `AppendProject` task (WP-11.6), which is the real entry point.
- **`DuplicateChunk` now REMAPS ids (WP-11.6 replaced the verbatim copy).**
  The 11.5b command used to keep the source's chunk-local ids and `next_id()`;
  it now routes through `core/chunk_ops.hpp`'s `duplicate_chunk`, so a copy can
  be merged with its source without an id collision and every internal reference
  (camera -> sensor, marker -> group, mask/depth map/orthomosaic label ->
  camera, model -> source asset, tie-point camera ids) points at the copy's own
  objects. `Undo.DuplicateChunk` and `ChunkOps.DuplicateIsUndoable` pin both the
  remapping and the exact undo/redo state.
- **Marker projections are normalised on add, not rejected.** A caller-built
  `Marker` may arrive with projections out of camera order (a GUI appends the
  one just clicked); `make_add_marker_command` sorts them into place. A
  repeated camera or a non-finite pixel is an error and nothing is staged.
- **`make_set_chunk_crs_command` is not declared.** Its parameter type,
  `crs::CoordinateSystem`, arrives with M12 WP-12.11; declaring a factory over
  a type that does not exist would not compile. The factory is added there,
  together with the rest of the CRS wiring.
- **The spill policy is implemented as a reusable buffer, not wired into
  commands yet.** `SpillBuffer` enforces the §4.7 rule (payloads above
  `kUndoAssetSpillBytes` = 64 MiB go to the cache directory; without a cache
  directory the payload is unbounded and `UndoStack::execute` refuses the
  command with `Error("command too large for the undo history")`). Asset
  commands in 11.5c are the first callers. `UndoStack::set_cache_dir` is the
  plumbing the caller must feed (`<base>.files/cache`, §4.9); the project
  layout WP (11.9) computes it, so until then it is empty and oversized
  payloads are refused rather than spilled.
- **`SpillBuffer` handles `Image` only.** That is the one payload with a
  self-describing byte layout available today (panorama assets are the `> 64
  MiB` case in practice). Point-cloud, mesh and depth-map payloads are not
  spillable yet; an asset command that replaced one above the cap reports
  itself unbounded. Generalising the buffer is part of 11.5c.
- **The depth cap drops history silently except at the wall.**
  `kMaxUndoDepth == 100`; dropped entries are counted (`dropped_count()`) and
  `undo_label()` appends `" (older steps dropped)"` **only** on the oldest
  retained step, which is where the user would first notice. There is no
  prompt and no "cannot undo further" dialog — the stack simply has nothing
  beneath it.
- **`dump_undo_stack` is a text aid, not a serializer.** `command_json.*` is
  debug-only (tests and crash reports); the name follows the plan's vocabulary
  but the output is the human-readable form, not JSON, and nothing parses it
  back.
- **Undo/redo is refused while a macro is open.** `begin_macro` applies its
  children immediately (that is what makes a macro usable in a UI event
  handler), so an undo mid-macro would have to reverse half a group. Rather
  than that, `undo()`/`redo()` throw `Error`. An unclosed macro is closed by
  the `UndoStack` destructor, innermost first, so no applied edit is left off
  the stack.
- **`StagedMutation` rolls back `commit()` failures, not `validate()`/
  `stage()` ones.** Both are required to be non-mutating, so a throw there
  leaves the document untouched by construction. A `commit()` that throws is
  restored from the staged snapshot and the error propagates;
  `CommandTransaction::apply` likewise reverts the children already applied
  before rethrowing.


### Chunk operations (WP-11.6)

- **`chunk_ops` lives in core, not algorithms.** The undo command layer needs
  the same id remapping, and core must not depend on algorithms, so
  `src/common/core/chunk_ops.{hpp,cpp}` owns `duplicate_chunk`, `append_chunk`,
  `crs_compatible` and `merge_chunks`. The task adapters under
  `common/algorithm/chunk/` are thin parameter/progress/registration wrappers.
- **Three id namespaces, not one.** Camera ids and frame ids are independent
  counters; every other object (markers, groups, scale bars, shapes,
  trajectories, assets, masks keyed by camera) draws from
  `Chunk::allocate_id()`. A copy remaps cameras and frames into the
  destination's `max(id) + 1` range and takes fresh chunk-local ids for
  everything else, so appending to the end keeps each container sorted.
- **A chunk with no sensors keeps `camera.sensor_id` verbatim.** The remap
  throws when a camera references a sensor the source does not have *and* the
  source has sensors at all. A sensor-less chunk (a bare test fixture, or an
  import that never resolved EXIF) has nothing to translate, so the field is
  carried through exactly as a `Chunk` copy would; otherwise the WP-11.5b
  `Undo.DuplicateChunk` fixture, which builds cameras without sensors, broke.
- **Sensor dedup is exact-identity only.** `(type, width, height,
  pixel_size_um, focal_length_mm, make, model, serial, calibration)`; `label`
  and `fixed_params` are presentation and are deliberately excluded. Two
  sensors that differ only in their fixed-parameter set therefore stay two
  sensors, which is the conservative direction the plan's risk note asks for.
- **Merged regions are the union of the sources' boxes; everything else comes
  from the first source.** CRS, transform and metadata are the first source's;
  camera poses are copied untouched, so a camera centre is preserved to
  bit-exact equality (the test allows 1e-12).
- **`AppendProject` preserves chunk-local ids and offsets only chunk ids.**
  Incoming chunk ids land at `destination.next_chunk_id() + source id`, which
  makes them collision-free with everything the destination has ever allocated
  (also across `Document::save(path, chunks)`).
- **`Document::save(path, chunks)` writes a v2 file holding only the selected
  chunks, with the source's `next_id` unchanged.** The chunk objects are shared
  with the in-memory document, not copied, and the active chunk falls back to
  the first kept chunk when the selection excludes it. The writer still runs
  its full container validation, so a subset save fails on the same defects a
  full save would.
- **`AppendProject`'s `append_doc` only fills gaps.** The source's global CRS
  is taken when this document has none, and each source metadata key is
  adopted only when this document does not already set it; nothing is ever
  overwritten.
- **`MergeChunks` with an empty `chunk_ids` merges the whole workspace** and
  refuses fewer than two chunks; a single-chunk "merge" is an error, not a
  copy, so a mis-bound parameter is never silent.

### Chunk alignment (WP-11.7, `AlignChunks`)

- **Marker-based and reference-based only.** `method="markers"` matches markers
  by exact `label` and needs each one's *estimated* `position` in its own chunk;
  `method="reference"` requires both chunks to be georeferenced in the same CRS
  and records the identity placement. Point-cloud-based alignment (AL-16 in
  full) is M17; there is no ICP or nearest-neighbour path here.
- **The similarity is Umeyama, reused not re-implemented.**
  `sfm::estimate_similarity_ransac(src, dst, threshold, 0x5eed)` from
  `algorithm/sfm/similarity.*` — the same solver `Georeference` and
  `AlignCameras` use. `src` is the ACTIVE chunk's marker positions, `dst` the
  REFERENCE chunk's, so the fit maps the active chunk into the reference frame.
- **Threshold, floor and gate.** The RANSAC inlier band is `3 x median sigma`
  over each pair's mean position sigma (from `Marker::position_covariance`,
  largest eigen-axis), floored at 0.01 m because zero covariances are common and
  a zero band would reject every point. `accuracy_m = 0` sets the residual gate
  to that same band; a positive value overrides the gate but not the band.
- **Camera poses carry the scale.** A similarity is not rigid, so the chunk's
  poses become `pose . S^-1` with `S = [sR | t]`; `A = pose[:3,:3]` is therefore
  `R/s`, and anything computing a camera centre must use `-A^-1 b`, not
  `-A^T b`. `Camera.center` in the Python bindings and the tests were updated
  for this; a caller-side helper that assumes orthonormality will be wrong by
  the scale factor after an alignment.
- **What moves:** camera poses, triangulated track points, marker positions,
  shape vertices, trajectory positions, point-cloud points and normals, mesh
  vertices, the region (axis-aligned bound of the mapped box corners), the
  chunk transform (`T' = T . S`) and the chunk CRS. Depth maps, orthomosaics,
  elevations and texture pages stay in the old frame: they are products of the
  pre-alignment geometry and rebuilding them is the honest fix. Mask and
  keypoint data are image-space and unaffected.
- **The edit is one command pair, not one command.** `make_align_chunks_command`
  installs the geometry, `make_set_chunk_meta_command` writes the reference
  chunk's record; the task wraps both in a `CommandTransaction`. Callers that
  want undo must keep the transaction (or use the same factories). The
  command-based record keys are `AlignChunks/<other chunk id>/matrix` in BOTH
  chunks (active → reference similarity, reference → its inverse), 16
  space-separated doubles, 17 significant digits, row-major. They are results,
  not task parameters, so `test_parity.py`'s drift check counts only
  two-segment `"<Task>/<param>"` keys.
- **Undo retention is unbounded for large payloads.** `payload_bytes()` reports
  the snapshotted point-cloud and mesh payloads, but `AlignChunksCommand` does
  not spill: `SpillBuffer` handles `Image` only (see above), so an alignment of
  a chunk with a multi-hundred-MB cloud retains it in RAM between apply and
  revert. For very large chunks, run the task and then clear/redo rather than
  keeping the entry, until the buffer is generalised.
- **Degenerate cases are refusals, never partial edits.** Fewer than
  `max(3, min_common_markers)` matched markers, a collinear set (Umeyama's own
  second/first singular value test, `> 1e-6`), a non-positive fitted scale, or
  an inlier RMS above the gate all throw before the document is touched. A
  block of exactly three markers cannot be outlier-checked: RANSAC falls back to
  the plain fit and the RMS gate is what rejects it.
- **`reference_chunk_id` defaults to the first other chunk.** That is
  convenient for scripts and unambiguous for two chunks; a three-chunk
  workspace should pass the id. Aligning a chunk onto itself is an error.
- **No Python-side undo hook yet.** `Document.align_chunks` runs the task
  directly (like every other task); the undo entry exists on the C++ side only.
  Wiring the Python undo stack is WP-11.11.
## M12 — Coordinate systems (PROJ), in progress

WP-12.1 lands the dependency, discovery and data layout. WP-12.2 adds
`crs::CoordinateSystem` (the value type) and the 48-entry built-in list, and
WP-12.3 adds `crs::CrsTransform` between two of them (axis-normalised, ballpark
refused by default, per-thread pipelines). Datums, geoids and raster
reprojection arrive with WP-12.4 … WP-12.12; until then the engine still uses
the legacy three-CRS stub (`geo::Projection`: local / EPSG:4326 / EPSG:326xx)
for DEM and ortho output, so a `CrsTransform` exists but no task consumes it
yet.

### WP-12.2 — `CoordinateSystem` and the built-in list

- **The built-in list is 48 codes, and §4.6's prose enumerates 56.** The plan's
  own acceptance is "exactly 48", so the list is exactly 48 and keeps every
  named category, trimming the NAD83 / UTM 11N–19N range to its endpoints and
  middle (26911, 26914, 26919) and leaving Tokyo (4301) and Pulkovo 1942
  (4284) to `from_epsg`. Nothing is unreachable: `from_epsg` is not limited to
  the list, only `builtin_crs_list()` and the Python `listBuiltinCRS` are.
- **Names come from PROJ, so the list renames itself.** EPSG:3006 is
  "SWEREF99 TM" in PROJ 9.4 (Ubuntu 24.04) and "ETRS89-SWE [SWEREF 99 TM]" in
  9.9 (Homebrew); EPSG:3035 is "ETRS89 / LAEA Europe" in 9.4 and
  "ETRS89-extended / LAEA Europe" in 9.9. The test asserts each name against
  `proj_get_name` on the machine it runs on, and the golden WKT fixture stores
  names stripped, so one fixture serves both PROJ versions.
- **`operator==` is string identity, not semantics.** It compares the local
  flag, the vertical component and then the authority name/code pair (or the
  canonical WKT when there is no code). Two different spellings of the same CRS
  — EPSG:32633 against `+proj=utm +zone=33 +datum=WGS84` — compare unequal;
  `is_equivalent_to()` is the semantic check and asks PROJ.
- **A stored WKT is re-parsed on every operation.** `CoordinateSystem` holds no
  `PJ*`: each `to_wkt`/`to_proj4`/`is_equivalent_to` builds one on the calling
  thread's context and destroys it. That keeps copies thread-safe and cheap, at
  one `proj_create` parse (~10 µs) per call. `CrsTransform` (WP-12.3) is where
  a long-lived object is the right shape: it parses the two CRSs once per
  thread and keeps the resulting pipeline.
- **WKT2:2019 output is informational between PROJ releases, not stable.** The
  same code renders differently across data releases (ensemble members, datum
  names, `ANCHOREPOCH`, optional `ID[]` blocks), so a stored WKT string is a
  lower-fidelity identity than the authority code. Projects keep both, and
  reading prefers the code; a project written by one PROJ version and read by
  another therefore lands on the same CRS, not the same bytes.
- **`CRS` is still the project DTO.** `Chunk::crs()`/`Document::global_crs()`
  keep returning the deprecated `{wkt, epsg}` pair so the doc.xml readers and
  writers did not have to change in this WP; `Chunk::coordinate_system()` and
  `Document::global_coordinate_system()` are the engine view, and setting one
  keeps the other in step. M13 removes the DTO. The conversions are declared
  unconditionally and throw in a `-DALPHASHAPE_USE_PROJ=OFF` build rather than
  being hidden behind a macro, because a class whose member set depended on a
  compile definition would be a different type per translation unit.
- **`from_name` is first-match, and "WGS 84" is ambiguous.** PROJ gives
  EPSG:4326, 4979 and 4978 the same name, so `from_name("WGS 84")` returns
  EPSG:4326 (the geographic 2D CRS everyone means) and the other two are
  reachable only by code. `builtin_crs_list()` is ordered so that this is a
  documented rule, not an accident of the table.
- **PROJ is not idempotent across `from_wkt` for every CRS.** EPSG:3857 loses
  its `ENSEMBLE` (re-parsed as a `DATUM`) and its `CONVERSION` name on the first
  re-parse, so `to_wkt()` after one `from_wkt()` is not byte-identical to the
  original authority rendering; the second generation is a fixed point, which
  is what `Crs.WktRoundTrip` asserts. Identity is preserved either way: the
  authority code, the name and PROJ equivalence all survive.
- **`from_wkt` does not preserve the input text.** It parses, then stores the
  canonical WKT2:2019 rendering, so `to_wkt()` after `from_wkt()` returns
  PROJ's normalised form rather than the caller's bytes. Equivalence (not string
  equality) is the contract for WKT round-trips.
- **A PROJ string that names a coordinate operation is rejected.**
  `from_proj` appends `+type=crs` when absent; a string that still does not
  describe a CRS (an invalid zone, a bare pipeline) throws `ValueError` rather
  than producing an object that fails later and further away.

- **PROJ data is not bundled yet.** The CRS engine reads `proj.db` and the
  geoid/NTv2 grids from the installed package (Homebrew `share/proj`, Ubuntu
  `/usr/share/proj`). `tools/stage_proj_data.sh` produces a self-contained
  `tools/proj-data/` subset (~100 MB: `proj.db`, `proj.ini`, EGM96, EGM2008,
  `ca_nrc_ntv2_0.tif`, `us_noaa_conus.tif`) for the packaged layout, but no
  build target copies it into an installer yet — that is M29. A machine without
  `proj-data` therefore runs with PROJ's error on the first grid-dependent
  transformation rather than with a bundled grid.
- **Network access is off.** `proj_context_initialize()` forces
  `PROJ_NETWORK=OFF` and every thread context has network access and grid
  caching disabled. A transformation that needs an uninstalled grid fails with
  PROJ's "File not found or invalid"; it never downloads one. This is the
  intended determinism trade-off (§4.1, §7), but it means a user cannot obtain
  a missing grid automatically.
- **One `PJ_CONTEXT` per thread.** `crs::thread_context()` is created on first
  use and destroyed at thread exit. PROJ documents one context per thread;
  sharing one across threads is not supported and is not done here. Code that
  spawns short-lived threads pays one context creation per thread.
- **Data-directory discovery is advisory, not fatal.** A `ALPHASHAPE_PROJ_DATA`
  that does not contain `proj.db` is logged and skipped rather than raising, so
  a stale environment variable cannot break startup; the consequence is that
  the process quietly falls through to a later search step (or to PROJ's own
  search) and may use a different data version than the caller intended.
  `crs::describe_environment()` names the directory that was actually chosen.
- **Version floor is 9.3, discovered at configure time only.** The CMake check
  rejects an older PROJ, but a binary run against a different `libproj` found
  at runtime (e.g. a relocated dylib) is not re-checked. `Crs.ProjVersionAtLeast9_3`
  asserts the runtime version of the linked library in the test suite.
- **`-DALPHASHAPE_USE_PROJ=OFF` is the fallback configuration.** It builds the
  legacy stub and the `Crs.*` tests are not compiled at all (rather than
  skipped), so a fallback build reports a smaller test count by design.

### WP-12.3 — the transform engine

- **A `CrsTransform` holds no `PJ*`; each thread caches its own pipeline in a
  thread-local registry.** PROJ operations are bound to the `PJ_CONTEXT` that
  created them and one context per thread is the documented rule, so the
  pipeline cannot be shared. It is created on the calling thread's first use
  and released either when that thread drops the transform or when the thread
  exits — always before the thread's `PJ_CONTEXT` is destroyed (the registry's
  constructor forces `thread_context()` to exist first, and thread-local
  destructors run in reverse construction order). `create()` builds the calling
  thread's pipeline eagerly so a broken or unavailable transform fails there,
  not inside a task.
- **The registry is keyed by the transform's address and validated by a
  `weak_ptr`.** A destroyed transform leaves its slot behind until the thread
  next looks at that address; the expired `weak_ptr` makes the thread free the
  old pipeline and rebuild rather than run a stale one, which
  `Crs.DestroyedTransformDoesNotPoisonTheThreadCache` covers. Entries for
  transforms a thread never uses again stay until thread exit.
- **Axis order is normalised, never configurable.** Every `CrsPoint` is
  east/lon, north/lat, up, via `proj_normalize_for_visualization()`. A caller
  holding authority-order coordinates (EPSG:4326 lat,lon) must swap them;
  there is no "raw" mode. `Crs.AxisOrderIsEastNorthUp` and
  `Crs.ProjectedToGeographicIsAxisNormalised` pin both directions.
- **Ballpark is refused by default, and the refusal is not a heuristic.** PROJ
  marks an operation as a ballpark (`proj_coordoperation_has_ballpark_transformation`);
  `create()` throws when the *selected* operation is one and `allow_ballpark`
  is false. The check runs twice: on the listed operation and on the pipeline
  that was actually built. The second check is load-bearing — an operation
  listed with a real accuracy can degrade to a ballpark at build time when the
  grid it needs is not installed, and that degradation must hit the same
  refusal. A pair with a real datum shift available is accepted regardless: the
  flag governs the operation actually chosen, not whether a ballpark exists
  anywhere in the list. With `allow_ballpark` the transform reports
  `is_ballpark()` and an accuracy of -1 (PROJ states none).
- **Accuracy is the operation's, not the pipeline's.** `accuracy_m()` is
  `proj_coordoperation_get_accuracy()` on the selected operation, -1 when PROJ
  reports 0 for a non-ballpark (a UTM conversion states 0 m, which is not the
  "unknown" sentinel upstream). WP-12.4 owns listing and ranking those
  operations.
- **`preferred_operation` pins an identifier PROJ already lists; it does not
  search `proj.db`.** "EPSG:16033" is matched against each candidate's
  `AUTH:CODE` (case-insensitive). An anonymous candidate (PROJ's grid
  operations often carry no id) cannot be pinned, and an unknown id is a
  creation error rather than a silent fallback to the automatic choice.
- **Out-of-domain detection relies on PROJ's per-point failure marker.** A
  single point fails when the output is not finite or `proj_errno` is set; a
  batch fails per point on non-finite outputs (`proj_trans_generic` does not
  report which point failed, so `HUGE_VAL`/NaN is the signal). PROJ is
  permissive about some nonsense input — it happily projects longitude 200° or
  latitude 89.9° outside a zone's nominal bounds, and `EPSG:4978 -> 4326`
  returns a value for the ECEF origin — so "outside the zone" is not "outside
  the domain". Only undefined inputs (latitude beyond ±90° for UTM) yield
  `nullopt`.
- **Vertical components are rejected by `create()`,** not silently ignored.
  `TransformOptions::src_vertical`/`dst_vertical` accept only the ellipsoidal
  kind until WP-12.5 can apply a geoid; a set geoid throws `ValueError`. The
  local engineering frame is likewise rejected with a pointer to `LocalFrame`
  (CRS-05, WP-12.6): a local frame is an origin plus rotation, not a CRS pair.
- **The `cs2cs` fixture is generated, not built.** `Crs.MatchesCs2cs` compares
  20 000 points against `tests/fixtures/crs/cs2cs_reference.csv`, generated by
  `tests/make_cs2cs_reference.sh` (PROJ CLI, `-f %.9f`). The file pins the PROJ
  data version it was made with; a PROJ release that changes a projection
  constant moves those last digits, and regenerating the fixture is the
  deliberate act that acknowledges it. CI needs no PROJ command-line tools.

