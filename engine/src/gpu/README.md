# GPU compute layer (`alphashape::gpu`)

Backend-neutral compute front end. The public API in `gpu.hpp` contains no
Vulkan types; algorithm code depends on `alphashape_gpu` (already linked into
`alphashape_algorithms`) and works entirely in terms of buffers, SPIR-V kernel
names and dispatches.

## What exists

- `available()` / `enabled()` / `set_enabled()` — process-wide switch, default
  from `ALPHASHAPE_GPU` (unset = on). `available()` lazily creates the default
  context; `unavailable_reason()` explains a false result.
- `enumerate_devices()` — physical devices with name, vendor/device id, memory,
  API version, workgroup limits, shared memory and subgroup size.
- `create_context(ContextOptions)` — extra contexts (e.g. `validation = true`,
  or a pinned `device_index`). `required_memory` rejects devices whose
  device-local heap is smaller than `required / 0.9`; selection is quiet first
  and logs every rejected device only when nothing qualifies.
- `Buffer` — storage buffer. `Buffer::of` / `read<T>` / `upload` / `download`.
  `make_buffer(context, bytes, MemoryKind)` picks the memory:
  `MemoryKind::DeviceLocal` (the default) is device-only memory on a discrete
  GPU and the host-visible heap on an integrated GPU (`info().integrated`),
  where both kinds are the same memory and no staging is needed.
  `MemoryKind::HostVisible` is the old behaviour. Copies to a DeviceLocal
  buffer that is not mappable stage through a per-context 64 MiB host-visible
  ring, chunked for larger transfers; host copies wait for any in-flight
  submission before touching the buffer, so the CPU never races the device.
  A CPU driver (llvmpipe) keeps the staged path even though its heap is
  unified, so the ring is exercised in CI.
- `Dispatch` + `run` — one command buffer, compute→compute memory barriers
  between dispatches, one submit, host wait. Throws `gpu::Error` for unknown
  kernels, a binding-count or push-constant-size mismatch, zero groups, a
  device-lost submit, and any Vulkan failure. A dispatch may mix storage
  buffers (`buffers`, filled in ascending binding order) with sampled images
  (`images` + `samplers` at explicit binding indices). Indices must be unique
  and must match the kernel's reflected interface; every image binding needs
  exactly one sampler at the same index. All of those checks run before any
  Vulkan call (see "Images and samplers" below).
- `Image2D` + `Sampler` (WP-10.2) — `Format{R8Unorm, R8G8B8A8Unorm,
  R32Float}`, `Filter{Nearest, Linear}`, `AddressMode{ClampToEdge, Repeat,
  MirroredRepeat}`.
  `Image2D::create(context, w, h, format, pixels = nullptr)` allocates a
  device-local 2D image (optimal tiling) and optionally uploads in the same
  call; `upload`/`download` move exactly `w*h*bytes_per_pixel` bytes and
  reject any other size with `gpu::Error`. Rows are tightly packed and row 0
  is the first host row. `Sampler::create(context, filter, address)` returns an
  immutable handle cached per context by `(filter, address)`, so requesting the
  same pair twice yields the same object. `image_sampling_supported(context,
  format)` reports whether the device advertises the sampled-image and transfer
  bits for the format; it logs the rejected `optimalTilingFeatures` once, and
  callers skip or fall back when it is false.
- `run_async(context, dispatches, Submission{.queue = n})` returns a `Ticket`;
  `wait(context, ticket)` blocks on its fence. `run` is exactly
  `run_async` + `wait`. At most `detail::kMaxInFlight` (4) submissions may be
  outstanding on one context: a further `run_async` throws `gpu::Error` rather
  than growing the ring. `detail::in_flight_count(context)` is the test hook.
  Ordering between submissions is the fence: `wait` on the producer's ticket
  before submitting work that reads its output. The ring is two command pools
  (slots index them by parity) and one fence per slot, so slots carry their own
  descriptor pool and command buffer; a slot is returned to the ring by `wait`.
  When the ring is full, the synchronous `run` drains the oldest ticket and
  retries; only `run_async` throws. An empty dispatch list is a no-op.
- `Submission{.queue = 1}` is honoured only when the selected compute family
  exposes at least two queues; otherwise it logs once and falls back to queue 0.
  There is no automatic splitting between queues. Cross-queue ordering is the
  caller's job via `wait`.
- `with_fallback(what, gpu_fn, cpu_fn)` — runs the CPU function when the layer
  is disabled or the GPU lambda throws; logs
  `GPU processing failed, switching to CPU mode: <what>` and bumps
  `failure_count()`.
- `compare(gpu, cpu)` — golden self-check helper (mean/max absolute error).

### Fence timeout

Every fence wait is sliced into 5 s polls and bounded: a wedged device raises
`gpu::Error("gpu: fence timeout")` instead of hanging. The total budget is 30 s
on hardware and 10 min on a CPU driver, because llvmpipe legitimately takes
minutes for a large Poisson batch and a false timeout there would turn a
slow-but-correct device into a silent CPU fallback.

### Memory kinds per algorithm

- Poisson (`poisson_gpu.cpp`): the large vectors (`p`, `p_new`, `z`, `q`, `x`,
  `r`, `inv_diag`, `weight`, `partials`) are `DeviceLocal`; the tiny scalar and
  counter buffers stay `HostVisible` so their per-batch readbacks never touch
  the ring.
- PatchMatch (`patch_match_gpu.cpp`): every working buffer stays
  `HostVisible` — the pre-decided fallback from the M10 plan, observable via
  `patch_match_gpu_prefers_host_memory()`. The whole grid is seeded from the
  host and read back each call, so DeviceLocal staging would add two grid-sized
  round trips per call without changing what the kernels compute.

Pipelines are built lazily on first use and cached per kernel. The interface of
each kernel (descriptor bindings and push-constant size) is parsed from the
SPIR-V module itself by `vulkan/spirv.cpp`; no reflection library is used.

### Images and samplers

Formats map to `VK_FORMAT_R8_UNORM`, `VK_FORMAT_R8G8B8A8_UNORM` and
`VK_FORMAT_R32_SFLOAT`; an image is device-local, optimal-tiled, one mip level
and one layer, with `SAMPLED | TRANSFER_SRC | TRANSFER_DST` usage.

`Image2D::create` rejects a non-positive size and a format whose
`optimalTilingFeatures` lack the sampled-image bit or either transfer bit, with
a `gpu::Error` and one log line naming the driver. `image_sampling_supported`
exposes the same query to callers (answers are cached per context).

Reflection decides the descriptor kind from the type chain, never the binding
number: an `OpTypeSampledImage` wrapping a 2D `OpTypeImage` in
`UniformConstant` storage is a `COMBINED_IMAGE_SAMPLER`; a bare `OpTypeImage` or
`OpTypeSampler` variable is rejected at pipeline-build time with "separate
images/samplers unsupported"; a non-2D `Dim` or a storage image is rejected the
same way. `OpTypePointer` variables in `StorageBuffer` storage stay storage
buffers.

Bindings are validated before anything is submitted: the dispatch must supply
exactly as many storage buffers and combined-image bindings as the shader
declares, every image index must be unique and must name an image binding, each
image must be paired with a sampler at the same index, both handles must come
from the same context, and the image must have been uploaded (`upload` has run)
— a never-uploaded image holds undefined contents and is refused. Any violation
is a `gpu::Error` with the offending binding in the message.

Layouts: an image lives in `SHADER_READ_ONLY_OPTIMAL` once uploaded. Upload
records `UNDEFINED -> TRANSFER_DST_OPTIMAL`, the copy, then
`TRANSFER_DST -> SHADER_READ_ONLY` with a compute-visible barrier; download
records `SHADER_READ_ONLY -> TRANSFER_SRC`, the copy, then back to
`SHADER_READ_ONLY`. Both run on a dedicated tiny command buffer + fence (not the
64 MiB staging ring) and wait for any in-flight submission first, so a dispatch
never has to record an image transition and two dispatches can read the same
image back to back with no barrier. The per-image staging buffer is temporary
and freed after its fence signals.

Unknown limits: the layer does not check
`maxPerStageDescriptorSampledImages` or `maxImageDimension2D`; a driver that
exceeds either fails at pipeline or image creation with `gpu::Error`, which
`with_fallback` converts into the CPU path.

## Loading and ICD selection

Nothing links `libvulkan`. `vulkan/loader.cpp` `dlopen`s, in order:

1. `$ALPHASHAPE_VULKAN_LIBRARY` (escape hatch for a different SDK),
2. the SDK loader baked at configure time (`libvulkan.1.dylib`),
3. `libvulkan.1.dylib`,
4. `libvulkan.dylib`.

Everything else is resolved through `vkGetInstanceProcAddr` into a function
table. Any failure is reported through `unavailable_reason()` — never abort.

MoltenVK is a portability driver, so the instance enables
`VK_KHR_portability_enumeration` (with the enumerate-portability flag) and the
device enables `VK_KHR_portability_subset` when advertised.

The SDK's MoltenVK ICD manifest path is baked into the binary. If neither
`VK_ICD_FILENAMES` nor `VK_DRIVER_FILES` is set, the layer exports
`VK_DRIVER_FILES` to that manifest for this process before creating the
instance. The standard variable was chosen over `VK_LUNARG_direct_driver_loading`
because it works with any loader, not just LunarG's.

With validation requested, `VK_LAYER_PATH` is pointed at the SDK's
`explicit_layer.d` directory when the caller has not set it, so the layer works
without environment setup. `detail::validation_active(context)` says whether the
layer actually loaded; the debug-utils messenger counts errors, and
`detail::validation_error_count()` lets a test assert zero.

## Build configuration

- `ALPHASHAPE_VULKAN_SDK` (default: `<source>/../tools/vulkan-sdk/1.4.363.0/macOS`,
  used only when it exists). `ALPHASHAPE_VULKAN_LIBRARY` / `..._ICD` /
  `..._GLSLC` / `..._SPIRV_VAL` override the individual pieces.
- `ALPHASHAPE_GPU` defaults to ON when `glslc`, `spirv-val`, the Vulkan headers
  and `vk_mem_alloc.h` are found, otherwise OFF with a STATUS message. With
  `OFF` the full API still exists: `available()` is false, every call reports
  the build-time reason, and tests skip.

`src/gpu/shaders/*.comp` are compiled by `glslc --target-env=vulkan1.2 -O` and
validated by `spirv-val` at build time, then embedded into a generated `.cpp`
as `uint32_t` arrays with a name→(pointer, words) table. Editing a shader
regenerates the blob on the next build.

## Adding a kernel

1. Drop `foo.comp` (GLSL 450, `local_size_*` declared) into `src/gpu/shaders/`.
   Bindings are declared `layout(binding = N) buffer ...` (storage buffers) or
   `layout(binding = N) uniform sampler2D ...` (combined image samplers) and
   parameters go in one `layout(push_constant)` struct. Descriptor set 0 only.
   A bare `texture2D`/`sampler` pair is rejected by reflection: declare a
   `sampler2D`.
2. Add it to `_shader_sources` in `src/gpu/CMakeLists.txt`.
3. Call it from algorithm code:

```cpp
Buffer gpu_x = Buffer::of(*context, x);
Buffer gpu_y = make_buffer(*context, y.size() * sizeof(float));
struct Params { float a; std::uint32_t n; };
gpu::Dispatch dispatch;
dispatch.kernel = "saxpy";
dispatch.buffers = {&gpu_x, &gpu_y};          // binding i = buffers[i]
dispatch.push_constants = gpu::push(Params{...});
dispatch.groups = {groups_for(n, 256), 1, 1};
gpu::run(*context, {dispatch});
```

With a sampled image the same dispatch adds the image and its sampler at the
same explicit index (the kernel then has `sampler2D` at that binding):

```cpp
Image2D image = Image2D::create(*context, w, h, Format::R8Unorm, pixels);
Sampler sampler = Sampler::create(*context, Filter::Linear, AddressMode::ClampToEdge);
dispatch.images = {{1u, &image}};
dispatch.samplers = {{1u, &sampler}};
```

Wrap the call in `gpu::with_fallback` so machines without a device (or a
driver failure) transparently use the CPU path. Add a golden test that compares
the GPU result with `gpu::compare` against the CPU implementation.

## Threading

`run` and `run_async` may be called from several threads on the same context.
Recording, submission and `wait` are serialised by a mutex inside the backend
(queue submission requires external synchronisation, and a slot must not be
recycled while another thread still waits on its fence). Buffers may be
uploaded/downloaded concurrently as long as two threads do not write the same
range; a host copy drains every in-flight submission first.
