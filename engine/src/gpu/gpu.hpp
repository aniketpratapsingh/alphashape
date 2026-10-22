#pragma once

// Backend-neutral GPU compute layer. The public surface deliberately contains
// no Vulkan (or other backend) types: callers describe work as buffers plus
// SPIR-V kernels, and everything device-specific stays in src/gpu/vulkan.
//
// Later algorithm milestones port individual stages onto this interface; keep
// it small. See src/gpu/README.md for the loading order and kernel workflow.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::gpu {

class Context;
class Buffer;
class Image2D;
class Sampler;
class Ticket;
struct Dispatch;
struct Submission;

/// Pixel formats an Image2D can hold. The backend mapping is fixed:
/// R8Unorm -> VK_FORMAT_R8_UNORM (1 byte/texel), R8G8B8A8Unorm ->
/// VK_FORMAT_R8G8B8A8_UNORM (4), R32Float -> VK_FORMAT_R32_SFLOAT (4).
enum class Format { R8Unorm, R8G8B8A8Unorm, R32Float };

/// Texel filtering for a Sampler. Nearest is exact; Linear blends the four
/// nearest texels with bilinear weights.
enum class Filter { Nearest, Linear };

/// Addressing outside [0, 1]. Repeat tiles; MirroredRepeat tiles with the
/// image reversed on every odd tile.
enum class AddressMode { ClampToEdge, Repeat, MirroredRepeat };

/// Bytes per texel of a Format (1 or 4).
std::size_t bytes_per_pixel(Format format);

/// Human-readable Format name, used in error messages.
const char* format_name(Format format);

/// Where a buffer lives. DeviceLocal asks for device-only memory; upload and
/// download then stage through a host-visible buffer. On unified memory
/// (info().integrated, an integrated GPU) both kinds resolve to the same
/// host-visible heap and the staging hop is skipped. A CPU driver
/// (info().software) keeps the staged path even though its heap is unified, so
/// the ring is exercised on software devices.
enum class MemoryKind { DeviceLocal, HostVisible };

namespace detail {
class Backend;
class BufferHandle;
class ImageHandle;
class SamplerHandle;
/// Accessor for the backend handle inside a Buffer: the public surface keeps
/// the handle private, but request translation (which lives in gpu.cpp, not in
/// the Buffer class) needs it.
struct BufferAccess {
    static const BufferHandle* handle(const Buffer& buffer);
};
/// Same for Image2D and Sampler; the dispatch translator needs the raw
/// backend handles to fill descriptor writes.
struct ImageAccess {
    static const ImageHandle* handle(const Image2D& image);
};
struct SamplerAccess {
    static const SamplerHandle* handle(const Sampler& sampler);
};
Buffer make_buffer(Context& context, const void* data, std::size_t bytes);
Buffer make_buffer(Context& context, const void* data, std::size_t bytes, MemoryKind kind);
Image2D make_image(Context& context, int width, int height, Format format,
                   const void* pixels);
Sampler make_sampler(Context& context, Filter filter, AddressMode address);
bool image_sampling_supported(Context& context, Format format);
Ticket submit_async(Context& context, const std::vector<Dispatch>& dispatches,
                    Submission submission);
std::uint64_t validation_error_count();
void add_validation_error();
void note_gpu_failure(const std::string& what);
bool backend_compiled_in();
/// True when this context was created with validation and the layer is active.
bool validation_active(const Context& context);
/// Un-waited submissions currently outstanding on this context (test hook).
std::uint32_t in_flight_count(Context& context);
/// Hard cap on un-waited async submissions per context; never grows.
constexpr std::uint32_t kMaxInFlight = 4;
}  // namespace detail

struct DeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    /// True for VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: unified memory, so
    /// MemoryKind::DeviceLocal and HostVisible resolve to the same heap.
    bool integrated = false;
    /// True when the driver is a CPU implementation (VK_PHYSICAL_DEVICE_TYPE_CPU).
    bool software = false;
    std::uint64_t memory_bytes = 0;
    std::uint32_t api_version = 0;
    std::array<std::uint32_t, 3> max_workgroup_size{0, 0, 0};
    std::uint32_t max_workgroup_invocations = 0;
    std::uint32_t max_shared_memory = 0;
    std::uint32_t subgroup_size = 0;
};

struct ContextOptions {
    // -1 selects automatically: the discrete device with the most memory wins,
    // otherwise the integrated one with the most memory.
    int device_index = -1;
    // Enables the Khronos validation layer and a debug-utils messenger.
    bool validation = false;
    // Rejects devices whose device-local heap cannot supply 90% of this.
    std::uint64_t required_memory = 0;
};

/// True when an initialised default context exists. Honours the process-wide
/// switch: ALPHASHAPE_GPU=0 (or set_enabled(false)) makes this false.
bool available();

/// Human-readable reason the layer is unavailable (empty when it is).
std::string unavailable_reason();

/// Process-wide switch. The initial value comes from ALPHASHAPE_GPU
/// (unset means enabled).
void set_enabled(bool enabled_value);
bool enabled();

class Context {
public:
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    const DeviceInfo& info() const;

private:
    friend std::shared_ptr<Context> create_context(const ContextOptions& options);
    friend void run(Context& context, const std::vector<Dispatch>& dispatches);
    friend Ticket detail::submit_async(Context&, const std::vector<Dispatch>&, Submission);
    friend void wait(Context& context, Ticket ticket);
    friend std::uint32_t detail::in_flight_count(Context& context);
    friend Buffer detail::make_buffer(Context&, const void*, std::size_t);
    friend Buffer detail::make_buffer(Context&, const void*, std::size_t, MemoryKind);
    friend Image2D detail::make_image(Context&, int, int, Format, const void*);
    friend Sampler detail::make_sampler(Context&, Filter, AddressMode);
    friend bool detail::image_sampling_supported(Context&, Format);
    friend bool detail::validation_active(const Context&);

    explicit Context(std::shared_ptr<detail::Backend> backend);

    std::shared_ptr<detail::Backend> backend_;
};

/// The lazily created context with default options; nullptr when unavailable.
std::shared_ptr<Context> default_context();

/// Creates an additional context (e.g. with validation on). Throws gpu::Error.
std::shared_ptr<Context> create_context(const ContextOptions& options = {});

/// Enumerates devices independently of the default context. Empty when the
/// layer is unavailable.
std::vector<DeviceInfo> enumerate_devices();

/// True when the device is a CPU implementation (Mesa lavapipe, SwiftShader,
/// ...) rather than a hardware GPU. Correctness is identical on both, but
/// throughput comparisons against the CPU must not be asserted there: the
/// "device" IS the CPU, so a speed-up target is meaningless.
bool is_software_device(const DeviceInfo& info);

/// True when this device can bind `format` as a sampled image (combining a
/// COMBINED_IMAGE_SAMPLER with the format's sampled-image bit). Callers that
/// want linear filtering should additionally check the format properties;
/// this layer reports what the sampler path itself requires, and the tests
/// skip with the reason when it is false.
bool image_sampling_supported(Context& context, Format format);

/// Host-visible storage buffer (device-local on unified memory). A Buffer is
/// a shared handle: copies refer to the same allocation.
class Buffer {
public:
    Buffer() = default;

    std::size_t size() const;
    /// Writes host bytes into the buffer. DeviceLocal buffers that cannot be
    /// mapped stage through the context's upload ring (chunked past 64 MiB).
    void upload(const void* data, std::size_t bytes, std::size_t offset = 0);
    /// Reads buffer bytes back to the host; DeviceLocal buffers stage through a
    /// download buffer.
    void download(void* data, std::size_t bytes, std::size_t offset = 0) const;

    template <class T>
    static Buffer of(Context& context, const std::vector<T>& values,
                     MemoryKind kind = MemoryKind::HostVisible) {
        static_assert(std::is_trivially_copyable_v<T>, "Buffer::of needs plain data");
        return detail::make_buffer(context, values.data(), values.size() * sizeof(T), kind);
    }

    template <class T>
    std::vector<T> read(std::size_t count) const {
        static_assert(std::is_trivially_copyable_v<T>, "Buffer::read needs plain data");
        std::vector<T> values(count);
        if (count != 0) {
            download(values.data(), count * sizeof(T), 0);
        }
        return values;
    }

private:
    friend struct detail::BufferAccess;
    friend Buffer detail::make_buffer(Context&, const void*, std::size_t);
    friend Buffer detail::make_buffer(Context&, const void*, std::size_t, MemoryKind);
    friend bool detail::validation_active(const Context&);
    friend void run(Context& context, const std::vector<Dispatch>& dispatches);
    friend Ticket detail::submit_async(Context&, const std::vector<Dispatch>&, Submission);
    explicit Buffer(std::shared_ptr<detail::BufferHandle> handle)
        : handle_(std::move(handle)) {}
    std::shared_ptr<detail::BufferHandle> handle_;
};

/// Allocates a buffer; contents are unspecified until uploaded. DeviceLocal is
/// the default: device-only memory on a discrete GPU (host copies stage through
/// the context's ring), the host-visible heap on an integrated GPU, where both
/// kinds are the same memory.
Buffer make_buffer(Context& context, std::size_t bytes,
                   MemoryKind kind = MemoryKind::HostVisible);

/// A device-local, sampled 2D image. The image is created in a
/// transfer-destination layout; upload() copies host bytes into it and leaves
/// it in SHADER_READ_ONLY_OPTIMAL, so any dispatch that binds it may read it
/// without further transitions on the caller's side. A shared handle: copies
/// refer to the same image.
///
/// Pixels are tightly packed, row-major, origin top-left (row 0 is the first
/// host row), with no row padding: width * height * bytes_per_pixel(format).
class Image2D {
public:
    Image2D() = default;
    ~Image2D() = default;
    Image2D(const Image2D&) = default;
    Image2D& operator=(const Image2D&) = default;
    Image2D(Image2D&&) noexcept = default;
    Image2D& operator=(Image2D&&) noexcept = default;

    /// Creates the image, optionally uploading `pixels` in the same call.
    /// Throws gpu::Error on a non-positive size, a format the device cannot
    /// sample, or an upload size mismatch.
    static Image2D create(Context& context, int width, int height, Format format,
                          const void* pixels = nullptr);

    /// Replaces the whole image. `bytes` must equal width*height*bytes_per_pixel.
    void upload(const void* pixels, std::size_t bytes);
    /// Reads the whole image back. `bytes` must equal width*height*bytes_per_pixel.
    void download(void* pixels, std::size_t bytes) const;

    int width() const { return width_; }
    int height() const { return height_; }
    Format format() const { return format_; }
    std::size_t byte_size() const { return byte_size_; }
    /// False for a default-constructed handle.
    explicit operator bool() const { return handle_ != nullptr; }

private:
    friend struct detail::ImageAccess;
    friend Image2D detail::make_image(Context&, int, int, Format, const void*);
    Image2D(std::shared_ptr<detail::ImageHandle> handle, int width, int height, Format format,
            std::size_t byte_size)
        : handle_(std::move(handle)), width_(width), height_(height), format_(format),
          byte_size_(byte_size) {}
    std::shared_ptr<detail::ImageHandle> handle_;
    int width_ = 0;
    int height_ = 0;
    Format format_ = Format::R8Unorm;
    std::size_t byte_size_ = 0;
};

/// Immutable sampling state. create() returns a handle cached per context by
/// (filter, address), so requesting the same pair twice yields the same
/// object. Default-constructed Sampler is invalid and cannot be bound.
class Sampler {
public:
    Sampler() = default;

    static Sampler create(Context& context, Filter filter, AddressMode address);

    Filter filter() const { return filter_; }
    AddressMode address() const { return address_; }
    explicit operator bool() const { return handle_ != nullptr; }

private:
    friend struct detail::SamplerAccess;
    friend Sampler detail::make_sampler(Context&, Filter, AddressMode);
    Sampler(std::shared_ptr<detail::SamplerHandle> handle, Filter filter, AddressMode address)
        : handle_(std::move(handle)), filter_(filter), address_(address) {}
    std::shared_ptr<detail::SamplerHandle> handle_;
    Filter filter_ = Filter::Nearest;
    AddressMode address_ = AddressMode::ClampToEdge;
};

/// True when this device can sample images of `format` (combined image sampler
/// with linear filtering is what the layer needs; a device that can only do
/// nearest still reports true and the caller must use Filter::Nearest).
bool image_sampling_supported(Context& context, Format format);

struct Dispatch {
    std::string kernel;
    // Binding i = buffers[i], descriptor set 0, storage buffers.
    std::vector<const Buffer*> buffers;
    // Binding index -> image; the descriptor is a COMBINED_IMAGE_SAMPLER, so
    // every image binding also needs exactly one sampler with the same index.
    std::vector<std::pair<std::uint32_t, const Image2D*>> images;
    std::vector<std::pair<std::uint32_t, const Sampler*>> samplers;
    std::vector<std::uint8_t> push_constants;
    std::array<std::uint32_t, 3> groups{1, 1, 1};
};

/// Serialises a trivially-copyable struct into push-constant bytes.
template <class T>
std::vector<std::uint8_t> push(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>, "push() needs a plain struct");
    std::vector<std::uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

/// Runs a list of dispatches in one command buffer, with compute->compute
/// barriers between them, then waits. Throws gpu::Error on any failure.
/// Equivalent to run_async(context, dispatches) followed by wait().
void run(Context& context, const std::vector<Dispatch>& dispatches);

/// Which queue a submission goes to. Queue 1 is only honoured when the selected
/// queue family exposes at least two queues; otherwise it falls back to queue 0
/// with one log line. Ordering between queues is the caller's responsibility:
/// wait() on the producing ticket before submitting to another queue.
struct Submission {
    int queue = 0;
};

/// Handle to one asynchronous submission. Copyable and comparable; waiting
/// twice is harmless. A default-constructed Ticket refers to nothing and waits
/// immediately.
class Ticket {
public:
    Ticket() = default;

    /// Implementation-private identity; 0 means "nothing to wait for".
    std::uint64_t id() const { return id_; }

private:
    friend Ticket detail::submit_async(Context&, const std::vector<Dispatch>&, Submission);
    friend void wait(Context& context, Ticket ticket);
    friend std::uint32_t detail::in_flight_count(Context& context);
    explicit Ticket(std::uint64_t id) : id_(id) {}
    std::uint64_t id_ = 0;
};

/// Submits work without waiting. At most detail::kMaxInFlight submissions may be
/// outstanding on one context; a further call throws Error rather than growing
/// any internal ring. Throws Error for the same interface problems as run().
Ticket run_async(Context& context, const std::vector<Dispatch>& dispatches,
                 Submission submission = {});

/// Blocks until the submission's fence signals. Every internal fence wait has a
/// 5 s timeout (Error("gpu: fence timeout")). Waiting on a Ticket that has
/// already been waited for is a no-op.
void wait(Context& context, Ticket ticket);

class Error : public alphashape::Error {
public:
    explicit Error(const std::string& what) : alphashape::Error(what) {}
};

/// Shader names compiled into this build.
std::vector<std::string> kernel_names();

/// GPU-vs-CPU comparison used by golden self-checks in ported kernels.
struct GoldReport {
    double mean_abs = 0.0;
    double max_abs = 0.0;
    std::size_t count = 0;
};
GoldReport compare(const std::vector<float>& gpu, const std::vector<float>& cpu);

/// Number of GPU failures the fallback path has absorbed this process.
std::uint64_t failure_count();

namespace detail {
/// Test hook: attempts to load one explicit Vulkan loader path and resolve the
/// global entry points, without touching the process-wide loader state.
/// Returns false with a human-readable reason on failure.
bool try_load_library(const std::string& path, std::string* reason);
}  // namespace detail

/// Runs gpu_fn when the layer is available, otherwise cpu_fn. Any exception
/// from gpu_fn is logged, counted, and answered by cpu_fn.
template <class R>
R with_fallback(const char* what, const std::function<R()>& gpu_fn,
                const std::function<R()>& cpu_fn) {
    if (!available() || !enabled()) {
        return cpu_fn();
    }
    try {
        return gpu_fn();
    } catch (const std::exception& error) {
        detail::note_gpu_failure(std::string(what) + ": " + error.what());
    } catch (...) {
        detail::note_gpu_failure(std::string(what) + ": unknown error");
    }
    return cpu_fn();
}

}  // namespace alphashape::gpu
