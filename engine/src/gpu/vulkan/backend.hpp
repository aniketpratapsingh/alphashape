#pragma once

// Internal contract between the public gpu:: API (gpu.cpp) and the Vulkan
// backend. Nothing here is visible to algorithm code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu/gpu.hpp"
#include "gpu/shaders_embedded.hpp"

namespace alphashape::gpu {

class Buffer;
class Context;
class Error;

namespace detail {

class BufferHandle;
class ImageHandle;
class SamplerHandle;

/// One combined-image-sampler binding in a dispatch: the binding index from
/// SPIR-V, the image, and the sampler that pairs with it.
struct ImageBinding {
    std::uint32_t binding = 0;
    const ImageHandle* image = nullptr;
    const SamplerHandle* sampler = nullptr;
};

struct DispatchRequest {
    const KernelEntry* kernel = nullptr;
    std::vector<const BufferHandle*> buffers;
    /// Sorted by binding. Every entry is a COMBINED_IMAGE_SAMPLER binding.
    std::vector<ImageBinding> images;
    std::vector<std::uint8_t> push_constants;
    std::array<std::uint32_t, 3> groups{1, 1, 1};
};

class BufferHandle {
public:
    virtual ~BufferHandle() = default;
    virtual std::size_t size() const = 0;
    virtual void upload(const void* data, std::size_t bytes, std::size_t offset) = 0;
    virtual void download(void* data, std::size_t bytes, std::size_t offset) const = 0;
};

/// A sampled 2D image owned by one backend. `owner()` identifies the backend,
/// so a dispatch rejects a handle created on a different context with a clean
/// gpu::Error instead of a driver fault or validation spam.
class ImageHandle {
public:
    virtual ~ImageHandle() = default;
    virtual const void* owner() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual Format format() const = 0;
    /// False until upload() has run: a dispatch that binds such an image is
    /// rejected before submit, because its contents would be undefined.
    virtual bool has_contents() const = 0;
    virtual void upload(const void* pixels, std::size_t bytes) = 0;
    virtual void download(void* pixels, std::size_t bytes) const = 0;
};

/// An immutable sampler; the backend caches one object per (filter, address).
class SamplerHandle {
public:
    virtual ~SamplerHandle() = default;
    virtual const void* owner() const = 0;
    virtual Filter filter() const = 0;
    virtual AddressMode address() const = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const DeviceInfo& info() const = 0;
    /// True when the validation layer was requested and is actually active.
    virtual bool validation_active() const = 0;
    virtual std::shared_ptr<BufferHandle> allocate(std::size_t bytes,
                                                   MemoryKind kind) = 0;
    /// Creates and (when pixels != nullptr) uploads a sampled image.
    virtual std::shared_ptr<ImageHandle> create_image(int width, int height, Format format,
                                                      const void* pixels) = 0;
    /// Returns the cached sampler for (filter, address), creating it first use.
    virtual std::shared_ptr<SamplerHandle> create_sampler(Filter filter,
                                                          AddressMode address) = 0;
    /// Format capability check; false makes callers fall back or skip.
    virtual bool image_sampling_supported(Format format) = 0;
    virtual void run(const std::vector<DispatchRequest>& dispatches) = 0;
    /// run() with an explicit fence budget for this submission (0 keeps the
    /// device default). The first submission of a batched operation is
    /// calibrated from its own duration, so it is allowed a generous budget
    /// instead of being declared a hang at the steady-state bound.
    virtual void run_with_fence_budget(const std::vector<DispatchRequest>& dispatches,
                                       std::uint64_t fence_budget_ns) = 0;
    /// Records and submits without waiting; returns the ticket id. When
    /// `block_when_full` is false (run_async) a full ring throws gpu::Error;
    /// when true (the synchronous path) the oldest pending submission is waited
    /// for first. The ticket keeps the slot until wait(ticket).
    virtual std::uint64_t submit(const std::vector<DispatchRequest>& dispatches, int queue,
                                 bool block_when_full) = 0;
    /// Blocks on the ticket's fence (5 s timeout). Unknown or already-completed
    /// tickets return immediately.
    virtual void wait(std::uint64_t ticket) = 0;
    /// Un-waited submissions on this backend.
    virtual std::uint32_t in_flight() const = 0;
};

/// Creates a backend. Throws gpu::Error with a descriptive message.
std::shared_ptr<Backend> create_backend(const ContextOptions& options);

/// dlopens one explicit path and resolves the global entry points.
bool probe_loader(const std::string& path, std::string* reason);

/// Enumerates physical devices without creating a backend. Returns false with
/// a reason when the loader or ICD is unavailable.
bool enumerate_devices(std::vector<DeviceInfo>* devices, std::string* reason);

}  // namespace detail
}  // namespace alphashape::gpu
