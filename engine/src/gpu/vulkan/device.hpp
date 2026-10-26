#pragma once

// Device-level internals shared by context.cpp and dispatch.cpp: the allocator,
// the pipeline cache, the buffer handle and the submission ring.

#include "gpu/vulkan/vk_headers.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gpu/vulkan/backend.hpp"
#include "gpu/vulkan/instance.hpp"
#include "gpu/vulkan/spirv.hpp"

namespace alphashape::gpu::detail {

struct Pipeline {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    SpirvInfo info;
};

class VulkanBackend final : public Backend {
public:
    /// One async submission in flight: its own fence, command buffer and
    /// descriptor pool, borrowed from one of the two command pools by index.
    struct Slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        std::uint64_t id = 0;
        int queue = 0;
        bool pending = false;
        /// Set by abandon_slot(): the fence was never observed to signal, so it
        /// must not be reset for a fresh submission. The slot is reclaimed (one
        /// bounded wait) only when a later submit() finds the fence signalled.
        bool abandoned = false;
    };

    /// Host-visible staging memory for device-local transfers. Fixed size; the
    /// offset is an explicit bounds-checked ring index.
    static constexpr std::size_t kStagingBytes = 64u * 1024u * 1024u;

    VulkanBackend(std::shared_ptr<Instance> instance, const DeviceRecord& device,
                  bool validation);
    ~VulkanBackend() override;

    const DeviceInfo& info() const override { return device_.info; }
    bool validation_active() const override { return validation_; }
    std::shared_ptr<BufferHandle> allocate(std::size_t bytes,
                                           MemoryKind kind) override;
    std::shared_ptr<ImageHandle> create_image(int width, int height, Format format,
                                              const void* pixels) override;
    std::shared_ptr<SamplerHandle> create_sampler(Filter filter, AddressMode address) override;
    bool image_sampling_supported(Format format) override;
    void run(const std::vector<DispatchRequest>& dispatches) override;
    void run_with_fence_budget(const std::vector<DispatchRequest>& dispatches,
                               std::uint64_t fence_budget_ns) override;
    std::uint64_t submit(const std::vector<DispatchRequest>& dispatches, int queue,
                         bool block_when_full) override;
    void wait(std::uint64_t ticket) override;
    std::uint32_t in_flight() const override;
    std::size_t orphaned_pool_count() const override {
        // A test hook, but it reads state the submission mutex protects.
        std::lock_guard<std::mutex> lock(mutex_);
        return orphaned_pools_.size();
    }

    VkDevice device() const { return device_handle_; }
    /// The queue a Submission's index resolves to (0 when out of range).
    VkQueue queue(int index) const;
    std::uint32_t queue_family() const { return queue_family_; }
    const Functions& fns() const { return instance_->fns; }
    VmaAllocator allocator() const { return allocator_; }
    /// True on unified memory (integrated GPU), where both MemoryKinds resolve
    /// to the same mappable heap.
    bool unified_memory() const { return unified_; }

    /// Streams an aligned window through the ring. read=true copies device ->
    /// host_buffer; read=false copies host_buffer -> device. Chunks the window
    /// when it is larger than the ring, with an explicit bounds-checked offset
    /// (never a modulo).
    void stream_window(VkBuffer device_buffer, std::size_t window_offset,
                       std::size_t window_bytes, const void* host_buffer, bool read);
    /// One staging copy of an aligned window (staging region <-> device range)
    /// plus its wait. Both offsets and the size must be multiples of 4. Only
    /// stream_window calls this; it runs with the transfer lock held.
    void run_transfer(VkBuffer device_buffer, VkDeviceSize staging_offset,
                      VkDeviceSize device_offset, VkDeviceSize bytes, bool to_device);
    /// Uploads a host range, preserving the device bytes outside it when the
    /// range is not 4-byte aligned. `capacity` is the allocation's reserved
    /// size (the aligned window is clipped to it).
    void upload_staged(VkBuffer destination, std::size_t capacity, const void* data,
                       std::size_t bytes, std::size_t offset);
    /// Downloads a host range, tolerating an unaligned offset.
    void download_staged(VkBuffer source, std::size_t capacity, void* data, std::size_t bytes,
                         std::size_t offset);
    /// The aligned read-modify-write implementation of both.
    void transfer_window(VkBuffer device_buffer, std::size_t capacity,
                         std::size_t device_offset, std::size_t bytes, const void* upload_src,
                         void* download_dst);
    /// Waits for every outstanding submission's fence (host access to buffers
    /// must not race with in-flight GPU work). Does not release the slots: they
    /// stay counted as un-waited until wait() is called for them. Throws
    /// gpu::Error once the device is lost: a host copy must not read a buffer
    /// the queue never finished writing.
    void wait_all_pending();
    /// The destructor form of wait_all_pending(): never throws, spends one
    /// shared deadline on the whole drain rather than one slice per fence, and
    /// marks the device lost when the drain does not complete. Destructors are
    /// implicitly noexcept, so throwing from one would be std::terminate.
    /// Returns true when every fence signalled.
    bool wait_all_pending_noexcept();
    /// VkFormat for a public Format (used by the image path and capability
    /// queries).
    VkFormat vk_format(Format format) const;
    /// Copies host bytes into `image` (whole image) and leaves it in
    /// SHADER_READ_ONLY_OPTIMAL: it transitions UNDEFINED -> TRANSFER_DST ->
    /// SHADER_READ_ONLY_OPTIMAL and ends with a compute-visible barrier, so
    /// the dispatch that reads it needs no transition of its own.
    void upload_image(VkImage image, int width, int height, Format format,
                      const void* pixels, std::size_t bytes, VkImageLayout old_layout);
    /// Copies `image` (SHADER_READ_ONLY_OPTIMAL) back to the host:
    /// SHADER_READ_ONLY -> TRANSFER_SRC -> SHADER_READ_ONLY.
    void download_image(VkImage image, int width, int height, Format format, void* pixels,
                        std::size_t bytes);
    /// Shared implementation of both: transitions, copies through a temporary
    /// host-visible buffer, transitions back to SHADER_READ_ONLY_OPTIMAL.
    void image_transfer(VkImage image, int width, int height, Format format, bool to_device,
                        const void* source, void* destination, VkImageLayout old_layout);
    /// Cached pipeline for a kernel, created on first use. Caller holds mutex().
    const Pipeline& pipeline_for(const KernelEntry& kernel);
    std::mutex& mutex() { return mutex_; }

private:
    void destroy_pipeline(Pipeline* pipeline);

    /// Records and submits `dispatches` into `slot`. Caller holds mutex_ and has
    /// reserved the slot. Throws gpu::Error on any failure (slot is released).
    void record_and_submit(Slot& slot, const std::vector<DispatchRequest>& dispatches);
    /// Waits for one fence in bounded slices under the default budget; throws
    /// Error("gpu: fence timeout"), and throws "device lost" without waiting on
    /// a device already marked lost. `budget_ns` overrides the device default
    /// (0 keeps it).
    void wait_fence(VkFence fence, std::uint64_t budget_ns = 0);
    /// One bounded, never-throwing wait used by the destructor drain. Returns
    /// true when the fence signalled.
    bool wait_fence_noexcept(VkFence fence, std::uint64_t timeout_ns) noexcept;
    /// True once the device is unusable: a real loss, or a drain that failed.
    bool lost() const;
    /// Drains every fence this backend owns with one shared deadline; the
    /// throwing and noexcept variants share it. Returns false when a fence did
    /// not signal inside the budget.
    bool drain_pending(std::uint64_t budget_ns, bool allow_throw);
    /// Reserves the next free slot; returns nullptr when kMaxInFlight are
    /// outstanding (the caller decides throw-or-drain).
    Slot* reserve_slot();
    /// Ticket id of the oldest pending submission (0 when none).
    std::uint64_t oldest_pending_id();
    void release_slot(Slot& slot);
    /// release_slot plus freeing the pools parked by *earlier* abandoned
    /// submissions on this slot's queue: the fence that just signalled proves
    /// the FIFO queue drained past them. Only the fence-signalled paths call
    /// this; a record_and_submit() failure and the destructor use the plain
    /// release and free_orphaned_pools() respectively.
    void release_slot_after_signal(Slot& slot);
    /// Destroys the descriptor pools parked by abandoned submissions. Called
    /// from ~VulkanBackend, after the device has been drained.
    void free_orphaned_pools();
    /// Frees the pools parked on one queue, once a fence on that FIFO queue has
    /// signalled (so it has drained past the abandoned submissions).
    void free_orphaned_pools_for_queue(int queue);
    /// Gives up a slot whose fence will never signal (a device loss). The
    /// pending state and ticket go, so no later wait() spins on it and the ring
    /// is not left permanently full. The descriptor pool moves to
    /// orphaned_pools_ rather than being freed or left in the slot: the queue
    /// may still reference it, and leaving it here would let the next
    /// record_and_submit() overwrite the handle and leak it.
    void abandon_slot(Slot& slot);
    void ensure_staging();
    /// Creates the image-copy command pool, buffer and fence on first use.
    void ensure_image_transfer();

    std::shared_ptr<Instance> instance_;
    DeviceRecord device_;
    VkDevice device_handle_ = VK_NULL_HANDLE;
    VkQueue queues_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::uint32_t queue_count_ = 0;
    std::uint32_t queue_family_ = 0;
    bool unified_ = false;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    std::unordered_map<std::string, Pipeline> pipelines_;
    mutable std::mutex mutex_;
    bool validation_ = false;

    // Two command pools (rings); four slots index them by parity.
    std::array<VkCommandPool, 2> command_pools_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<Slot, kMaxInFlight> slots_{};
    // Descriptor pools owned by submissions that were abandoned (a fence
    // timeout or a lost device) and so never released normally, paired with the
    // queue they were submitted on. They must not be destroyed while the
    // abandoned submission might still reference them, so they wait here for
    // the next release on the same queue -- a FIFO queue that signalled a later
    // fence has drained past them -- or for ~VulkanBackend.
    std::vector<std::pair<VkDescriptorPool, int>> orphaned_pools_;
    std::uint64_t next_ticket_ = 1;

    // Staging ring, created on first device-local transfer.
    std::mutex transfer_mutex_;
    VkBuffer staging_ = VK_NULL_HANDLE;
    VmaAllocation staging_allocation_ = VK_NULL_HANDLE;
    void* staging_mapped_ = nullptr;
    std::size_t staging_bytes_ = 0;
    std::size_t staging_offset_ = 0;
    VkCommandPool transfer_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer transfer_commands_ = VK_NULL_HANDLE;
    VkFence transfer_fence_ = VK_NULL_HANDLE;

    // Image copies have their own tiny command buffer and fence so they do not
    // have to allocate the 64 MiB staging ring just to get a command pool.
    VkCommandPool image_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer image_commands_ = VK_NULL_HANDLE;
    VkFence image_fence_ = VK_NULL_HANDLE;

    // Post-loss destructor drain: one deadline shared by every buffer that the
    // same failed operation destroys, so the fallback unwind is bounded in
    // total rather than per handle. Keyed to the loss generation so a later,
    // unrelated loss arms a fresh window. Guarded by mutex_.
    bool lost_drain_started_ = false;
    std::uint64_t lost_drain_generation_ = 0;
    std::chrono::steady_clock::time_point lost_drain_deadline_{};

    // Samplers are immutable and cached per (filter, address). The key packs
    // the two enums into one integer; the value is the shared handle.
    std::mutex sampler_mutex_;
    std::unordered_map<std::uint32_t, std::shared_ptr<SamplerHandle>> samplers_;
    /// Format support answers, cached per Format (queries are expensive). The
    /// cache is written under the submission mutex so a concurrent create_image
    /// cannot race the query.
    std::array<int, 3> format_support_{{-1, -1, -1}};
};

/// The concrete storage buffer. Host-visible storage is mapped for the life of
/// the buffer; device-local storage routes copies through the staging ring.
class VulkanBuffer final : public BufferHandle {
public:
    VulkanBuffer(VulkanBackend* backend, VkBuffer buffer, VmaAllocation allocation,
                 VmaAllocationInfo allocation_info, std::size_t size,
                 std::size_t reserved, bool staged);
    ~VulkanBuffer() override;

    std::size_t size() const override { return size_; }
    void upload(const void* data, std::size_t bytes, std::size_t offset) override;
    void download(void* data, std::size_t bytes, std::size_t offset) const override;

    VkBuffer handle() const { return buffer_; }
    /// True when host copies go through the staging ring (DeviceLocal on a
    /// non-unified device), whether or not the heap happens to be mappable.
    bool staged() const { return staged_; }
    /// The allocation's size, rounded up to 4 bytes so every VkBufferCopy the
    /// staging ring issues has 4-multiple offsets and size.
    std::size_t reserved() const { return reserved_; }

private:
    VulkanBackend* backend_;
    VkBuffer buffer_;
    VmaAllocation allocation_;
    void* mapped_ = nullptr;
    std::size_t size_ = 0;
    std::size_t reserved_ = 0;
    bool staged_ = false;
};

/// A device-local sampled image with stable identity: exactly one layout
/// (SHADER_READ_ONLY_OPTIMAL once uploaded), no per-dispatch transitions.
class VulkanImage final : public ImageHandle {
public:
    VulkanImage(VulkanBackend* backend, VkImage image, VmaAllocation allocation,
                VkImageView view, int width, int height, Format format);
    ~VulkanImage() override;

    const void* owner() const override { return backend_; }
    int width() const override { return width_; }
    int height() const override { return height_; }
    Format format() const override { return format_; }
    bool has_contents() const override { return uploaded_; }
    void upload(const void* pixels, std::size_t bytes) override;
    void download(void* pixels, std::size_t bytes) const override;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    /// Records the pending-layout marker after a successful upload/download.
    void set_uploaded() { uploaded_ = true; }

private:
    VulkanBackend* backend_;
    VkImage image_;
    VmaAllocation allocation_;
    VkImageView view_;
    int width_;
    int height_;
    Format format_;
    bool uploaded_ = false;
};

class VulkanSampler final : public SamplerHandle {
public:
    VulkanSampler(VulkanBackend* backend, VkSampler sampler, Filter filter,
                  AddressMode address);
    ~VulkanSampler() override;

    const void* owner() const override { return backend_; }
    Filter filter() const override { return filter_; }
    AddressMode address() const override { return address_; }

    VkSampler handle() const { return sampler_; }

private:
    VulkanBackend* backend_;
    VkSampler sampler_;
    Filter filter_;
    AddressMode address_;
};

}  // namespace alphashape::gpu::detail
