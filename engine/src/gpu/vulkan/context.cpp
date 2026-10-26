#include "gpu/vulkan/device.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "base/utils/log.hpp"
#include "gpu/gpu_config.hpp"
#include "gpu/vulkan/loader.hpp"

namespace alphashape::gpu::detail {
namespace {

bool has_device_extension(VkPhysicalDevice device, const Functions& fns, const char* name) {
    std::uint32_t count = 0;
    if (fns.EnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> properties(count);
    if (fns.EnumerateDeviceExtensionProperties(device, nullptr, &count, properties.data()) !=
        VK_SUCCESS) {
        return false;
    }
    for (const VkExtensionProperties& property : properties) {
        if (std::strcmp(property.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

std::string vk_result_string(const char* call, VkResult result) {
    return std::string(call) + " failed with VkResult " + std::to_string(result);
}

}  // namespace

VulkanBackend::VulkanBackend(std::shared_ptr<Instance> instance, const DeviceRecord& device,
                             bool validation)
    : instance_(std::move(instance)), device_(device) {
    const Functions& fns = instance_->fns;
    // The instance only enables the layer when it is actually present.
    validation_ = validation && instance_->validation;

    // MoltenVK advertises the portability subset; it must be enabled explicitly.
    std::vector<const char*> extensions;
    if (has_device_extension(device_.handle, fns, "VK_KHR_portability_subset")) {
        extensions.push_back("VK_KHR_portability_subset");
    }

    const float priorities[2] = {1.0f, 1.0f};
    // Two compute queues when the family exposes them: Submission{.queue = 1}
    // is the only way to use the second one (no automatic splitting).
    queue_count_ = std::min<std::uint32_t>(device_.queue_count, 2U);
    if (queue_count_ == 0) {
        queue_count_ = 1;
    }
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = device_.queue_family;
    queue_info.queueCount = queue_count_;
    queue_info.pQueuePriorities = priorities;

    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

    const VkResult created = fns.CreateDevice(device_.handle, &create_info, nullptr, &device_handle_);
    if (created != VK_SUCCESS) {
        throw Error(vk_result_string("vkCreateDevice", created));
    }
    for (std::uint32_t index = 0; index < queue_count_; ++index) {
        fns.GetDeviceQueue(device_handle_, device_.queue_family, index, &queues_[index]);
        if (queues_[index] == VK_NULL_HANDLE) {
            throw Error("vkGetDeviceQueue returned a null queue");
        }
    }
    if (queue_count_ < 2) {
        log(LogLevel::Info) << "GPU: queue family exposes one compute queue; "
                               "Submission{.queue = 1} falls back to queue 0";
    }

    // Unified memory (Apple silicon): device-local and host-visible memory are
    // the same heap, so DeviceLocal allocates the mappable type and no staging
    // hop is needed. A CPU driver has one heap too, but keeps the staged path
    // (see allocate) so the ring is exercised on software devices as well.
    unified_ = device_.info.integrated;

    // The async ring: two transient command pools, four fences.
    for (VkCommandPool& pool : command_pools_) {
        VkCommandPoolCreateInfo pool_create{};
        pool_create.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_create.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_create.queueFamilyIndex = device_.queue_family;
        const VkResult pool_result =
            fns.CreateCommandPool(device_handle_, &pool_create, nullptr, &pool);
        if (pool_result != VK_SUCCESS) {
            throw Error(vk_result_string("vkCreateCommandPool", pool_result));
        }
    }

    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroup;
    auto get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        instance_->fns.GetInstanceProcAddr(instance_->handle, "vkGetPhysicalDeviceProperties2KHR"));
    if (get_properties2 == nullptr) {
        get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            instance_->fns.GetInstanceProcAddr(instance_->handle, "vkGetPhysicalDeviceProperties2"));
    }
    if (get_properties2 != nullptr) {
        get_properties2(device_.handle, &properties2);
        if (subgroup.subgroupSize != 0) {
            device_.info.subgroup_size = subgroup.subgroupSize;
        }
    }

    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = instance_->fns.GetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = instance_->fns.GetDeviceProcAddr;

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.physicalDevice = device_.handle;
    allocator_info.device = device_handle_;
    allocator_info.instance = instance_->handle;
    allocator_info.pVulkanFunctions = &functions;
    allocator_info.vulkanApiVersion =
        std::min(device_.info.api_version, 0x00402000u);
    const VkResult allocator_result = vmaCreateAllocator(&allocator_info, &allocator_);
    if (allocator_result != VK_SUCCESS) {
        throw Error(vk_result_string("vmaCreateAllocator", allocator_result));
    }
    log(LogLevel::Info) << "GPU backend ready: " << device_.info.name << " (subgroup "
                        << device_.info.subgroup_size << ")";
}

VulkanBackend::~VulkanBackend() {
    if (device_handle_ != VK_NULL_HANDLE) {
        const Functions& fns = instance_->fns;
        // Command pools and fences may only go once no submission is running.
        // vkDeviceWaitIdle is unbounded, so on a lost device it is skipped
        // entirely -- the driver never finishes the abandoned submissions and
        // the call would block the fallback unwind forever. The bounded fence
        // drain below covers the same requirement in that case.
        if (!lost()) {
            fns.DeviceWaitIdle(device_handle_);
        } else {
            (void)wait_all_pending_noexcept();
        }
        // Cached samplers are owned by this backend; a Sampler handle that
        // outlives it holds a dangling pointer only if it also outlives the
        // context, which is a caller error (the layer's handles are borrowed).
        {
            std::lock_guard<std::mutex> lock(sampler_mutex_);
            for (auto& entry : samplers_) {
                auto sampler = dynamic_cast<VulkanSampler*>(entry.second.get());
                if (sampler != nullptr && sampler->handle() != VK_NULL_HANDLE) {
                    fns.DestroySampler(device_handle_, sampler->handle(), nullptr);
                }
            }
            samplers_.clear();
        }
        for (Slot& slot : slots_) {
            if (slot.fence != VK_NULL_HANDLE) {
                fns.DestroyFence(device_handle_, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            if (slot.descriptor_pool != VK_NULL_HANDLE) {
                fns.DestroyDescriptorPool(device_handle_, slot.descriptor_pool, nullptr);
                slot.descriptor_pool = VK_NULL_HANDLE;
            }
            slot.pending = false;
        }
        if (transfer_fence_ != VK_NULL_HANDLE) {
            fns.DestroyFence(device_handle_, transfer_fence_, nullptr);
            transfer_fence_ = VK_NULL_HANDLE;
        }
        if (transfer_pool_ != VK_NULL_HANDLE) {
            fns.DestroyCommandPool(device_handle_, transfer_pool_, nullptr);
            transfer_pool_ = VK_NULL_HANDLE;
            transfer_commands_ = VK_NULL_HANDLE;
        }
        if (image_fence_ != VK_NULL_HANDLE) {
            fns.DestroyFence(device_handle_, image_fence_, nullptr);
            image_fence_ = VK_NULL_HANDLE;
        }
        if (image_pool_ != VK_NULL_HANDLE) {
            fns.DestroyCommandPool(device_handle_, image_pool_, nullptr);
            image_pool_ = VK_NULL_HANDLE;
            image_commands_ = VK_NULL_HANDLE;
        }
        if (staging_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, staging_, staging_allocation_);
            staging_ = VK_NULL_HANDLE;
            staging_allocation_ = VK_NULL_HANDLE;
            staging_mapped_ = nullptr;
        }
        for (VkCommandPool& pool : command_pools_) {
            if (pool != VK_NULL_HANDLE) {
                fns.DestroyCommandPool(device_handle_, pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
        for (auto& entry : pipelines_) {
            destroy_pipeline(&entry.second);
        }
        pipelines_.clear();
        if (allocator_ != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator_);
            allocator_ = VK_NULL_HANDLE;
        }
        fns.DestroyDevice(device_handle_, nullptr);
    }
}

VkQueue VulkanBackend::queue(int index) const {
    if (index <= 0 || queue_count_ < 2) {
        return queues_[0];
    }
    return queues_[1];
}

void VulkanBackend::destroy_pipeline(Pipeline* pipeline) {
    const Functions& fns = instance_->fns;
    if (pipeline->pipeline != VK_NULL_HANDLE) {
        fns.DestroyPipeline(device_handle_, pipeline->pipeline, nullptr);
    }
    if (pipeline->layout != VK_NULL_HANDLE) {
        fns.DestroyPipelineLayout(device_handle_, pipeline->layout, nullptr);
    }
    if (pipeline->set_layout != VK_NULL_HANDLE) {
        fns.DestroyDescriptorSetLayout(device_handle_, pipeline->set_layout, nullptr);
    }
    if (pipeline->module != VK_NULL_HANDLE) {
        fns.DestroyShaderModule(device_handle_, pipeline->module, nullptr);
    }
    *pipeline = Pipeline{};
}

std::shared_ptr<BufferHandle> VulkanBackend::allocate(std::size_t bytes, MemoryKind kind) {
    // Round the allocation up so the staging ring can always copy 4-byte
    // multiples (VkBufferCopy requires aligned offsets and size). A zero-byte
    // request still gets a legal 4-byte allocation.
    const std::size_t reserved = std::max<std::size_t>((bytes + 3u) & ~std::size_t{3u}, 4u);
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = static_cast<VkDeviceSize>(reserved);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    // Staging is used for DeviceLocal copies unless the device has unified
    // memory (an integrated GPU, where the host-visible heap IS the device
    // memory, so a staging hop would be a pure extra copy). A CPU device
    // (llvmpipe) is unified in the same sense but keeps the staged path so the
    // ring is exercised in CI, not only on discrete hardware.
    const bool staged = kind == MemoryKind::DeviceLocal && !device_.info.integrated;
    if (staged) {
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    } else {
        // HostVisible, or an integrated GPU where both kinds are one heap.
        // On unified memory (Apple silicon, llvmpipe) this resolves to
        // host-visible device-local memory; elsewhere VMA picks the best type
        // it can map.
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo result_info{};
    const VkResult result =
        vmaCreateBuffer(allocator_, &info, &allocation_info, &buffer, &allocation, &result_info);
    if (result != VK_SUCCESS) {
        throw Error("buffer allocation of " + std::to_string(bytes) + " bytes failed (VkResult " +
                    std::to_string(result) + ")");
    }
    // A staged buffer never uses the mapping even if the device only has a
    // mappable heap: the staging path is then exercised on every driver, not
    // just discrete ones with VRAM.
    return std::make_shared<VulkanBuffer>(this, buffer, allocation, result_info, bytes, reserved,
                                          staged);
}

// -- images and samplers -----------------------------------------------------
//
// Images are device-local, sampled, and have exactly one steady-state layout
// (SHADER_READ_ONLY_OPTIMAL after their first upload). The layout transitions
// that get them there are recorded on the transfer command buffer, which is
// already the synchronisation point for a host copy: the dispatch that later
// reads the image is submitted after the transfer fence signals, and the
// transfer command buffer ends with a compute-visible barrier. That keeps the
// dispatch path free of per-image transitions (WP-10.2 design, §4.1: the
// transitions happen once per upload).

VkFormat VulkanBackend::vk_format(Format format) const {
    switch (format) {
        case Format::R8Unorm:
            return VK_FORMAT_R8_UNORM;
        case Format::R8G8B8A8Unorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::R32Float:
            return VK_FORMAT_R32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

bool VulkanBackend::image_sampling_supported(Format format) {
    const std::size_t index = static_cast<std::size_t>(format);
    // Serialised with submissions: the query result is cached in a shared
    // array, and create_image may be called from any thread.
    std::lock_guard<std::mutex> lock(mutex_);
    if (index < format_support_.size() && format_support_[index] >= 0) {
        return format_support_[index] != 0;
    }
    const Functions& fns = instance_->fns;
    VkFormatProperties properties{};
    fns.GetPhysicalDeviceFormatProperties(device_.handle, vk_format(format), &properties);
    // The dispatch binds a combined image sampler with a sampler object, so
    // the format needs the sampled-image bit. Transfer bits are what upload
    // and download need; they are required as well, because an image with no
    // legal copy path cannot be populated.
    const VkFormatFeatureFlags sampled = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    const VkFormatFeatureFlags transfer =
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const bool supported = (properties.optimalTilingFeatures & sampled) != 0 &&
                           (properties.optimalTilingFeatures & transfer) == transfer;
    if (!supported) {
        log(LogLevel::Warn) << "GPU: " << device_.info.name << " cannot sample "
                            << format_name(format) << " (optimalTilingFeatures=0x" << std::hex
                            << properties.optimalTilingFeatures << std::dec << ")";
    }
    if (index < format_support_.size()) {
        format_support_[index] = supported ? 1 : 0;
    }
    return supported;
}

std::shared_ptr<ImageHandle> VulkanBackend::create_image(int width, int height, Format format,
                                                         const void* pixels) {
    if (width <= 0 || height <= 0) {
        throw Error("image dimensions must be positive, got " + std::to_string(width) + "x" +
                    std::to_string(height));
    }
    if (!image_sampling_supported(format)) {
        throw Error(std::string("device cannot sample ") + format_name(format) + " images");
    }
    const Functions& fns = instance_->fns;

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = vk_format(format);
    info.extent.width = static_cast<std::uint32_t>(width);
    info.extent.height = static_cast<std::uint32_t>(height);
    info.extent.depth = 1;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkResult result = vmaCreateImage(allocator_, &info, &allocation_info, &image, &allocation,
                                     nullptr);
    if (result != VK_SUCCESS) {
        throw Error("image allocation of " + std::to_string(width) + "x" +
                    std::to_string(height) + " failed (VkResult " + std::to_string(result) + ")");
    }

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = info.format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    result = fns.CreateImageView(device_handle_, &view_info, nullptr, &view);
    if (result != VK_SUCCESS) {
        vmaDestroyImage(allocator_, image, allocation);
        throw Error(vk_result_string("vkCreateImageView", result));
    }

    auto handle = std::make_shared<VulkanImage>(this, image, allocation, view, width, height,
                                                format);
    if (pixels != nullptr) {
        handle->upload(pixels, static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                    bytes_per_pixel(format));
    }
    return handle;
}

std::shared_ptr<SamplerHandle> VulkanBackend::create_sampler(Filter filter,
                                                             AddressMode address) {
    const std::uint32_t key = (static_cast<std::uint32_t>(filter) << 8u) |
                              static_cast<std::uint32_t>(address);
    {
        std::lock_guard<std::mutex> lock(sampler_mutex_);
        auto existing = samplers_.find(key);
        if (existing != samplers_.end()) {
            return existing->second;
        }
    }

    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    info.minFilter = info.magFilter;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    const VkSamplerAddressMode mode = address == AddressMode::Repeat
                                          ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                      : address == AddressMode::MirroredRepeat
                                          ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT
                                          : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeU = mode;
    info.addressModeV = mode;
    info.addressModeW = mode;
    info.minLod = 0.0f;
    info.maxLod = 0.0f;
    info.compareEnable = VK_FALSE;
    info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

    VkSampler sampler = VK_NULL_HANDLE;
    const VkResult result =
        instance_->fns.CreateSampler(device_handle_, &info, nullptr, &sampler);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkCreateSampler", result));
    }
    auto handle = std::shared_ptr<SamplerHandle>(
        new VulkanSampler(this, sampler, filter, address));
    std::lock_guard<std::mutex> lock(sampler_mutex_);
    // Another thread may have won the race; keep the first object so every
    // caller sees the same handle, and destroy the loser.
    auto inserted = samplers_.emplace(key, handle);
    if (!inserted.second) {
        instance_->fns.DestroySampler(device_handle_, sampler, nullptr);
        return inserted.first->second;
    }
    return handle;
}

VulkanImage::VulkanImage(VulkanBackend* backend, VkImage image, VmaAllocation allocation,
                         VkImageView view, int width, int height, Format format)
    : backend_(backend), image_(image), allocation_(allocation), view_(view), width_(width),
      height_(height), format_(format) {}

VulkanImage::~VulkanImage() {
    if (image_ != VK_NULL_HANDLE) {
        const Functions& fns = backend_->fns();
        // A dispatch submitted before the last reference dropped may still read
        // the image; every fence is waited for before the handle goes. The
        // noexcept drain is what a destructor may call: the throwing one would
        // terminate the process when the device is lost, which is exactly the
        // case this unwind runs in.
        (void)backend_->wait_all_pending_noexcept();
        fns.DestroyImageView(backend_->device(), view_, nullptr);
        vmaDestroyImage(backend_->allocator(), image_, allocation_);
    }
}

void VulkanImage::upload(const void* pixels, std::size_t bytes) {
    if (pixels == nullptr && bytes != 0) {
        throw Error("image upload from a null pointer");
    }
    const std::size_t expected = static_cast<std::size_t>(width_) *
                                 static_cast<std::size_t>(height_) * bytes_per_pixel(format_);
    if (bytes != expected) {
        throw Error("image upload needs exactly " + std::to_string(expected) + " byte(s) for " +
                    std::to_string(width_) + "x" + std::to_string(height_) + " " +
                    format_name(format_) + ", got " + std::to_string(bytes));
    }
    // A re-upload starts from SHADER_READ_ONLY (the layout the first upload
    // left); the copy still overwrites every texel, so the old contents do
    // not survive. UNDEFINED on the first call lets the driver discard.
    const VkImageLayout previous =
        uploaded_ ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    backend_->upload_image(image_, width_, height_, format_, pixels, bytes, previous);
    uploaded_ = true;
}

void VulkanImage::download(void* pixels, std::size_t bytes) const {
    if (!uploaded_) {
        throw Error("download from an image that has never been uploaded");
    }
    if (pixels == nullptr && bytes != 0) {
        throw Error("image download to a null pointer");
    }
    const std::size_t expected = static_cast<std::size_t>(width_) *
                                 static_cast<std::size_t>(height_) * bytes_per_pixel(format_);
    if (bytes != expected) {
        throw Error("image download needs exactly " + std::to_string(expected) +
                    " byte(s) for " + std::to_string(width_) + "x" + std::to_string(height_) +
                    " " + format_name(format_) + ", got " + std::to_string(bytes));
    }
    backend_->download_image(image_, width_, height_, format_, pixels, bytes);
}

VulkanSampler::VulkanSampler(VulkanBackend* backend, VkSampler sampler, Filter filter,
                             AddressMode address)
    : backend_(backend), sampler_(sampler), filter_(filter), address_(address) {}

VulkanSampler::~VulkanSampler() = default;

VulkanBuffer::VulkanBuffer(VulkanBackend* backend, VkBuffer buffer, VmaAllocation allocation,
                           VmaAllocationInfo allocation_info, std::size_t size,
                           std::size_t reserved, bool staged)
    : backend_(backend),
      buffer_(buffer),
      allocation_(allocation),
      mapped_(allocation_info.pMappedData),
      size_(size),
      reserved_(reserved),
      staged_(staged) {}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        // A submission may still read or write this memory -- including one
        // whose fence wait threw (a device-lost report), which unwinds the
        // caller straight into this destructor. Freeing while the queue still
        // references the range is a use-after-free the driver punishes with a
        // segfault, so drain first. The catch: a destructor is implicitly
        // noexcept, so the throwing drain would be std::terminate on exactly
        // the lost-device path this exists for.
        (void)backend_->wait_all_pending_noexcept();
        vmaDestroyBuffer(backend_->allocator(), buffer_, allocation_);
    }
}

void VulkanBuffer::upload(const void* data, std::size_t bytes, std::size_t offset) {
    if (data == nullptr && bytes != 0) {
        throw Error("upload from a null pointer");
    }
    if (offset > size_ || bytes > size_ - offset) {
        throw Error("upload range " + std::to_string(offset) + "+" + std::to_string(bytes) +
                    " exceeds buffer size " + std::to_string(size_));
    }
    if (bytes == 0) {
        return;
    }
    if (staged_) {
        backend_->upload_staged(buffer_, reserved_, data, bytes, offset);
        return;
    }
    if (mapped_ != nullptr) {
        // Unified-memory aliasing: the host write must not race an in-flight
        // read of this buffer, so drain any outstanding submissions first.
        backend_->wait_all_pending();
        std::memcpy(static_cast<std::uint8_t*>(mapped_) + offset, data, bytes);
        vmaFlushAllocation(backend_->allocator(), allocation_, offset, bytes);
        return;
    }
    void* target = nullptr;
    const VkResult mapped = vmaMapMemory(backend_->allocator(), allocation_, &target);
    if (mapped == VK_SUCCESS && target != nullptr) {
        backend_->wait_all_pending();
        std::memcpy(static_cast<std::uint8_t*>(target) + offset, data, bytes);
        vmaFlushAllocation(backend_->allocator(), allocation_, offset, bytes);
        vmaUnmapMemory(backend_->allocator(), allocation_);
        return;
    }
    // Not host-mappable (device-only memory on a discrete GPU): stage.
    backend_->upload_staged(buffer_, reserved_, data, bytes, offset);
}

void VulkanBuffer::download(void* data, std::size_t bytes, std::size_t offset) const {
    if (data == nullptr && bytes != 0) {
        throw Error("download to a null pointer");
    }
    if (offset > size_ || bytes > size_ - offset) {
        throw Error("download range " + std::to_string(offset) + "+" + std::to_string(bytes) +
                    " exceeds buffer size " + std::to_string(size_));
    }
    if (bytes == 0) {
        return;
    }
    if (staged_) {
        backend_->download_staged(buffer_, reserved_, data, bytes, offset);
        return;
    }
    if (mapped_ != nullptr) {
        // Drain in-flight work: the GPU may still be writing this range.
        backend_->wait_all_pending();
        vmaInvalidateAllocation(backend_->allocator(), allocation_, offset, bytes);
        std::memcpy(data, static_cast<const std::uint8_t*>(mapped_) + offset, bytes);
        return;
    }
    void* source = nullptr;
    const VkResult mapped = vmaMapMemory(backend_->allocator(), allocation_, &source);
    if (mapped == VK_SUCCESS && source != nullptr) {
        backend_->wait_all_pending();
        vmaInvalidateAllocation(backend_->allocator(), allocation_, offset, bytes);
        std::memcpy(data, static_cast<const std::uint8_t*>(source) + offset, bytes);
        vmaUnmapMemory(backend_->allocator(), allocation_);
        return;
    }
    // Device-only memory: copy out through the staging ring.
    backend_->download_staged(buffer_, reserved_, data, bytes, offset);
}

// -- staging ring ------------------------------------------------------------
//
// One fixed host-visible buffer (kStagingBytes) carries every device-local
// transfer. Every transfer's fence is waited for before the host touches the
// staging region again, so there is never an outstanding copy into or out of
// the region: the offset is an explicit bounds-checked index that advances by
// the chunk size and resets to 0 when it would overrun the ring. Window sizes
// are rounded to the 4 bytes VkBufferCopy requires, and a request larger than
// the ring is split into ring-sized chunks copied one at a time. There is no
// modulo and no dangling region, so no live ticket can alias what a transfer
// is reading or writing.

void VulkanBackend::ensure_staging() {
    if (staging_ != VK_NULL_HANDLE && transfer_pool_ != VK_NULL_HANDLE &&
        transfer_fence_ != VK_NULL_HANDLE) {
        return;
    }
    const Functions& fns = instance_->fns;
    // A failed previous attempt may have left the buffer without its command
    // pool: tear it down so this call rebuilds all three pieces.
    if (staging_ != VK_NULL_HANDLE && (transfer_pool_ == VK_NULL_HANDLE ||
                                       transfer_fence_ == VK_NULL_HANDLE)) {
        if (transfer_fence_ != VK_NULL_HANDLE) {
            fns.DestroyFence(device_handle_, transfer_fence_, nullptr);
            transfer_fence_ = VK_NULL_HANDLE;
        }
        if (transfer_pool_ != VK_NULL_HANDLE) {
            fns.DestroyCommandPool(device_handle_, transfer_pool_, nullptr);
            transfer_pool_ = VK_NULL_HANDLE;
            transfer_commands_ = VK_NULL_HANDLE;
        }
        vmaDestroyBuffer(allocator_, staging_, staging_allocation_);
        staging_ = VK_NULL_HANDLE;
        staging_allocation_ = VK_NULL_HANDLE;
        staging_mapped_ = nullptr;
    }
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = static_cast<VkDeviceSize>(kStagingBytes);
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo result_info{};
    const VkResult created = vmaCreateBuffer(allocator_, &info, &allocation_info, &staging_,
                                             &staging_allocation_, &result_info);
    if (created != VK_SUCCESS || result_info.pMappedData == nullptr) {
        staging_ = VK_NULL_HANDLE;
        throw Error("staging buffer of " + std::to_string(kStagingBytes) +
                    " bytes failed (VkResult " + std::to_string(created) + ")");
    }
    staging_mapped_ = result_info.pMappedData;
    staging_bytes_ = kStagingBytes;
    staging_offset_ = 0;

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_;
    VkResult result = fns.CreateCommandPool(device_handle_, &pool_info, nullptr, &transfer_pool_);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkCreateCommandPool", result));
    }
    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = transfer_pool_;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    result = fns.AllocateCommandBuffers(device_handle_, &allocate_info, &transfer_commands_);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkAllocateCommandBuffers", result));
    }
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = fns.CreateFence(device_handle_, &fence_info, nullptr, &transfer_fence_);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkCreateFence", result));
    }
}

/// Runs one staging copy: a whole region of the staging buffer to/from the
/// device buffer, then waits for its fence.
void VulkanBackend::run_transfer(VkBuffer device_buffer, VkDeviceSize staging_offset,
                                 VkDeviceSize device_offset, VkDeviceSize bytes,
                                 bool to_device) {
    const Functions& fns = instance_->fns;
    VkResult result = fns.ResetCommandBuffer(transfer_commands_, 0);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkResetCommandBuffer", result));
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = fns.BeginCommandBuffer(transfer_commands_, &begin);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkBeginCommandBuffer", result));
    }
    VkBufferCopy copy{};
    copy.srcOffset = to_device ? staging_offset : device_offset;
    copy.dstOffset = to_device ? device_offset : staging_offset;
    copy.size = bytes;
    if (to_device) {
        // The host wrote the staging region: make those writes visible to the
        // transfer read that follows (no automatic host->transfer sync).
        VkMemoryBarrier host_write{};
        host_write.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_write.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_write.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        fns.CmdPipelineBarrier(transfer_commands_, VK_PIPELINE_STAGE_HOST_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &host_write, 0, nullptr, 0,
                               nullptr);
    }
    fns.CmdCopyBuffer(transfer_commands_, to_device ? staging_ : device_buffer,
                      to_device ? device_buffer : staging_, 1, &copy);
    if (!to_device) {
        // The copy writes the host-visible staging buffer: make it visible to
        // the host read that follows.
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = staging_;
        barrier.offset = staging_offset;
        barrier.size = bytes;
        fns.CmdPipelineBarrier(transfer_commands_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0,
                               nullptr);
    } else {
        // The device buffer's writes must be visible to later dispatches. The
        // compute submission that reads it starts after this transfer's fence,
        // and a submit-order dependency covers queue-side visibility, but an
        // explicit transfer->compute barrier keeps the rule stated. (No host
        // access mask here: VK_ACCESS_HOST_* only pairs with the HOST stage.)
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = device_buffer;
        barrier.offset = device_offset;
        barrier.size = bytes;
        fns.CmdPipelineBarrier(transfer_commands_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier,
                               0, nullptr);
    }
    result = fns.EndCommandBuffer(transfer_commands_);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkEndCommandBuffer", result));
    }
    result = fns.ResetFences(device_handle_, 1, &transfer_fence_);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkResetFences", result));
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transfer_commands_;
    {
        // Queue submission is serialised against compute submissions, as
        // vkQueueSubmit requires external synchronisation.
        std::lock_guard<std::mutex> lock(mutex_);
        result = fns.QueueSubmit(queues_[0], 1, &submit, transfer_fence_);
    }
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkQueueSubmit", result));
    }
    wait_fence(transfer_fence_);
}

void VulkanBackend::ensure_image_transfer() {
    if (image_pool_ != VK_NULL_HANDLE && image_commands_ != VK_NULL_HANDLE &&
        image_fence_ != VK_NULL_HANDLE) {
        return;
    }
    const Functions& fns = instance_->fns;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_;
    VkResult result = fns.CreateCommandPool(device_handle_, &pool_info, nullptr, &image_pool_);
    if (result != VK_SUCCESS) {
        image_pool_ = VK_NULL_HANDLE;
        throw Error(vk_result_string("vkCreateCommandPool", result));
    }
    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = image_pool_;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    result = fns.AllocateCommandBuffers(device_handle_, &allocate_info, &image_commands_);
    if (result != VK_SUCCESS) {
        fns.DestroyCommandPool(device_handle_, image_pool_, nullptr);
        image_pool_ = VK_NULL_HANDLE;
        image_commands_ = VK_NULL_HANDLE;
        throw Error(vk_result_string("vkAllocateCommandBuffers", result));
    }
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = fns.CreateFence(device_handle_, &fence_info, nullptr, &image_fence_);
    if (result != VK_SUCCESS) {
        fns.DestroyCommandPool(device_handle_, image_pool_, nullptr);
        image_pool_ = VK_NULL_HANDLE;
        image_commands_ = VK_NULL_HANDLE;
        image_fence_ = VK_NULL_HANDLE;
        throw Error(vk_result_string("vkCreateFence", result));
    }
}

void VulkanBackend::upload_staged(VkBuffer destination, std::size_t capacity, const void* data,
                                  std::size_t bytes, std::size_t offset) {
    transfer_window(destination, capacity, offset, bytes, data, nullptr);
}

void VulkanBackend::download_staged(VkBuffer source, std::size_t capacity, void* data,
                                    std::size_t bytes, std::size_t offset) {
    transfer_window(source, capacity, offset, bytes, nullptr, data);
}

// -- image copies ------------------------------------------------------------
//
// A temporary host-visible buffer carries the whole image and is freed as soon
// as the copy has been waited for, so images never compete with the staging
// ring for offsets and can be any size (subject to the caller's budget). The
// command buffer transitions into the transfer layout, copies, then
// transitions to SHADER_READ_ONLY_OPTIMAL and ends with a compute-visible
// barrier, so the next dispatch may read the image with no extra transition.

void VulkanBackend::upload_image(VkImage image, int width, int height, Format format,
                                 const void* pixels, std::size_t bytes,
                                 VkImageLayout old_layout) {
    if (bytes == 0) {
        return;
    }
    image_transfer(image, width, height, format, true, pixels, nullptr, old_layout);
}

void VulkanBackend::download_image(VkImage image, int width, int height, Format format,
                                   void* pixels, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    image_transfer(image, width, height, format, false, nullptr, pixels,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanBackend::image_transfer(VkImage image, int width, int height, Format format,
                                   bool to_device, const void* source, void* destination,
                                   VkImageLayout old_layout) {
    const Functions& fns = instance_->fns;
    const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                              bytes_per_pixel(format);
    std::lock_guard<std::mutex> lock(transfer_mutex_);
    // In-flight dispatches may still read the image.
    wait_all_pending();

    // Temporary host-visible staging buffer for the whole image.
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = static_cast<VkDeviceSize>(bytes);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo staging_info{};
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;
    VkResult result = vmaCreateBuffer(allocator_, &buffer_info, &allocation_info, &staging,
                                      &staging_allocation, &staging_info);
    if (result != VK_SUCCESS || staging_info.pMappedData == nullptr) {
        if (staging != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, staging, staging_allocation);
        }
        throw Error("image staging buffer of " + std::to_string(bytes) +
                    " bytes failed (VkResult " + std::to_string(result) + ")");
    }
    if (to_device) {
        std::memcpy(staging_info.pMappedData, source, bytes);
        vmaFlushAllocation(allocator_, staging_allocation, 0, bytes);
    }

    // The image copy has its own command buffer and fence (see
    // ensure_image_transfer), so it neither allocates the 64 MiB staging ring
    // nor contends with the buffer ring's command buffer.
    ensure_image_transfer();
    VkCommandBuffer commands = image_commands_;
    VkFence fence = image_fence_;

    result = fns.ResetCommandBuffer(commands, 0);
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator_, staging, staging_allocation);
        throw Error(vk_result_string("vkResetCommandBuffer", result));
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = fns.BeginCommandBuffer(commands, &begin);
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator_, staging, staging_allocation);
        throw Error(vk_result_string("vkBeginCommandBuffer", result));
    }

    // 1. Transition to the transfer layout.
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = to_device ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                  : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        // First use: no prior contents to keep.
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fns.CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                               &barrier);
    } else {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = to_device ? VK_ACCESS_TRANSFER_WRITE_BIT
                                          : VK_ACCESS_TRANSFER_READ_BIT;
        fns.CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                               &barrier);
    }

    // 2. The copy itself.
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;    // tightly packed
    region.bufferImageHeight = 0;  // tightly packed
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<std::uint32_t>(width),
                          static_cast<std::uint32_t>(height), 1};
    if (to_device) {
        // Host writes must be visible to the transfer read.
        VkMemoryBarrier host_write{};
        host_write.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_write.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_write.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        fns.CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_HOST_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &host_write, 0, nullptr, 0,
                               nullptr);
        fns.CmdCopyBufferToImage(commands, staging, image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else {
        fns.CmdCopyImageToBuffer(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging,
                                 1, &region);
    }

    // 3. Transition to SHADER_READ_ONLY_OPTIMAL, visible to the next dispatch.
    VkImageMemoryBarrier finish{};
    finish.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    finish.oldLayout = to_device ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                 : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    finish.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    finish.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finish.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finish.image = image;
    finish.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    finish.subresourceRange.baseMipLevel = 0;
    finish.subresourceRange.levelCount = 1;
    finish.subresourceRange.baseArrayLayer = 0;
    finish.subresourceRange.layerCount = 1;
    finish.srcAccessMask = to_device ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    finish.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    fns.CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                           &finish);
    if (!to_device) {
        // The copy wrote the staging buffer: make it visible to the host read.
        VkBufferMemoryBarrier host_read{};
        host_read.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        host_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host_read.buffer = staging;
        host_read.offset = 0;
        host_read.size = bytes;
        fns.CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host_read, 0,
                               nullptr);
    }

    result = fns.EndCommandBuffer(commands);
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator_, staging, staging_allocation);
        throw Error(vk_result_string("vkEndCommandBuffer", result));
    }
    result = fns.ResetFences(device_handle_, 1, &fence);
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator_, staging, staging_allocation);
        throw Error(vk_result_string("vkResetFences", result));
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    {
        std::lock_guard<std::mutex> queue_lock(mutex_);
        result = fns.QueueSubmit(queues_[0], 1, &submit, fence);
    }
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator_, staging, staging_allocation);
        throw Error(vk_result_string("vkQueueSubmit", result));
    }
    // wait_fence may throw; the staging buffer must still be freed, so capture
    // the exception, clean up and rethrow.
    try {
        wait_fence(fence);
    } catch (...) {
        vmaDestroyBuffer(allocator_, staging, staging_allocation);
        throw;
    }
    if (!to_device) {
        vmaInvalidateAllocation(allocator_, staging_allocation, 0, bytes);
        std::memcpy(destination, staging_info.pMappedData, bytes);
    }
    vmaDestroyBuffer(allocator_, staging, staging_allocation);
}

/// Aligns a host/device byte range to the 4-byte granularity VkBufferCopy
/// requires, clipping the aligned window to the allocation.
namespace {
struct AlignedWindow {
    std::size_t offset = 0;  ///< aligned device (and staging) offset
    std::size_t bytes = 0;   ///< aligned byte count
    std::size_t lead = 0;    ///< bytes at the window start before the caller's range
};

AlignedWindow align_window(std::size_t offset, std::size_t bytes, std::size_t capacity) {
    constexpr std::size_t kAlignment = 4;
    AlignedWindow window;
    window.offset = offset - (offset % kAlignment);
    window.lead = offset - window.offset;
    const std::size_t end = offset + bytes;
    std::size_t aligned_end = end + ((kAlignment - (end % kAlignment)) % kAlignment);
    aligned_end = std::min(aligned_end, capacity);
    window.bytes = aligned_end > window.offset ? aligned_end - window.offset : 0;
    return window;
}
}  // namespace

void VulkanBackend::transfer_window(VkBuffer device_buffer, std::size_t capacity,
                                    std::size_t device_offset, std::size_t bytes,
                                    const void* upload_src, void* download_dst) {
    std::lock_guard<std::mutex> lock(transfer_mutex_);
    // In-flight dispatches may still touch the range.
    wait_all_pending();
    ensure_staging();
    const bool to_device = upload_src != nullptr;
    // The caller already range-checked against the buffer's size.
    const AlignedWindow window = align_window(device_offset, bytes, capacity);
    if (window.bytes == 0) {
        return;
    }
    const std::size_t lead = window.lead;
    const bool aligned = lead == 0 && window.bytes == bytes;
    if (to_device && !aligned) {
        // VkBufferCopy is 4-byte granular: read the aligned window, patch the
        // requested bytes into it, write it back. Rare (misaligned host offset)
        // and obviously correct; the aligned case below is a single copy.
        std::vector<std::uint8_t> staging(window.bytes);
        stream_window(device_buffer, window.offset, window.bytes, staging.data(), true);
        std::memcpy(staging.data() + lead, upload_src, bytes);
        stream_window(device_buffer, window.offset, window.bytes, staging.data(), false);
        return;
    }
    if (to_device) {
        stream_window(device_buffer, window.offset, window.bytes, upload_src, false);
        return;
    }
    // Download: fetch the aligned window, then hand the caller its slice.
    std::vector<std::uint8_t> staging(window.bytes);
    stream_window(device_buffer, window.offset, window.bytes, staging.data(), true);
    std::memcpy(download_dst, staging.data() + lead, bytes);
}

void VulkanBackend::stream_window(VkBuffer device_buffer, std::size_t window_offset,
                                  std::size_t window_bytes, const void* host_buffer, bool read) {
    // stream_window owns the host pointer; when reading, it writes through it,
    // so the const qualification is only on the callers' convenience.
    std::uint8_t* host = const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(host_buffer));
    // Every transfer is waited for before stream_window returns, so the ring
    // region is never in use by an outstanding copy: wrapping is always safe.
    // window_offset and window_bytes are 4-byte multiples (align_window), and
    // chunking preserves that, because VkBufferCopy requires both.
    constexpr std::size_t kCopyAlignment = 4;
    constexpr std::size_t kFlushAlignment = 64;
    std::size_t done = 0;
    while (done < window_bytes) {
        std::size_t reachable = staging_bytes_ - staging_offset_;
        reachable -= reachable % kCopyAlignment;
        if (reachable == 0) {
            // Fewer than 4 bytes left at the tail: start the ring over.
            staging_offset_ = 0;
            reachable = staging_bytes_;
        }
        const std::size_t chunk = std::min(window_bytes - done, reachable);
        if (read) {
            run_transfer(device_buffer, staging_offset_, window_offset + done, chunk, false);
            // Invalidate the whole atom-aligned window that covers the chunk.
            const std::size_t flush_offset = staging_offset_ - (staging_offset_ % kFlushAlignment);
            std::size_t flush_end = staging_offset_ + chunk;
            flush_end += (kFlushAlignment - (flush_end % kFlushAlignment)) % kFlushAlignment;
            flush_end = std::min(flush_end, staging_bytes_);
            vmaInvalidateAllocation(allocator_, staging_allocation_, flush_offset,
                                    flush_end - flush_offset);
            std::memcpy(host + done,
                        static_cast<const std::uint8_t*>(staging_mapped_) + staging_offset_,
                        chunk);
        } else {
            std::memcpy(static_cast<std::uint8_t*>(staging_mapped_) + staging_offset_,
                        static_cast<const std::uint8_t*>(host_buffer) + done, chunk);
            const std::size_t flush_offset = staging_offset_ - (staging_offset_ % kFlushAlignment);
            std::size_t flush_end = staging_offset_ + chunk;
            flush_end += (kFlushAlignment - (flush_end % kFlushAlignment)) % kFlushAlignment;
            flush_end = std::min(flush_end, staging_bytes_);
            vmaFlushAllocation(allocator_, staging_allocation_, flush_offset,
                               flush_end - flush_offset);
            run_transfer(device_buffer, staging_offset_, window_offset + done, chunk, true);
        }
        staging_offset_ += chunk;
        if (staging_offset_ >= staging_bytes_) {
            staging_offset_ = 0;
        }
        done += chunk;
    }
}

}  // namespace alphashape::gpu::detail
