#pragma once

// Vulkan instance ownership and physical-device discovery.

#include "gpu/vulkan/vk_headers.hpp"

#include <memory>
#include <string>
#include <vector>

#include "gpu/gpu.hpp"
#include "gpu/vulkan/loader.hpp"

namespace alphashape::gpu::detail {

struct Instance {
    VkInstance handle = VK_NULL_HANDLE;
    Functions fns;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    bool validation = false;

    ~Instance();
    Instance() = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
};

/// Loads the loader, configures the ICD and creates an instance. Throws
/// gpu::Error with the first failing step's reason.
std::shared_ptr<Instance> create_instance(bool validation);

/// Physical device plus its parsed DeviceInfo, in enumeration order.
struct DeviceRecord {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    DeviceInfo info;
    std::uint64_t device_local_bytes = 0;
    std::uint32_t queue_family = 0;
    /// Queues exposed by the selected compute family (at least 1).
    std::uint32_t queue_count = 1;
};

/// Enumerates physical devices. Throws gpu::Error.
std::vector<DeviceRecord> list_devices(const Instance& instance);

/// Chooses a device: explicit index, else the discrete device with the most
/// memory, else the integrated one with the most memory. Applies
/// required_memory as a quiet filter with a loud explanation on failure.
std::size_t select_device(const std::vector<DeviceRecord>& devices,
                          const ContextOptions& options);

}  // namespace alphashape::gpu::detail
