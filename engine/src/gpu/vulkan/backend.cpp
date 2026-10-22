#include "gpu/gpu.hpp"

#include <memory>
#include <string>
#include <vector>

#include "gpu/vulkan/backend.hpp"

#include "gpu/vulkan/device.hpp"
#include "gpu/vulkan/instance.hpp"
#include "gpu/vulkan/loader.hpp"

namespace alphashape::gpu::detail {

bool enumerate_devices(std::vector<DeviceInfo>* devices, std::string* reason) {
    devices->clear();
    try {
        if (!ensure_icd_configured(reason)) {
            return false;
        }
        auto instance = create_instance(false);
        for (const DeviceRecord& record : list_devices(*instance)) {
            devices->push_back(record.info);
        }
        return true;
    } catch (const std::exception& error) {
        if (reason != nullptr) {
            *reason = error.what();
        }
        return false;
    }
}

std::shared_ptr<Backend> create_backend(const ContextOptions& options) {
    std::string reason;
    if (!ensure_icd_configured(&reason)) {
        throw Error("GPU unavailable: " + reason);
    }
    auto instance = create_instance(options.validation);
    const std::vector<DeviceRecord> devices = list_devices(*instance);
    const std::size_t chosen = select_device(devices, options);
    return std::make_shared<VulkanBackend>(std::move(instance), devices[chosen],
                                           options.validation);
}

}  // namespace alphashape::gpu::detail
