// Vulkan backend stubs used when ALPHASHAPE_GPU=OFF or the SDK is missing.
// The public API still exists; every entry point reports unavailability.

#include "gpu/vulkan/backend.hpp"

#include "gpu/gpu.hpp"

namespace alphashape::gpu::detail {

std::shared_ptr<Backend> create_backend(const ContextOptions& /*options*/) {
    throw Error("GPU support disabled at build time (ALPHASHAPE_GPU=OFF)");
}

bool enumerate_devices(std::vector<DeviceInfo>* devices, std::string* reason) {
    devices->clear();
    if (reason != nullptr) {
        *reason = "GPU support disabled at build time (ALPHASHAPE_GPU=OFF)";
    }
    return false;
}

bool probe_loader(const std::string& /*path*/, std::string* reason) {
    if (reason != nullptr) {
        *reason = "GPU support disabled at build time (ALPHASHAPE_GPU=OFF)";
    }
    return false;
}

const KernelEntry* find_kernel(const std::string& /*name*/) { return nullptr; }
std::vector<std::string> all_kernel_names() { return {}; }

}  // namespace alphashape::gpu::detail
