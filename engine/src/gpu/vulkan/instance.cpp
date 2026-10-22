#include "gpu/vulkan/instance.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "base/utils/log.hpp"
#include "gpu/gpu_config.hpp"

namespace alphashape::gpu::detail {
namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* /*user*/) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        add_validation_error();
        log(LogLevel::Error) << "Vulkan validation: " << (data->pMessage ? data->pMessage : "");
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        log(LogLevel::Warn) << "Vulkan validation: " << (data->pMessage ? data->pMessage : "");
    }
    // Validation messages must never abort the process: the driver may still
    // produce correct results (and the fallback path covers the rest).
    return VK_FALSE;
}

bool has_instance_extension(const Instance& instance, const char* name) {
    std::uint32_t count = 0;
    if (instance.fns.EnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> properties(count);
    if (instance.fns.EnumerateInstanceExtensionProperties(nullptr, &count, properties.data()) !=
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

bool has_layer(const Instance& instance, const char* name) {
    std::uint32_t count = 0;
    if (instance.fns.EnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkLayerProperties> properties(count);
    if (instance.fns.EnumerateInstanceLayerProperties(&count, properties.data()) != VK_SUCCESS) {
        return false;
    }
    for (const VkLayerProperties& property : properties) {
        if (std::strcmp(property.layerName, name) == 0) {
            return true;
        }
    }
    return false;
}

std::uint32_t api_version_1_2() { return 0x00402000u; }  // VK_MAKE_API_VERSION(0,1,2,0)

std::string version_string(std::uint32_t api_version) {
    // VK_VERSION_MAJOR/MINOR/PATCH are macro wrappers around masked shifts;
    // spell the masks out so -Wold-style-cast stays quiet.
    return std::to_string((api_version >> 22u) & 0x7Fu) + "." +
           std::to_string((api_version >> 12u) & 0x3FFu) + "." +
           std::to_string(api_version & 0xFFFu);
}

}  // namespace

Instance::~Instance() {
    if (messenger != VK_NULL_HANDLE && fns.DestroyDebugUtilsMessengerEXT != nullptr) {
        fns.DestroyDebugUtilsMessengerEXT(handle, messenger, nullptr);
    }
    if (handle != VK_NULL_HANDLE && fns.DestroyInstance != nullptr) {
        fns.DestroyInstance(handle, nullptr);
    }
}

std::shared_ptr<Instance> create_instance(bool validation) {
    auto instance = std::make_shared<Instance>();
    std::string reason;
    const Loader* loader = load_library(&reason);
    if (loader == nullptr) {
        throw Error("Vulkan loader unavailable: " + reason);
    }
    instance->fns = loader->fns;
    if (!ensure_icd_configured(&reason)) {
        throw Error("Vulkan ICD unavailable: " + reason);
    }

    // MoltenVK is a portability driver: without the enumeration extension no
    // physical device is reported on this platform.
    std::vector<const char*> extensions;
    std::vector<const char*> layers;
    if (has_instance_extension(*instance, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
    if (validation) {
        const bool layer_present = has_layer(*instance, "VK_LAYER_KHRONOS_validation");
        const bool debug_utils =
            has_instance_extension(*instance, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (layer_present && debug_utils) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            instance->validation = true;
            log(LogLevel::Info) << "Vulkan validation layer enabled";
        } else {
            log(LogLevel::Warn) << "Vulkan validation layers requested but "
                                << (layer_present ? "VK_EXT_debug_utils" : "VK_LAYER_KHRONOS_validation")
                                << " is not available; continuing without them";
        }
    }

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "alphashape";
    application.applicationVersion = 0;
    application.pEngineName = "alphashape";
    application.engineVersion = 0;
    application.apiVersion = api_version_1_2();

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application;
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    create_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    if (std::find(extensions.begin(), extensions.end(),
                  VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != extensions.end()) {
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    const VkResult created = instance->fns.CreateInstance(&create_info, nullptr, &instance->handle);
    if (created != VK_SUCCESS) {
        throw Error("vkCreateInstance failed with VkResult " + std::to_string(created));
    }
    // The loader returns instance-level commands only once the instance exists.
    if (!resolve_instance(instance->fns.GetInstanceProcAddr, instance->handle, &instance->fns,
                          &reason)) {
        throw Error("Vulkan entry points unavailable: " + reason);
    }

    if (instance->validation && instance->fns.CreateDebugUtilsMessengerEXT != nullptr) {
        VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
        messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        messenger_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messenger_info.pfnUserCallback = debug_callback;
        const VkResult messenger_result = instance->fns.CreateDebugUtilsMessengerEXT(
            instance->handle, &messenger_info, nullptr, &instance->messenger);
        if (messenger_result != VK_SUCCESS) {
            log(LogLevel::Warn) << "vkCreateDebugUtilsMessengerEXT failed with VkResult "
                                << messenger_result;
        }
    }
    return instance;
}

std::vector<DeviceRecord> list_devices(const Instance& instance) {
    std::uint32_t count = 0;
    VkResult result = instance.fns.EnumeratePhysicalDevices(instance.handle, &count, nullptr);
    if (result != VK_SUCCESS || count == 0) {
        throw Error(count == 0 ? "no Vulkan physical devices reported (driver/ICD problem)"
                               : "vkEnumeratePhysicalDevices failed with VkResult " +
                                     std::to_string(result));
    }
    std::vector<VkPhysicalDevice> handles(count);
    result = instance.fns.EnumeratePhysicalDevices(instance.handle, &count, handles.data());
    if (result != VK_SUCCESS) {
        throw Error("vkEnumeratePhysicalDevices failed with VkResult " + std::to_string(result));
    }
    handles.resize(count);

    std::vector<DeviceRecord> records;
    records.reserve(handles.size());
    for (VkPhysicalDevice handle : handles) {
        VkPhysicalDeviceProperties properties{};
        instance.fns.GetPhysicalDeviceProperties(handle, &properties);

        VkPhysicalDeviceMemoryProperties memory{};
        instance.fns.GetPhysicalDeviceMemoryProperties(handle, &memory);
        std::uint64_t device_local = 0;
        for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
            if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                device_local = std::max<std::uint64_t>(device_local, memory.memoryHeaps[heap].size);
            }
        }

        std::uint32_t family_count = 0;
        instance.fns.GetPhysicalDeviceQueueFamilyProperties(handle, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        instance.fns.GetPhysicalDeviceQueueFamilyProperties(handle, &family_count, families.data());
        std::uint32_t queue_family = 0;
        std::uint32_t queue_count = 0;
        bool have_queue = false;
        for (std::uint32_t family = 0; family < family_count; ++family) {
            if (families[family].queueCount > 0 &&
                (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                queue_family = family;
                queue_count = families[family].queueCount;
                have_queue = true;
                break;
            }
        }
        if (!have_queue) {
            continue;
        }

        DeviceRecord record;
        record.handle = handle;
        record.device_local_bytes = device_local;
        record.queue_family = queue_family;
        record.queue_count = std::max<std::uint32_t>(queue_count, 1U);
        record.info.name = properties.deviceName;
        record.info.vendor_id = properties.vendorID;
        record.info.device_id = properties.deviceID;
        record.info.integrated =
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        record.info.software = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        record.info.memory_bytes = device_local;
        record.info.api_version = properties.apiVersion;
        record.info.max_workgroup_size = {properties.limits.maxComputeWorkGroupSize[0],
                                          properties.limits.maxComputeWorkGroupSize[1],
                                          properties.limits.maxComputeWorkGroupSize[2]};
        record.info.max_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
        record.info.max_shared_memory = properties.limits.maxComputeSharedMemorySize;
        // Subgroup size is filled from VkPhysicalDeviceSubgroupProperties by the
        // backend once the device is created; the limit here is a safe fallback.
        record.info.subgroup_size = 32;
        records.push_back(std::move(record));
    }
    if (records.empty()) {
        throw Error("no Vulkan device with a compute queue");
    }
    return records;
}

std::size_t select_device(const std::vector<DeviceRecord>& devices,
                          const ContextOptions& options) {
    if (options.device_index >= 0) {
        if (static_cast<std::size_t>(options.device_index) >= devices.size()) {
            throw Error("requested GPU device index " + std::to_string(options.device_index) +
                        " but only " + std::to_string(devices.size()) + " device(s) exist");
        }
        const DeviceRecord& chosen = devices[static_cast<std::size_t>(options.device_index)];
        if (options.required_memory != 0 &&
            static_cast<double>(chosen.device_local_bytes) * 0.9 <
                static_cast<double>(options.required_memory)) {
            throw Error("GPU device '" + chosen.info.name + "' has " +
                        std::to_string(chosen.device_local_bytes) +
                        " bytes device-local but " +
                        std::to_string(options.required_memory) + " are required");
        }
        return static_cast<std::size_t>(options.device_index);
    }

    std::size_t best = 0;
    bool have_best = false;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const DeviceRecord& candidate = devices[i];
        if (options.required_memory != 0 &&
            static_cast<double>(candidate.device_local_bytes) * 0.9 <
                static_cast<double>(options.required_memory)) {
            continue;  // quiet pass: no log per rejected device
        }
        if (!have_best) {
            best = i;
            have_best = true;
            continue;
        }
        const DeviceRecord& current = devices[best];
        // Rank: hardware > software, and among hardware discrete > integrated;
        // ties go to the larger device-local heap. A CPU driver reports
        // VK_PHYSICAL_DEVICE_TYPE_CPU (neither integrated nor discrete) and
        // often reports a larger heap than a small iGPU, so without this it
        // wins on a machine that has a real GPU (AMD iGPU + llvmpipe on the
        // Linux test box).
        const auto rank = [](const DeviceRecord& device) {
            if (device.info.software) {
                return 0;
            }
            return device.info.integrated ? 1 : 2;
        };
        if (rank(candidate) > rank(current) ||
            (rank(candidate) == rank(current) &&
             candidate.device_local_bytes > current.device_local_bytes)) {
            best = i;
        }
    }
    if (!have_best) {
        // Loud pass: explain why every device was rejected.
        log(LogLevel::Error) << "no GPU device satisfies required_memory = "
                             << options.required_memory << " bytes (90% of device-local heap):";
        for (const DeviceRecord& device : devices) {
            log(LogLevel::Error) << "  '" << device.info.name << "': "
                                 << device.device_local_bytes << " bytes";
        }
        throw Error("no GPU device satisfies required_memory");
    }
    const DeviceInfo& selected = devices[best].info;
    log(LogLevel::Info) << "GPU device selected: " << selected.name << " ("
                        << version_string(selected.api_version) << ", "
                        << (selected.software ? "software"
                                              : selected.integrated ? "integrated" : "discrete")
                        << ", " << selected.memory_bytes << " bytes)";
    return best;
}

}  // namespace alphashape::gpu::detail
