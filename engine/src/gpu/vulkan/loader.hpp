#pragma once

// Runtime dlopen loading of the Vulkan loader. No libvulkan is linked: the
// headers are compiled with VK_NO_PROTOTYPES and every entry point is resolved
// through vkGetInstanceProcAddr into the table below.
//
// Load order: $ALPHASHAPE_VULKAN_LIBRARY, the SDK path baked at configure time
// (macOS SDK builds only), then the platform sonames -- "libvulkan.1.dylib" /
// "libvulkan.dylib" on macOS, "libvulkan.so.1" / "libvulkan.so" on Linux. When
// no ICD is configured through VK_ICD_FILENAMES / VK_DRIVER_FILES, a baked
// MoltenVK manifest (macOS) is exported as VK_DRIVER_FILES for this process
// before the instance is created. (We set the standard environment variable
// rather than using VK_LUNARG_direct_driver_loading, so the same path works for
// any loader.) With a distribution Vulkan install nothing is baked and the
// loader finds the system ICDs and layers itself.

#include "gpu/vulkan/vk_headers.hpp"

#include <string>

#include "base/platform/dynamic_library.hpp"

namespace alphashape::gpu::detail {

struct Functions {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;

    // Global commands (resolvable with a null instance).
    PFN_vkCreateInstance CreateInstance = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties = nullptr;
    PFN_vkEnumerateInstanceLayerProperties EnumerateInstanceLayerProperties = nullptr;

    // Instance and device commands (resolved against a live instance).
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkCreateDevice CreateDevice = nullptr;
    PFN_vkDestroyDevice DestroyDevice = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;

    PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT = nullptr;

    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkCreateBufferView CreateBufferView = nullptr;
    PFN_vkDestroyBufferView DestroyBufferView = nullptr;
    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkCreateImageView CreateImageView = nullptr;
    PFN_vkDestroyImageView DestroyImageView = nullptr;
    PFN_vkCreateSampler CreateSampler = nullptr;
    PFN_vkDestroySampler DestroySampler = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer = nullptr;
    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines CreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers FreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants CmdPushConstants = nullptr;
    PFN_vkCmdDispatch CmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer CmdCopyBuffer = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
    PFN_vkDestroyFence DestroyFence = nullptr;
    PFN_vkResetCommandBuffer ResetCommandBuffer = nullptr;
};

/// The process-global loader. Empty until load() succeeds.
///
/// The DynamicLibrary member is what keeps the module mapped: every resolved
/// entry point is a code address inside it, so it must outlive the Functions
/// table. It is therefore held for the life of the process, as the POSIX
/// version held its dlopen handle.
struct Loader {
    platform::DynamicLibrary library;
    Functions fns;
    std::string library_path;
    bool loaded = false;
};

/// Loads once (thread-safe), or returns nullptr with a reason.
const Loader* load_library(std::string* reason);

/// Resolves the instance/device commands against a live instance. Returns false
/// with a reason when a required entry point is missing.
bool resolve_instance(PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                      Functions* fns, std::string* reason);

/// Test hook: tries one explicit path without touching the process-global
/// loader. Returns true when the library opened and its global entry points
/// resolved; otherwise fills *reason.
bool probe_loader(const std::string& path, std::string* reason);

/// Exports the baked ICD manifest as VK_DRIVER_FILES unless the caller already
/// selected one. Returns true if an ICD is now configured -- including the case
/// where nothing is baked and the loader's own search paths apply.
bool ensure_icd_configured(std::string* reason);

}  // namespace alphashape::gpu::detail
