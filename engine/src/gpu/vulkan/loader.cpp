#include "gpu/vulkan/loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "base/platform/dynamic_library.hpp"
#include "base/platform/env.hpp"
#include "gpu/gpu_config.hpp"

#ifdef __APPLE__
#include <dlfcn.h>
#endif

namespace alphashape::gpu::detail {
namespace {

std::mutex g_mutex;
// Never destroyed (so never dlclose'd): the default GPU context outlives static
// destruction and still calls into the library.
Loader& g_loader = *new Loader();
std::string g_reason;
bool g_attempted = false;

bool getenv_set(const char* name) { return !platform::get_env(name).empty(); }

#ifdef __APPLE__
/// Directory of the Mach-O image (executable or Python extension module) this code was linked
/// into. dladdr is used instead of the executable path so the Python module, which runs inside
/// somebody else's interpreter, finds its own app bundle too.
std::filesystem::path image_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&image_directory), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    std::error_code ec;
    return std::filesystem::weakly_canonical(info.dli_fname, ec).parent_path();
}

/// First existing <ancestor>/<relative>, starting at this image's directory and climbing three
/// levels. That covers Contents/MacOS/<exe> and Contents/Resources/python/<module>, which both
/// reach Contents/ within two steps. Empty when this is not an app bundle.
std::filesystem::path find_in_bundle(const char* relative) {
    std::filesystem::path dir = image_directory();
    for (int level = 0; level < 4 && !dir.empty(); ++level) {
        std::error_code ec;
        const std::filesystem::path candidate = dir / relative;
        if (std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
        const std::filesystem::path parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }
    return {};
}
#endif

template <class T>
T lookup(PFN_vkGetInstanceProcAddr gipa, const char* name) {
    return reinterpret_cast<T>(gipa(nullptr, name));
}

/// Global (instance-independent) entry points, resolved with a null instance.
bool resolve_global(PFN_vkGetInstanceProcAddr gipa, Functions* fns, std::string* reason) {
    fns->GetInstanceProcAddr = gipa;
    fns->CreateInstance = lookup<PFN_vkCreateInstance>(gipa, "vkCreateInstance");
    fns->EnumerateInstanceExtensionProperties =
        lookup<PFN_vkEnumerateInstanceExtensionProperties>(gipa, "vkEnumerateInstanceExtensionProperties");
    fns->EnumerateInstanceLayerProperties =
        lookup<PFN_vkEnumerateInstanceLayerProperties>(gipa, "vkEnumerateInstanceLayerProperties");
    if (fns->CreateInstance == nullptr || fns->EnumerateInstanceExtensionProperties == nullptr ||
        fns->EnumerateInstanceLayerProperties == nullptr) {
        *reason = "loader does not export the core global entry points";
        return false;
    }
    return true;
}

struct Candidate {
    std::string path;
    void* handle;
    Functions fns;
};

}  // namespace

bool resolve_instance(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, Functions* fns,
                      std::string* reason) {
    if (gipa == nullptr) {
        *reason = "vkGetInstanceProcAddr is null";
        return false;
    }
    // The loader exposes vkGetDeviceProcAddr globally; the rest are resolved
    // through the instance so the device-level trampolines come from the driver.
    fns->GetInstanceProcAddr = gipa;
    fns->GetDeviceProcAddr =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
#define ALPHASHAPE_RESOLVE(type, member, name) \
    fns->member = reinterpret_cast<type>(gipa(instance, name))
    ALPHASHAPE_RESOLVE(PFN_vkDestroyInstance, DestroyInstance, "vkDestroyInstance");
    ALPHASHAPE_RESOLVE(PFN_vkEnumeratePhysicalDevices, EnumeratePhysicalDevices,
                       "vkEnumeratePhysicalDevices");
    ALPHASHAPE_RESOLVE(PFN_vkGetPhysicalDeviceProperties, GetPhysicalDeviceProperties,
                       "vkGetPhysicalDeviceProperties");
    ALPHASHAPE_RESOLVE(PFN_vkGetPhysicalDeviceMemoryProperties, GetPhysicalDeviceMemoryProperties,
                       "vkGetPhysicalDeviceMemoryProperties");
    ALPHASHAPE_RESOLVE(PFN_vkGetPhysicalDeviceQueueFamilyProperties,
                       GetPhysicalDeviceQueueFamilyProperties,
                       "vkGetPhysicalDeviceQueueFamilyProperties");
    ALPHASHAPE_RESOLVE(PFN_vkGetPhysicalDeviceFormatProperties, GetPhysicalDeviceFormatProperties,
                       "vkGetPhysicalDeviceFormatProperties");
    ALPHASHAPE_RESOLVE(PFN_vkEnumerateDeviceExtensionProperties, EnumerateDeviceExtensionProperties,
                       "vkEnumerateDeviceExtensionProperties");
    ALPHASHAPE_RESOLVE(PFN_vkCreateDevice, CreateDevice, "vkCreateDevice");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyDevice, DestroyDevice, "vkDestroyDevice");
    ALPHASHAPE_RESOLVE(PFN_vkGetDeviceQueue, GetDeviceQueue, "vkGetDeviceQueue");
    ALPHASHAPE_RESOLVE(PFN_vkDeviceWaitIdle, DeviceWaitIdle, "vkDeviceWaitIdle");
    ALPHASHAPE_RESOLVE(PFN_vkCreateDebugUtilsMessengerEXT, CreateDebugUtilsMessengerEXT,
                       "vkCreateDebugUtilsMessengerEXT");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyDebugUtilsMessengerEXT, DestroyDebugUtilsMessengerEXT,
                       "vkDestroyDebugUtilsMessengerEXT");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyBuffer, DestroyBuffer, "vkDestroyBuffer");
    ALPHASHAPE_RESOLVE(PFN_vkCreateBufferView, CreateBufferView, "vkCreateBufferView");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyBufferView, DestroyBufferView, "vkDestroyBufferView");
    ALPHASHAPE_RESOLVE(PFN_vkCreateImage, CreateImage, "vkCreateImage");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyImage, DestroyImage, "vkDestroyImage");
    ALPHASHAPE_RESOLVE(PFN_vkGetImageMemoryRequirements, GetImageMemoryRequirements,
                       "vkGetImageMemoryRequirements");
    ALPHASHAPE_RESOLVE(PFN_vkCreateImageView, CreateImageView, "vkCreateImageView");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyImageView, DestroyImageView, "vkDestroyImageView");
    ALPHASHAPE_RESOLVE(PFN_vkCreateSampler, CreateSampler, "vkCreateSampler");
    ALPHASHAPE_RESOLVE(PFN_vkDestroySampler, DestroySampler, "vkDestroySampler");
    ALPHASHAPE_RESOLVE(PFN_vkCmdCopyBufferToImage, CmdCopyBufferToImage, "vkCmdCopyBufferToImage");
    ALPHASHAPE_RESOLVE(PFN_vkCmdCopyImageToBuffer, CmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");
    ALPHASHAPE_RESOLVE(PFN_vkCreateShaderModule, CreateShaderModule, "vkCreateShaderModule");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyShaderModule, DestroyShaderModule, "vkDestroyShaderModule");
    ALPHASHAPE_RESOLVE(PFN_vkCreateDescriptorSetLayout, CreateDescriptorSetLayout,
                       "vkCreateDescriptorSetLayout");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyDescriptorSetLayout, DestroyDescriptorSetLayout,
                       "vkDestroyDescriptorSetLayout");
    ALPHASHAPE_RESOLVE(PFN_vkCreatePipelineLayout, CreatePipelineLayout, "vkCreatePipelineLayout");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyPipelineLayout, DestroyPipelineLayout,
                       "vkDestroyPipelineLayout");
    ALPHASHAPE_RESOLVE(PFN_vkCreateComputePipelines, CreateComputePipelines,
                       "vkCreateComputePipelines");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyPipeline, DestroyPipeline, "vkDestroyPipeline");
    ALPHASHAPE_RESOLVE(PFN_vkCreateDescriptorPool, CreateDescriptorPool, "vkCreateDescriptorPool");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyDescriptorPool, DestroyDescriptorPool,
                       "vkDestroyDescriptorPool");
    ALPHASHAPE_RESOLVE(PFN_vkAllocateDescriptorSets, AllocateDescriptorSets,
                       "vkAllocateDescriptorSets");
    ALPHASHAPE_RESOLVE(PFN_vkUpdateDescriptorSets, UpdateDescriptorSets, "vkUpdateDescriptorSets");
    ALPHASHAPE_RESOLVE(PFN_vkCreateCommandPool, CreateCommandPool, "vkCreateCommandPool");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyCommandPool, DestroyCommandPool, "vkDestroyCommandPool");
    ALPHASHAPE_RESOLVE(PFN_vkAllocateCommandBuffers, AllocateCommandBuffers,
                       "vkAllocateCommandBuffers");
    ALPHASHAPE_RESOLVE(PFN_vkFreeCommandBuffers, FreeCommandBuffers, "vkFreeCommandBuffers");
    ALPHASHAPE_RESOLVE(PFN_vkBeginCommandBuffer, BeginCommandBuffer, "vkBeginCommandBuffer");
    ALPHASHAPE_RESOLVE(PFN_vkEndCommandBuffer, EndCommandBuffer, "vkEndCommandBuffer");
    ALPHASHAPE_RESOLVE(PFN_vkCmdBindPipeline, CmdBindPipeline, "vkCmdBindPipeline");
    ALPHASHAPE_RESOLVE(PFN_vkCmdBindDescriptorSets, CmdBindDescriptorSets,
                       "vkCmdBindDescriptorSets");
    ALPHASHAPE_RESOLVE(PFN_vkCmdPushConstants, CmdPushConstants, "vkCmdPushConstants");
    ALPHASHAPE_RESOLVE(PFN_vkCmdDispatch, CmdDispatch, "vkCmdDispatch");
    ALPHASHAPE_RESOLVE(PFN_vkCmdPipelineBarrier, CmdPipelineBarrier, "vkCmdPipelineBarrier");
    ALPHASHAPE_RESOLVE(PFN_vkCmdCopyBuffer, CmdCopyBuffer, "vkCmdCopyBuffer");
    ALPHASHAPE_RESOLVE(PFN_vkQueueSubmit, QueueSubmit, "vkQueueSubmit");
    ALPHASHAPE_RESOLVE(PFN_vkCreateFence, CreateFence, "vkCreateFence");
    ALPHASHAPE_RESOLVE(PFN_vkWaitForFences, WaitForFences, "vkWaitForFences");
    ALPHASHAPE_RESOLVE(PFN_vkResetFences, ResetFences, "vkResetFences");
    ALPHASHAPE_RESOLVE(PFN_vkDestroyFence, DestroyFence, "vkDestroyFence");
    ALPHASHAPE_RESOLVE(PFN_vkResetCommandBuffer, ResetCommandBuffer, "vkResetCommandBuffer");
#undef ALPHASHAPE_RESOLVE

    const struct {
        const void* fn;
        const char* name;
    } required[] = {
        {reinterpret_cast<const void*>(fns->GetDeviceProcAddr), "vkGetDeviceProcAddr"},
        {reinterpret_cast<const void*>(fns->EnumeratePhysicalDevices), "vkEnumeratePhysicalDevices"},
        {reinterpret_cast<const void*>(fns->GetPhysicalDeviceProperties), "vkGetPhysicalDeviceProperties"},
        {reinterpret_cast<const void*>(fns->GetPhysicalDeviceMemoryProperties), "vkGetPhysicalDeviceMemoryProperties"},
        {reinterpret_cast<const void*>(fns->GetPhysicalDeviceQueueFamilyProperties), "vkGetPhysicalDeviceQueueFamilyProperties"},
        {reinterpret_cast<const void*>(fns->EnumerateDeviceExtensionProperties), "vkEnumerateDeviceExtensionProperties"},
        {reinterpret_cast<const void*>(fns->CreateDevice), "vkCreateDevice"},
        {reinterpret_cast<const void*>(fns->GetDeviceQueue), "vkGetDeviceQueue"},
        {reinterpret_cast<const void*>(fns->DeviceWaitIdle), "vkDeviceWaitIdle"},
        {reinterpret_cast<const void*>(fns->CreateShaderModule), "vkCreateShaderModule"},
        {reinterpret_cast<const void*>(fns->CreateDescriptorSetLayout), "vkCreateDescriptorSetLayout"},
        {reinterpret_cast<const void*>(fns->CreatePipelineLayout), "vkCreatePipelineLayout"},
        {reinterpret_cast<const void*>(fns->CreateComputePipelines), "vkCreateComputePipelines"},
        {reinterpret_cast<const void*>(fns->CreateDescriptorPool), "vkCreateDescriptorPool"},
        {reinterpret_cast<const void*>(fns->AllocateDescriptorSets), "vkAllocateDescriptorSets"},
        {reinterpret_cast<const void*>(fns->UpdateDescriptorSets), "vkUpdateDescriptorSets"},
        {reinterpret_cast<const void*>(fns->CreateCommandPool), "vkCreateCommandPool"},
        {reinterpret_cast<const void*>(fns->AllocateCommandBuffers), "vkAllocateCommandBuffers"},
        {reinterpret_cast<const void*>(fns->BeginCommandBuffer), "vkBeginCommandBuffer"},
        {reinterpret_cast<const void*>(fns->EndCommandBuffer), "vkEndCommandBuffer"},
        {reinterpret_cast<const void*>(fns->CmdDispatch), "vkCmdDispatch"},
        {reinterpret_cast<const void*>(fns->CmdPipelineBarrier), "vkCmdPipelineBarrier"},
        {reinterpret_cast<const void*>(fns->CmdCopyBuffer), "vkCmdCopyBuffer"},
        {reinterpret_cast<const void*>(fns->CreateImage), "vkCreateImage"},
        {reinterpret_cast<const void*>(fns->CreateImageView), "vkCreateImageView"},
        {reinterpret_cast<const void*>(fns->CreateSampler), "vkCreateSampler"},
        {reinterpret_cast<const void*>(fns->CmdCopyBufferToImage), "vkCmdCopyBufferToImage"},
        {reinterpret_cast<const void*>(fns->CmdCopyImageToBuffer), "vkCmdCopyImageToBuffer"},
        {reinterpret_cast<const void*>(fns->QueueSubmit), "vkQueueSubmit"},
        {reinterpret_cast<const void*>(fns->CreateFence), "vkCreateFence"},
        {reinterpret_cast<const void*>(fns->WaitForFences), "vkWaitForFences"},
        {reinterpret_cast<const void*>(fns->ResetCommandBuffer), "vkResetCommandBuffer"},
    };
    for (const auto& entry : required) {
        if (entry.fn == nullptr) {
            *reason = std::string(entry.name) + " not resolved from the loader";
            return false;
        }
    }
    return true;
}

bool probe_loader(const std::string& path, std::string* reason) {
    if (path.empty()) {
        *reason = "empty library path";
        return false;
    }
    std::string open_error;
    platform::DynamicLibrary library = platform::DynamicLibrary::open(path, &open_error);
    if (!library.is_open()) {
        *reason = open_error;
        return false;
    }
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        library.symbol("vkGetInstanceProcAddr", nullptr));
    if (gipa == nullptr) {
        *reason = std::string("'") + path + "' has no vkGetInstanceProcAddr";
        return false;
    }
    Functions scratch;
    std::string scratch_reason;
    if (!resolve_global(gipa, &scratch, &scratch_reason)) {
        *reason = std::string("'") + path + "': " + scratch_reason;
        return false;
    }
    // Library closes on scope exit; the entry points above are values, not
    // pointers into the module, and only their continued validity matters.
    return true;
}

bool ensure_icd_configured(std::string* reason) {
    // A distribution Vulkan install (Linux) has no baked manifest: its ICDs and
    // layers live in the standard system directories, which the loader finds on
    // its own. Baking nothing here is deliberate -- setting VK_DRIVER_FILES to a
    // single system manifest would hide every other installed driver, and an
    // empty VK_LAYER_PATH would hide the installed layers.
#ifdef __APPLE__
    // An app bundle ships its own loader and MoltenVK manifest. Use them unless the caller chose
    // a driver explicitly, so the app runs on a Mac with no Vulkan SDK or Homebrew.
    if (!getenv_set("VK_ICD_FILENAMES") && !getenv_set("VK_DRIVER_FILES")) {
        const std::filesystem::path bundled = find_in_bundle("Resources/vulkan/icd.d/MoltenVK_icd.json");
        if (!bundled.empty()) {
            (void)platform::set_env("VK_DRIVER_FILES", bundled.string());
            return true;
        }
    }
#endif
    const bool baked_icd = kIcdPath[0] != '\0';
    const bool baked_layers = kValidationLayerDir[0] != '\0';
    if (!baked_icd && !baked_layers) {
        return true;
    }
    // The layer directory is independent of the ICD choice: set it whenever the
    // caller has not, so validation works without environment setup.
    if (baked_layers && !getenv_set("VK_LAYER_PATH")) {
        std::error_code layer_error;
        if (std::filesystem::exists(kValidationLayerDir, layer_error)) {
            (void)platform::set_env("VK_LAYER_PATH", std::string(kValidationLayerDir));
        }
    }
    if (!baked_icd || getenv_set("VK_ICD_FILENAMES") || getenv_set("VK_DRIVER_FILES")) {
        return true;
    }
    std::error_code error;
    if (!std::filesystem::exists(kIcdPath, error)) {
        *reason = std::string("ICD manifest not found: ") + kIcdPath;
        return false;
    }
    (void)platform::set_env("VK_DRIVER_FILES", std::string(kIcdPath));
    return true;
}

const Loader* load_library(std::string* reason) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_attempted) {
        g_attempted = true;
        std::vector<std::string> candidates;
        const std::string library_override = platform::get_env("ALPHASHAPE_VULKAN_LIBRARY");
        if (!library_override.empty()) {
            candidates.emplace_back(library_override);
        }
#ifdef __APPLE__
        {
            const std::filesystem::path bundled = find_in_bundle("Frameworks/libvulkan.1.dylib");
            if (!bundled.empty()) {
                candidates.emplace_back(bundled.string());
            }
        }
#endif
        if (kLibraryPath[0] != '\0') {
            candidates.emplace_back(kLibraryPath);
        }
        // Platform sonames, in the order the platform's own loader would try.
        // On Linux the unversioned libvulkan.so is only present with the
        // development package, hence the versioned name comes first.
        //
        // Windows: vulkan-1.dll is the loader, and every graphics driver that
        // supports Vulkan installs one into System32. It is deliberately not
        // bundled with the installer -- a private copy would shadow the
        // driver's loader and could not find the driver's ICDs, and the
        // Windows loader has no LD_LIBRARY_PATH-style override to recover from
        // that. LoadLibraryW resolves a bare name through the application
        // directory, System32 and the PATH, which is exactly the search order
        // a user expects.
#if defined(__APPLE__)
        candidates.emplace_back("libvulkan.1.dylib");
        candidates.emplace_back("libvulkan.dylib");
#elif defined(__linux__)
        candidates.emplace_back("libvulkan.so.1");
        candidates.emplace_back("libvulkan.so");
#elif defined(_WIN32)
        candidates.emplace_back("vulkan-1.dll");
#endif

        std::string last_reason = "no loader candidates";
        for (const std::string& candidate : candidates) {
            std::string open_error;
            platform::DynamicLibrary library = platform::DynamicLibrary::open(candidate, &open_error);
            if (!library.is_open()) {
                last_reason = open_error;
                continue;
            }
            auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                library.symbol("vkGetInstanceProcAddr", nullptr));
            Functions resolved;
            std::string why;
            if (gipa == nullptr) {
                why = "no vkGetInstanceProcAddr";
            } else if (!resolve_global(gipa, &resolved, &why)) {
                // why filled
            } else {
                // The module must stay mapped: the entry points above are code
                // addresses inside it. The Loader keeps the handle for the life
                // of the process (RAII closes it at exit, never earlier).
                g_loader.library = std::move(library);
                g_loader.fns = resolved;
                g_loader.library_path = candidate;
                g_loader.loaded = true;
                g_reason.clear();
                break;
            }
            last_reason = std::string("'") + candidate + "': " + why;
        }
        if (!g_loader.loaded) {
            g_reason = last_reason;
#if defined(_WIN32)
            // A bare "LoadLibrary 'vulkan-1.dll' failed" reads like a defect in
            // the program. It is not: Windows ships no Vulkan runtime of its
            // own, it arrives with the graphics driver. Say so, because this
            // string is what the app and `alphashape-cli --gpu-info` show a
            // user whose machine has none.
            g_reason += " (no Vulkan runtime found; Windows installs vulkan-1.dll with the "
                        "graphics driver, so install or update the GPU driver to enable "
                        "GPU acceleration -- every task runs on the CPU meanwhile)";
#endif
        }
    }
    if (!g_loader.loaded) {
        if (reason != nullptr) {
            *reason = g_reason;
        }
        return nullptr;
    }
    return &g_loader;
}

}  // namespace alphashape::gpu::detail
