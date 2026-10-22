#pragma once

// The Vulkan headers use old-style casts inside their macros (VK_MAKE_API_VERSION
// and friends), which our -Wold-style-cast -Werror rejects. Suppress locally for
// the third-party include, as done for other imported headers in this tree.
// <vulkan/vulkan.h> is included with VK_NO_PROTOTYPES: nothing is linked, all
// entry points come from the runtime function table.

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined(__GNUC__)
// The same third-party noise from the Vulkan headers under g++: their macros
// use old-style casts and the extension types are not our design to fix.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include <vulkan/vulkan.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// VMA is third-party as well; every TU that includes it wraps it here so the
// backend code itself still compiles under the full warning set.
//
// The header is installed either as <include>/vma/vk_mem_alloc.h (Vulkan SDK
// layout, macOS SDK, Fedora-style packages) or directly as <include>/
// vk_mem_alloc.h (Ubuntu's vulkan-headers), so probe both.
#if defined(ALPHASHAPE_VMA_IMPLEMENTATION)
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wpedantic"
#elif defined(__GNUC__)
// VMA's implementation trips a long tail of g++ warnings that are either
// stylistic or -Wmaybe-uninitialized false positives from its template soup.
// The implementation TU additionally compiles with -w (see src/gpu/CMakeLists),
// but every other TU including this header sees VMA's template definitions too.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wnull-dereference"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if __has_include(<vma/vk_mem_alloc.h>)
#include <vma/vk_mem_alloc.h>
#elif __has_include(<vk_mem_alloc.h>)
#include <vk_mem_alloc.h>
#else
#error "VMA header not found: expected <vma/vk_mem_alloc.h> or <vk_mem_alloc.h>"
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
