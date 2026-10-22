#pragma once

// Interface between the GPU core and the generated SPIR-V blob.

#include <cstdint>
#include <string>
#include <vector>

namespace alphashape::gpu::detail {

struct KernelEntry {
    const char* name;
    const std::uint32_t* words;
    std::size_t word_count;
};

/// nullptr when the name is not one of the compiled shaders.
const KernelEntry* find_kernel(const std::string& name);
std::vector<std::string> all_kernel_names();

}  // namespace alphashape::gpu::detail
