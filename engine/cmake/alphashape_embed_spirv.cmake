# Build-time SPIR-V embedding: turns each .spv into a uint32_t array plus a
# name -> (pointer, word count) table. Invoked by alphashape_vulkan.cmake with
#   -DALPHASHAPE_EMBED_OUTPUT=<generated .cpp>
#   -DALPHASHAPE_EMBED_INPUTS=<semicolon-separated .spv paths>

if(NOT ALPHASHAPE_EMBED_OUTPUT OR NOT ALPHASHAPE_EMBED_INPUTS)
    message(FATAL_ERROR "alphashape_embed_spirv: output and inputs are required")
endif()

set(_definitions "")
set(_entries "")

foreach(_spv IN LISTS ALPHASHAPE_EMBED_INPUTS)
    get_filename_component(_name "${_spv}" NAME_WE)
    file(READ "${_spv}" _hex HEX)
    string(LENGTH "${_hex}" _hex_length)
    math(EXPR _words "${_hex_length} / 8")
    # SPIR-V is little-endian; each 8 hex chars are one word big-endian, so the
    # four byte pairs are reversed when formatting.
    string(REGEX REPLACE
           "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])"
           "0x\\4\\3\\2\\1, " _bytes "${_hex}")
    string(REGEX REPLACE "(0x[0-9a-f]+, 0x[0-9a-f]+, 0x[0-9a-f]+, 0x[0-9a-f]+, 0x[0-9a-f]+, 0x[0-9a-f]+, 0x[0-9a-f]+, 0x[0-9a-f]+, )"
           "\\1\n    " _bytes "${_bytes}")
    string(APPEND _definitions
        "constexpr std::uint32_t k_${_name}[] = {\n    ${_bytes}\n};\n\n")
    string(APPEND _entries
        "        {\"${_name}\", k_${_name}, sizeof(k_${_name}) / sizeof(std::uint32_t)},\n")
endforeach()

file(WRITE "${ALPHASHAPE_EMBED_OUTPUT}"
"// Generated from src/gpu/shaders/*.comp at build time. Do not edit.
#include <cstdint>
#include <string>
#include <vector>

#include \"gpu/shaders_embedded.hpp\"

namespace alphashape::gpu::detail {
namespace {

${_definitions}}  // namespace

const KernelEntry* find_kernel(const std::string& name) {
    static constexpr KernelEntry kTable[] = {
${_entries}    };
    for (const KernelEntry& entry : kTable) {
        if (name == entry.name) {
            return &entry;
        }
    }
    return nullptr;
}

std::vector<std::string> all_kernel_names() {
    static constexpr KernelEntry kTable[] = {
${_entries}    };
    std::vector<std::string> names;
    names.reserve(sizeof(kTable) / sizeof(kTable[0]));
    for (const KernelEntry& entry : kTable) {
        names.emplace_back(entry.name);
    }
    return names;
}

}  // namespace alphashape::gpu::detail
")
