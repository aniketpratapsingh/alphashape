#pragma once

// Parsed reflection facts extracted from a SPIR-V module, enough to build a
// compute pipeline: the descriptor bindings at descriptor set 0 (storage
// buffers and combined image samplers, in declaration order) and the
// push-constant block size.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace alphashape::gpu::detail {

/// What kind of descriptor a binding holds. The kind comes from the SPIR-V
/// type chain, never from the binding number.
enum class BindingKind {
    /// `buffer` block / runtime array of a pointer in StorageBuffer storage.
    StorageBuffer,
    /// `OpTypeSampledImage` (an image combined with a sampler) declared as a
    /// uniform of 2D dimension. Bound as VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER.
    CombinedImageSampler,
    /// `OpTypeImage` or `OpTypeSampler` declared on its own. The layer rejects
    /// these because Vulkan requires two descriptors and this API's Dispatch
    /// models one binding per image and sampler.
    SeparateImageOrSampler,
};

struct BindingInfo {
    std::uint32_t binding = 0;
    BindingKind kind = BindingKind::StorageBuffer;
    bool read_only = false;
};

struct SpirvInfo {
    // Sorted by binding.
    std::vector<BindingInfo> bindings;
    // Size in bytes of the PushConstant block (0 when the shader has none).
    std::uint32_t push_constant_size = 0;
    // Non-zero when validation failed; the message is human-readable.
    std::string error;

    bool ok() const { return error.empty(); }

    /// Number of bindings of one kind (used by the dispatch validator to say
    /// how many buffers/images it expects).
    std::size_t count(BindingKind kind) const;
    /// The binding at `binding`, or nullptr.
    const BindingInfo* find(std::uint32_t binding) const;
};

/// Parses `word_count` SPIR-V words (little-endian host). Returns false and
/// fills info->error on a malformed or unsupported module.
bool reflect_spirv(const std::uint32_t* words, std::size_t word_count, SpirvInfo* info);

}  // namespace alphashape::gpu::detail
