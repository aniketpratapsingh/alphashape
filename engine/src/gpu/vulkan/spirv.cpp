#include "gpu/vulkan/spirv.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace alphashape::gpu::detail {
namespace {

// SPIR-V 1.x opcodes and decorations (SPIR-V spec, chapter 3).
constexpr std::uint16_t kOpTypeInt = 21;
constexpr std::uint16_t kOpTypeFloat = 22;
constexpr std::uint16_t kOpTypeVector = 23;
constexpr std::uint16_t kOpTypeMatrix = 24;
constexpr std::uint16_t kOpTypeArray = 28;
constexpr std::uint16_t kOpTypeRuntimeArray = 29;
constexpr std::uint16_t kOpTypeStruct = 30;
constexpr std::uint16_t kOpTypePointer = 32;
constexpr std::uint16_t kOpTypeImage = 25;
constexpr std::uint16_t kOpTypeSampler = 26;
constexpr std::uint16_t kOpTypeSampledImage = 27;
constexpr std::uint16_t kOpConstant = 43;
constexpr std::uint16_t kOpVariable = 59;
constexpr std::uint16_t kOpDecorate = 71;
constexpr std::uint16_t kOpMemberDecorate = 72;

constexpr std::uint32_t kDecorationBinding = 33;
constexpr std::uint32_t kDecorationDescriptorSet = 34;
constexpr std::uint32_t kDecorationOffset = 35;
constexpr std::uint16_t kStorageClassStorageBuffer = 12;
constexpr std::uint16_t kStorageClassUniformConstant = 0;
constexpr std::uint16_t kStorageClassPushConstant = 9;
// Dim enum (SPIR-V spec): 1 is 2D.
constexpr std::uint16_t kDim2D = 1;
constexpr std::uint32_t kMagic = 0x07230203;

struct TypeInfo {
    std::uint16_t opcode = 0;
    std::uint32_t id = 0;
    std::uint32_t operand = 0;    // element / pointee / component type
    std::uint32_t count = 0;      // vector size, array length id, member count
    std::uint16_t bit_width = 0;  // scalar width in bits
    std::uint16_t storage = 0;    // pointer storage class
    std::uint32_t first_member = 0;  // index into member_type_ids
    std::uint16_t dim = 0;        // OpTypeImage Dim operand (1 = 2D)
    std::uint16_t image_sampled = 0;  // OpTypeImage Sampled operand (1 = sampled)
};

struct Layout {
    std::vector<TypeInfo> types;
    std::vector<std::uint32_t> member_type_ids;  // concatenated, per struct
    std::vector<std::pair<std::uint32_t, std::uint32_t>> constants;  // result id -> value
    std::vector<std::uint32_t> variable_ids;
    std::vector<std::uint32_t> variable_types;
    std::vector<std::uint16_t> variable_storage;

    struct Decoration {
        std::uint32_t target;
        std::uint32_t decoration;
        std::uint32_t value;
    };
    std::vector<Decoration> decorations;
    struct MemberDecoration {
        std::uint32_t type;
        std::uint32_t member;
        std::uint32_t offset;
    };
    std::vector<MemberDecoration> member_decorations;
};

const TypeInfo* find_type(const Layout& layout, std::uint32_t id) {
    for (const TypeInfo& type : layout.types) {
        if (type.id == id) {
            return &type;
        }
    }
    return nullptr;
}

/// Byte size of a scalar/vector/matrix/array type. 0 means unknown.
std::uint64_t type_size(const Layout& layout, std::uint32_t id, int depth) {
    if (depth > 16) {
        return 0;
    }
    const TypeInfo* type = find_type(layout, id);
    if (type == nullptr) {
        return 0;
    }
    switch (type->opcode) {
        case kOpTypeInt:
        case kOpTypeFloat:
            return type->bit_width / 8u;
        case kOpTypeVector:
        case kOpTypeMatrix:
            return static_cast<std::uint64_t>(type->count) * type_size(layout, type->operand, depth + 1);
        case kOpTypeArray: {
            std::uint64_t length = 0;
            for (const auto& constant : layout.constants) {
                if (constant.first == type->count) {
                    length = constant.second;
                }
            }
            return length * type_size(layout, type->operand, depth + 1);
        }
        default:
            return 0;
    }
}

bool parse(const std::uint32_t* words, std::size_t word_count, Layout* layout) {
    std::size_t i = 5;
    while (i < word_count) {
        const std::uint32_t header = words[i];
        const std::uint16_t opcode = static_cast<std::uint16_t>(header & 0xFFFFu);
        const std::uint16_t length = static_cast<std::uint16_t>(header >> 16u);
        if (length == 0 || i + length > word_count) {
            return false;
        }
        switch (opcode) {
            case kOpTypeInt:
                if (length < 4) return false;
                layout->types.push_back({opcode, words[i + 1], 0, 0,
                                         static_cast<std::uint16_t>(words[i + 2]), 0, 0});
                break;
            case kOpTypeFloat:
                if (length < 3) return false;
                layout->types.push_back({opcode, words[i + 1], 0, 0,
                                         static_cast<std::uint16_t>(words[i + 2]), 0, 0});
                break;
            case kOpTypeVector:
            case kOpTypeMatrix:
                if (length < 4) return false;
                layout->types.push_back(
                    {opcode, words[i + 1], words[i + 2], words[i + 3], 0, 0, 0});
                break;
            case kOpTypeArray:
                if (length < 4) return false;
                layout->types.push_back(
                    {opcode, words[i + 1], words[i + 2], words[i + 3], 0, 0, 0});
                break;
            case kOpTypeRuntimeArray:
                if (length < 3) return false;
                layout->types.push_back({opcode, words[i + 1], words[i + 2], 0, 0, 0, 0});
                break;
            case kOpTypeStruct: {
                if (length < 3) return false;
                TypeInfo info;
                info.opcode = opcode;
                info.id = words[i + 1];
                info.count = static_cast<std::uint32_t>(length - 2u);
                info.first_member = static_cast<std::uint32_t>(layout->member_type_ids.size());
                for (std::uint16_t m = 2; m < length; ++m) {
                    layout->member_type_ids.push_back(words[i + m]);
                }
                layout->types.push_back(info);
                break;
            }
            case kOpTypeImage:
                // Result Id, Sampled Type, Dim, Depth, Arrayed, MS, Sampled, Format.
                if (length < 9) return false;
                layout->types.push_back({opcode, words[i + 1], words[i + 2], 0, 0, 0, 0,
                                         static_cast<std::uint16_t>(words[i + 3]),
                                         static_cast<std::uint16_t>(words[i + 7])});
                break;
            case kOpTypeSampler:
                if (length < 2) return false;
                layout->types.push_back({opcode, words[i + 1], 0, 0, 0, 0, 0, 0, 0});
                break;
            case kOpTypeSampledImage:
                // Result Id, Image Type Id.
                if (length < 3) return false;
                layout->types.push_back({opcode, words[i + 1], words[i + 2], 0, 0, 0, 0, 0, 0});
                break;
            case kOpTypePointer:
                if (length < 4) return false;
                layout->types.push_back({opcode, words[i + 1], words[i + 3], 0, 0,
                                         static_cast<std::uint16_t>(words[i + 2]), 0});
                break;
            case kOpConstant:
                if (length >= 4) {
                    layout->constants.emplace_back(words[i + 2], words[i + 3]);
                }
                break;
            case kOpVariable:
                if (length < 4) return false;
                layout->variable_ids.push_back(words[i + 2]);
                layout->variable_types.push_back(words[i + 1]);
                layout->variable_storage.push_back(
                    static_cast<std::uint16_t>(words[i + 3]));
                break;
            case kOpDecorate:
                if (length >= 4) {
                    layout->decorations.push_back({words[i + 1], words[i + 2], words[i + 3]});
                }
                break;
            case kOpMemberDecorate:
                // OpMemberDecorate: type, member, decoration, operands...
                if (length >= 5 && words[i + 3] == kDecorationOffset) {
                    layout->member_decorations.push_back(
                        {words[i + 1], words[i + 2], words[i + 4]});
                }
                break;
            default:
                break;
        }
        i += length;
    }
    return true;
}

}  // namespace

bool reflect_spirv(const std::uint32_t* words, std::size_t word_count, SpirvInfo* info) {
    info->bindings.clear();
    info->push_constant_size = 0;
    info->error.clear();
    if (words == nullptr || word_count < 5) {
        info->error = "SPIR-V blob is truncated";
        return false;
    }
    if (words[0] != kMagic) {
        info->error = "bad SPIR-V magic number";
        return false;
    }
    Layout layout;
    if (!parse(words, word_count, &layout)) {
        info->error = "malformed SPIR-V instruction stream";
        return false;
    }

    for (std::size_t v = 0; v < layout.variable_ids.size(); ++v) {
        const std::uint16_t storage = layout.variable_storage[v];
        if (storage != kStorageClassStorageBuffer && storage != kStorageClassUniformConstant) {
            continue;
        }
        std::uint32_t binding = 0;
        std::uint32_t set = 0;
        bool have_binding = false;
        for (const Layout::Decoration& decoration : layout.decorations) {
            if (decoration.target != layout.variable_ids[v]) {
                continue;
            }
            if (decoration.decoration == kDecorationBinding) {
                binding = decoration.value;
                have_binding = true;
            } else if (decoration.decoration == kDecorationDescriptorSet) {
                set = decoration.value;
            }
        }
        if (!have_binding) {
            info->error = storage == kStorageClassStorageBuffer
                              ? "storage buffer without a Binding decoration"
                              : "image/sampler without a Binding decoration";
            return false;
        }
        if (set != 0) {
            info->error = "descriptor outside set 0 is not supported";
            return false;
        }
        if (storage == kStorageClassStorageBuffer) {
            info->bindings.push_back({binding, BindingKind::StorageBuffer, false});
            continue;
        }
        // UniformConstant: the variable's type is a pointer to either a
        // combined sampled image (the only supported form) or a bare image or
        // sampler (which Vulkan binds through two descriptors; this API models
        // one binding per object, so reject it here rather than guess).
        const TypeInfo* pointer = find_type(layout, layout.variable_types[v]);
        if (pointer == nullptr || pointer->opcode != kOpTypePointer) {
            info->error = "uniform-constant variable has a non-pointer type";
            return false;
        }
        const TypeInfo* pointee = find_type(layout, pointer->operand);
        if (pointee == nullptr) {
            info->error = "uniform-constant variable points at an unknown type";
            return false;
        }
        if (pointee->opcode == kOpTypeSampledImage) {
            const TypeInfo* image = find_type(layout, pointee->operand);
            if (image == nullptr || image->opcode != kOpTypeImage) {
                info->error = "sampled image does not wrap an OpTypeImage";
                return false;
            }
            if (image->dim != kDim2D) {
                info->error = "only 2D images are supported (binding " +
                              std::to_string(binding) + " has Dim " +
                              std::to_string(image->dim) + ")";
                return false;
            }
            if (image->image_sampled == 2) {
                // Sampled == 2 means "used as storage image", which this layer
                // does not bind.
                info->error = "storage images are not supported (binding " +
                              std::to_string(binding) + ")";
                return false;
            }
            info->bindings.push_back({binding, BindingKind::CombinedImageSampler, true});
            continue;
        }
        if (pointee->opcode == kOpTypeImage) {
            info->error = "separate images/samplers unsupported (binding " +
                          std::to_string(binding) +
                          " is a bare OpTypeImage; declare it as sampler2D)";
            return false;
        }
        if (pointee->opcode == kOpTypeSampler) {
            info->error = "separate images/samplers unsupported (binding " +
                          std::to_string(binding) +
                          " is a bare OpTypeSampler; declare a sampler2D)";
            return false;
        }
        info->error = "unsupported uniform-constant descriptor type at binding " +
                      std::to_string(binding);
        return false;
    }
    std::sort(info->bindings.begin(), info->bindings.end(),
              [](const BindingInfo& a, const BindingInfo& b) { return a.binding < b.binding; });
    for (std::size_t i = 1; i < info->bindings.size(); ++i) {
        if (info->bindings[i].binding == info->bindings[i - 1].binding) {
            info->error = "duplicate binding " + std::to_string(info->bindings[i].binding);
            return false;
        }
    }

    for (std::size_t v = 0; v < layout.variable_ids.size(); ++v) {
        if (layout.variable_storage[v] != kStorageClassPushConstant) {
            continue;
        }
        const TypeInfo* pointer = find_type(layout, layout.variable_types[v]);
        if (pointer == nullptr || pointer->opcode != kOpTypePointer) {
            info->error = "push constant variable has a non-pointer type";
            return false;
        }
        const TypeInfo* block = find_type(layout, pointer->operand);
        if (block == nullptr || block->opcode != kOpTypeStruct) {
            info->error = "push constant is not a struct";
            return false;
        }
        std::uint32_t size = 0;
        for (std::uint32_t member = 0; member < block->count; ++member) {
            std::uint32_t offset = 0;
            bool have_offset = false;
            for (const Layout::MemberDecoration& decoration : layout.member_decorations) {
                if (decoration.type == block->id && decoration.member == member) {
                    offset = decoration.offset;
                    have_offset = true;
                }
            }
            if (!have_offset) {
                info->error = "push constant member without an Offset decoration";
                return false;
            }
            const std::uint32_t member_type =
                layout.member_type_ids[block->first_member + member];
            const std::uint64_t member_size = type_size(layout, member_type, 0);
            size = std::max(size, static_cast<std::uint32_t>(offset + member_size));
        }
        info->push_constant_size = size;
    }
    return true;
}

std::size_t SpirvInfo::count(BindingKind kind) const {
    std::size_t total = 0;
    for (const BindingInfo& binding : bindings) {
        if (binding.kind == kind) {
            ++total;
        }
    }
    return total;
}

const BindingInfo* SpirvInfo::find(std::uint32_t binding) const {
    for (const BindingInfo& entry : bindings) {
        if (entry.binding == binding) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace alphashape::gpu::detail
