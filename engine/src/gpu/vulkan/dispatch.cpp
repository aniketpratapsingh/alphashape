#include "gpu/vulkan/device.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "base/utils/log.hpp"

namespace alphashape::gpu::detail {
namespace {

std::string vk_result_string(const char* call, VkResult result) {
    return std::string(call) + " failed with VkResult " + std::to_string(result);
}

/// Compact textual form of the bindings of one kind, for validation messages.
std::string binding_list(const std::vector<BindingInfo>& bindings, BindingKind kind) {
    std::string text;
    for (const BindingInfo& binding : bindings) {
        if (binding.kind != kind) {
            continue;
        }
        if (!text.empty()) {
            text += ", ";
        }
        text += std::to_string(binding.binding);
        if (kind == BindingKind::CombinedImageSampler) {
            text += ":image";
        }
    }
    return text.empty() ? "(none)" : text;
}

const VulkanBuffer* as_vulkan(const BufferHandle* handle) {
    return dynamic_cast<const VulkanBuffer*>(handle);
}

const VulkanImage* as_vulkan(const ImageHandle* handle) {
    return dynamic_cast<const VulkanImage*>(handle);
}

const VulkanSampler* as_vulkan(const SamplerHandle* handle) {
    return dynamic_cast<const VulkanSampler*>(handle);
}

VkDescriptorType descriptor_type(BindingKind kind) {
    return kind == BindingKind::CombinedImageSampler
               ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
               : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

}  // namespace

const Pipeline& VulkanBackend::pipeline_for(const KernelEntry& kernel) {
    auto existing = pipelines_.find(kernel.name);
    if (existing != pipelines_.end()) {
        return existing->second;
    }

    const Functions& fns = instance_->fns;
    Pipeline pipeline;
    if (!reflect_spirv(kernel.words, kernel.word_count, &pipeline.info)) {
        throw Error(std::string("kernel '") + kernel.name + "' reflection failed: " +
                    pipeline.info.error);
    }

    if (!pipeline.info.bindings.empty()) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(pipeline.info.bindings.size());
        for (const BindingInfo& binding : pipeline.info.bindings) {
            VkDescriptorSetLayoutBinding entry{};
            entry.binding = binding.binding;
            entry.descriptorType = descriptor_type(binding.kind);
            entry.descriptorCount = 1;
            entry.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(entry);
        }
        VkDescriptorSetLayoutCreateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        set_info.pBindings = bindings.data();
        VkResult result =
            fns.CreateDescriptorSetLayout(device_handle_, &set_info, nullptr, &pipeline.set_layout);
        if (result != VK_SUCCESS) {
            destroy_pipeline(&pipeline);
            throw Error(vk_result_string("vkCreateDescriptorSetLayout", result));
        }
    }

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = kernel.word_count * sizeof(std::uint32_t);
    module_info.pCode = kernel.words;
    VkResult result = fns.CreateShaderModule(device_handle_, &module_info, nullptr, &pipeline.module);
    if (result != VK_SUCCESS) {
        destroy_pipeline(&pipeline);
        throw Error(std::string("kernel '") + kernel.name + "': " +
                    vk_result_string("vkCreateShaderModule", result));
    }

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = pipeline.info.push_constant_size;
    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = pipeline.set_layout != VK_NULL_HANDLE ? 1u : 0u;
    layout_info.pSetLayouts = pipeline.set_layout != VK_NULL_HANDLE ? &pipeline.set_layout : nullptr;
    layout_info.pushConstantRangeCount = pipeline.info.push_constant_size != 0 ? 1u : 0u;
    layout_info.pPushConstantRanges = pipeline.info.push_constant_size != 0 ? &range : nullptr;
    result = fns.CreatePipelineLayout(device_handle_, &layout_info, nullptr, &pipeline.layout);
    if (result != VK_SUCCESS) {
        destroy_pipeline(&pipeline);
        throw Error(std::string("kernel '") + kernel.name + "': " +
                    vk_result_string("vkCreatePipelineLayout", result));
    }

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = pipeline.module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipeline.layout;
    result = fns.CreateComputePipelines(device_handle_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                        &pipeline.pipeline);
    if (result != VK_SUCCESS) {
        destroy_pipeline(&pipeline);
        throw Error(std::string("kernel '") + kernel.name + "': " +
                    vk_result_string("vkCreateComputePipelines", result));
    }

    auto inserted = pipelines_.emplace(std::string(kernel.name), pipeline);
    return inserted.first->second;
}

bool VulkanBackend::lost() const { return device_lost(); }

bool VulkanBackend::wait_fence_noexcept(VkFence fence, std::uint64_t timeout_ns) noexcept {
    if (fence == VK_NULL_HANDLE) {
        return true;
    }
    // This deliberately waits even when the device is already marked lost. A
    // declared loss (the test hook, or a timeout that raced completion) may
    // still have the work in the queue, and the caller is about to free memory
    // it reads: freeing a range the queue still references is a use-after-free
    // the driver punishes with a segfault. The *cost* is bounded instead -- the
    // shared post-loss deadline in drain_pending means only the first drain
    // after a loss pays, and every later one returns at once.
    //
    // Poll in short slices so a fence that is merely slow still completes
    // inside the budget instead of being declared a hang on its first slice.
    constexpr std::uint64_t kPollNs = 20'000'000ULL;
    std::uint64_t waited = 0;
    while (true) {
        const std::uint64_t left = waited >= timeout_ns ? 0 : timeout_ns - waited;
        if (left == 0) {
            note_device_lost("no fence signal within the destructor drain deadline");
            return false;
        }
        const std::uint64_t slice = std::min(left, kPollNs);
        const VkResult result =
            instance_->fns.WaitForFences(device_handle_, 1, &fence, VK_TRUE, slice);
        if (result == VK_SUCCESS) {
            return true;
        }
        if (result == VK_TIMEOUT) {
            waited += slice;
            continue;
        }
        // A destructor cannot throw (implicit noexcept -> std::terminate), so
        // the failure is recorded instead: the backend stays marked lost and no
        // later operation touches the device again.
        note_device_lost(result == VK_ERROR_DEVICE_LOST
                             ? "VK_ERROR_DEVICE_LOST while draining fences in a destructor"
                             : "fence wait failed while draining fences in a destructor");
        return false;
    }
}

bool VulkanBackend::drain_pending(std::uint64_t budget_ns, bool allow_throw) {
    // One shared deadline for the whole drain (a wall clock instant, not a
    // slice per fence): after a loss every fence is unsignalled, and paying a
    // bounded slice per slot per buffer is what made the fallback unwind slow.
    // The slots stay pending (their tickets own them) -- only the waiting is
    // bounded here.
    //
    // A throwing drain uses the per-fence fence budget (budget_ns == 0 means
    // "the device default", which wait_fence applies). A no-throw drain is the
    // destructor path and is bounded by `budget_ns`, further capped by the
    // shared post-loss window below.
    constexpr std::uint64_t kPostLossDrainNs = 1'000'000'000ULL;
    if (!device_lost()) {
        // A healthy drain (the normal host-copy path, or a device restored by
        // the test-level reset) disarms the window, so the next genuine loss
        // gets its own full budget.
        lost_drain_started_ = false;
    } else if (!allow_throw) {
        // The no-throw drain runs from destructors during a fallback unwind,
        // where one slice per buffer would multiply the cost: the post-loss
        // budget is shared by the whole backend, keyed to the loss that armed
        // it. The first drain after a loss pays it, every later one returns
        // immediately -- so an operation that destroys a dozen buffers still
        // costs at most one second in total. A different loss (the generation
        // moved on) arms a fresh window.
        const std::uint64_t generation = device_lost_generation();
        const auto now = std::chrono::steady_clock::now();
        if (!lost_drain_started_ || lost_drain_generation_ != generation) {
            lost_drain_started_ = true;
            lost_drain_generation_ = generation;
            lost_drain_deadline_ = now + std::chrono::nanoseconds(kPostLossDrainNs);
        }
        if (now >= lost_drain_deadline_) {
            return false;
        }
        const std::uint64_t left = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(lost_drain_deadline_ - now)
                .count());
        budget_ns = budget_ns == 0 ? left : std::min(budget_ns, left);
    }
    if (allow_throw) {
        // Each fence gets its own full budget: a wait that fails is a device
        // failure and throws, so there is nothing to share a deadline with.
        for (const Slot& slot : slots_) {
            if (slot.fence != VK_NULL_HANDLE) {
                wait_fence(slot.fence);
            }
        }
        for (VkFence fence : {transfer_fence_, image_fence_}) {
            if (fence != VK_NULL_HANDLE) {
                wait_fence(fence);
            }
        }
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(budget_ns);
    const auto remaining_ns = [&]() -> std::uint64_t {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return 0;
        }
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count());
    };
    bool drained = true;
    const auto wait_one = [&](VkFence fence) {
        if (fence == VK_NULL_HANDLE) {
            return;
        }
        const std::uint64_t left = remaining_ns();
        if (left == 0) {
            // The shared deadline is gone. Do not issue a zero-length wait (it
            // would report a loss for a fence that may still signal); the
            // device is marked lost below, once.
            drained = false;
            return;
        }
        if (!wait_fence_noexcept(fence, left)) {
            drained = false;
        }
    };
    // Every slot that owns a fence, not just the pending ones: a slot whose wait
    // threw was abandoned (its ticket cleared, so nothing will ever wait it
    // again), yet the queue may still be reading its buffers -- and the unwind
    // destroying those buffers is exactly why this drain runs. A signalled fence
    // returns immediately, so including the idle slots costs nothing.
    for (const Slot& slot : slots_) {
        if (slot.fence != VK_NULL_HANDLE) {
            wait_one(slot.fence);
        }
    }
    // Transfers live outside the ring: a copy that threw a fence timeout left
    // its submission queued, and the unwind is about to free what it targets.
    for (VkFence fence : {transfer_fence_, image_fence_}) {
        if (fence != VK_NULL_HANDLE) {
            wait_one(fence);
        }
    }
    if (!drained && !lost()) {
        // The shared deadline ran out before every fence signalled: record it
        // so no later operation touches the device. (A fence that reported a
        // loss has already logged the precise reason, and note_device_lost only
        // records the first one.)
        note_device_lost("fence drain did not complete");
    }
    return drained;
}

void VulkanBackend::wait_fence(VkFence fence, std::uint64_t budget_ns) {
    const Functions& fns = instance_->fns;
    // The device is already gone. Waiting on it never returns and its fences
    // never signal, so report the failure rather than return success: a caller
    // that saw a throw must never be handed buffers the queue never finished
    // writing, and the ring's slot bookkeeping assumes the work ran.
    if (lost()) {
        throw Error("gpu: device lost (" + device_lost_reason() + ")");
    }
    // Test hook: behave exactly like a wedged device without hanging the suite.
    // Mark first so this precise reason wins over the generic drain message; the
    // drain then refuses every fence without touching Vulkan, which is exactly
    // what a genuinely wedged device costs.
    if (consume_forced_fence_timeout()) {
        note_device_lost("forced fence timeout (test hook)");
        drain_pending(200'000'000ULL, /*allow_throw=*/false);
        throw Error("gpu: fence timeout (forced by test hook)");
    }
    // Fence waits poll in one-second slices and are bounded, so a wedged device
    // raises Error instead of hanging. The budget is small now that every long
    // loop splits its submissions (gpu::run_batched): a submission is meant to
    // run for tens of milliseconds, so a wait of seconds already means the
    // device hung. On a shared display GPU (Apple silicon, an iGPU) waiting
    // longer does not help -- the compositor is already starved and the machine
    // can freeze -- so the hardware budget is 5 s, not the old 30 s. A CPU
    // driver (llvmpipe) runs the "device" in host threads; it cannot starve a
    // display and one Poisson step can legitimately take minutes, so software
    // keeps the generous 10 min budget. A caller-supplied budget may only
    // lengthen it: run_batched's calibrating submission is the one place whose
    // duration is not yet known.
    constexpr std::uint64_t kSliceNs = 1'000'000'000ULL;
    constexpr std::uint64_t kHardwareBudgetNs = 5'000'000'000ULL;
    constexpr std::uint64_t kSoftwareBudgetNs = 600'000'000'000ULL;
    const std::uint64_t default_budget_ns =
        device_.info.software ? kSoftwareBudgetNs : kHardwareBudgetNs;
    const std::uint64_t effective_budget_ns =
        budget_ns == 0 ? default_budget_ns : std::max(budget_ns, default_budget_ns);
    std::uint64_t waited_ns = 0;
    while (waited_ns < effective_budget_ns) {
        const std::uint64_t slice_ns = std::min(kSliceNs, effective_budget_ns - waited_ns);
        const VkResult result = fns.WaitForFences(device_handle_, 1, &fence, VK_TRUE, slice_ns);
        if (result == VK_SUCCESS) {
            return;
        }
        if (result == VK_TIMEOUT) {
            waited_ns += slice_ns;
            continue;
        }
        if (result == VK_ERROR_DEVICE_LOST) {
            note_device_lost("VK_ERROR_DEVICE_LOST while waiting for GPU completion");
            throw Error("Vulkan device lost while waiting for GPU completion");
        }
        throw Error(vk_result_string("vkWaitForFences", result));
    }
    // A timeout is a device failure, not a slow device: mark the GPU lost so
    // every later call takes its CPU branch, and let with_fallback rerun this
    // operation on the CPU. Mark first, then give the outstanding fences one
    // shared bounded drain: the unwind frees this batch's buffers, and a queue
    // still reading them would fault.
    note_device_lost("no fence signal within " +
                     std::to_string(effective_budget_ns / 1'000'000'000ULL) + " s");
    drain_pending(200'000'000ULL, /*allow_throw=*/false);
    throw Error("gpu: fence timeout");
}

VulkanBackend::Slot* VulkanBackend::reserve_slot() {
    for (Slot& slot : slots_) {
        if (!slot.pending) {
            return &slot;
        }
    }
    return nullptr;
}

std::uint64_t VulkanBackend::oldest_pending_id() {
    std::uint64_t oldest = 0;
    for (const Slot& slot : slots_) {
        if (slot.pending && (oldest == 0 || slot.id < oldest)) {
            oldest = slot.id;
        }
    }
    return oldest;
}

void VulkanBackend::release_slot(Slot& slot) {
    const Functions& fns = instance_->fns;
    if (slot.descriptor_pool != VK_NULL_HANDLE) {
        fns.DestroyDescriptorPool(device_handle_, slot.descriptor_pool, nullptr);
        slot.descriptor_pool = VK_NULL_HANDLE;
    }
    slot.pending = false;
    slot.id = 0;
}

void VulkanBackend::abandon_slot(Slot& slot) {
    // The fence is unsignalled and will stay that way (the device is lost). Drop
    // the booking so no later wait() or destructor waits on it, but keep the
    // descriptor pool: the abandoned submission may still reference it, and
    // nothing is submitted to a lost device to reuse the slot. It is freed by
    // ~VulkanBackend (or by a release_slot on a restored device).
    slot.pending = false;
    slot.id = 0;
}

void VulkanBackend::record_and_submit(Slot& slot,
                                      const std::vector<DispatchRequest>& dispatches) {
    const Functions& fns = instance_->fns;
    VkResult result = fns.ResetCommandBuffer(slot.command_buffer, 0);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkResetCommandBuffer", result));
    }

    // Descriptors: one set per dispatch, from a pool sized to fit them all.
    // The pool must cover every descriptor actually bound: kernels with more
    // than eight bindings (pm_* use fourteen) would otherwise fail
    // vkAllocateDescriptorSets with VK_ERROR_OUT_OF_POOL_MEMORY. A pool size
    // with descriptorCount == 0 is itself a validation error
    // (VUID-VkDescriptorPoolSize-descriptorCount-00302), so each kind is only
    // declared when the batch actually uses it.
    std::uint32_t total_sets = 0;
    std::uint32_t total_buffers = 0;
    std::uint32_t total_images = 0;
    for (const DispatchRequest& dispatch : dispatches) {
        const bool has_bindings =
            !dispatch.buffers.empty() || !dispatch.images.empty();
        total_sets += static_cast<std::uint32_t>(has_bindings ? 1 : 0);
        total_buffers += static_cast<std::uint32_t>(dispatch.buffers.size());
        total_images += static_cast<std::uint32_t>(dispatch.images.size());
    }
    if (total_sets != 0) {
        VkDescriptorPoolSize sizes[2]{};
        std::uint32_t size_count = 0;
        if (total_buffers != 0) {
            sizes[size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[size_count].descriptorCount = total_buffers;
            ++size_count;
        }
        if (total_images != 0) {
            sizes[size_count].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[size_count].descriptorCount = total_images;
            ++size_count;
        }
        VkDescriptorPoolCreateInfo descriptor_info{};
        descriptor_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_info.maxSets = total_sets;
        descriptor_info.poolSizeCount = size_count;
        descriptor_info.pPoolSizes = sizes;
        result = fns.CreateDescriptorPool(device_handle_, &descriptor_info, nullptr,
                                          &slot.descriptor_pool);
        if (result != VK_SUCCESS) {
            throw Error(vk_result_string("vkCreateDescriptorPool", result));
        }
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = fns.BeginCommandBuffer(slot.command_buffer, &begin_info);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkBeginCommandBuffer", result));
    }

    for (std::size_t index = 0; index < dispatches.size(); ++index) {
        const DispatchRequest& dispatch = dispatches[index];
        const Pipeline& pipeline = pipeline_for(*dispatch.kernel);

        {
            // Storage-buffer writes from the previous dispatch -- or from the
            // previous submission, when a long loop was split across command
            // buffers (gpu::run_batched) -- must be visible to this one. The
            // barrier is issued unconditionally, including at index 0: the
            // first dispatch of a split submission needs the same dependency
            // the unsplit command buffer gets for free from its predecessor's
            // barrier, and a barrier with no prior writes is a no-op.
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            fns.CmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                                   nullptr, 0, nullptr);
        }

        fns.CmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.pipeline);
        if (!dispatch.buffers.empty() || !dispatch.images.empty()) {
            VkDescriptorSetAllocateInfo set_allocate{};
            set_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            set_allocate.descriptorPool = slot.descriptor_pool;
            set_allocate.descriptorSetCount = 1;
            set_allocate.pSetLayouts = &pipeline.set_layout;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
            result = fns.AllocateDescriptorSets(device_handle_, &set_allocate, &descriptor_set);
            if (result != VK_SUCCESS) {
                throw Error(vk_result_string("vkAllocateDescriptorSets", result));
            }
            const std::size_t write_count = dispatch.buffers.size() + dispatch.images.size();
            std::vector<VkDescriptorBufferInfo> buffer_infos(dispatch.buffers.size());
            std::vector<VkDescriptorImageInfo> image_infos(dispatch.images.size());
            std::vector<VkWriteDescriptorSet> writes(write_count);
            // Buffers fill the STORAGE_BUFFER bindings in ascending order;
            // images carry their explicit binding (validated in submit()).
            std::size_t storage_index = 0;
            for (std::size_t b = 0; b < pipeline.info.bindings.size(); ++b) {
                if (pipeline.info.bindings[b].kind != BindingKind::StorageBuffer) {
                    continue;
                }
                if (storage_index >= dispatch.buffers.size()) {
                    break;
                }
                const VulkanBuffer* buffer = as_vulkan(dispatch.buffers[storage_index]);
                if (buffer == nullptr) {
                    throw Error("buffer passed to GPU dispatch did not come from this layer");
                }
                buffer_infos[storage_index].buffer = buffer->handle();
                buffer_infos[storage_index].offset = 0;
                buffer_infos[storage_index].range = VK_WHOLE_SIZE;
                writes[storage_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[storage_index].dstSet = descriptor_set;
                writes[storage_index].dstBinding = pipeline.info.bindings[b].binding;
                writes[storage_index].descriptorCount = 1;
                writes[storage_index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[storage_index].pBufferInfo = &buffer_infos[storage_index];
                ++storage_index;
            }
            for (std::size_t i = 0; i < dispatch.images.size(); ++i) {
                const VulkanImage* image = as_vulkan(dispatch.images[i].image);
                const VulkanSampler* sampler = as_vulkan(dispatch.images[i].sampler);
                if (image == nullptr || sampler == nullptr) {
                    throw Error(
                        "image or sampler passed to GPU dispatch did not come from this layer");
                }
                image_infos[i].sampler = sampler->handle();
                image_infos[i].imageView = image->view();
                image_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                const std::size_t write_index = dispatch.buffers.size() + i;
                writes[write_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[write_index].dstSet = descriptor_set;
                writes[write_index].dstBinding = dispatch.images[i].binding;
                writes[write_index].descriptorCount = 1;
                writes[write_index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[write_index].pImageInfo = &image_infos[i];
            }
            fns.UpdateDescriptorSets(device_handle_, static_cast<std::uint32_t>(writes.size()),
                                     writes.data(), 0, nullptr);
            fns.CmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.layout, 0, 1, &descriptor_set, 0, nullptr);
        }
        if (!dispatch.push_constants.empty()) {
            fns.CmdPushConstants(slot.command_buffer, pipeline.layout,
                                 VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 static_cast<std::uint32_t>(dispatch.push_constants.size()),
                                 dispatch.push_constants.data());
        }
        fns.CmdDispatch(slot.command_buffer, dispatch.groups[0], dispatch.groups[1],
                        dispatch.groups[2]);
    }

    // Make the final writes visible to host reads and to a later staging copy.
    VkMemoryBarrier final_barrier{};
    final_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    final_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    final_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    fns.CmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                           &final_barrier, 0, nullptr, 0, nullptr);

    result = fns.EndCommandBuffer(slot.command_buffer);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkEndCommandBuffer", result));
    }

    if (slot.fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = fns.CreateFence(device_handle_, &fence_info, nullptr, &slot.fence);
        if (result != VK_SUCCESS) {
            throw Error(vk_result_string("vkCreateFence", result));
        }
    }
    result = fns.ResetFences(device_handle_, 1, &slot.fence);
    if (result != VK_SUCCESS) {
        throw Error(vk_result_string("vkResetFences", result));
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.command_buffer;
    result = fns.QueueSubmit(queue(slot.queue), 1, &submit, slot.fence);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            note_device_lost("VK_ERROR_DEVICE_LOST from vkQueueSubmit");
        }
        throw Error(vk_result_string("vkQueueSubmit", result));
    }
}

std::uint64_t VulkanBackend::submit(const std::vector<DispatchRequest>& dispatches, int queue_index,
                                    bool block_when_full) {
    if (dispatches.empty()) {
        return 0;
    }
    // The device stopped responding earlier in this process. Refuse before
    // touching Vulkan: resubmitting to a wedged GPU is what the whole
    // device-lost path exists to prevent, and on a shared display GPU the
    // second hang is the one that takes the machine down. A thread already
    // inside run_batched stops here, at its next submission.
    if (lost()) {
        throw Error("gpu: device lost (" + device_lost_reason() +
                    "); refusing to submit more work");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    // Validate every dispatch against its kernel's reflected interface before
    // touching Vulkan, so a bad call is a clean gpu::Error.
    for (const DispatchRequest& dispatch : dispatches) {
        if (dispatch.kernel == nullptr) {
            throw Error("null kernel in GPU dispatch");
        }
        const Pipeline& pipeline = pipeline_for(*dispatch.kernel);
        const std::string name = dispatch.kernel->name;
        const std::size_t expected_buffers =
            pipeline.info.count(BindingKind::StorageBuffer);
        const std::size_t expected_images =
            pipeline.info.count(BindingKind::CombinedImageSampler);
        if (dispatch.buffers.size() != expected_buffers) {
            throw Error("kernel '" + name + "' expects " +
                        std::to_string(expected_buffers) +
                        " storage buffer binding(s) [" +
                        binding_list(pipeline.info.bindings, BindingKind::StorageBuffer) +
                        "] but the dispatch supplied " + std::to_string(dispatch.buffers.size()));
        }
        for (std::size_t i = 0; i < dispatch.buffers.size(); ++i) {
            if (dispatch.buffers[i] == nullptr) {
                throw Error("null buffer at index " + std::to_string(i) + " of kernel '" + name +
                            "'");
            }
        }
        if (dispatch.images.size() != expected_images) {
            throw Error("kernel '" + name + "' expects " + std::to_string(expected_images) +
                        " combined image sampler binding(s) [" +
                        binding_list(pipeline.info.bindings, BindingKind::CombinedImageSampler) +
                        "] but the dispatch supplied " + std::to_string(dispatch.images.size()));
        }
        // Every image binding must name a binding the shader declares as a
        // combined image sampler, exactly once, and must be paired with a
        // sampler that comes from this backend with contents uploaded.
        for (std::size_t i = 0; i < dispatch.images.size(); ++i) {
            const ImageBinding& entry = dispatch.images[i];
            const BindingInfo* declared = pipeline.info.find(entry.binding);
            if (declared == nullptr) {
                throw Error("kernel '" + name + "' has no binding " +
                            std::to_string(entry.binding) + " (declared: [" +
                            binding_list(pipeline.info.bindings, BindingKind::StorageBuffer) +
                            "] buffers, [" +
                            binding_list(pipeline.info.bindings, BindingKind::CombinedImageSampler) +
                            "] images)");
            }
            if (declared->kind != BindingKind::CombinedImageSampler) {
                throw Error("binding " + std::to_string(entry.binding) + " of kernel '" + name +
                            "' is a storage buffer, not a combined image sampler");
            }
            if (i != 0 && dispatch.images[i - 1].binding == entry.binding) {
                throw Error("binding " + std::to_string(entry.binding) +
                            " appears twice in the dispatch of kernel '" + name + "'");
            }
            if (entry.image == nullptr || entry.sampler == nullptr) {
                throw Error("binding " + std::to_string(entry.binding) + " of kernel '" + name +
                            "' has an image without a sampler");
            }
            const VulkanImage* image = as_vulkan(entry.image);
            const VulkanSampler* sampler = as_vulkan(entry.sampler);
            if (image == nullptr) {
                throw Error("image at binding " + std::to_string(entry.binding) + " of kernel '" +
                            name + "' did not come from this layer");
            }
            if (sampler == nullptr) {
                throw Error("sampler at binding " + std::to_string(entry.binding) + " of kernel '" +
                            name + "' did not come from this layer");
            }
            if (image->owner() != this || sampler->owner() != this) {
                throw Error("image or sampler at binding " + std::to_string(entry.binding) +
                            " of kernel '" + name + "' belongs to a different GPU context");
            }
            if (!image->has_contents()) {
                throw Error("image at binding " + std::to_string(entry.binding) + " of kernel '" +
                            name + "' has never been uploaded");
            }
        }
        if (dispatch.push_constants.size() != pipeline.info.push_constant_size) {
            throw Error("kernel '" + name + "' expects " +
                        std::to_string(pipeline.info.push_constant_size) +
                        " push-constant byte(s) but the dispatch supplied " +
                        std::to_string(dispatch.push_constants.size()));
        }
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (dispatch.groups[axis] == 0) {
                throw Error("kernel '" + name + "' dispatch has a zero workgroup count");
            }
        }
    }

    // Acquire a slot. run_async (block_when_full == false) throws on a full
    // ring; the synchronous path drains the oldest ticket and retries, bounded
    // because a sibling thread may re-take the slot in between.
    Slot* reserved = nullptr;
    for (int attempt = 0; attempt <= static_cast<int>(kMaxInFlight); ++attempt) {
        reserved = reserve_slot();
        if (reserved != nullptr) {
            break;
        }
        if (!block_when_full) {
            break;
        }
        const std::uint64_t oldest = oldest_pending_id();
        if (oldest == 0) {
            break;
        }
        // Outside the lock: the fence wait must be able to release the slot.
        lock.unlock();
        wait(oldest);
        lock.lock();
    }
    if (reserved == nullptr) {
        throw Error("gpu: " + std::to_string(kMaxInFlight) +
                    " submissions already in flight on this context; wait() one before "
                    "submitting");
    }
    Slot& slot = *reserved;
    const std::size_t index = static_cast<std::size_t>(reserved - slots_.data());
    if (slot.command_buffer == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = command_pools_[index % command_pools_.size()];
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        const VkResult result = instance_->fns.AllocateCommandBuffers(
            device_handle_, &allocate_info, &slot.command_buffer);
        if (result != VK_SUCCESS) {
            throw Error(vk_result_string("vkAllocateCommandBuffers", result));
        }
    }
    if (queue_index == 1 && queue_count_ < 2) {
        // One log line per process: the fallback is silent otherwise.
        static bool reported = false;
        if (!reported) {
            reported = true;
            log(LogLevel::Info) << "GPU: Submission{.queue = 1} requested but the queue family "
                                   "has one queue; using queue 0";
        }
    }
    slot.queue = queue_index == 1 && queue_count_ >= 2 ? 1 : 0;
    slot.pending = true;
    slot.id = next_ticket_++;
    try {
        record_and_submit(slot, dispatches);
    } catch (...) {
        release_slot(slot);
        throw;
    }
    return slot.id;
}

void VulkanBackend::wait(std::uint64_t ticket) {
    if (ticket == 0) {
        return;
    }
    // The fence wait runs under the submission mutex: a slot must not be
    // recycled (and its fence reset for a new submission) while another thread
    // is still waiting on it, and a single queue serialises the work anyway.
    std::lock_guard<std::mutex> lock(mutex_);
    for (Slot& slot : slots_) {
        if (slot.pending && slot.id == ticket) {
            try {
                wait_fence(slot.fence);
            } catch (...) {
                // A failed wait is a device failure: the fence will never
                // signal, so the slot is abandoned rather than held forever.
                // Nothing else can submit to a lost device, so no later
                // wait()/destructor spins on it.
                abandon_slot(slot);
                throw;
            }
            release_slot(slot);
            return;
        }
    }
    // Already completed (waited for by another call), or an unknown ticket.
}

std::uint32_t VulkanBackend::in_flight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint32_t count = 0;
    for (const Slot& slot : slots_) {
        if (slot.pending) {
            ++count;
        }
    }
    return count;
}

void VulkanBackend::wait_all_pending() {
    // Called from the host-copy paths (upload/download/staging), which are
    // about to touch memory the queue may still be writing. Holding the
    // submission mutex keeps slots from being recycled while their fences are
    // waited on; the transfer mutex is always taken before this one, so the
    // order transfer_mutex_ -> mutex_ is consistent with run()/wait().
    std::lock_guard<std::mutex> lock(mutex_);
    // A lost device's fences never signal: throw instead of returning, because
    // the caller would otherwise read (or overwrite) a buffer the queue never
    // finished with. Nothing is submitted after a loss, so the samples the copy
    // is about to take would be garbage either way.
    if (lost()) {
        throw Error("gpu: device lost (" + device_lost_reason() + ")");
    }
    drain_pending(0, /*allow_throw=*/true);
    // The slots stay pending: they are owned by their tickets until wait().
}

bool VulkanBackend::wait_all_pending_noexcept() {
    // Destructor path: a throw here is std::terminate (destructors are
    // implicitly noexcept). drain_pending bounds the drain, and once the device
    // is lost the budget is *shared* across the whole fallback unwind (see
    // kPostLossDrainNs), so destroying a dozen buffers after a hang costs one
    // second in total rather than one bound per handle.
    constexpr std::uint64_t kHealthyDrainBudgetNs = 5'000'000'000ULL;
    constexpr std::uint64_t kSoftwareDrainBudgetNs = 600'000'000'000ULL;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint64_t budget =
            device_.info.software ? kSoftwareDrainBudgetNs : kHealthyDrainBudgetNs;
        return drain_pending(budget, /*allow_throw=*/false);
    } catch (...) {
        // Defensive only: nothing under the lock is expected to throw here, and
        // a failure is reported by the fence drain itself (device marked lost).
        note_device_lost("exception while draining fences in a destructor");
        return false;
    }
}

void VulkanBackend::run(const std::vector<DispatchRequest>& dispatches) {
    run_with_fence_budget(dispatches, 0);
}

void VulkanBackend::run_with_fence_budget(const std::vector<DispatchRequest>& dispatches,
                                          std::uint64_t fence_budget_ns) {
    if (dispatches.empty()) {
        return;
    }
    // The synchronous path drains the oldest pending submission instead of
    // rejecting a full ring: run() is documented to wait, and under
    // concurrency a run_async from another thread may hold every slot.
    const std::uint64_t ticket = submit(dispatches, 0, true);
    if (fence_budget_ns == 0) {
        wait(ticket);
        return;
    }
    // One submission with a caller-chosen fence budget. The wait below mirrors
    // wait(), including abandoning the slot on failure, so the ring accounting
    // is unchanged -- only the fence budget differs.
    std::lock_guard<std::mutex> lock(mutex_);
    for (Slot& slot : slots_) {
        if (slot.pending && slot.id == ticket) {
            try {
                wait_fence(slot.fence, fence_budget_ns);
            } catch (...) {
                abandon_slot(slot);
                throw;
            }
            release_slot(slot);
            return;
        }
    }
}

}  // namespace alphashape::gpu::detail
