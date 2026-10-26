#include "gpu/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/log.hpp"
#include "gpu/gpu_config.hpp"
#include "gpu/vulkan/backend.hpp"

namespace alphashape::gpu {
namespace {

std::once_flag g_env_once;
bool g_enabled = true;

std::once_flag g_default_once;
// Deliberately never destroyed: at process exit the Vulkan driver (MoltenVK in
// particular) may already have run its own static teardown, so destroying the
// device from our static destructors crashes. The OS reclaims it instead.
std::shared_ptr<Context>& g_default_context = *new std::shared_ptr<Context>();

std::mutex g_state_mutex;
std::string g_reason;
std::string g_attempted_reason;

std::atomic<std::uint64_t> g_failure_count{0};
std::atomic<std::uint64_t> g_validation_errors{0};

/// Set once the device has stopped responding. Not a user switch (that is
/// g_enabled): a lost device stays unavailable for the life of the process, so
/// a shared display GPU is never submitted to again after a hang.
std::atomic<bool> g_device_lost{false};
std::mutex g_device_lost_mutex;
std::string g_device_lost_detail;
/// Test hook: the next fence wait reports a timeout instead of waiting.
std::atomic<bool> g_forced_fence_timeout{false};

void read_environment() {
    const char* value = std::getenv("ALPHASHAPE_GPU");
    if (value == nullptr) {
        g_enabled = true;
        return;
    }
    std::string text(value);
    for (char& character : text) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    g_enabled = !(text == "0" || text == "false" || text == "off" || text == "no");
}

void record_reason(const std::string& reason) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_reason = reason;
}

std::shared_ptr<Context> init_default_context() {
    if (!detail::kCompiledIn) {
        record_reason("GPU support disabled at build time (ALPHASHAPE_GPU=OFF)");
        return nullptr;
    }
    try {
        return create_context(ContextOptions{});
    } catch (const std::exception& error) {
        record_reason(error.what());
        return nullptr;
    }
}

}  // namespace

std::size_t bytes_per_pixel(Format format) {
    switch (format) {
        case Format::R8Unorm:
            return 1;
        case Format::R8G8B8A8Unorm:
        case Format::R32Float:
            return 4;
    }
    return 0;
}

const char* format_name(Format format) {
    switch (format) {
        case Format::R8Unorm:
            return "R8Unorm";
        case Format::R8G8B8A8Unorm:
            return "R8G8B8A8Unorm";
        case Format::R32Float:
            return "R32Float";
    }
    return "unknown";
}

namespace detail {

void note_gpu_failure(const std::string& what) {
    g_failure_count.fetch_add(1);
    if (g_device_lost.load()) {
        // The device-lost line (logged once, by note_device_lost) already says
        // why the GPU is off; a second "switching to CPU" line per operation
        // would just repeat it.
        log(LogLevel::Debug) << "GPU operation fell back to the CPU: " << what;
        return;
    }
    log(LogLevel::Warn) << "GPU processing failed, switching to CPU mode: " << what;
}

void note_device_lost(const std::string& detail) {
    // Once only: a second timeout must not print a second line or race the
    // first reason out of the record.
    if (g_device_lost.exchange(true)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_device_lost_mutex);
        g_device_lost_detail = detail;
    }
    log(LogLevel::Warn) << "GPU stopped responding; continuing on the CPU ("
                        << "device lost: " << detail << "). The GPU layer stays off for this "
                        << "process.";
}

bool device_lost() { return g_device_lost.load(); }

std::string device_lost_reason() {
    std::lock_guard<std::mutex> lock(g_device_lost_mutex);
    return g_device_lost_detail;
}

void force_fence_timeout_once() { g_forced_fence_timeout.store(true); }

bool consume_forced_fence_timeout() { return g_forced_fence_timeout.exchange(false); }

void reset_device_lost() {
    g_device_lost.store(false);
    {
        std::lock_guard<std::mutex> lock(g_device_lost_mutex);
        g_device_lost_detail.clear();
    }
    g_forced_fence_timeout.store(false);
}

std::uint64_t validation_error_count() { return g_validation_errors.load(); }

void add_validation_error() { g_validation_errors.fetch_add(1); }

}  // namespace detail

bool enabled() {
    std::call_once(g_env_once, read_environment);
    return g_enabled;
}

void set_enabled(bool enabled_value) {
    std::call_once(g_env_once, read_environment);
    // A lost device cannot be switched back on: the driver keeps its state, and
    // resubmitting to a wedged GPU is exactly what this guards against. Tests
    // clear it with detail::reset_device_lost().
    g_enabled = enabled_value;
}

bool available() {
    if (!enabled()) {
        return false;
    }
    if (detail::device_lost()) {
        return false;
    }
    return default_context() != nullptr;
}

std::string unavailable_reason() {
    std::call_once(g_env_once, read_environment);
    if (!g_enabled) {
        return "GPU layer disabled (ALPHASHAPE_GPU=0 or set_enabled(false))";
    }
    if (detail::device_lost()) {
        return "GPU stopped responding; continuing on the CPU";
    }
    if (!detail::kCompiledIn) {
        return "GPU support disabled at build time (ALPHASHAPE_GPU=OFF)";
    }
    if (default_context() != nullptr) {
        return {};
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_reason;
}

std::shared_ptr<Context> default_context() {
    std::call_once(g_env_once, read_environment);
    if (!g_enabled || detail::device_lost()) {
        return nullptr;
    }
    std::call_once(g_default_once, [] { g_default_context = init_default_context(); });
    return g_default_context;
}

std::shared_ptr<Context> create_context(const ContextOptions& options) {
    if (!detail::kCompiledIn) {
        throw Error("GPU support disabled at build time (ALPHASHAPE_GPU=OFF)");
    }
    return std::shared_ptr<Context>(new Context(detail::create_backend(options)));
}

Context::Context(std::shared_ptr<detail::Backend> backend) : backend_(std::move(backend)) {}
Context::~Context() = default;
const DeviceInfo& Context::info() const { return backend_->info(); }

std::vector<DeviceInfo> enumerate_devices() {
    std::vector<DeviceInfo> devices;
    std::string reason;
    if (!detail::enumerate_devices(&devices, &reason)) {
        record_reason(reason);
        return {};
    }
    return devices;
}

bool is_software_device(const DeviceInfo& info) {
    // The device type is authoritative when the driver reports it; some CPU
    // drivers report an "other" type, so the names of known CPU implementations
    // are matched as well (never a heuristic on performance).
    if (info.software) {
        return true;
    }
    static const char* const kKnown[] = {"llvmpipe", "lavapipe", "swiftshader",
                                         "softpipe", "swrast"};
    for (const char* needle : kKnown) {
        if (info.name.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::size_t Buffer::size() const { return handle_ ? handle_->size() : 0; }

void Buffer::upload(const void* data, std::size_t bytes, std::size_t offset) {
    if (!handle_) {
        throw Error("upload on an empty GPU buffer");
    }
    handle_->upload(data, bytes, offset);
}

void Buffer::download(void* data, std::size_t bytes, std::size_t offset) const {
    if (!handle_) {
        throw Error("download from an empty GPU buffer");
    }
    handle_->download(data, bytes, offset);
}

Buffer make_buffer(Context& context, std::size_t bytes, MemoryKind kind) {
    return detail::make_buffer(context, nullptr, bytes, kind);
}

Image2D Image2D::create(Context& context, int width, int height, Format format,
                        const void* pixels) {
    return detail::make_image(context, width, height, format, pixels);
}

void Image2D::upload(const void* pixels, std::size_t bytes) {
    if (!handle_) {
        throw Error("upload on an empty GPU image");
    }
    handle_->upload(pixels, bytes);
}

void Image2D::download(void* pixels, std::size_t bytes) const {
    if (!handle_) {
        throw Error("download from an empty GPU image");
    }
    handle_->download(pixels, bytes);
}

Sampler Sampler::create(Context& context, Filter filter, AddressMode address) {
    return detail::make_sampler(context, filter, address);
}

bool image_sampling_supported(Context& context, Format format) {
    return detail::image_sampling_supported(context, format);
}

namespace {

/// Translates the public dispatch list into the backend's request list.
///
/// Binding semantics: `buffers` fill the STORAGE_BUFFER bindings in ascending
/// binding order; `images`/`samplers` fill the COMBINED_IMAGE_SAMPLER bindings
/// at the explicit indices they carry. Both lists may be mixed in one
/// dispatch. Indices must be unique and must match the kernel's reflected
/// interface; a mismatch throws here, before any Vulkan call.
std::vector<detail::DispatchRequest> requests_for(const std::vector<Dispatch>& dispatches) {
    std::vector<detail::DispatchRequest> requests;
    requests.reserve(dispatches.size());
    for (const Dispatch& dispatch : dispatches) {
        detail::DispatchRequest request;
        request.kernel = detail::find_kernel(dispatch.kernel);
        if (request.kernel == nullptr) {
            throw Error("unknown GPU kernel: " + dispatch.kernel);
        }
        request.push_constants = dispatch.push_constants;
        request.groups = dispatch.groups;

        for (const Buffer* buffer : dispatch.buffers) {
            if (buffer == nullptr) {
                throw Error("null buffer in GPU dispatch of '" + dispatch.kernel + "'");
            }
            request.buffers.push_back(detail::BufferAccess::handle(*buffer));
        }

        // Images and samplers arrive as separate (binding, pointer) lists and
        // are paired here; the backend re-checks uniqueness, kinds and
        // ownership against the same reflected interface.
        std::vector<std::pair<std::uint32_t, const detail::ImageHandle*>> images;
        images.reserve(dispatch.images.size());
        for (const auto& entry : dispatch.images) {
            if (entry.second == nullptr) {
                throw Error("null image at binding " + std::to_string(entry.first) +
                            " of kernel '" + dispatch.kernel + "'");
            }
            images.emplace_back(entry.first, detail::ImageAccess::handle(*entry.second));
        }
        for (const auto& entry : dispatch.samplers) {
            if (entry.second == nullptr) {
                throw Error("null sampler at binding " + std::to_string(entry.first) +
                            " of kernel '" + dispatch.kernel + "'");
            }
            const detail::SamplerHandle* sampler = detail::SamplerAccess::handle(*entry.second);
            bool paired = false;
            for (auto& image : images) {
                if (image.first == entry.first) {
                    request.images.push_back({entry.first, image.second, sampler});
                    image.second = nullptr;  // mark paired
                    paired = true;
                    break;
                }
            }
            if (!paired) {
                throw Error("sampler at binding " + std::to_string(entry.first) +
                            " has no matching image in kernel '" + dispatch.kernel + "'");
            }
        }
        for (const auto& image : images) {
            if (image.second != nullptr) {
                throw Error("image at binding " + std::to_string(image.first) +
                            " has no matching sampler in kernel '" + dispatch.kernel + "'");
            }
        }
        // Deterministic order: the backend walks bindings and depends on it.
        std::sort(request.images.begin(), request.images.end(),
                  [](const detail::ImageBinding& a, const detail::ImageBinding& b) {
                      return a.binding < b.binding;
                  });
        requests.push_back(std::move(request));
    }
    return requests;
}

}  // namespace

void run(Context& context, const std::vector<Dispatch>& dispatches) {
    if (dispatches.empty()) {
        return;
    }
    if (!context.backend_) {
        throw Error("GPU work requested from an empty context");
    }
    // The synchronous path never fails on a full ring: the backend drains the
    // oldest pending submission instead (run == run_async + wait, and run is
    // documented to wait, not to reject under concurrency).
    context.backend_->run(requests_for(dispatches));
}

BatchLimiter::BatchLimiter(std::size_t dispatches_per_step, std::size_t initial_steps,
                           std::size_t max_steps)
    : per_step_(std::max<std::size_t>(dispatches_per_step, 1)),
      steps_(std::max<std::size_t>(initial_steps, 1)),
      max_steps_(std::max<std::size_t>(max_steps, 1)) {
    if (const std::size_t forced = override_steps(); forced != 0) {
        steps_ = std::min(forced, max_steps_);
    }
}

std::size_t BatchLimiter::override_steps() {
    const char* text = std::getenv("ALPHASHAPE_GPU_MAX_BATCH");
    if (text == nullptr || *text == '\0') {
        return 0;
    }
    // Strict parse. strtoull accepts a leading sign and leading whitespace, and
    // "-2" would wrap to a huge positive size: an override that silently became
    // something else would hide itself, so anything but plain digits is ignored
    // (adaptive), never guessed at.
    for (const char* digit = text; *digit != '\0'; ++digit) {
        if (*digit < '0' || *digit > '9') {
            return 0;
        }
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == nullptr || *end != '\0' || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        return 0;
    }
    return static_cast<std::size_t>(value);
}

std::size_t BatchLimiter::steps(std::size_t remaining_steps) const {
    if (remaining_steps == 0) {
        return 0;
    }
    return std::min(steps_, remaining_steps);
}

void BatchLimiter::observe(std::size_t steps_done, double seconds) {
    if (steps_done == 0 || !(seconds > 0.0) || !(target_ > 0.0)) {
        return;
    }
    const double per_step = seconds / static_cast<double>(steps_done);
    if (!(per_step > 0.0)) {
        return;
    }
    // Aim at the target, but move at most 4x per measurement so one noisy first
    // submission (or a pipeline compilation charged to it) cannot jump to a
    // multi-second batch.
    const double ideal = target_ / per_step;
    const double lowest = static_cast<double>(steps_done) / 4.0;
    const double highest = static_cast<double>(steps_done) * 4.0;
    const double scaled = std::min(std::max(ideal, lowest), highest);
    const double rounded = std::floor(scaled + 0.5);
    std::size_t next = rounded < 1.0 ? 1U : static_cast<std::size_t>(rounded);
    steps_ = std::min(next, max_steps_);
    if (const std::size_t forced = override_steps(); forced != 0) {
        steps_ = std::min(forced, max_steps_);
    }
}

std::size_t run_batched(Context& context, const std::vector<Dispatch>& dispatches,
                        BatchLimiter& limiter, std::size_t leading_dispatches,
                        std::size_t trailing_dispatches) {
    if (dispatches.empty()) {
        return 0;
    }
    if (!context.backend_) {
        throw Error("GPU work requested from an empty context");
    }
    // Resolve the target once per call: 80 ms on hardware, generous on a CPU
    // driver, where submissions are cheap to outlive but expensive to make.
    if (!(limiter.target_seconds() > 0.0)) {
        limiter.set_target_seconds(is_software_device(context.info())
                                       ? BatchLimiter::kSoftwareTargetSeconds
                                       : BatchLimiter::kHardwareTargetSeconds);
    }
    const std::size_t per_step = limiter.dispatches_per_step();
    // Leading and trailing dispatches are outside the step arithmetic; clamp
    // them so a caller cannot make the split skip anything.
    const std::size_t lead = std::min(leading_dispatches, dispatches.size());
    const std::size_t trail = std::min(trailing_dispatches, dispatches.size() - lead);
    const std::size_t body_end = dispatches.size() - trail;
    const std::size_t body_steps = (body_end - lead) / per_step;
    std::size_t submissions = 0;

    const auto submit_range = [&](std::size_t begin, std::size_t end) {
        if (end <= begin) {
            return;
        }
        run(context, std::vector<Dispatch>(
                         dispatches.begin() + static_cast<std::ptrdiff_t>(begin),
                         dispatches.begin() + static_cast<std::ptrdiff_t>(end)));
        ++submissions;
    };

    std::size_t steps_done = 0;
    std::size_t cursor = lead;
    while (steps_done < body_steps) {
        const std::size_t steps_here = limiter.steps(body_steps - steps_done);
        const bool last = steps_done + steps_here == body_steps;
        // The first submission carries the leading dispatches; the last one
        // carries everything after the body (the settle statistic, or a
        // malformed tail), so neither pays an extra round trip.
        const std::size_t begin = steps_done == 0 ? 0 : cursor;
        cursor = lead + (steps_done + steps_here) * per_step;
        const std::size_t end = last ? dispatches.size() : cursor;
        const auto start = std::chrono::steady_clock::now();
        submit_range(begin, end);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        limiter.observe(steps_here, seconds);
        steps_done += steps_here;
    }
    // No whole steps at all (leading/trailing work only): one submission.
    if (submissions == 0) {
        submit_range(0, dispatches.size());
    }
    return submissions;
}

Ticket run_async(Context& context, const std::vector<Dispatch>& dispatches,
                 Submission submission) {
    return detail::submit_async(context, dispatches, submission);
}

void wait(Context& context, Ticket ticket) {
    if (!context.backend_) {
        throw Error("wait on an empty GPU context");
    }
    context.backend_->wait(ticket.id());
}

std::vector<std::string> kernel_names() { return detail::all_kernel_names(); }

GoldReport compare(const std::vector<float>& gpu_values, const std::vector<float>& cpu_values) {
    GoldReport report;
    if (gpu_values.size() != cpu_values.size()) {
        report.count = std::min(gpu_values.size(), cpu_values.size());
        report.mean_abs = report.max_abs = std::numeric_limits<double>::infinity();
        return report;
    }
    report.count = gpu_values.size();
    double sum = 0.0;
    double worst = 0.0;
    for (std::size_t i = 0; i < gpu_values.size(); ++i) {
        const double difference =
            static_cast<double>(gpu_values[i]) - static_cast<double>(cpu_values[i]);
        if (!std::isfinite(difference)) {
            worst = std::numeric_limits<double>::infinity();
        } else {
            const double magnitude = std::fabs(difference);
            sum += magnitude;
            worst = std::max(worst, magnitude);
        }
    }
    report.mean_abs = report.count != 0 ? sum / static_cast<double>(report.count) : 0.0;
    report.max_abs = worst;
    return report;
}

std::uint64_t failure_count() { return g_failure_count.load(); }

}  // namespace alphashape::gpu

namespace alphashape::gpu::detail {

bool backend_compiled_in() { return kCompiledIn; }

bool validation_active(const Context& context) {
    return context.backend_ != nullptr && context.backend_->validation_active();
}

bool try_load_library(const std::string& path, std::string* reason) {
#if ALPHASHAPE_GPU_HAS_VULKAN
    return probe_loader(path, reason);
#else
    (void)path;
    if (reason != nullptr) {
        *reason = "GPU support disabled at build time (ALPHASHAPE_GPU=OFF)";
    }
    return false;
#endif
}

Buffer make_buffer(Context& context, const void* data, std::size_t bytes) {
    return make_buffer(context, data, bytes, MemoryKind::DeviceLocal);
}

Buffer make_buffer(Context& context, const void* data, std::size_t bytes, MemoryKind kind) {
    if (!context.backend_) {
        throw Error("GPU buffer requested from an empty context");
    }
    auto handle = context.backend_->allocate(bytes, kind);
    if (bytes != 0 && data != nullptr) {
        handle->upload(data, bytes, 0);
    }
    return Buffer(std::move(handle));
}

Image2D make_image(Context& context, int width, int height, Format format,
                   const void* pixels) {
    if (!context.backend_) {
        throw Error("GPU image requested from an empty context");
    }
    auto handle = context.backend_->create_image(width, height, format, pixels);
    return Image2D(std::move(handle), width, height, format,
                   static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                       bytes_per_pixel(format));
}

Sampler make_sampler(Context& context, Filter filter, AddressMode address) {
    if (!context.backend_) {
        throw Error("GPU sampler requested from an empty context");
    }
    return Sampler(context.backend_->create_sampler(filter, address), filter, address);
}

bool image_sampling_supported(Context& context, Format format) {
    if (!context.backend_) {
        throw Error("GPU image query on an empty context");
    }
    return context.backend_->image_sampling_supported(format);
}

Ticket submit_async(Context& context, const std::vector<Dispatch>& dispatches,
                    Submission submission) {
    if (!context.backend_) {
        throw Error("GPU work requested from an empty context");
    }
    // An empty dispatch list is a no-op submission: nothing to wait for.
    if (dispatches.empty()) {
        return Ticket();
    }
    if (submission.queue < 0 || submission.queue > 1) {
        throw Error("GPU submission queue must be 0 or 1, got " +
                    std::to_string(submission.queue));
    }
    const std::uint64_t id =
        context.backend_->submit(requests_for(dispatches), submission.queue, false);
    return Ticket(id);
}

std::uint32_t in_flight_count(Context& context) {
    return context.backend_ ? context.backend_->in_flight() : 0;
}

const BufferHandle* BufferAccess::handle(const Buffer& buffer) {
    return buffer.handle_.get();
}

const ImageHandle* ImageAccess::handle(const Image2D& image) {
    return image.handle_.get();
}

const SamplerHandle* SamplerAccess::handle(const Sampler& sampler) {
    return sampler.handle_.get();
}
}  // namespace alphashape::gpu::detail