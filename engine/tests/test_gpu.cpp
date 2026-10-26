#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include "base/platform/process.hpp"
#include "base/utils/log.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/mesh/poisson_gpu.hpp"
#include "common/algorithm/mesh/poisson_internal.hpp"
#include "gpu/gpu.hpp"
#include "model/mesh/point_cloud.hpp"

namespace {

using namespace alphashape::gpu;

// Path to the CLI binary, from tests/CMakeLists.txt. Empty when the CLI is not
// built, which the one test that needs it skips on.
constexpr const char* kCommandLinePath =
#if defined(ALPHASHAPE_CLI_EXE)
    ALPHASHAPE_CLI_EXE;
#else
    "";
#endif

// Every device test reports why it is skipped rather than silently passing.
#define REQUIRE_GPU()                                       \
    do {                                                    \
        if (!available()) {                                 \
            GTEST_SKIP() << unavailable_reason();           \
        }                                                   \
    } while (false)

// Timing/speed-up assertions compare the device against the CPU. On a software
// device (llvmpipe, SwiftShader, ...) the "device" is the CPU, so a speed-up
// target is meaningless: skip those assertions, never the correctness ones.
#define REQUIRE_HARDWARE_GPU(what)                                                   \
    do {                                                                             \
        const std::shared_ptr<Context> _ctx = default_context();                      \
        if (_ctx != nullptr && is_software_device(_ctx->info())) {                    \
            GTEST_SKIP() << what << " needs a hardware GPU; device is "               \
                         << _ctx->info().name << " (software rasteriser)";            \
        }                                                                            \
    } while (false)

std::vector<float> random_floats(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    // Portable draw: std::uniform_real_distribution is implementation-defined
    // across libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    std::vector<float> values(count);
    for (float& value : values) {
        value = static_cast<float>(alphashape::rng::uniform_real(generator, -10.0, 10.0));
    }
    return values;
}

std::uint32_t groups_for(std::size_t count, std::uint32_t local_size) {
    return static_cast<std::uint32_t>((count + local_size - 1) / local_size);
}

/// A byte pattern that keeps every position distinguishable, so a shifted or
/// chunked wrongly transfer cannot compare equal.
std::vector<std::uint8_t> pattern_bytes(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint8_t> bytes(count);
    std::mt19937 generator(seed);
    for (std::uint8_t& byte : bytes) {
        byte = static_cast<std::uint8_t>(generator() & 0xFFu);
    }
    return bytes;
}

}  // namespace

TEST(Gpu, EnumerateUpstreamDevice) {
    REQUIRE_GPU();
    const std::vector<DeviceInfo> devices = enumerate_devices();
    ASSERT_FALSE(devices.empty());
    for (const DeviceInfo& device : devices) {
        std::printf("[gpu] %s vendor=0x%04x device=0x%04x integrated=%d memory=%llu api=%u.%u.%u "
                    "workgroup=%ux%ux%u invocations=%u shared=%u subgroup=%u\n",
                    device.name.c_str(), device.vendor_id, device.device_id,
                    device.integrated ? 1 : 0,
                    static_cast<unsigned long long>(device.memory_bytes),
                    (device.api_version >> 22) & 0x7Fu, (device.api_version >> 12) & 0x3FFu,
                    device.api_version & 0xFFFu, device.max_workgroup_size[0],
                    device.max_workgroup_size[1], device.max_workgroup_size[2],
                    device.max_workgroup_invocations, device.max_shared_memory,
                    device.subgroup_size);
    }
    // The enumeration contract is what this asserts, on any vendor: a named
    // device with a real compute limit and a known type (hardware or software).
    EXPECT_FALSE(devices[0].name.empty());
    EXPECT_GT(devices[0].max_workgroup_invocations, 0u) << devices[0].name;
    EXPECT_GT(devices[0].subgroup_size, 0u) << devices[0].name;
    EXPECT_FALSE(devices[0].name.empty());
    EXPECT_GT(devices[0].memory_bytes, 0u);
    EXPECT_GT(devices[0].max_workgroup_invocations, 0u);
}

TEST(Gpu, DefaultContextReportsDevice) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);
    EXPECT_FALSE(context->info().name.empty());
    EXPECT_TRUE(kernel_names().size() >= 4);
}

TEST(Gpu, BufferRoundTrip) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kBytes = 1024 * 1024;
    std::vector<std::uint8_t> source(kBytes);
    std::mt19937 generator(1234);
    for (std::uint8_t& byte : source) {
        byte = static_cast<std::uint8_t>(generator() & 0xFFu);
    }

    Buffer buffer = make_buffer(*context, kBytes);
    EXPECT_EQ(buffer.size(), kBytes);
    buffer.upload(source.data(), source.size());

    std::vector<std::uint8_t> result(kBytes);
    buffer.download(result.data(), result.size());
    EXPECT_EQ(result, source);

    // Offset writes and reads.
    const std::vector<std::uint8_t> patch{1, 2, 3, 4, 5, 6, 7, 8};
    buffer.upload(patch.data(), patch.size(), 4096);
    std::vector<std::uint8_t> check(patch.size());
    buffer.download(check.data(), check.size(), 4096);
    EXPECT_EQ(check, patch);
    std::vector<std::uint8_t> head(patch.size());
    buffer.download(head.data(), head.size(), 0);
    EXPECT_EQ(head, std::vector<std::uint8_t>(source.begin(), source.begin() + 8));

    EXPECT_THROW(buffer.download(result.data(), kBytes + 16), Error);
    EXPECT_THROW(buffer.upload(source.data(), 8, kBytes - 4), Error);
}

TEST(Gpu, SaxpyMatchesCpu) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kCount = 1000003;
    constexpr float kScale = 2.5f;
    std::vector<float> x = random_floats(kCount, 7);
    std::vector<float> y = random_floats(kCount, 11);

    std::vector<float> expected(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        expected[i] = kScale * x[i] + y[i];
    }

    Buffer gpu_x = Buffer::of(*context, x);
    Buffer gpu_y = Buffer::of(*context, y);
    struct Params {
        float a;
        std::uint32_t n;
    };
    const Params params{kScale, static_cast<std::uint32_t>(kCount)};
    Dispatch dispatch;
    dispatch.kernel = "saxpy";
    dispatch.buffers = {&gpu_x, &gpu_y};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(kCount, 256), 1, 1};
    ASSERT_NO_THROW(run(*context, {dispatch}));

    const std::vector<float> result = gpu_y.read<float>(kCount);
    const GoldReport report = compare(result, expected);
    EXPECT_EQ(report.count, kCount);
    // Relative bound: the GPU may fuse a*x+y into one FMA while the CPU reference
    // (built with -ffp-contract=off) rounds twice, so they can differ by ~1 ulp.
    double worst_rel = 0.0;
    for (std::size_t i = 0; i < kCount; ++i) {
        const double e = static_cast<double>(expected[i]);
        const double rel = std::abs(static_cast<double>(result[i]) - e) / std::max(1.0, std::abs(e));
        worst_rel = std::max(worst_rel, rel);
    }
    EXPECT_LE(worst_rel, 1e-6) << "mean abs " << report.mean_abs << " max abs " << report.max_abs;
}

TEST(Gpu, ReduceSumMatchesCpu) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kCount = 1u << 20;
    std::vector<float> values = random_floats(kCount, 42);
    double expected = 0.0;
    for (float value : values) {
        expected += static_cast<double>(value);
    }

    const std::uint32_t groups = groups_for(kCount, 256);
    Buffer gpu_values = Buffer::of(*context, values);
    Buffer gpu_partials = make_buffer(*context, groups * sizeof(float));
    struct Params {
        std::uint32_t n;
    };
    const Params params{static_cast<std::uint32_t>(kCount)};
    Dispatch dispatch;
    dispatch.kernel = "reduce_sum";
    dispatch.buffers = {&gpu_values, &gpu_partials};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups, 1, 1};
    ASSERT_NO_THROW(run(*context, {dispatch}));

    const std::vector<float> partials = gpu_partials.read<float>(groups);
    double total = 0.0;
    for (float partial : partials) {
        total += static_cast<double>(partial);
    }
    const double relative = std::fabs(total - expected) / std::max(std::fabs(expected), 1.0);
    EXPECT_LE(relative, 1e-3) << "gpu " << total << " cpu " << expected;
}

TEST(Gpu, ImageBox3MatchesCpu) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr int kWidth = 1031;
    constexpr int kHeight = 517;
    const std::size_t count = static_cast<std::size_t>(kWidth) * kHeight;
    std::vector<float> source = random_floats(count, 99);

    std::vector<float> expected(count);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = std::clamp(y + dy, 0, kHeight - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::clamp(x + dx, 0, kWidth - 1);
                    sum += source[static_cast<std::size_t>(yy) * static_cast<std::size_t>(kWidth) +
                                static_cast<std::size_t>(xx)];
                }
            }
            expected[static_cast<std::size_t>(y) * static_cast<std::size_t>(kWidth) +
                     static_cast<std::size_t>(x)] = sum / 9.0f;
        }
    }

    Buffer gpu_source = Buffer::of(*context, source);
    Buffer gpu_result = make_buffer(*context, count * sizeof(float));
    struct Params {
        std::uint32_t w;
        std::uint32_t h;
    };
    const Params params{static_cast<std::uint32_t>(kWidth),
                        static_cast<std::uint32_t>(kHeight)};
    Dispatch dispatch;
    dispatch.kernel = "image_box3";
    dispatch.buffers = {&gpu_source, &gpu_result};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(static_cast<std::size_t>(kWidth), 16),
                       groups_for(static_cast<std::size_t>(kHeight), 16), 1};
    ASSERT_NO_THROW(run(*context, {dispatch}));

    const std::vector<float> result = gpu_result.read<float>(count);
    const GoldReport report = compare(result, expected);
    EXPECT_EQ(report.count, count);
    EXPECT_LE(report.max_abs, 1e-5) << "mean " << report.mean_abs << " max " << report.max_abs;
}

TEST(Gpu, BarriersBetweenDispatches) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kCount = 5000;
    std::vector<float> zeros(kCount, 0.0f);
    Buffer buffer = Buffer::of(*context, zeros);

    struct Params {
        std::uint32_t n;
    };
    const Params params{static_cast<std::uint32_t>(kCount)};
    std::vector<Dispatch> chain;
    chain.reserve(100);
    for (int step = 0; step < 100; ++step) {
        Dispatch dispatch;
        dispatch.kernel = "chain_add";
        dispatch.buffers = {&buffer};
        dispatch.push_constants = push(params);
        dispatch.groups = {groups_for(kCount, 256), 1, 1};
        chain.push_back(std::move(dispatch));
    }
    ASSERT_NO_THROW(run(*context, chain));

    const std::vector<float> result = buffer.read<float>(kCount);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 100.0f);
    }
}

TEST(Gpu, InterfaceMismatchesThrow) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    std::vector<float> values(256, 1.0f);
    Buffer a = Buffer::of(*context, values);
    Buffer b = Buffer::of(*context, values);
    struct Params {
        std::uint32_t n;
    };
    const Params params{256};

    // Too few bindings for saxpy (which needs two).
    Dispatch too_few;
    too_few.kernel = "saxpy";
    too_few.buffers = {&a};
    too_few.push_constants = push(params);
    EXPECT_THROW(run(*context, {too_few}), Error);

    // Too many bindings.
    Dispatch too_many;
    too_many.kernel = "chain_add";
    too_many.buffers = {&a, &b};
    too_many.push_constants = push(params);
    EXPECT_THROW(run(*context, {too_many}), Error);

    // Wrong push-constant size (saxpy wants 8 bytes, not 4).
    Dispatch wrong_push;
    wrong_push.kernel = "saxpy";
    wrong_push.buffers = {&a, &b};
    wrong_push.push_constants = push(params);
    EXPECT_THROW(run(*context, {wrong_push}), Error);

    // Unknown kernel name.
    Dispatch unknown;
    unknown.kernel = "no_such_kernel";
    EXPECT_THROW(run(*context, {unknown}), Error);

    // A zero workgroup count.
    Dispatch zero_groups;
    zero_groups.kernel = "chain_add";
    zero_groups.buffers = {&a};
    zero_groups.push_constants = push(params);
    zero_groups.groups = {0, 1, 1};
    EXPECT_THROW(run(*context, {zero_groups}), Error);

    // The layer must still be usable afterwards.
    Dispatch valid;
    valid.kernel = "chain_add";
    valid.buffers = {&a};
    valid.push_constants = push(params);
    valid.groups = {1, 1, 1};
    EXPECT_NO_THROW(run(*context, {valid}));
    EXPECT_FLOAT_EQ(a.read<float>(1)[0], 2.0f);
}

TEST(Gpu, FallbackWhenDisabled) {
    const std::uint64_t before = failure_count();
    set_enabled(false);
    EXPECT_FALSE(available());
    EXPECT_TRUE(with_fallback<int>("disabled test", [] { return 1; }, [] { return 2; }) == 2);
    EXPECT_EQ(failure_count(), before);
    set_enabled(true);
}

TEST(Gpu, FallbackWhenGpuThrows) {
    if (!available()) {
        // Without a device (GPU=OFF or no driver) the GPU lambda is never
        // called by design, so there is nothing to fall back from.
        GTEST_SKIP() << "no GPU device: " << unavailable_reason();
    }

    // A throwing GPU lambda bumps the counter and answers on the CPU.
    const std::uint64_t baseline = failure_count();
    const int answer = with_fallback<int>(
        "throwing test", []() -> int { throw Error("synthetic GPU failure"); },
        [] { return 7; });
    EXPECT_EQ(answer, 7);
    EXPECT_EQ(failure_count(), baseline + 1);

    // Non-gpu exceptions are covered too.
    const int answer2 = with_fallback<int>(
        "std exception", []() -> int { throw std::runtime_error("boom"); }, [] { return 9; });
    EXPECT_EQ(answer2, 9);
    EXPECT_EQ(failure_count(), baseline + 2);
}

TEST(Gpu, MissingLibraryIsReportedNotFatal) {
    if (!detail::backend_compiled_in()) {
        GTEST_SKIP() << "GPU support disabled at build time (ALPHASHAPE_GPU=OFF)";
    }
    std::string reason;
    // Spelled per platform so the assertion below is about our own message and
    // not about a path separator that happens to be unrepresentable.
#if defined(_WIN32)
    const std::string missing = "Z:\\nonexistent\\alphashape\\nope-vulkan-1.dll";
#elif defined(__APPLE__)
    const std::string missing = "/nonexistent/alphashape/libNopeVulkan.dylib";
#else
    const std::string missing = "/nonexistent/alphashape/libNopeVulkan.so.1";
#endif
    const bool loaded = detail::try_load_library(missing, &reason);
    EXPECT_FALSE(loaded);
    EXPECT_FALSE(reason.empty());
    EXPECT_NE(reason.find("NopeVulkan"), std::string::npos) << reason;
}

// The Windows failure mode this exists for: a build with the GPU layer compiled
// in, started on a machine with no Vulkan runtime, must run on the CPU rather
// than fail to load a DLL. It runs the real CLI in a child process because a
// missing DLL is decided by the loader before main() in the failing case, which
// no in-process assertion can observe.
//
// The child may legitimately find a device (the CI lanes install lavapipe);
// what is asserted is the contract either way: the process starts, prints a
// device line or an explained "unavailable", and never claims the layer was
// left out of the build.
TEST(Gpu, CommandLineReportsGpuStateWithoutAborting) {
    if (!detail::backend_compiled_in()) {
        GTEST_SKIP() << "GPU support disabled at build time (ALPHASHAPE_GPU=OFF)";
    }
    if (std::string(kCommandLinePath).empty()) {
        GTEST_SKIP() << "the CLI was not part of this build (ALPHASHAPE_BUILD_CLI=OFF)";
    }
    // The path is baked at build time; a build tree moved to a different
    // absolute path (a fork's runner, a relocated unpack) is an environment
    // difference, not a product defect, so it skips rather than fails.
    {
        std::error_code error;
        if (!std::filesystem::exists(kCommandLinePath, error)) {
            GTEST_SKIP() << "no alphashape-cli at " << kCommandLinePath;
        }
    }
    std::vector<std::string> argv{kCommandLinePath, "--gpu-info"};
    const alphashape::platform::ProcessResult result =
        alphashape::platform::run_process_capture(argv);
    ASSERT_TRUE(result.started) << result.output;
    // 0 = at least one device listed, 1 = explained as unavailable. Anything
    // else is a crash, an unhandled exception, or a usage error.
    EXPECT_TRUE(result.exit_code == 0 || result.exit_code == 1)
        << "exit " << result.exit_code << ": " << result.output;
    EXPECT_EQ(result.output.find("disabled at build time"), std::string::npos)
        << "the build reports the GPU layer as compiled out: " << result.output;
    if (result.exit_code == 0) {
        EXPECT_NE(result.output.find("gpu device 0:"), std::string::npos) << result.output;
    } else {
        EXPECT_NE(result.output.find("gpu: unavailable"), std::string::npos) << result.output;
        // A bare "LoadLibrary failed" is not enough for a user to act on.
        // "ulkan" matches both the capitalised prose on Windows and the
        // lowercase soname in the POSIX dlopen error.
        EXPECT_NE(result.output.find("ulkan"), std::string::npos) << result.output;
    }
}

TEST(Gpu, ValidationModeReportsNoErrors) {
    REQUIRE_GPU();
    ContextOptions options;
    options.validation = true;
    std::shared_ptr<Context> context;
    try {
        context = create_context(options);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "validation layer unavailable: " << error.what();
    }
    if (!detail::validation_active(*context)) {
        GTEST_SKIP() << "validation layer is not installed";
    }
    const std::uint64_t before = detail::validation_error_count();
    const std::size_t count = 4096;
    std::vector<float> x = random_floats(count, 3);
    std::vector<float> y = random_floats(count, 5);
    Buffer gpu_x = Buffer::of(*context, x);
    Buffer gpu_y = Buffer::of(*context, y);
    struct Params {
        float a;
        std::uint32_t n;
    };
    const Params params{1.5f, static_cast<std::uint32_t>(count)};
    Dispatch dispatch;
    dispatch.kernel = "saxpy";
    dispatch.buffers = {&gpu_x, &gpu_y};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(count, 256), 1, 1};
    ASSERT_NO_THROW(run(*context, {dispatch}));
    EXPECT_EQ(detail::validation_error_count(), before);
}

TEST(Gpu, ConcurrentSaxpy) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr int kThreads = 4;
    constexpr int kIterations = 20;
    constexpr std::size_t kCount = 1 << 14;
    struct Params {
        float a;
        std::uint32_t n;
    };
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int thread = 0; thread < kThreads; ++thread) {
        threads.emplace_back([&, thread] {
            try {
                std::vector<float> x = random_floats(kCount, static_cast<std::uint32_t>(thread));
                std::vector<float> y = random_floats(kCount, 100 + static_cast<std::uint32_t>(thread));
                Buffer gpu_x = Buffer::of(*context, x);
                Buffer gpu_y = Buffer::of(*context, y);
                const Params params{static_cast<float>(thread + 1),
                                    static_cast<std::uint32_t>(kCount)};
                for (int iteration = 0; iteration < kIterations; ++iteration) {
                    Dispatch dispatch;
                    dispatch.kernel = "saxpy";
                    dispatch.buffers = {&gpu_x, &gpu_y};
                    dispatch.push_constants = push(params);
                    dispatch.groups = {groups_for(kCount, 256), 1, 1};
                    run(*context, {dispatch});
                }
                const std::vector<float> result = gpu_y.read<float>(kCount);
                for (std::size_t i = 0; i < kCount; ++i) {
                    const float expected =
                        static_cast<float>(thread + 1) * kIterations * x[i] + y[i];
                    const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected));
                    if (std::fabs(result[i] - expected) > tolerance) {
                        failures.fetch_add(1);
                        break;
                    }
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "thread %d failed: %s\n", thread, error.what());
                failures.fetch_add(1);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(failures.load(), 0);
}

TEST(Gpu, SaxpyTiming) {
    REQUIRE_GPU();
    REQUIRE_HARDWARE_GPU("Gpu.SaxpyTiming");
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kCount = 1u << 24;
    std::vector<float> x = random_floats(kCount, 21);
    std::vector<float> y = random_floats(kCount, 22);
    struct Params {
        float a;
        std::uint32_t n;
    };
    const Params params{1.25f, static_cast<std::uint32_t>(kCount)};

    Buffer gpu_x = Buffer::of(*context, x);
    Buffer gpu_y = Buffer::of(*context, y);
    Dispatch dispatch;
    dispatch.kernel = "saxpy";
    dispatch.buffers = {&gpu_x, &gpu_y};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(kCount, 256), 1, 1};

    // Warm the pipeline so the first-dispatch cost does not dominate.
    run(*context, {dispatch});
    const auto gpu_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < 5; ++iteration) {
        run(*context, {dispatch});
    }
    const auto gpu_end = std::chrono::steady_clock::now();
    const double gpu_seconds =
        std::chrono::duration<double>(gpu_end - gpu_start).count() / 5.0;

    // Volatile sink: without an observable result the optimiser deletes the
    // whole loop and the CPU time prints as 0.
    volatile float cpu_sink = 0.0f;
    const auto cpu_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kCount; ++i) {
        cpu_sink = params.a * x[i] + y[i];
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_seconds = std::chrono::duration<double>(cpu_end - cpu_start).count();
    (void)cpu_sink;

    std::printf("[gpu] saxpy 16M: GPU %.3f ms, CPU %.3f ms (single thread)\n",
                gpu_seconds * 1e3, cpu_seconds * 1e3);
    SUCCEED();
}

TEST(Gpu, ContextOptionRejections) {
    REQUIRE_GPU();
    // Device index past the end is a clean Error, not a crash.
    ContextOptions bad_index;
    bad_index.device_index = 1000;
    EXPECT_THROW(create_context(bad_index), Error);

    // An impossible memory requirement is rejected (the quiet pass rejects
    // every device; the loud pass logs them). 1 PiB exceeds any current GPU.
    ContextOptions impossible;
    impossible.required_memory = std::uint64_t{1} << 50;
    try {
        auto context = create_context(impossible);
        // If a host really has 1 PiB, the requirement is satisfiable: accept it.
        EXPECT_GT(context->info().memory_bytes, 0u);
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("required_memory"), std::string::npos);
    }
}

TEST(Gpu, CompareHelper) {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{1.5f, 2.0f, 2.0f};
    const GoldReport report = compare(a, b);
    EXPECT_EQ(report.count, 3u);
    EXPECT_DOUBLE_EQ(report.max_abs, 1.0);
    EXPECT_NEAR(report.mean_abs, 0.5, 1e-12);

    const GoldReport mismatch = compare(a, std::vector<float>{1.0f});
    EXPECT_EQ(mismatch.count, 1u);
    EXPECT_TRUE(std::isinf(mismatch.max_abs));
}

// ---------------------------------------------------------------------------
// WP-10.3: device-local storage, staging and asynchronous submission
// ---------------------------------------------------------------------------

TEST(Gpu, DeviceLocalRoundTrip) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    // Small buffer, both memory kinds: byte-exact in both directions, plus an
    // offset round trip (the offset path must not be a full-buffer copy).
    constexpr std::size_t kBytes = 1024 * 1024;
    const std::vector<std::uint8_t> source = pattern_bytes(kBytes, 77);
    for (const MemoryKind kind : {MemoryKind::DeviceLocal, MemoryKind::HostVisible}) {
        Buffer buffer = make_buffer(*context, kBytes, kind);
        ASSERT_EQ(buffer.size(), kBytes);
        buffer.upload(source.data(), source.size());

        std::vector<std::uint8_t> result(kBytes);
        buffer.download(result.data(), result.size());
        EXPECT_EQ(result, source) << "kind " << (kind == MemoryKind::DeviceLocal ? "device" : "host");

        const std::vector<std::uint8_t> patch = pattern_bytes(4096, 78);
        buffer.upload(patch.data(), patch.size(), 12345);
        std::vector<std::uint8_t> check(patch.size());
        buffer.download(check.data(), check.size(), 12345);
        EXPECT_EQ(check, patch);
        std::vector<std::uint8_t> head(check.size());
        buffer.download(head.data(), head.size(), 0);
        EXPECT_EQ(head, std::vector<std::uint8_t>(source.begin(), source.begin() + 4096));

        // Misaligned offsets and sizes: VkBufferCopy is 4-byte granular, so the
        // staged path has to preserve the neighbouring bytes. A clash there
        // would corrupt the bytes before or after the patched range.
        const std::vector<std::uint8_t> odd_patch = pattern_bytes(37, 80);
        buffer.upload(odd_patch.data(), odd_patch.size(), 999);
        std::vector<std::uint8_t> odd_check(odd_patch.size());
        buffer.download(odd_check.data(), odd_check.size(), 999);
        EXPECT_EQ(odd_check, odd_patch);
        // A 3-byte patch at an odd offset is the smallest read-modify-write.
        const std::vector<std::uint8_t> tiny{0xA1, 0xB2, 0xC3};
        buffer.upload(tiny.data(), tiny.size(), 2049);
        std::vector<std::uint8_t> tiny_check(tiny.size());
        buffer.download(tiny_check.data(), tiny_check.size(), 2049);
        EXPECT_EQ(tiny_check, tiny);
        // The byte just before the tiny patch must still be the source pattern.
        std::vector<std::uint8_t> before(1);
        buffer.download(before.data(), 1, 2048);
        EXPECT_EQ(before[0], source[2048]);
    }

    // A transfer larger than the 64 MiB staging ring: the ring must chunk, and
    // every chunk must land at the right offset (a wrap that aliased would show
    // up as a block of the wrong pattern).
    constexpr std::size_t kLargeBytes = 70 * 1024 * 1024;
    const std::vector<std::uint8_t> large = pattern_bytes(kLargeBytes, 79);
    Buffer big = make_buffer(*context, kLargeBytes);
    ASSERT_EQ(big.size(), kLargeBytes);
    big.upload(large.data(), large.size());
    std::vector<std::uint8_t> readback(kLargeBytes);
    big.download(readback.data(), readback.size());
    EXPECT_EQ(readback, large);

    // Device-local storage must still be dispatchable (the staging path leaves
    // it a normal storage buffer).
    struct Params {
        std::uint32_t n;
    };
    const std::uint32_t count = 4096;
    const Params params{count};
    std::vector<float> values(count, 0.0f);
    Buffer device = Buffer::of(*context, values, MemoryKind::DeviceLocal);
    Dispatch dispatch;
    dispatch.kernel = "chain_add";
    dispatch.buffers = {&device};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(count, 256), 1, 1};
    ASSERT_NO_THROW(run(*context, {dispatch}));
    const std::vector<float> after = device.read<float>(count);
    for (float value : after) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }
}

TEST(Gpu, AsyncTwoSubmissionsOrdered) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kCount = 1 << 16;
    std::vector<float> x = random_floats(kCount, 31);
    std::vector<float> zeros(kCount, 0.0f);
    Buffer gpu_x = Buffer::of(*context, x);
    Buffer gpu_y = Buffer::of(*context, zeros);

    struct SaxpyParams {
        float a;
        std::uint32_t n;
    };
    struct Params {
        std::uint32_t n;
    };
    const SaxpyParams saxpy{2.0f, static_cast<std::uint32_t>(kCount)};
    const Params chain{static_cast<std::uint32_t>(kCount)};

    // Submission 1 writes y = 2 x. Submission 2 (queued only after the first
    // ticket was waited for) increments y. If the two raced, the increments
    // would be lost and the comparison below would fail.
    Dispatch first;
    first.kernel = "saxpy";
    first.buffers = {&gpu_x, &gpu_y};
    first.push_constants = push(saxpy);
    first.groups = {groups_for(kCount, 256), 1, 1};
    const Ticket ticket = run_async(*context, {first});
    EXPECT_NE(ticket.id(), 0u);
    EXPECT_EQ(detail::in_flight_count(*context), 1u);
    wait(*context, ticket);
    EXPECT_EQ(detail::in_flight_count(*context), 0u);

    Dispatch second;
    second.kernel = "chain_add";
    second.buffers = {&gpu_y};
    second.push_constants = push(chain);
    second.groups = {groups_for(kCount, 256), 1, 1};
    ASSERT_NO_THROW(wait(*context, run_async(*context, {second})));

    const std::vector<float> result = gpu_y.read<float>(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        ASSERT_FLOAT_EQ(result[i], 2.0f * x[i] + 1.0f) << "at " << i;
    }
}

TEST(Gpu, AsyncRunMatchesSync) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    // A multi-kernel chain (saxpy then a run of chain_add steps) driven through
    // both paths. The same kernels in the same order must give bit-identical
    // results: the async path may change when work is submitted, never what it
    // computes.
    constexpr std::size_t kCount = 1 << 20;
    constexpr int kChainSteps = 32;
    std::vector<float> x = random_floats(kCount, 51);
    std::vector<float> y = random_floats(kCount, 52);

    struct SaxpyParams {
        float a;
        std::uint32_t n;
    };
    struct Params {
        std::uint32_t n;
    };
    const SaxpyParams saxpy{1.25f, static_cast<std::uint32_t>(kCount)};
    const Params chain{static_cast<std::uint32_t>(kCount)};

    const auto build = [&](const Buffer& gpu_x, const Buffer& gpu_y) {
        std::vector<Dispatch> dispatches;
        dispatches.reserve(1 + kChainSteps);
        Dispatch first;
        first.kernel = "saxpy";
        first.buffers = {&gpu_x, &gpu_y};
        first.push_constants = push(saxpy);
        first.groups = {groups_for(kCount, 256), 1, 1};
        dispatches.push_back(std::move(first));
        for (int step = 0; step < kChainSteps; ++step) {
            Dispatch add;
            add.kernel = "chain_add";
            add.buffers = {&gpu_y};
            add.push_constants = push(chain);
            add.groups = {groups_for(kCount, 256), 1, 1};
            dispatches.push_back(std::move(add));
        }
        return dispatches;
    };

    Buffer sync_x = Buffer::of(*context, x);
    Buffer sync_y = Buffer::of(*context, y);
    ASSERT_NO_THROW(run(*context, build(sync_x, sync_y)));
    const std::vector<float> sync_result = sync_y.read<float>(kCount);

    Buffer async_x = Buffer::of(*context, x);
    Buffer async_y = Buffer::of(*context, y);
    const Ticket ticket = run_async(*context, build(async_x, async_y));
    wait(*context, ticket);
    const std::vector<float> async_result = async_y.read<float>(kCount);

    ASSERT_EQ(async_result.size(), sync_result.size());
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < sync_result.size(); ++i) {
        if (std::memcmp(&async_result[i], &sync_result[i], sizeof(float)) != 0) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "async and sync results differ in " << mismatches << " values";
}

TEST(Gpu, InFlightLimitRespected) {
    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    constexpr std::size_t kCount = 1 << 14;
    const std::vector<float> zeros(kCount, 0.0f);
    struct Params {
        std::uint32_t n;
    };
    const Params params{static_cast<std::uint32_t>(kCount)};
    const auto make_dispatch = [&](Buffer& buffer) {
        Dispatch dispatch;
        dispatch.kernel = "chain_add";
        dispatch.buffers = {&buffer};
        dispatch.push_constants = push(params);
        dispatch.groups = {groups_for(kCount, 256), 1, 1};
        return dispatch;
    };

    // Four independent buffers so the queued work does not serialise on a
    // common target (it would still be correct, but the point is the ring).
    std::vector<Buffer> buffers;
    buffers.reserve(detail::kMaxInFlight);
    for (std::uint32_t i = 0; i < detail::kMaxInFlight; ++i) {
        buffers.push_back(Buffer::of(*context, zeros));
    }

    std::vector<Ticket> tickets;
    tickets.reserve(detail::kMaxInFlight);
    for (std::uint32_t i = 0; i < detail::kMaxInFlight; ++i) {
        tickets.push_back(run_async(*context, {make_dispatch(buffers[i])}));
        EXPECT_NE(tickets.back().id(), 0u);
        EXPECT_EQ(detail::in_flight_count(*context), i + 1);
    }
    EXPECT_EQ(detail::in_flight_count(*context), detail::kMaxInFlight);

    // The fifth submission must fail cleanly rather than grow the ring.
    EXPECT_THROW(run_async(*context, {make_dispatch(buffers[0])}), Error);
    EXPECT_EQ(detail::in_flight_count(*context), detail::kMaxInFlight);

    for (const Ticket& ticket : tickets) {
        wait(*context, ticket);
    }
    EXPECT_EQ(detail::in_flight_count(*context), 0u);

    // The layer is still usable, and the synchronous path is unaffected by a
    // ring that has been filled and drained.
    ASSERT_NO_THROW(run(*context, {make_dispatch(buffers[0])}));
    EXPECT_FLOAT_EQ(buffers[0].read<float>(1)[0], 2.0f);
    EXPECT_EQ(detail::in_flight_count(*context), 0u);
}

TEST(Gpu, SecondQueueFallsBackOrRuns) {    REQUIRE_GPU();
    auto context = default_context();
    ASSERT_NE(context, nullptr);

    // Queue 1 must either run on the second queue or fall back to queue 0, and
    // either way produce the right answer with no gpu::Error.
    constexpr std::size_t kCount = 4096;
    std::vector<float> zeros(kCount, 0.0f);
    Buffer buffer = Buffer::of(*context, zeros);
    struct Params {
        std::uint32_t n;
    };
    const Params params{static_cast<std::uint32_t>(kCount)};
    Dispatch dispatch;
    dispatch.kernel = "chain_add";
    dispatch.buffers = {&buffer};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(kCount, 256), 1, 1};

    Submission second;
    second.queue = 1;
    const Ticket ticket = run_async(*context, {dispatch}, second);
    wait(*context, ticket);
    for (float value : buffer.read<float>(kCount)) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }

    // An out-of-range queue index is a clean Error before any submission.
    Submission invalid;
    invalid.queue = 2;
    EXPECT_THROW(run_async(*context, {dispatch}, invalid), Error);
    EXPECT_EQ(detail::in_flight_count(*context), 0u);
}

TEST(Gpu, PoissonDeterministicWithAsync) {
    REQUIRE_GPU();
    // The Poisson solve now reaches the device through run() == run_async() +
    // wait(). Two solves of the same assembled system must be bit-identical:
    // the submission path changed, the arithmetic did not.
    alphashape::PointCloud cloud;
    constexpr std::size_t kPoints = 3000;
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    cloud.points.reserve(kPoints);
    cloud.normals.reserve(kPoints);
    for (std::size_t i = 0; i < kPoints; ++i) {
        const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(kPoints);
        const double z = 1.0 - 2.0 * u;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double theta = 2.0 * M_PI * static_cast<double>(i) / golden;
        const Eigen::Vector3d point(r * std::cos(theta), r * std::sin(theta), z);
        cloud.points.push_back(point);
        cloud.normals.push_back(point.normalized().cast<float>());
    }

    alphashape::meshing::PoissonOptions options;
    options.max_resolution = 32;
    const Eigen::Vector3d box = Eigen::Vector3d::Constant(1.5);
    const alphashape::meshing::detail::PoissonSystem system =
        alphashape::meshing::detail::assemble_poisson_system(cloud, -box, box, options);

    alphashape::meshing::detail::PoissonCgStats first_stats;
    alphashape::meshing::detail::PoissonCgStats second_stats;
    const std::vector<double> first =
        alphashape::meshing::poisson_solve_device(system, options, first_stats);
    const std::vector<double> second =
        alphashape::meshing::poisson_solve_device(system, options, second_stats);
    ASSERT_EQ(first.size(), second.size());
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (std::memcmp(&first[i], &second[i], sizeof(double)) != 0) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "Poisson results differ in " << mismatches
                              << " voxels between two async-driven solves";
    EXPECT_EQ(first_stats.iterations, second_stats.iterations);
    if (const std::shared_ptr<Context> context = default_context()) {
        EXPECT_EQ(detail::in_flight_count(*context), 0u);
    }
}

// -- images and samplers (WP-10.2) ------------------------------------------

namespace {

/// An 8x8 R8 gradient: value(x, y) = (x * 8 + y) * 2, so every texel differs
/// from its neighbours by at least 2/255 and a wrong texel is visible.
std::vector<std::uint8_t> gradient_r8(int width, int height) {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) *
                                     static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)] =
                static_cast<std::uint8_t>((x + y * width) * 2);
        }
    }
    return pixels;
}

/// CPU reference for nearest/linear sampling of an R8 image with ClampToEdge.
double sample_r8_clamped(const std::vector<std::uint8_t>& pixels, int width, int height,
                         double u, double v, Filter filter) {
    // Texel centres sit at (i + 0.5) / width; clamp the address mode first.
    double x = std::clamp(u, 0.0, 1.0) * static_cast<double>(width) - 0.5;
    double y = std::clamp(v, 0.0, 1.0) * static_cast<double>(height) - 0.5;
    auto texel = [&](int ix, int iy) {
        ix = std::clamp(ix, 0, width - 1);
        iy = std::clamp(iy, 0, height - 1);
        return static_cast<double>(
            pixels[static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(ix)]) /
            255.0;
    };
    if (filter == Filter::Nearest) {
        return texel(static_cast<int>(std::floor(x + 0.5)), static_cast<int>(std::floor(y + 0.5)));
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double value = (1.0 - fx) * (1.0 - fy) * texel(x0, y0) + fx * (1.0 - fy) * texel(x0 + 1, y0) +
                         (1.0 - fx) * fy * texel(x0, y0 + 1) + fx * fy * texel(x0 + 1, y0 + 1);
    return value;
}

/// Runs the image_sample_test kernel over normalised coordinates and returns
/// one float per coordinate pair.
std::vector<float> sample_image(Context& context, const Image2D& image, const Sampler& sampler,
                               const std::vector<float>& uv) {
    const std::size_t count = uv.size() / 2;
    Buffer out = make_buffer(context, count * sizeof(float));
    Buffer coords = Buffer::of(context, uv);
    struct Params {
        std::uint32_t count;
        std::uint32_t stride;
        float scale;
    };
    const Params params{static_cast<std::uint32_t>(count), 2, 1.0f};
    Dispatch dispatch;
    dispatch.kernel = "image_sample_test";
    dispatch.buffers = {&out, &coords};
    dispatch.images = {{1u, &image}};
    dispatch.samplers = {{1u, &sampler}};
    dispatch.push_constants = push(params);
    dispatch.groups = {groups_for(count, 64), 1, 1};
    run(context, {dispatch});
    return out.read<float>(count);
}

}  // namespace

TEST(Gpu, ImageRoundTripLinearSampler) {
    REQUIRE_GPU();
    std::shared_ptr<Context> context = default_context();
    ASSERT_NE(context, nullptr);
    if (!image_sampling_supported(*context, Format::R8Unorm)) {
        GTEST_SKIP() << context->info().name << " cannot sample R8Unorm";
    }

    constexpr int kWidth = 8;
    constexpr int kHeight = 8;
    const std::vector<std::uint8_t> pixels = gradient_r8(kWidth, kHeight);
    Image2D image = Image2D::create(*context, kWidth, kHeight, Format::R8Unorm, pixels.data());
    Sampler sampler = Sampler::create(*context, Filter::Linear, AddressMode::ClampToEdge);

    // Half-pixel coordinates land exactly between texel centres, which is
    // where bilinear weights are 0.5/0.5 and an off-by-half-texel convention
    // shows up as a full texel of error.
    std::vector<float> uv;
    for (int y = 0; y < kHeight - 1; ++y) {
        for (int x = 0; x < kWidth - 1; ++x) {
            uv.push_back((static_cast<float>(x) + 1.0f) / static_cast<float>(kWidth));
            uv.push_back((static_cast<float>(y) + 1.0f) / static_cast<float>(kHeight));
        }
    }
    const std::vector<float> values = sample_image(*context, image, sampler, uv);
    ASSERT_EQ(values.size(), uv.size() / 2);
    // 1/255 for the 8-bit source quantum and 2 ulp for the float path.
    const double tolerance = 1.0 / 255.0 + 2.0 * std::pow(2.0, -24.0);
    double worst = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double expected =
            sample_r8_clamped(pixels, kWidth, kHeight, static_cast<double>(uv[2 * i]),
                              static_cast<double>(uv[2 * i + 1]), Filter::Linear);
        worst = std::max(worst, std::fabs(static_cast<double>(values[i]) - expected));
    }
    EXPECT_LE(worst, tolerance) << "max abs error " << worst << " over " << values.size()
                                << " bilinear samples";

    // The whole image survives the upload/download path byte for byte.
    std::vector<std::uint8_t> round_trip(pixels.size());
    image.download(round_trip.data(), round_trip.size());
    EXPECT_EQ(round_trip, pixels);
}

TEST(Gpu, ImageClampToEdge) {
    REQUIRE_GPU();
    std::shared_ptr<Context> context = default_context();
    ASSERT_NE(context, nullptr);
    if (!image_sampling_supported(*context, Format::R8Unorm)) {
        GTEST_SKIP() << context->info().name << " cannot sample R8Unorm";
    }

    constexpr int kWidth = 8;
    constexpr int kHeight = 8;
    const std::vector<std::uint8_t> pixels = gradient_r8(kWidth, kHeight);
    Image2D image = Image2D::create(*context, kWidth, kHeight, Format::R8Unorm, pixels.data());
    Sampler nearest = Sampler::create(*context, Filter::Nearest, AddressMode::ClampToEdge);

    // Sample far outside the image in every direction; ClampToEdge must return
    // the corner texel, not a wrap or a zero.
    const std::vector<float> uv{-1.0f, -1.0f, 2.0f, 2.0f, -0.5f, 0.5f, 1.5f, 0.5f, -0.001f, -0.001f};
    const std::vector<float> values = sample_image(*context, image, nearest, uv);
    ASSERT_EQ(values.size(), 5u);
    const auto texel = [&](int x, int y) {
        return static_cast<float>(pixels[static_cast<std::size_t>(y) *
                                          static_cast<std::size_t>(kWidth) +
                                          static_cast<std::size_t>(x)]) /
               255.0f;
    };
    EXPECT_NEAR(values[0], texel(0, 0), 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(values[1], texel(kWidth - 1, kHeight - 1), 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(values[2], texel(0, 4), 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(values[3], texel(kWidth - 1, 4), 1.0 / 255.0 + 1e-6);
    EXPECT_NEAR(values[4], texel(0, 0), 1.0 / 255.0 + 1e-6);
}

TEST(Gpu, ImageBindingMismatchThrows) {
    REQUIRE_GPU();
    std::shared_ptr<Context> context = default_context();
    ASSERT_NE(context, nullptr);
    if (!image_sampling_supported(*context, Format::R8Unorm)) {
        GTEST_SKIP() << context->info().name << " cannot sample R8Unorm";
    }

    const std::vector<std::uint8_t> pixels(8 * 8, 7);
    Image2D image = Image2D::create(*context, 8, 8, Format::R8Unorm, pixels.data());
    Sampler sampler = Sampler::create(*context, Filter::Nearest, AddressMode::ClampToEdge);
    Buffer out = make_buffer(*context, 4 * sizeof(float));
    Buffer coords = Buffer::of(*context, std::vector<float>{0.5f, 0.5f});
    struct Params {
        std::uint32_t count;
        std::uint32_t stride;
        float scale;
    };
    const Params params{1, 2, 1.0f};

    // The kernel has a sampler at binding 1 and buffers at 0 and 2.
    auto base_dispatch = [&] {
        Dispatch dispatch;
        dispatch.kernel = "image_sample_test";
        dispatch.buffers = {&out, &coords};
        dispatch.push_constants = push(params);
        dispatch.groups = {1, 1, 1};
        return dispatch;
    };

    // Image at a storage-buffer binding: unknown to the reflection as an image.
    {
        Dispatch dispatch = base_dispatch();
        dispatch.images = {{9u, &image}};
        dispatch.samplers = {{9u, &sampler}};
        EXPECT_THROW(run(*context, {dispatch}), Error);
    }
    // Image with no sampler.
    {
        Dispatch dispatch = base_dispatch();
        dispatch.images = {{1u, &image}};
        EXPECT_THROW(run(*context, {dispatch}), Error);
    }
    // Sampler with no image.
    {
        Dispatch dispatch = base_dispatch();
        dispatch.samplers = {{1u, &sampler}};
        EXPECT_THROW(run(*context, {dispatch}), Error);
    }
    // Buffer bound where the shader wants the image (the declared image
    // binding is missing, so the count check fires).
    {
        Dispatch dispatch = base_dispatch();
        dispatch.images.clear();
        dispatch.samplers.clear();
        EXPECT_THROW(run(*context, {dispatch}), Error);
    }
    // Duplicate binding index.
    {
        Dispatch dispatch = base_dispatch();
        dispatch.images = {{1u, &image}, {1u, &image}};
        dispatch.samplers = {{1u, &sampler}, {1u, &sampler}};
        EXPECT_THROW(run(*context, {dispatch}), Error);
    }
    // Wrong image upload size.
    EXPECT_THROW(image.upload(pixels.data(), pixels.size() + 1), Error);
    EXPECT_THROW(image.download(const_cast<std::uint8_t*>(pixels.data()), 0), Error);
}

TEST(Gpu, MixedBufferAndImageDispatch) {
    REQUIRE_GPU();
    std::shared_ptr<Context> context = default_context();
    ASSERT_NE(context, nullptr);
    if (!image_sampling_supported(*context, Format::R32Float)) {
        GTEST_SKIP() << context->info().name << " cannot sample R32Float";
    }

    // A 4x4 R32Float image plus a storage buffer in the same dispatch: the
    // buffer is the output and the UV list, the image the sampled source.
    constexpr int kWidth = 4;
    constexpr int kHeight = 4;
    std::vector<float> pixels(static_cast<std::size_t>(kWidth) * kHeight);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<float>(i) * 0.25f;
    }
    Image2D image = Image2D::create(*context, kWidth, kHeight, Format::R32Float, pixels.data());
    Sampler sampler = Sampler::create(*context, Filter::Nearest, AddressMode::ClampToEdge);

    // Sample each texel centre exactly.
    std::vector<float> uv;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            uv.push_back((static_cast<float>(x) + 0.5f) / static_cast<float>(kWidth));
            uv.push_back((static_cast<float>(y) + 0.5f) / static_cast<float>(kHeight));
        }
    }
    const std::vector<float> values = sample_image(*context, image, sampler, uv);
    ASSERT_EQ(values.size(), pixels.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(values[i], pixels[i], 1e-6) << "texel " << i;
    }

    // And the image's own download round-trips the float data exactly.
    std::vector<float> round_trip(pixels.size());
    image.download(round_trip.data(), round_trip.size() * sizeof(float));
    EXPECT_EQ(round_trip, pixels);
}

TEST(Gpu, ImageWithoutUploadThrows) {
    REQUIRE_GPU();
    std::shared_ptr<Context> context = default_context();
    ASSERT_NE(context, nullptr);
    if (!image_sampling_supported(*context, Format::R8Unorm)) {
        GTEST_SKIP() << context->info().name << " cannot sample R8Unorm";
    }

    Image2D image = Image2D::create(*context, 8, 8, Format::R8Unorm);
    Sampler sampler = Sampler::create(*context, Filter::Nearest, AddressMode::ClampToEdge);
    Buffer out = make_buffer(*context, sizeof(float));
    Buffer coords = Buffer::of(*context, std::vector<float>{0.5f, 0.5f});
    struct Params {
        std::uint32_t count;
        std::uint32_t stride;
        float scale;
    };
    const Params params{1, 2, 1.0f};
    Dispatch dispatch;
    dispatch.kernel = "image_sample_test";
    dispatch.buffers = {&out, &coords};
    dispatch.images = {{1u, &image}};
    dispatch.samplers = {{1u, &sampler}};
    dispatch.push_constants = push(params);
    dispatch.groups = {1, 1, 1};
    EXPECT_THROW(run(*context, {dispatch}), Error);
    // Download from an un-uploaded image is rejected too.
    std::vector<std::uint8_t> pixels(64);
    EXPECT_THROW(image.download(pixels.data(), pixels.size()), Error);
}

TEST(Gpu, ImageDispatchValidationClean) {
    REQUIRE_GPU();
    ContextOptions options;
    options.validation = true;
    std::shared_ptr<Context> context;
    try {
        context = create_context(options);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "validation layer unavailable: " << error.what();
    }
    if (!detail::validation_active(*context)) {
        GTEST_SKIP() << "validation layer is not installed";
    }
    if (!image_sampling_supported(*context, Format::R8G8B8A8Unorm)) {
        GTEST_SKIP() << context->info().name << " cannot sample R8G8B8A8Unorm";
    }
    const std::uint64_t before = detail::validation_error_count();

    constexpr int kWidth = 16;
    constexpr int kHeight = 16;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth) * kHeight * 4);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<std::uint8_t>(i * 7u);
    }
    Image2D image =
        Image2D::create(*context, kWidth, kHeight, Format::R8G8B8A8Unorm, pixels.data());
    Sampler sampler = Sampler::create(*context, Filter::Linear, AddressMode::Repeat);
    std::vector<float> uv;
    for (int i = 0; i < 64; ++i) {
        uv.push_back(static_cast<float>(i % 7) / 6.0f);
        uv.push_back(static_cast<float>(i % 5) / 4.0f);
    }
    const std::vector<float> values = sample_image(*context, image, sampler, uv);
    EXPECT_EQ(values.size(), 64u);
    EXPECT_EQ(detail::validation_error_count(), before);
}
