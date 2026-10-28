// GPU PatchMatch tests (M10). Every device test skips with a reason when the
// layer is unavailable, so the suite runs unchanged on a machine without a
// Vulkan device; ALPHASHAPE_GPU=OFF skips the whole file.
//
// The agreement and accuracy tests run the CPU implementation on the same
// inputs as the reference, which is what makes the bounds meaningful: the GPU
// result must land in the same acceptance region the CPU dense tests assert,
// not merely reproduce a particular run.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial_fixture.hpp"
#include "dense_fixture.hpp"

#include "base/platform/env.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/stereo/patch_match.hpp"
#include "common/algorithm/stereo/patch_match_gpu.hpp"
#include "common/algorithm/stereo/view_selection.hpp"
#include "common/core/document.hpp"
#include "gpu/gpu.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "image/operators/downsample.hpp"
#include "model/camera/calibration.hpp"
#include "perf_gate.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace {

using stereo::PMView;
using stereo::PatchMatchOptions;
using stereo::PatchMatchResult;

#define REQUIRE_GPU_PM()                     \
    do {                                     \
        if (!gpu::available()) {             \
            GTEST_SKIP() << gpu::unavailable_reason(); \
        }                                    \
    } while (false)

// Speed-up versus the CPU is only meaningful on real hardware: on a software
// device (llvmpipe/lavapipe) the "device" is the CPU itself. Correctness tests
// run everywhere; only the timing ones are skipped here.
#define REQUIRE_HARDWARE_GPU_PM(what)                                              \
    do {                                                                           \
        const std::shared_ptr<gpu::Context> _ctx = gpu::default_context();          \
        if (_ctx != nullptr && gpu::is_software_device(_ctx->info())) {              \
            GTEST_SKIP() << what << " needs a hardware GPU; device is "             \
                         << _ctx->info().name << " (software rasteriser)";          \
        }                                                                           \
    } while (false)

double time_seconds(const std::function<PatchMatchResult()>& run) {
    const auto start = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::size_t valid_pixels(const PatchMatchResult& result) {
    std::size_t n = 0;
    for (const float d : result.depth) {
        if (d > 0.0F) {
            ++n;
        }
    }
    return n;
}

/// Fraction of pixels valid in both whose depth differs by less than `tolerance`
/// (relative), and how many such pixels there were.
struct Agreement {
    double fraction = 0.0;
    std::size_t both = 0;
};

Agreement relative_agreement(const PatchMatchResult& a, const PatchMatchResult& b, double tolerance) {
    Agreement agreement;
    std::size_t within = 0;
    for (std::size_t p = 0; p < a.depth.size() && p < b.depth.size(); ++p) {
        if (!(a.depth[p] > 0.0F) || !(b.depth[p] > 0.0F)) {
            continue;
        }
        ++agreement.both;
        const double relative = std::abs(static_cast<double>(a.depth[p]) -
                                         static_cast<double>(b.depth[p])) /
                                static_cast<double>(b.depth[p]);
        if (relative < tolerance) {
            ++within;
        }
    }
    if (agreement.both > 0) {
        agreement.fraction = static_cast<double>(within) / static_cast<double>(agreement.both);
    }
    return agreement;
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// The dense scene's depth-map inputs at a chosen downscale, built exactly as
/// BuildDepthMaps builds them (same loading, downsampling and scaled calibration).
struct DenseDepthInputs {
    std::vector<FloatImage> images;
    std::vector<Calibration> calibrations;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> translations;
    std::vector<std::int32_t> camera_ids;
    std::vector<stereo::ViewNeighbourhood> neighbourhoods;

    int index_of(std::int32_t camera_id) const {
        for (std::size_t i = 0; i < camera_ids.size(); ++i) {
            if (camera_ids[i] == camera_id) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    PMView view(std::int32_t camera_id) const {
        const int index = index_of(camera_id);
        PMView view;
        view.image = &images[static_cast<std::size_t>(index)];
        view.calibration = calibrations[static_cast<std::size_t>(index)];
        view.R = rotations[static_cast<std::size_t>(index)];
        view.t = translations[static_cast<std::size_t>(index)];
        return view;
    }
};

std::shared_ptr<DenseDepthInputs> dense_inputs(int downscale) {
    auto inputs = std::make_shared<DenseDepthInputs>();
    Document document = dense_fixture::aligned_document(dense_fixture::seq_files());
    Chunk& chunk = *document.active_chunk();
    const std::vector<dense_fixture::TruthImage> truth =
        dense_fixture::parse_truth_images("seq_truth.json");
    dense_fixture::set_truth_poses(document, truth);
    dense_fixture::set_truth_calibration(document, 480, 360, 500.0);
    dense_fixture::retriangulate_tracks(chunk);

    inputs->neighbourhoods = stereo::select_views(chunk, stereo::ViewSelectionOptions{});
    for (const Camera& camera : chunk.cameras()) {
        const int n = std::atoi(camera.label.c_str() + 4);
        const Image image = read_image(dense_fixture::fixture("seq_" + std::to_string(n) + ".jpg"));
        inputs->images.push_back(downsample_box(image.to_luminance(), downscale));
        inputs->calibrations.push_back(scaled_calibration(
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration, downscale));
        inputs->rotations.push_back(camera.pose.block<3, 3>(0, 0));
        inputs->translations.push_back(camera.pose.block<3, 1>(0, 3));
        inputs->camera_ids.push_back(camera.id);
    }
    return inputs;
}

// ---------------------------------------------------------------------------
// 1. Feature/device reporting
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, DeviceReported) {
    REQUIRE_GPU_PM();
    const auto context = gpu::default_context();
    ASSERT_NE(context, nullptr);
    std::printf("[gpu] device: %s, max workgroup %u\n", context->info().name.c_str(),
                context->info().max_workgroup_invocations);
    EXPECT_FALSE(context->info().name.empty());
}

// The M10 WP-10.3 pre-decided fallback: PatchMatch's working set stays on
// host-visible memory, because the whole grid is seeded from the host and read
// back on every call, so DeviceLocal staging would add two grid-sized round
// trips per call. Poisson's large vectors do use DeviceLocal. Asserted so a
// future change to that decision is a visible test change, not a silent one.
TEST(PatchMatchGpu, WorkingSetPrefersHostMemory) {
    EXPECT_TRUE(stereo::patch_match_gpu_prefers_host_memory());
}

// ---------------------------------------------------------------------------
// 2. Synthetic plane: GPU meets the CPU's quality contract
// ---------------------------------------------------------------------------

/// A fronto-parallel textured plane seen by a reference and four neighbours,
/// plus its analytic expected depth. Modelled on test_patch_match.cpp.
struct PlaneScene {
    FloatImage reference;
    std::vector<FloatImage> neighbours;
    Calibration calibration;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> translations;
    double plane_depth = 10.0;
};

float plane_texture(double a, double b) {
    return static_cast<float>(0.5 + 0.22 * std::sin(1.7 * a + 0.3) +
                              0.18 * std::sin(2.3 * b + 1.1) +
                              0.12 * std::sin(1.1 * a + 1.9 * b + 0.7));
}

PlaneScene make_plane_scene(int width, int height, double focal, double k1, double k2) {
    PlaneScene scene;
    scene.calibration = Calibration::initial(CameraType::Frame, width, height, focal);
    scene.calibration.k1 = k1;
    scene.calibration.k2 = k2;
    const Eigen::Vector3d centres[5] = {
        Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.8, 0.0, 0.0),
        Eigen::Vector3d(-0.6, 0.35, 0.0), Eigen::Vector3d(0.4, -0.9, 0.25),
        Eigen::Vector3d(-0.5, -0.45, -0.3)};
    const Eigen::Vector3d target(0.0, 0.0, 10.0);
    const Eigen::Vector3d normal(0.0, 0.0, -1.0);
    for (int v = 0; v < 5; ++v) {
        const Eigen::Vector3d c = centres[v];
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        if (c.norm() > 1e-12) {
            const Eigen::Vector3d forward = (target - c).normalized();
            Eigen::Vector3d x = Eigen::Vector3d(0.0, 1.0, 0.0).cross(forward);
            if (x.norm() < 1e-6) {
                x = Eigen::Vector3d(1.0, 0.0, 0.0);
            }
            x.normalize();
            const Eigen::Vector3d y = forward.cross(x);
            r.row(0) = x;
            r.row(1) = y;
            r.row(2) = forward;
        }
        scene.rotations.push_back(r);
        scene.translations.push_back(-r * c);
        FloatImage image(width, height, 0.0F);
        const Eigen::Matrix3d rt = r.transpose();
        const Eigen::Vector3d centre = -rt * scene.translations.back();
        const double d_plane = normal.dot(Eigen::Vector3d(0.0, 0.0, 10.0));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Eigen::Vector3d b = scene.calibration.unproject(
                    Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
                const Eigen::Vector3d dir = rt * b;
                const double denom = normal.dot(dir);
                if (std::abs(denom) < 1e-9) {
                    continue;
                }
                const double depth = (d_plane - normal.dot(centre)) / denom;
                if (depth > 0.0) {
                    const Eigen::Vector3d p = centre + depth * dir;
                    image.at(x, y) = plane_texture(p.x(), p.y());
                }
            }
        }
        if (v == 0) {
            scene.reference = image;
        } else {
            scene.neighbours.push_back(image);
        }
    }
    return scene;
}

std::vector<PMView> plane_views(PlaneScene& scene) {
    std::vector<PMView> views;
    for (std::size_t i = 0; i < scene.neighbours.size(); ++i) {
        PMView view;
        view.image = &scene.neighbours[i];
        view.calibration = scene.calibration;
        view.R = scene.rotations[i + 1];
        view.t = scene.translations[i + 1];
        views.push_back(view);
    }
    return views;
}

/// Relative depth error over the interior against the analytic plane.
struct PlaneAccuracy {
    double coverage = 0.0;
    double median = 0.0;
    double p90 = 0.0;
};

PlaneAccuracy measure_plane(const PlaneScene& scene, const PatchMatchResult& result, int radius) {
    std::vector<double> errors;
    const int width = scene.reference.width();
    const int height = scene.reference.height();
    const Eigen::Matrix3d rt = scene.rotations[0].transpose();
    const Eigen::Vector3d centre = -rt * scene.translations[0];
    const Eigen::Vector3d normal(0.0, 0.0, -1.0);
    const double d_plane = normal.dot(Eigen::Vector3d(0.0, 0.0, 10.0));
    int interior = 0;
    int covered = 0;
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            const Eigen::Vector3d b = scene.calibration.unproject(
                Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
            const double denom = normal.dot(rt * b);
            if (std::abs(denom) < 1e-9) {
                continue;
            }
            const double expected = (d_plane - normal.dot(centre)) / denom;
            if (!(expected > 0.0)) {
                continue;
            }
            ++interior;
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x);
            if (!(result.depth[index] > 0.0F)) {
                continue;
            }
            ++covered;
            errors.push_back(std::abs(static_cast<double>(result.depth[index]) - expected) /
                             expected);
        }
    }
    std::sort(errors.begin(), errors.end());
    PlaneAccuracy accuracy;
    accuracy.coverage =
        interior > 0 ? static_cast<double>(covered) / static_cast<double>(interior) : 0.0;
    accuracy.median = dense_fixture::percentile(errors, 0.50);
    accuracy.p90 = dense_fixture::percentile(errors, 0.90);
    return accuracy;
}

PatchMatchOptions plane_options() {
    PatchMatchOptions options;
    options.depth_min = 5.0;
    options.depth_max = 20.0;
    options.window_radius = 5;
    options.window_step = 2;
    options.iterations = 6;
    options.max_cost = 0.5;
    options.seed = 12345;
    // One level: these tests calibrate against the pre-pyramid search. The
    // pyramid has its own agreement test (PyramidAgreesWithCpu).
    options.pyramid_levels = 1;
    return options;
}

// ---------------------------------------------------------------------------
// 3. GPU vs CPU on a synthetic plane, clean and distorted
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, PlanarSceneMatchesCpu) {
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(160, 120, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];

    const PatchMatchOptions options = plane_options();
    ASSERT_TRUE(stereo::patch_match_gpu_would_run(reference, neighbours, options));

    const PatchMatchResult gpu_result =
        stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult cpu_result =
        stereo::run_patch_match(reference, neighbours, options);

    const PlaneAccuracy gpu_accuracy = measure_plane(scene, gpu_result, 5);
    const PlaneAccuracy cpu_accuracy = measure_plane(scene, cpu_result, 5);
    const Agreement agreement = relative_agreement(gpu_result, cpu_result, 0.01);
    const std::size_t gpu_valid = valid_pixels(gpu_result);
    const std::size_t cpu_valid = valid_pixels(cpu_result);
    std::printf("[gpu plane] gpu: coverage %.3f median %.5f p90 %.5f | cpu: coverage %.3f "
                "median %.5f p90 %.5f | agree<1%%: %.1f%% of %zu | valid gpu %zu cpu %zu\n",
                gpu_accuracy.coverage, gpu_accuracy.median, gpu_accuracy.p90,
                cpu_accuracy.coverage, cpu_accuracy.median, cpu_accuracy.p90,
                100.0 * agreement.fraction, agreement.both, gpu_valid, cpu_valid);

    // Same contract the CPU plane tests assert.
    EXPECT_GT(gpu_accuracy.coverage, 0.85);
    EXPECT_LT(gpu_accuracy.median, 0.005);
    EXPECT_LT(gpu_accuracy.p90, 0.02);
    // Agreement and validity are within the M10 acceptance window.
    EXPECT_GT(agreement.fraction, 0.90);
    EXPECT_GT(gpu_valid, 0);
    EXPECT_NEAR(static_cast<double>(gpu_valid), static_cast<double>(cpu_valid),
                0.05 * static_cast<double>(cpu_valid));
}

TEST(PatchMatchGpu, DistortedCameraMatchesCpu) {
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(160, 120, 150.0, -0.08, 0.02);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];

    const PatchMatchOptions options = plane_options();
    const PatchMatchResult gpu_result =
        stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult cpu_result =
        stereo::run_patch_match(reference, neighbours, options);

    const PlaneAccuracy gpu_accuracy = measure_plane(scene, gpu_result, 5);
    const PlaneAccuracy cpu_accuracy = measure_plane(scene, cpu_result, 5);
    const Agreement agreement = relative_agreement(gpu_result, cpu_result, 0.01);
    std::printf("[gpu distorted] gpu: coverage %.3f median %.5f p90 %.5f | cpu: coverage %.3f "
                "median %.5f p90 %.5f | agree<1%%: %.1f%% of %zu\n",
                gpu_accuracy.coverage, gpu_accuracy.median, gpu_accuracy.p90,
                cpu_accuracy.coverage, cpu_accuracy.median, cpu_accuracy.p90,
                100.0 * agreement.fraction, agreement.both);

    EXPECT_GT(gpu_accuracy.coverage, 0.85);
    EXPECT_LT(gpu_accuracy.median, 0.005);
    EXPECT_LT(gpu_accuracy.p90, 0.02);
    EXPECT_GT(agreement.fraction, 0.90);
}

// ---------------------------------------------------------------------------
// 4. Determinism
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, DeterministicRunToRun) {
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(128, 96, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];
    const PatchMatchOptions options = plane_options();

    const PatchMatchResult first = stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult second = stereo::run_patch_match_device(reference, neighbours, options);
    EXPECT_EQ(first.depth, second.depth);
    EXPECT_EQ(first.normals, second.normals);
    EXPECT_EQ(first.cost, second.cost);
}

// Splitting is a scheduling change, not an algorithmic one: a submission
// boundary only adds the same compute->compute barrier the unsplit command
// buffer already carries between dispatches, and a row band writes no pixel
// another band of the same half-sweep reads (every propagation offset is odd, so
// a source is always the opposite colour). Both splits are therefore compared
// bit for bit against the unsplit run, with memcmp rather than == so a -0.0/NaN
// difference cannot slip through.
TEST(PatchMatchGpu, SplitSubmissionsAreBitIdentical) {
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(128, 96, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];
    PatchMatchOptions options = plane_options();
    // Keep the per-call cost low but leave several sweeps to schedule.
    options.iterations = 4;

    struct EnvGuard {
        const char* name;
        std::string previous;
        explicit EnvGuard(const char* variable)
            : name(variable), previous(alphashape::platform::get_env(variable)) {}
        ~EnvGuard() { alphashape::platform::set_env(name, previous); }
    };
    const auto identical = [](const std::vector<float>& a, const std::vector<float>& b) {
        return a.size() == b.size() &&
               (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    };
    const auto expect_identical = [&](const PatchMatchResult& split,
                                      const PatchMatchResult& reference_run,
                                      const char* what) {
        EXPECT_EQ(split.width, reference_run.width);
        EXPECT_EQ(split.height, reference_run.height);
        EXPECT_TRUE(identical(split.depth, reference_run.depth))
            << what << " changed depth values";
        EXPECT_TRUE(identical(split.normals, reference_run.normals))
            << what << " changed normals";
        EXPECT_TRUE(identical(split.cost, reference_run.cost)) << what << " changed costs";
    };

    // Unsplit reference: one submission per half-sweep, no row bands.
    PatchMatchResult unsplit;
    {
        EnvGuard batch("ALPHASHAPE_GPU_MAX_BATCH");
        EnvGuard band("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");
        // A band target far above the image keeps every half-sweep in one
        // dispatch, whatever the default is.
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS", "100000000");
        unsplit = stereo::run_patch_match_device(reference, neighbours, options);
    }
    ASSERT_FALSE(unsplit.depth.empty());
    ASSERT_GT(valid_pixels(unsplit), 0u);

    // Maximum splitting: one step per submission.
    {
        EnvGuard batch("ALPHASHAPE_GPU_MAX_BATCH");
        EnvGuard band("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "1");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS", "100000000");
        expect_identical(
            stereo::run_patch_match_device(reference, neighbours, options), unsplit,
            "ALPHASHAPE_GPU_MAX_BATCH=1");
    }

    // Maximum row banding: a target below one tile row, so every half-sweep is
    // split into one dispatch per 16-row tile band.
    {
        EnvGuard batch("ALPHASHAPE_GPU_MAX_BATCH");
        EnvGuard band("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS", "1");
        expect_identical(
            stereo::run_patch_match_device(reference, neighbours, options), unsplit,
            "ALPHASHAPE_GPU_MAX_BAND_PIXELS=1");
    }

    // Both at once -- the shape a slow device takes: maximum banding and one
    // step per submission.
    {
        EnvGuard batch("ALPHASHAPE_GPU_MAX_BATCH");
        EnvGuard band("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "1");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS", "1");
        expect_identical(
            stereo::run_patch_match_device(reference, neighbours, options), unsplit,
            "maximum splitting and maximum banding");
    }

    // The adaptive limiter (no override) must reach the same answer too.
    {
        EnvGuard batch("ALPHASHAPE_GPU_MAX_BATCH");
        EnvGuard band("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS", "1");
        expect_identical(
            stereo::run_patch_match_device(reference, neighbours, options), unsplit,
            "the adaptive limiter with banding");
    }
}

TEST(PatchMatchGpu, SweepProgressReportsMonotoneFractions) {
    // The hook must describe the WHOLE search, not one level of it: a level's
    // share is its own half-sweeps over the run's, spent across that level's
    // sweeps. Getting the scale wrong is silent -- the bar simply never gets
    // far -- so the shape is pinned here rather than eyeballed: with 1 / the
    // run's sweep count used as the level's share, the progress was divided by
    // the sweep count a second time and a finished level reported 1/12 of the
    // progress it had actually made.
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(96, 72, 100.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    neighbours.resize(1);  // the progress values do not depend on the count
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];

    // One dispatch per submission, so every half-sweep's milestone is reported
    // regardless of how fast this device is. The adaptive limiter is free to
    // batch several sweeps into one submission, which is correct for the bar but
    // would make the exact values below depend on the machine.
    struct EnvGuard {
        const char* name;
        std::string previous;
        explicit EnvGuard(const char* variable)
            : name(variable), previous(alphashape::platform::get_env(variable)) {}
        ~EnvGuard() { alphashape::platform::set_env(name, previous); }
    };
    EnvGuard batch("ALPHASHAPE_GPU_MAX_BATCH");
    EnvGuard band("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
    alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "1");
    alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS", "100000000");

    const auto progress_of = [&](PatchMatchOptions options) {
        std::vector<double> reported;
        options.on_sweep = [&](double fraction) { reported.push_back(fraction); };
        stereo::run_patch_match_device(reference, neighbours, options);
        return reported;
    };
    const auto expect_shape = [](const std::vector<double>& reported, const char* what) {
        ASSERT_GE(reported.size(), 2U)
            << what << ": the hook must report more than the final tick";
        for (std::size_t i = 0; i < reported.size(); ++i) {
            EXPECT_GE(reported[i], 0.0) << what << " report " << i;
            EXPECT_LE(reported[i], 1.0) << what << " report " << i;
            if (i > 0) {
                EXPECT_GE(reported[i], reported[i - 1])
                    << what << " report " << i << " went backwards (" << reported[i - 1]
                    << " -> " << reported[i] << ")";
            }
        }
        EXPECT_DOUBLE_EQ(reported.back(), 1.0) << what << " did not finish at 1.0";
    };

    // A single level with six iterations is twelve half-sweeps, so the last
    // report before the final tick has 11/12 behind it. Anything much below that
    // means the level's share was scaled down again.
    PatchMatchOptions single = plane_options();
    single.pyramid_levels = 1;
    single.iterations = 6;
    single.window_radius = 3;
    single.window_step = 3;
    const std::vector<double> single_reported = progress_of(single);
    expect_shape(single_reported, "single level");
    ASSERT_GE(single_reported.size(), 2U);
    EXPECT_GE(single_reported[single_reported.size() - 2], 0.8)
        << "last pre-final report was " << single_reported[single_reported.size() - 2]
        << "; the sweeps lower a single-level run reports are 1/12 apart";

    // Enough sweeps that the second-to-last report is within 1% of the end: the
    // bar must get there from the SWEEPS, not from a single trailing tick.
    PatchMatchOptions long_run = single;
    long_run.iterations = 60;  // 120 half-sweeps, so the pre-final report is 119/120
    const std::vector<double> long_reported = progress_of(long_run);
    expect_shape(long_reported, "long run");
    ASSERT_GE(long_reported.size(), 2U);
    EXPECT_GE(long_reported[long_reported.size() - 2], 0.99)
        << "last pre-final report was " << long_reported[long_reported.size() - 2];

    // A pyramid must be monotone ACROSS level boundaries, and each level must
    // spend its own share rather than restarting at its base.
    PatchMatchOptions pyramid = single;
    pyramid.iterations = 6;
    pyramid.pyramid_levels = 2;
    const std::vector<int> plan =
        stereo::patch_match_iteration_plan(scene.reference.width(), scene.reference.height(),
                                           pyramid);
    ASSERT_EQ(plan.size(), 2U) << "the two-level request should survive here";
    const std::vector<double> pyramid_reported = progress_of(pyramid);
    expect_shape(pyramid_reported, "two levels");
    // The coarse level owns 2 * plan[0] of the 2 * (plan[0] + plan[1]) sweeps, so
    // some report must land strictly inside its share -- not only the 0 it
    // starts at and not only the boundary value.
    const double coarse_share =
        static_cast<double>(2 * plan[0]) / static_cast<double>(2 * (plan[0] + plan[1]));
    bool coarse_reported = false;
    for (const double value : pyramid_reported) {
        if (value > 0.0 && value < coarse_share - 1e-12) {
            coarse_reported = true;
        }
    }
    EXPECT_TRUE(coarse_reported) << "the coarse level reported no progress of its own";
}

// ---------------------------------------------------------------------------
// 5. Fallback semantics
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, DisabledFallsBackToCpu) {
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(128, 96, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];
    const PatchMatchOptions options = plane_options();

    gpu::set_enabled(false);
    const PatchMatchResult disabled =
        stereo::run_patch_match_device(reference, neighbours, options);
    EXPECT_FALSE(stereo::patch_match_gpu_would_run(reference, neighbours, options));
    gpu::set_enabled(true);
    const PatchMatchResult cpu = stereo::run_patch_match(reference, neighbours, options);

    EXPECT_EQ(disabled.depth, cpu.depth);
    EXPECT_EQ(disabled.normals, cpu.normals);
    EXPECT_EQ(disabled.cost, cpu.cost);
}

TEST(PatchMatchGpu, ForcedFailureFallsBackAndCounts) {
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(128, 96, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];
    const PatchMatchOptions options = plane_options();

    const std::uint64_t before = gpu::failure_count();
    stereo::patch_match_gpu_force_failure(true);
    const PatchMatchResult result = stereo::run_patch_match_device(reference, neighbours, options);
    EXPECT_EQ(gpu::failure_count(), before + 1);
    EXPECT_EQ(result.depth, stereo::run_patch_match(reference, neighbours, options).depth);
}

// ---------------------------------------------------------------------------
// 6. Dense fixture: accuracy contract and timing
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, DenseFixtureAccuracy) {
    REQUIRE_GPU_PM();
    const auto inputs = dense_inputs(2);

    // One reference (the first camera with a usable neighbourhood).
    const stereo::ViewNeighbourhood& hood = inputs->neighbourhoods.front();
    std::vector<PMView> neighbours;
    for (const std::int32_t id : hood.neighbours) {
        neighbours.push_back(inputs->view(id));
    }
    const PMView reference = inputs->view(hood.camera_id);
    PatchMatchOptions options;
    options.depth_min = hood.depth_min;
    options.depth_max = hood.depth_max;
    options.seed = 7;
    // Single level: this test's 0.90 floor was calibrated on the flat search,
    // and PyramidAgreesWithCpu is the pyramid's own gate.
    options.pyramid_levels = 1;

    const PatchMatchResult gpu_result =
        stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult cpu_result = stereo::run_patch_match(reference, neighbours, options);
    const Agreement agreement = relative_agreement(gpu_result, cpu_result, 0.01);
    const std::size_t gpu_valid = valid_pixels(gpu_result);
    const std::size_t cpu_valid = valid_pixels(cpu_result);
    std::printf("[gpu dense] camera %d: valid gpu %zu (%zu px) cpu %zu (%zu px) | agree<1%%: "
                "%.1f%% of %zu\n",
                hood.camera_id, gpu_valid, gpu_result.depth.size(), cpu_valid,
                cpu_result.depth.size(), 100.0 * agreement.fraction, agreement.both);

    EXPECT_GT(agreement.both, 1000U);
    EXPECT_GT(agreement.fraction, 0.90);
    EXPECT_NEAR(static_cast<double>(gpu_valid), static_cast<double>(cpu_valid),
                0.05 * static_cast<double>(cpu_valid));
}

// The random initial planes are a pure function of (seed, pixel), so with no
// sweeps the GPU and CPU must pick the same plane for every pixel: the same
// depth, the same normal and (up to float32 rounding) the same cost. This is
// the cheapest check that the kernels were compiled and executed correctly on a
// device; a driver that mishandles the module shows up here as a wrong normal
// long before the full search's agreement figure moves.
TEST(PatchMatchGpu, InitialPlanesMatchCpu) {
    REQUIRE_GPU_PM();
    const auto inputs = dense_inputs(2);
    const stereo::ViewNeighbourhood& hood = inputs->neighbourhoods.front();
    std::vector<PMView> neighbours;
    for (const std::int32_t id : hood.neighbours) {
        neighbours.push_back(inputs->view(id));
    }
    const PMView reference = inputs->view(hood.camera_id);
    PatchMatchOptions options;
    options.depth_min = hood.depth_min;
    options.depth_max = hood.depth_max;
    options.seed = 7;
    options.iterations = 0;
    options.max_cost = 3.0;  // keep every initialised pixel
    // Zero sweeps is a single random initialisation at the reference
    // resolution; the pyramid would seed the fine level from a coarse one.
    options.pyramid_levels = 1;

    const PatchMatchResult gpu_result =
        stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult cpu_result = stereo::run_patch_match(reference, neighbours, options);
    ASSERT_EQ(gpu_result.depth.size(), cpu_result.depth.size());

    std::size_t both = 0, depth_ok = 0, normal_ok = 0, cost_ok = 0;
    for (std::size_t i = 0; i < gpu_result.depth.size(); ++i) {
        if (!(gpu_result.depth[i] > 0.0F) || !(cpu_result.depth[i] > 0.0F)) {
            continue;
        }
        ++both;
        const double cpu_depth = static_cast<double>(cpu_result.depth[i]);
        if (std::abs(static_cast<double>(gpu_result.depth[i]) - cpu_depth) / cpu_depth < 1e-4) {
            ++depth_ok;
        }
        double dot = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            dot += static_cast<double>(gpu_result.normals[3 * i + k]) *
                   static_cast<double>(cpu_result.normals[3 * i + k]);
        }
        if (dot > 1.0 - 1e-4) {
            ++normal_ok;
        }
        if (std::abs(static_cast<double>(gpu_result.cost[i]) -
                     static_cast<double>(cpu_result.cost[i])) < 1e-2) {
            ++cost_ok;
        }
    }
    const double n = static_cast<double>(std::max<std::size_t>(both, 1));
    std::printf("[gpu init] %zu pixels: depth %.2f%% normal %.2f%% cost(<0.01) %.2f%%\n", both,
                100.0 * static_cast<double>(depth_ok) / n,
                100.0 * static_cast<double>(normal_ok) / n,
                100.0 * static_cast<double>(cost_ok) / n);
    ASSERT_GT(both, 1000U);
    EXPECT_GT(static_cast<double>(depth_ok) / n, 0.999);
    EXPECT_GT(static_cast<double>(normal_ok) / n, 0.999);
    EXPECT_GT(static_cast<double>(cost_ok) / n, 0.99);
}

// The refine kernel's own gate. InitialPlanesMatchCpu runs with iterations = 0,
// so it only exercises pm_init and pm_finalize; a regression inside the descent
// half-sweep (pm_refine) is invisible to it. This test runs real sweeps on the
// same dense fixture and compares the result with the CPU reference.
//
// What breaks this agreement is a stray fused multiply-add on the
// depth/plane-to-pixel path: it moves floor() to a neighbouring texel and flips a
// winner. That is not hypothetical -- it is exactly what an optimised pm_refine
// did on the Tesla T4 before the projection chain was marked `precise`, and
// DenseFixtureAccuracy at its 0.90 floor did not notice. The measured history on
// that GPU, same fixture, same settings: 100.0% on the unoptimised module (main),
// 93.4% with only the projected coordinates marked, 85.6% before the shaders'
// undefined behaviour was fixed.
//
// The floor is therefore 0.99 rather than 0.999, and that is calibrated, not
// arbitrary. It has to fail the 93.4% regression with room to spare, which 0.99
// does, while not failing on ordinary last-digit rounding: the same optimiser
// settings that give 100.0% here put the plane and distorted-camera fixtures at
// 99.9% and 99.8% on that GPU, so a floor inside that band would be asserting
// bit-identity of transcendental functions across two different math libraries.
// At 0.99 a failure still means what this test is for -- thousands of pixels
// landing on the wrong texel -- and not one ulp in sin/cos.
//
// A software device gets a looser bound: llvmpipe/lavapipe is not the reference
// the regression appeared on, and its own transcendental implementations are a
// separate variable from the one being gated.
TEST(PatchMatchGpu, RefineSweepsMatchCpu) {
    REQUIRE_GPU_PM();
    const auto inputs = dense_inputs(2);
    const stereo::ViewNeighbourhood& hood = inputs->neighbourhoods.front();
    std::vector<PMView> neighbours;
    for (const std::int32_t id : hood.neighbours) {
        neighbours.push_back(inputs->view(id));
    }
    const PMView reference = inputs->view(hood.camera_id);
    PatchMatchOptions options;
    options.depth_min = hood.depth_min;
    options.depth_max = hood.depth_max;
    options.seed = 7;
    // The shipped default, written out both so this test cannot silently turn
    // into a second InitialPlanesMatchCpu (iterations = 0) and so its figure is
    // directly comparable with DenseFixtureAccuracy's, which is the measurement
    // the floor above is calibrated against.
    options.iterations = 6;
    // The gate measures pm_refine, so it runs the search it has always run: one
    // level at the reference resolution. The pyramid is gated separately by
    // PyramidAgreesWithCpu.
    options.pyramid_levels = 1;

    const std::shared_ptr<gpu::Context> context = gpu::default_context();
    const bool software = context != nullptr && gpu::is_software_device(context->info());

    const PatchMatchResult gpu_result =
        stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult cpu_result = stereo::run_patch_match(reference, neighbours, options);
    ASSERT_EQ(gpu_result.depth.size(), cpu_result.depth.size());

    const Agreement agreement = relative_agreement(gpu_result, cpu_result, 0.01);
    // Normals are scored only where the depths already agree, so a wrong search
    // cannot hide behind a normal that happens to be close.
    std::size_t normal_ok = 0;
    for (std::size_t i = 0; i < gpu_result.depth.size(); ++i) {
        if (!(gpu_result.depth[i] > 0.0F) || !(cpu_result.depth[i] > 0.0F)) {
            continue;
        }
        const double cpu_depth = static_cast<double>(cpu_result.depth[i]);
        if (std::abs(static_cast<double>(gpu_result.depth[i]) - cpu_depth) / cpu_depth >= 0.01) {
            continue;
        }
        double dot = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            dot += static_cast<double>(gpu_result.normals[3 * i + k]) *
                   static_cast<double>(cpu_result.normals[3 * i + k]);
        }
        if (dot > 1.0 - 1e-4) {
            ++normal_ok;
        }
    }
    const double both = static_cast<double>(std::max<std::size_t>(agreement.both, 1));
    const double normal_fraction = static_cast<double>(normal_ok) / both;
    std::printf("[gpu refine] camera %d, %d sweeps: agree<1%% %.2f%% of %zu | normals %.2f%% of "
                "the depth-agreeing pixels | %s\n",
                hood.camera_id, options.iterations, 100.0 * agreement.fraction, agreement.both,
                100.0 * normal_fraction, software ? "software device" : "hardware device");

    ASSERT_GT(agreement.both, 1000U);
    EXPECT_GT(agreement.fraction, software ? 0.90 : 0.99)
        << "pm_refine disagrees with the CPU on " << agreement.both
        << " comparable pixels; a fused multiply-add on the depth/plane-to-pixel path (see the "
           "note at the top of pm_math.glsl) is the usual cause";
    EXPECT_GT(normal_fraction, software ? 0.90 : 0.99);
}

// The RefineSweepsMatchCpu floor above is calibrated at pyramid_levels=1, which
// is the search it has always measured. Pin that here so a later change to the
// default cannot quietly move the gate's meaning -- the pyramid has its own gate
// (PyramidAgreesWithCpu) with a bound of its own.
TEST(PatchMatchGpu, RefineSweepsGateRunsSingleLevel) {
    const PatchMatchOptions defaults;
    EXPECT_EQ(defaults.pyramid_levels, 0) << "the shipped default is automatic";
    PatchMatchOptions single;
    single.pyramid_levels = 1;
    EXPECT_EQ(single.pyramid_levels, 1);
}

// ---------------------------------------------------------------------------
// 6b. Coarse-to-fine on the GPU: agreement with the CPU at the pyramid default
// ---------------------------------------------------------------------------

/// The pyramid default (automatic levels) on the dense fixture, against the
/// CPU's own pyramid run over the same inputs. The two paths build their level
/// plans from the same `patch_match_iteration_plan`, apply the same
/// `pyramid_upsample` to the carried map and search from the same
/// `pyramid_search_span`, so what differs is only the kernel arithmetic -- the
/// same precision difference RefineSweepsMatchCpu measures, now on the whole
/// pyramid rather than on one level.
///
/// It is a separate test rather than a second assertion in RefineSweepsMatchCpu
/// because that one is the calibrated NVIDIA regression gate for pm_refine: if
/// the pyramid also failed there, a real kernel regression and a
/// coarse-to-fine bookkeeping bug would arrive as one indistinguishable number.
TEST(PatchMatchGpu, PyramidAgreesWithCpu) {
    REQUIRE_GPU_PM();
    const auto inputs = dense_inputs(2);
    const stereo::ViewNeighbourhood& hood = inputs->neighbourhoods.front();
    std::vector<PMView> neighbours;
    for (const std::int32_t id : hood.neighbours) {
        neighbours.push_back(inputs->view(id));
    }
    const PMView reference = inputs->view(hood.camera_id);

    PatchMatchOptions options;
    options.depth_min = hood.depth_min;
    options.depth_max = hood.depth_max;
    options.seed = 7;
    options.iterations = 6;
    // Automatic: 240x180 at the 256 default floor is already one level, so ask
    // for a pyramid explicitly with a floor that the fixture can reach.
    options.pyramid_levels = 0;
    options.pyramid_min_size = 64;

    const std::vector<int> plan =
        stereo::patch_match_iteration_plan(reference.image->width(),
                                           reference.image->height(), options);
    ASSERT_GE(plan.size(), 2U) << "the fixture must exercise a real pyramid";

    const std::shared_ptr<gpu::Context> context = gpu::default_context();
    const bool software = context != nullptr && gpu::is_software_device(context->info());

    const PatchMatchResult gpu_result =
        stereo::run_patch_match_device(reference, neighbours, options);
    const PatchMatchResult cpu_result = stereo::run_patch_match(reference, neighbours, options);
    ASSERT_EQ(gpu_result.depth.size(), cpu_result.depth.size());

    const Agreement agreement = relative_agreement(gpu_result, cpu_result, 0.01);
    const std::size_t gpu_valid = valid_pixels(gpu_result);
    const std::size_t cpu_valid = valid_pixels(cpu_result);
    std::printf("[gpu pyramid] camera %d, %zu levels: agree<1%% %.2f%% of %zu | valid gpu %zu "
                "cpu %zu | %s\n",
                hood.camera_id, plan.size(), 100.0 * agreement.fraction, agreement.both, gpu_valid,
                cpu_valid, software ? "software device" : "hardware device");

    ASSERT_GT(agreement.both, 1000U);
    // The same shape of bound as RefineSweepsMatchCpu, one notch looser: a
    // pyramid runs the level plan twice (once per path) and a hole in the coarse
    // map is a different random plane on each, so a handful more pixels land on
    // a different local minimum. A real port regression moves this by tens of
    // percent, not by one.
    EXPECT_GT(agreement.fraction, software ? 0.88 : 0.98);
    EXPECT_NEAR(static_cast<double>(gpu_valid), static_cast<double>(cpu_valid),
                0.05 * static_cast<double>(cpu_valid));
}

// ---------------------------------------------------------------------------
// 7. Fused-cloud accuracy: the same contract as BuildDenseTruthPoses
// ---------------------------------------------------------------------------

/// Nearest of the two truth planes (as test_build_dense.cpp's plane_stats).
struct PlaneStats {
    std::size_t n = 0;
    double median = 0.0, p90 = 0.0;
};

PlaneStats plane_stats(const std::vector<Eigen::Vector3d>& points) {
    std::vector<double> distances;
    distances.reserve(points.size());
    for (const Eigen::Vector3d& p : points) {
        distances.push_back(std::min(std::abs(p.z()), std::abs(p.y())));
    }
    std::sort(distances.begin(), distances.end());
    PlaneStats stats;
    stats.n = distances.size();
    stats.median = dense_fixture::percentile(distances, 0.50);
    stats.p90 = dense_fixture::percentile(distances, 0.90);
    return stats;
}

/// Full dense path (BuildDepthMaps at downscale 2 + BuildPointCloud) on the
/// truth poses, returning the fused cloud in the truth frame.
std::vector<Eigen::Vector3d> run_dense_cloud() {
    Document document = dense_fixture::aligned_document(dense_fixture::seq_files());
    Chunk& chunk = *document.active_chunk();
    dense_fixture::set_truth_poses(document, dense_fixture::parse_truth_images("seq_truth.json"));
    dense_fixture::set_truth_calibration(document, 480, 360, 500.0);
    dense_fixture::retriangulate_tracks(chunk);
    dense_fixture::run_build_depth_maps(document, /*downscale=*/2, "mild");
    dense_fixture::run_build_point_cloud(document);
    return document.active_chunk()->point_cloud().points;
}

TEST(PatchMatchGpu, DenseFusedCloudMeetsAccuracyContract) {
    REQUIRE_GPU_PM();
    gpu::set_enabled(true);
    const std::vector<Eigen::Vector3d> gpu_points = run_dense_cloud();
    const PlaneStats gpu_stats = plane_stats(gpu_points);

    // CPU reference on the same scene, for the side-by-side print and to prove
    // the device switch is the only difference.
    gpu::set_enabled(false);
    const std::vector<Eigen::Vector3d> cpu_points = run_dense_cloud();
    gpu::set_enabled(true);
    const PlaneStats cpu_stats = plane_stats(cpu_points);

    std::printf("[gpu fused] gpu: %zu points median %.4f p90 %.4f | cpu: %zu points median "
                "%.4f p90 %.4f\n",
                gpu_stats.n, gpu_stats.median, gpu_stats.p90, cpu_stats.n, cpu_stats.median,
                cpu_stats.p90);

    // The dense-stereo accuracy contract (test_build_dense.cpp).
    EXPECT_LT(gpu_stats.median, 0.03) << "median plane distance " << gpu_stats.median;
    EXPECT_LT(gpu_stats.p90, 0.10) << "p90 plane distance " << gpu_stats.p90;
    EXPECT_GT(gpu_stats.n, 10000U);
    // The GPU cloud must land where the CPU's does, not somewhere merely valid.
    EXPECT_NEAR(static_cast<double>(gpu_stats.n), static_cast<double>(cpu_stats.n),
                0.05 * static_cast<double>(cpu_stats.n));
    EXPECT_NEAR(gpu_stats.median, cpu_stats.median, 0.005);
}

// ---------------------------------------------------------------------------
// 8. Aerial block: the k_eff visibility vote rule
// ---------------------------------------------------------------------------

/// Terrain and roof points for the aerial chunk, as test_build_dem.cpp builds
/// them (BuildDepthMaps needs triangulated tracks for depth ranges and views).
void add_aerial_tie_points(Chunk& chunk) {
    std::vector<Track>& tracks = chunk.tie_points().tracks;
    tracks.clear();
    const auto push_point = [&](const Eigen::Vector3d& world) {
        Track track;
        for (const Camera& camera : chunk.cameras()) {
            const Eigen::Vector3d cam =
                camera.pose.block<3, 3>(0, 0) * world + camera.pose.block<3, 1>(0, 3);
            if (!(cam.z() > 1.0)) {
                continue;
            }
            const double u = aerial_fixture::kImageWidth / 2.0 +
                             aerial_fixture::kFocal * cam.x() / cam.z();
            const double v = aerial_fixture::kImageHeight / 2.0 +
                             aerial_fixture::kFocal * cam.y() / cam.z();
            if (u < 0.0 || v < 0.0 || u >= aerial_fixture::kImageWidth ||
                v >= aerial_fixture::kImageHeight) {
                continue;
            }
            Observation observation;
            observation.camera_id = camera.id;
            observation.keypoint_index = 0;
            track.observations.push_back(observation);
        }
        if (track.observations.size() < 2) {
            return;
        }
        track.point = world;
        tracks.push_back(std::move(track));
    };
    for (double x = -55.0; x <= 55.0 + 1e-6; x += 5.0) {
        for (double y = -40.0; y <= 40.0 + 1e-6; y += 5.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
    for (double x = -19.0; x <= -5.0 + 1e-6; x += 2.0) {
        for (double y = 6.0; y <= 16.0 + 1e-6; y += 2.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::kBuildTop));
        }
    }
    for (double x = 1.0; x <= 19.0 + 1e-6; x += 2.0) {
        for (double y = 16.0; y <= 29.0 + 1e-6; y += 2.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
}

TEST(PatchMatchGpu, AerialBlockValidFractionMatchesCpu) {
    REQUIRE_GPU_PM();
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    add_aerial_tie_points(chunk);
    const std::vector<stereo::ViewNeighbourhood> hoods =
        stereo::select_views(chunk, stereo::ViewSelectionOptions{});

    std::vector<FloatImage> images;
    std::vector<Calibration> calibrations;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> translations;
    std::vector<std::int32_t> ids;
    const std::vector<std::string> files = aerial_fixture::image_files();
    for (const Camera& camera : chunk.cameras()) {
        images.push_back(read_image(files[ids.size()]).to_luminance());
        calibrations.push_back(scaled_calibration(
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration, 1));
        rotations.push_back(camera.pose.block<3, 3>(0, 0));
        translations.push_back(camera.pose.block<3, 1>(0, 3));
        ids.push_back(camera.id);
    }
    const auto view = [&](std::int32_t camera_id) {
        PMView result;
        std::size_t index = 0;
        for (; index < ids.size(); ++index) {
            if (ids[index] == camera_id) {
                break;
            }
        }
        result.image = &images[index];
        result.calibration = calibrations[index];
        result.R = rotations[index];
        result.t = translations[index];
        return result;
    };

    // The low-overlap nadir block: every camera's neighbourhood is large and
    // only part of each footprint overlaps, which is what exposed the vote bug.
    std::size_t checked = 0;
    for (const stereo::ViewNeighbourhood& hood : hoods) {
        if (hood.neighbours.size() < 4 || checked >= 3) {
            continue;
        }
        std::vector<PMView> neighbours;
        for (const std::int32_t id : hood.neighbours) {
            neighbours.push_back(view(id));
        }
        const PMView reference = view(hood.camera_id);
        PatchMatchOptions options;
        options.depth_min = hood.depth_min;
        options.depth_max = hood.depth_max;
        options.seed = 11;
        // Single level: this test's figures are the flat search's, and the
        // aerial block's 0.90 agreement floor was calibrated against them.
        options.pyramid_levels = 1;

        const double gpu_seconds = time_seconds(
            [&] { return stereo::run_patch_match_device(reference, neighbours, options); });
        const double cpu_seconds =
            time_seconds([&] { return stereo::run_patch_match(reference, neighbours, options); });
        const PatchMatchResult gpu_result =
            stereo::run_patch_match_device(reference, neighbours, options);
        const PatchMatchResult cpu_result = stereo::run_patch_match(reference, neighbours, options);
        const double total = static_cast<double>(cpu_result.depth.size());
        const double gpu_fraction = static_cast<double>(valid_pixels(gpu_result)) / total;
        const double cpu_fraction = static_cast<double>(valid_pixels(cpu_result)) / total;
        const Agreement agreement = relative_agreement(gpu_result, cpu_result, 0.01);
        std::printf("[gpu aerial] camera %d (%zu neighbours): valid gpu %.1f%% cpu %.1f%% | "
                    "agree<1%%: %.1f%% | gpu %.2f s cpu %.2f s (%.2fx)\n",
                    hood.camera_id, hood.neighbours.size(), 100.0 * gpu_fraction,
                    100.0 * cpu_fraction, 100.0 * agreement.fraction, gpu_seconds,
                    cpu_seconds, cpu_seconds / gpu_seconds);
        EXPECT_NEAR(gpu_fraction, cpu_fraction, 0.05);
        EXPECT_GT(agreement.fraction, 0.90);
        ++checked;
    }
    EXPECT_GT(checked, 0U) << "no aerial camera had a wide neighbourhood";
}

// ---------------------------------------------------------------------------
// 9. Timing: GPU against the CPU on one map
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, TimingAgainstCpu) {
    REQUIRE_GPU_PM();
    REQUIRE_HARDWARE_GPU_PM("PatchMatchGpu.TimingAgainstCpu");
    const auto inputs = dense_inputs(1);
    const stereo::ViewNeighbourhood& hood = inputs->neighbourhoods.front();
    std::vector<PMView> neighbours;
    for (const std::int32_t id : hood.neighbours) {
        neighbours.push_back(inputs->view(id));
    }
    const PMView reference = inputs->view(hood.camera_id);
    PatchMatchOptions options;
    options.depth_min = hood.depth_min;
    options.depth_max = hood.depth_max;
    options.seed = 3;
    // Single level, so the reported ratio stays the pm_refine figure
    // LIMITATIONS.md documents.
    options.pyramid_levels = 1;

    // Warm the device so pipeline compilation is not charged to the benchmark.
    ASSERT_TRUE(stereo::patch_match_gpu_would_run(reference, neighbours, options));
    stereo::run_patch_match_device(reference, neighbours, options);

    const double gpu_seconds = time_seconds(
        [&] { return stereo::run_patch_match_device(reference, neighbours, options); });
    gpu::set_enabled(false);
    const double cpu_seconds =
        time_seconds([&] { return stereo::run_patch_match(reference, neighbours, options); });
    gpu::set_enabled(true);
    const double speedup = cpu_seconds / gpu_seconds;
    std::printf("[gpu timing] dense ds1 %dx%d, %zu neighbours: gpu %.2f s, cpu %.2f s, "
                "speed-up %.2fx\n",
                reference.image->width(), reference.image->height(), neighbours.size(),
                gpu_seconds, cpu_seconds, speedup);
    RecordProperty("gpu_seconds", gpu_seconds);
    RecordProperty("cpu_seconds", cpu_seconds);
    RecordProperty("speedup", speedup);
    // Target was >= 3x; measured 2.4-2.9x on the M1 and reported in
    // LIMITATIONS.md. The floor below catches a real regression (the first
    // working version was 0.4x) without failing on the documented gap.
    if (test::perf_asserts_enabled()) {
        EXPECT_GT(speedup, 2.0) << "target >= 3x on the M1 at downscale 1";
    }
}

// ---------------------------------------------------------------------------
// 8. Census cost: declined on the GPU, answered by the CPU
// ---------------------------------------------------------------------------

TEST(PatchMatchGpu, CensusDeclinesBeforeDispatchAndMatchesCpu) {
    // The census cost is not implemented in the kernels (see patch_match_gpu.cpp
    // fits_kernel_limits): a census run must be declined before anything is
    // dispatched -- no failure is recorded -- and answered by the CPU
    // implementation, bit for bit.
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(128, 96, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];

    PatchMatchOptions options = plane_options();
    options.cost_type = stereo::CostType::Census;

    EXPECT_FALSE(stereo::patch_match_gpu_supported(reference, neighbours, options))
        << "the GPU must decline the census cost";
    EXPECT_FALSE(stereo::patch_match_gpu_would_run(reference, neighbours, options));

    const std::uint64_t before = gpu::failure_count();
    const PatchMatchResult result =
        stereo::run_patch_match_device(reference, neighbours, options);
    EXPECT_EQ(gpu::failure_count(), before) << "a decline is not a device failure";

    const PatchMatchResult cpu = stereo::run_patch_match(reference, neighbours, options);
    EXPECT_EQ(result.depth, cpu.depth);
    EXPECT_EQ(result.normals, cpu.normals);
    EXPECT_EQ(result.cost, cpu.cost);

    // And the census result is still a depth map of the plane (the decline must
    // not silently produce an empty map).
    std::size_t valid = 0;
    for (const float d : result.depth) {
        if (d > 0.0F) {
            ++valid;
        }
    }
    EXPECT_GT(valid, result.depth.size() / 2);
}

TEST(PatchMatchGpu, CensusAgreesWithZnccOnThePlane) {
    // A clean plane is the case both costs handle: the census result must land in
    // the same accuracy region as the ZNCC one (it runs on the CPU either way, so
    // this checks the cost's own correctness rather than the device's).
    REQUIRE_GPU_PM();
    PlaneScene scene = make_plane_scene(160, 120, 150.0, 0.0, 0.0);
    std::vector<PMView> neighbours = plane_views(scene);
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];

    PatchMatchOptions census_options = plane_options();
    census_options.cost_type = stereo::CostType::Census;
    PatchMatchOptions zncc_options = plane_options();
    zncc_options.cost_type = stereo::CostType::Zncc;

    const PatchMatchResult census =
        stereo::run_patch_match_device(reference, neighbours, census_options);
    const PatchMatchResult zncc =
        stereo::run_patch_match_device(reference, neighbours, zncc_options);

    const PlaneAccuracy census_accuracy = measure_plane(scene, census, /*radius=*/5);
    const PlaneAccuracy zncc_accuracy = measure_plane(scene, zncc, /*radius=*/5);
    std::printf("[census gpu] census cov=%.4f median=%.5f, zncc cov=%.4f median=%.5f\n",
                census_accuracy.coverage, census_accuracy.median, zncc_accuracy.coverage,
                zncc_accuracy.median);
    // The census descriptor is coarser than 8-bit intensity on a smooth gradient,
    // so the census plane is expected to be a little less sharp; the assertion is
    // that it still reconstructs the plane (coverage and error in the same
    // region), not that it matches the intensity cost number for number.
    EXPECT_GT(census_accuracy.coverage, 0.7);
    EXPECT_LT(census_accuracy.median, 0.05);
    EXPECT_LT(census_accuracy.median, zncc_accuracy.median * 5.0 + 0.005);
}

}  // namespace
}  // namespace alphashape
