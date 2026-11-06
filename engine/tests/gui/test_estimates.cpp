// Estimates.*: the cold model, measured throughput history, persistence and
// the RAM warning rule (WP-14.18).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <QLabel>
#include <QSettings>
#include <QTemporaryDir>

#include "apps/gui/jobs/estimate.hpp"
#include "apps/gui/task_dialog.hpp"
#include "apps/gui/jobs/estimate_model.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/tasks/registry.hpp"
#include "../perf_gate.hpp"

using namespace alphashape;
using namespace alphashape::gui;

namespace {

Chunk make_chunk(int cameras, int width = 6000, int height = 4000) {
    Chunk chunk;
    Sensor sensor;
    sensor.width = width;
    sensor.height = height;
    chunk.sensors().push_back(sensor);
    for (int i = 0; i < cameras; ++i) {
        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.enabled = true;
        chunk.cameras().push_back(camera);
    }
    return chunk;
}

Params depth_params() {
    return make_params(builtin_registry().require("BuildDepthMaps"));
}

}  // namespace

TEST(Estimates, Monotonic) {
    ThroughputHistory history;
    const Params params = depth_params();
    const Chunk four = make_chunk(4);
    const Chunk eight = make_chunk(8);
    const Estimate small =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, four, history);
    const Estimate big =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, eight, history);
    EXPECT_GE(big.seconds, small.seconds * 1.5);

    const Chunk narrow = make_chunk(6, 3000, 2000);
    const Chunk wide = make_chunk(6, 6000, 4000);
    const Estimate narrow_estimate =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, narrow, history);
    const Estimate wide_estimate =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, wide, history);
    EXPECT_GE(wide_estimate.seconds, narrow_estimate.seconds * 2.5);
}

TEST(Estimates, UsesObservedRates) {
    ThroughputHistory history;
    const Params params = depth_params();
    const Chunk chunk = make_chunk(4);
    const Estimate cold =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, chunk, history);
    ASSERT_GT(cold.seconds, 0.0);
    for (int i = 0; i < 3; ++i) {
        ThroughputRow row;
        row.task = "BuildDepthMaps";
        row.cameras = 4.0;
        row.megapixels = 1.5;  // 6000x4000 at downscale 4
        row.seconds = 2.0 * cold.seconds;
        history.add(row);
    }
    const Estimate measured =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, chunk, history);
    EXPECT_TRUE(measured.confident);
    EXPECT_NEAR(measured.seconds, 2.0 * cold.seconds, 0.25 * 2.0 * cold.seconds);
    EXPECT_NE(measured.basis.find("measured"), std::string::npos);
}

TEST(Estimates, RamWarningThreshold) {
    Estimate estimate;
    const std::uint64_t budget = 1000;
    estimate.peak_ram_bytes = 750;
    EXPECT_FALSE(needs_ram_warning(estimate, budget));
    estimate.peak_ram_bytes = 751;
    EXPECT_TRUE(needs_ram_warning(estimate, budget));
    EXPECT_FALSE(needs_ram_warning(estimate, 0));
}

TEST(Estimates, DialogShowsWarningAboveThreshold) {
    QTemporaryDir directory;
    QSettings settings(directory.filePath(QStringLiteral("settings.ini")),
                       QSettings::IniFormat);
    const Chunk chunk = make_chunk(4);
    ThroughputHistory history;
    TaskDialog dialog("BuildDepthMaps", builtin_registry(), settings);
    dialog.set_estimate_context(chunk_facts(chunk), &history);
    ASSERT_NE(dialog.estimate_label(), nullptr);
    EXPECT_TRUE(dialog.estimate_label()->text().contains(QStringLiteral("Estimated:")));
    EXPECT_TRUE(dialog.estimate_label()->text().contains(QStringLiteral("cold model")));

    dialog.set_memory_budget(700ULL * 1000ULL * 1000ULL);
    ASSERT_NE(dialog.ram_warning_label(), nullptr);
    EXPECT_FALSE(dialog.ram_warning_label()->isHidden());
    EXPECT_TRUE(
        dialog.ram_warning_label()->text().contains(QStringLiteral("4-image block")));

    dialog.set_memory_budget(1000ULL * 1000ULL * 1000ULL);
    EXPECT_TRUE(dialog.ram_warning_label()->isHidden());
}

TEST(Estimates, UnknownTaskFallsBack) {
    ThroughputHistory history;
    const Chunk chunk = make_chunk(4);
    const Params empty;
    const Estimate estimate =
        estimate_task(builtin_registry(), "NoSuchTask", empty, chunk, history);
    EXPECT_FALSE(estimate.confident);
    EXPECT_EQ(estimate.basis, "cold model (generic)");
    EXPECT_GT(estimate.seconds, 0.0);
    EXPECT_EQ(estimate.seconds_sigma_rel, 0.35);
}

TEST(Estimates, HistoryKeepsLast20) {
    ThroughputHistory history;
    for (int i = 0; i < 25; ++i) {
        ThroughputRow row;
        row.task = "BuildDepthMaps";
        row.cameras = 4.0;
        row.megapixels = 1.0;
        row.seconds = static_cast<double>(i);
        history.add(row);
    }
    const std::vector<ThroughputRow>* rows = history.rows("BuildDepthMaps");
    ASSERT_NE(rows, nullptr);
    ASSERT_EQ(rows->size(), ThroughputHistory::kMaxRowsPerTask);
    EXPECT_DOUBLE_EQ(rows->front().seconds, 5.0);
    EXPECT_DOUBLE_EQ(rows->back().seconds, 24.0);

    ThroughputRow other;
    other.task = "BuildModel";
    other.cameras = 1.0;
    other.megapixels = 1.0;
    other.seconds = 1.0;
    history.add(other);
    EXPECT_EQ(history.rows("BuildModel")->size(), 1U);
}

TEST(Estimates, HistoryFileRoundTrip) {
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const std::string project = directory.filePath("scene.psx").toStdString();
    const std::string path = history_path_for_project(project);
    // Compare through std::filesystem: its string() is native, while
    // QTemporaryDir::filePath() uses forward slashes on every platform.
    const std::string base =
        std::filesystem::path(directory.filePath("scene").toStdString()).string();
    ASSERT_EQ(path, base + ".files/job_history.tsv");

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    {
        std::ofstream file(path);
        // A malformed line BETWEEN two valid rows must lose neither valid row.
        file << "\"BuildDepthMaps\"\t4\t1.5\t12.25\t123456789\n";
        file << "this line is not a history row\n";
        file << "\"BuildDepthMaps\"\t4\t1.5\t13.5\t0\n";
        // Zero seconds is harmless (the rate median skips it); NaN/infinite
        // seconds are rejected, and a 1e300 peak saturates instead of UB.
        file << "\"BuildDepthMaps\"\t4\t1.5\t0\t0\n";
        file << "\"BuildDepthMaps\"\t4\t1.5\tnan\t0\n";
        file << "\"BuildDepthMaps\"\t4\t1.5\tinf\t0\n";
        file << "\"BuildDepthMaps\"\t4\t1.5\t14\t1e300\n";
    }

    FileThroughputStore store(path);
    const ThroughputHistory loaded = store.load();
    const std::vector<ThroughputRow>* rows = loaded.rows("BuildDepthMaps");
    ASSERT_NE(rows, nullptr);
    ASSERT_EQ(rows->size(), 4U);
    EXPECT_DOUBLE_EQ((*rows)[0].seconds, 12.25);
    EXPECT_DOUBLE_EQ((*rows)[1].seconds, 13.5);
    EXPECT_DOUBLE_EQ((*rows)[2].seconds, 0.0);
    EXPECT_DOUBLE_EQ((*rows)[3].seconds, 14.0);
    EXPECT_EQ((*rows)[1].peak_rss_bytes, 0U);
    EXPECT_EQ((*rows)[3].peak_rss_bytes, UINT64_MAX);

    // The extreme row must not make the estimate non-finite.
    const Estimate estimate = estimate_task(builtin_registry(), "BuildDepthMaps",
                                            depth_params(), make_chunk(4), loaded);
    EXPECT_TRUE(std::isfinite(estimate.seconds));
    EXPECT_FALSE(std::filesystem::exists(path + ".tmp"));
}

TEST(Estimates, SigmaFloor) {
    ThroughputHistory history;
    const Params params = depth_params();
    const Chunk chunk = make_chunk(4);
    for (int i = 0; i < 3; ++i) {
        ThroughputRow row;
        row.task = "BuildDepthMaps";
        row.cameras = 4.0;
        row.megapixels = 1.5;
        row.seconds = 30.0;
        history.add(row);
    }
    const Estimate estimate =
        estimate_task(builtin_registry(), "BuildDepthMaps", params, chunk, history);
    EXPECT_TRUE(estimate.confident);
    EXPECT_DOUBLE_EQ(estimate.seconds_sigma_rel, 0.10);
}

TEST(Estimates, PerfUnderFiveMilliseconds) {
    if (!test::perf_asserts_enabled()) {
        GTEST_SKIP() << "perf assertions disabled";
    }
    ThroughputHistory history;
    const Params params = depth_params();
    const Chunk chunk = make_chunk(200);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        const Estimate estimate =
            estimate_task(builtin_registry(), "BuildDepthMaps", params, chunk, history);
        ASSERT_GT(estimate.seconds, 0.0);
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start);
    EXPECT_LE(elapsed.count() / 1000.0, 5.0);
}

TEST(Estimates, PeakRssOnlyWhenRaised) {
    // The counter is process-wide: only a run that lifts it may claim it.
    EXPECT_EQ(recorded_peak_rss(100, 200), 200U);
    EXPECT_EQ(recorded_peak_rss(200, 100), 0U);
    EXPECT_EQ(recorded_peak_rss(100, 100), 0U);
    EXPECT_EQ(recorded_peak_rss(0, 0), 0U);
}

TEST(Estimates, MeasuredRamFromHistory) {
    const auto row = [](double peak) {
        ThroughputRow r;
        r.task = "BuildDepthMaps";
        r.cameras = 4.0;
        r.megapixels = 1.5;  // units 6
        r.peak_rss_bytes = static_cast<std::uint64_t>(peak);
        return r;
    };
    const std::vector<ThroughputRow> sparse{row(0), row(0), row(1000)};
    EXPECT_EQ(measured_ram_from_history(sparse, 6.0, 4242), 4242U);

    // Median peak 2000 at units 6, scaled to the current 12 units -> 4000.
    const std::vector<ThroughputRow> enough{row(1000), row(2000), row(3000), row(0)};
    EXPECT_EQ(measured_ram_from_history(enough, 12.0, 4242), 4000U);
    EXPECT_EQ(measured_ram_from_history(enough, 6.0, 4242), 2000U);
    EXPECT_EQ(measured_ram_from_history(enough, 0.0, 4242), 4242U);
}

TEST(Estimates, HistoryPathForPszProject) {
    const std::filesystem::path project("scene.psz");
    EXPECT_EQ(history_path_for_project(project.string()),
              project.string() + ".history.tsv");
}

TEST(Estimates, PointCloudUsesDepthMapResolution) {
    // 12 cameras of 24 MP would suggest half an hour; the depth maps they
    // consumed are 240x180 = 0.0432 Mpx each, which is the real input.
    Chunk chunk = make_chunk(12, 6000, 4000);
    std::vector<DepthMap>& maps = chunk.depth_maps();
    for (int i = 0; i < 12; ++i) {
        DepthMap map;
        map.camera_id = i;
        map.width = 240;
        map.height = 180;
        map.downscale = 4;
        maps.push_back(map);
    }
    const Estimate estimate = estimate_task(
        builtin_registry(), "BuildPointCloud",
        make_params(builtin_registry().require("BuildPointCloud")), chunk,
        ThroughputHistory{});
    EXPECT_LT(estimate.seconds, 5.0);
}

TEST(Estimates, MatchPhotosOnlyDetectsCamerasWithoutKeypoints) {
    Chunk chunk = make_chunk(4);
    for (int i = 0; i < 4; ++i) {
        auto frame = std::make_shared<Frame>(
            "image" + std::to_string(i) + ".jpg", 6000, 4000);
        frame->set_camera_id(i);
        chunk.frames().push_back(frame);
    }
    const Params params = make_params(builtin_registry().require("MatchPhotos"));
    const Estimate with_detection = estimate_task(
        builtin_registry(), "MatchPhotos", params, chunk, ThroughputHistory{});
    for (int i = 0; i < 4; ++i) {
        KeypointSet set;
        set.camera_id = i;
        chunk.tie_points().keypoint_sets.push_back(set);
    }
    const Estimate matching_only = estimate_task(
        builtin_registry(), "MatchPhotos", params, chunk, ThroughputHistory{});
    EXPECT_LT(matching_only.seconds, with_detection.seconds);
}

