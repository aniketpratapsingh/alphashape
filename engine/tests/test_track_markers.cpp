// WP-13.10: TrackMarkers, pyramidal Lucas-Kanade with a forward-backward
// check. The synthetic 20-frame sequence is rendered by make_marker_sequence.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "common/algorithm/markers/detect_markers_task.hpp"
#include "common/algorithm/markers/track_markers_task.hpp"
#include "common/commands/command.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "image/io/image_io.hpp"
#include "make_marker_sequence.hpp"

namespace alphashape {
namespace {

using test::MarkerSequence;
using test::MarkerSequenceSpec;
using Vec2 = Eigen::Vector2d;

constexpr int kFrames = 20;

std::string label_for(const test::MarkerSequenceTarget& target, int index) {
    return target.cross ? "cross " + std::to_string(index)
                        : "target " + std::to_string(target.code);
}

Marker* find_marker(Chunk& chunk, std::int32_t marker_id) {
    for (Marker& marker : chunk.markers()) {
        if (marker.id == marker_id) {
            return &marker;
        }
    }
    return nullptr;
}

const MarkerProjection* projection_for(const Marker& marker,
                                       std::int32_t camera_id) {
    for (const MarkerProjection& projection : marker.projections) {
        if (projection.camera_id == camera_id) {
            return &projection;
        }
    }
    return nullptr;
}

/// Attach one marker with a single projection in `temporal`.
std::int32_t seed_marker(MarkerSequence& scene, std::size_t target,
                         std::size_t temporal,
                         const Vec2& offset = Vec2::Zero()) {
    const std::optional<Vec2> truth = scene.truth(target, temporal);
    if (!truth) {
        return -1;
    }
    Marker marker;
    marker.id = scene.chunk->allocate_id();
    marker.label = label_for(scene.targets[target], static_cast<int>(target));
    marker.type = MarkerType::Fiducial;
    marker.valid = true;
    // The marker's own style lets one run redetect a coded target and a cross
    // without a single task-wide target_type.
    marker.meta.set("target_type",
                    scene.targets[target].cross ? "Cross" : "Circular12");
    marker.projections.push_back(
        {scene.frames[temporal].camera_id, *truth + offset, true});
    scene.chunk->markers().push_back(std::move(marker));
    return scene.chunk->markers().back().id;
}

std::vector<std::int32_t> seed_all(MarkerSequence& scene,
                                   std::size_t temporal) {
    std::vector<std::int32_t> ids;
    for (std::size_t target = 0; target < scene.targets.size(); ++target) {
        ids.push_back(seed_marker(scene, target, temporal));
    }
    return ids;
}

TrackMarkersOptions base_options() {
    TrackMarkersOptions options;
    options.target_type.clear();
    return options;
}

TrackMarkersStats run_track(MarkerSequence& scene,
                            const TrackMarkersOptions& options) {
    TrackMarkersPlan plan =
        plan_track_markers(scene.document, scene.chunk->id(), options);
    const TrackMarkersStats stats = plan.stats;
    if (plan.command) {
        plan.command->apply(scene.document);
    }
    return stats;
}

struct ErrorSummary {
    double rms = 0.0;
    double maximum = 0.0;
    std::size_t count = 0;
};

/// Error of every present projection against the target centre's truth.
ErrorSummary errors_for(const MarkerSequence& scene, const Marker& marker,
                        std::size_t target) {
    double squared = 0.0;
    ErrorSummary summary;
    for (const MarkerProjection& projection : marker.projections) {
        for (std::size_t temporal = 0; temporal < scene.frames.size();
             ++temporal) {
            if (scene.frames[temporal].camera_id != projection.camera_id) {
                continue;
            }
            const std::optional<Vec2> truth = scene.truth(target, temporal);
            if (!truth) {
                continue;
            }
            const double error = (projection.pixel - *truth).norm();
            squared += error * error;
            summary.maximum = std::max(summary.maximum, error);
            ++summary.count;
        }
    }
    if (summary.count > 0) {
        summary.rms =
            std::sqrt(squared / static_cast<double>(summary.count));
    }
    return summary;
}

TEST(Markers, TrackAcrossSequence) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    MarkerSequence scene = make_marker_sequence(spec);
    const std::vector<std::int32_t> ids = seed_all(scene, 0);
    ASSERT_EQ(ids.size(), 4U);
    for (const std::int32_t id : ids) {
        ASSERT_GE(id, 0);
    }
    TrackMarkersOptions options = base_options();
    const TrackMarkersStats stats = run_track(scene, options);
    for (std::size_t target = 0; target < ids.size(); ++target) {
        const Marker* marker = find_marker(*scene.chunk, ids[target]);
        ASSERT_NE(marker, nullptr);
        EXPECT_EQ(marker->projections.size(), static_cast<std::size_t>(kFrames))
            << "target " << target;
        const ErrorSummary summary = errors_for(scene, *marker, target);
        std::printf("[track rms] target %zu labels %s count %zu RMS %.4f max "
                    "%.4f px\n",
                    target, marker->label.c_str(), summary.count, summary.rms,
                    summary.maximum);
        EXPECT_EQ(summary.count, static_cast<std::size_t>(kFrames));
        EXPECT_LE(summary.rms, 0.3) << "target " << target;
        EXPECT_LE(summary.maximum, 1.0) << "target " << target;
    }
    EXPECT_EQ(stats.markers_extended, 4U);
    EXPECT_EQ(stats.markers_unchanged, 0U);
    EXPECT_EQ(stats.markers_skipped, 0U);
    EXPECT_EQ(stats.stopped_markers, 0U);
}

TEST(Markers, TrackOrdersByCaptureTime) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    spec.import_order = {7,  3,  15, 0,  11, 18, 5,  1,  14, 9,
                         2,  19, 6,  12, 4,  17, 8,  10, 16, 13};
    {
        MarkerSequence scene = make_marker_sequence(spec);
        const std::vector<std::int32_t> ids = seed_all(scene, 0);
        const TrackMarkersStats stats = run_track(scene, base_options());
        for (std::size_t target = 0; target < ids.size(); ++target) {
            const Marker* marker = find_marker(*scene.chunk, ids[target]);
            ASSERT_NE(marker, nullptr);
            EXPECT_EQ(marker->projections.size(),
                      static_cast<std::size_t>(kFrames))
                << "shuffled import, target " << target;
            EXPECT_LE(errors_for(scene, *marker, target).rms, 0.3);
        }
        EXPECT_EQ(stats.stopped_markers, 0U);
    }
    // One camera lacks a time, so ordering must fall back to import order.
    // import order is temporal here while the written times are reversed; a
    // time-ordered run would therefore go backwards and fail.
    MarkerSequenceSpec fallback;
    fallback.frames = kFrames;
    fallback.reverse_capture_times = true;
    fallback.missing_time_temporal = 5;
    MarkerSequence scene = make_marker_sequence(fallback);
    const std::vector<std::int32_t> ids = seed_all(scene, 0);
    const TrackMarkersStats stats = run_track(scene, base_options());
    for (std::size_t target = 0; target < ids.size(); ++target) {
        const Marker* marker = find_marker(*scene.chunk, ids[target]);
        ASSERT_NE(marker, nullptr);
        EXPECT_EQ(marker->projections.size(),
                  static_cast<std::size_t>(kFrames))
            << "fallback import order, target " << target;
        const ErrorSummary summary = errors_for(scene, *marker, target);
        EXPECT_LE(summary.rms, 0.3);
        EXPECT_LE(summary.maximum, 1.0);
    }
    EXPECT_EQ(stats.stopped_markers, 0U);
}

TEST(Markers, TrackStopsAtOcclusion) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    spec.occlude_first = 8;
    spec.occlude_last = 10;
    spec.occluded_target = 0;
    MarkerSequence scene = make_marker_sequence(spec);
    const std::vector<std::int32_t> ids = seed_all(scene, 0);
    TrackMarkersOptions options = base_options();
    options.redetect = false;
    options.max_gap = 2;
    const TrackMarkersStats stats = run_track(scene, options);
    const Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    for (std::size_t temporal = 8; temporal < static_cast<std::size_t>(kFrames);
         ++temporal) {
        const std::int32_t camera_id = scene.frames[temporal].camera_id;
        EXPECT_EQ(projection_for(*marker, camera_id), nullptr)
            << "occluded frame " << temporal;
    }
    EXPECT_EQ(marker->projections.size(), 8U);
    EXPECT_GE(stats.misses, 3U);
    EXPECT_EQ(stats.stopped_markers, 1U);
    for (std::size_t target = 1; target < ids.size(); ++target) {
        const Marker* other = find_marker(*scene.chunk, ids[target]);
        ASSERT_NE(other, nullptr);
        EXPECT_EQ(other->projections.size(),
                  static_cast<std::size_t>(kFrames));
    }
}

TEST(Markers, TrackRedetectRecovers) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    spec.occlude_first = 8;
    spec.occlude_last = 10;
    spec.occluded_target = 0;
    MarkerSequence scene = make_marker_sequence(spec);
    const std::vector<std::int32_t> ids = seed_all(scene, 0);
    // The redetection path depends on the decoder, so first prove that the
    // rendered target is decodable at all: same style, same code.
    const FloatImage probe = read_image(scene.frames[1].path).to_luminance();
    const std::vector<ImageDetection> probe_detections =
        detect_targets_in_image(probe, "Circular12");
    bool code_found = false;
    for (const ImageDetection& detection : probe_detections) {
        code_found = code_found ||
                     (detection.coded &&
                      detection.code == scene.targets[0].code);
    }
    std::printf("[track redetect] frame 1 detections %zu:",
                probe_detections.size());
    for (const ImageDetection& detection : probe_detections) {
        std::printf(" code=%d coded=%d r=%.2f", detection.code,
                    detection.coded ? 1 : 0, detection.radius);
    }
    std::printf("; expected code %d, found %d\n", scene.targets[0].code,
                code_found ? 1 : 0);
    const std::vector<ImageDetection> probe_blobs =
        detect_targets_in_image(probe, "Circular");
    std::printf("[track redetect] frame 1 non-coded blobs %zu\n",
                probe_blobs.size());
    ASSERT_TRUE(code_found);
    TrackMarkersOptions options = base_options();
    options.redetect = true;
    // LK cannot meet a 0.001 px forward-backward bound under this noise, so
    // every non-seeded projection in this test comes from redetection. One
    // level keeps that deliberate failure cheap.
    options.max_error_px = 0.001;
    options.pyramid_levels = 1;
    options.window = 15;
    options.max_gap = 3;
    const TrackMarkersStats stats = run_track(scene, options);
    const Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    for (std::size_t temporal = 8; temporal <= 10; ++temporal) {
        EXPECT_EQ(projection_for(*marker, scene.frames[temporal].camera_id),
                  nullptr)
            << "occluded frame " << temporal;
    }
    for (std::size_t temporal = 11;
         temporal < static_cast<std::size_t>(kFrames); ++temporal) {
        EXPECT_NE(projection_for(*marker, scene.frames[temporal].camera_id),
                  nullptr)
            << "recovered frame " << temporal;
    }
    EXPECT_GE(stats.redetected, 1U);
    const std::optional<Vec2> truth = scene.truth(0, 11);
    ASSERT_TRUE(truth.has_value());
    const MarkerProjection* recovered =
        projection_for(*marker, scene.frames[11].camera_id);
    ASSERT_NE(recovered, nullptr);
    std::printf("[track redetect] frame 11 error %.4f px\n",
                (recovered->pixel - *truth).norm());
    EXPECT_LE((recovered->pixel - *truth).norm(), 1.0);
}

TEST(Markers, TrackBackward) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    MarkerSequence scene = make_marker_sequence(spec);
    const std::vector<std::int32_t> ids = seed_all(scene, 10);
    TrackMarkersOptions options = base_options();
    options.direction = "backward";
    const TrackMarkersStats stats = run_track(scene, options);
    for (std::size_t target = 0; target < ids.size(); ++target) {
        const Marker* marker = find_marker(*scene.chunk, ids[target]);
        ASSERT_NE(marker, nullptr);
        EXPECT_EQ(marker->projections.size(), 11U) << "target " << target;
        for (std::size_t temporal = 11;
             temporal < static_cast<std::size_t>(kFrames); ++temporal) {
            EXPECT_EQ(projection_for(*marker,
                                     scene.frames[temporal].camera_id),
                      nullptr)
                << "forward frame " << temporal;
        }
        const ErrorSummary summary = errors_for(scene, *marker, target);
        EXPECT_LE(summary.rms, 0.3);
        EXPECT_LE(summary.maximum, 1.0);
    }
    EXPECT_EQ(stats.markers_extended, 4U);
}

TEST(Markers, TrackBoth) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    MarkerSequence scene = make_marker_sequence(spec);
    const std::vector<std::int32_t> ids = seed_all(scene, 10);
    TrackMarkersOptions options = base_options();
    options.direction = "both";
    const TrackMarkersStats stats = run_track(scene, options);
    for (std::size_t target = 0; target < ids.size(); ++target) {
        const Marker* marker = find_marker(*scene.chunk, ids[target]);
        ASSERT_NE(marker, nullptr);
        EXPECT_EQ(marker->projections.size(),
                  static_cast<std::size_t>(kFrames))
            << "target " << target;
        const ErrorSummary summary = errors_for(scene, *marker, target);
        std::printf("[track both] target %zu RMS %.4f max %.4f px\n", target,
                    summary.rms, summary.maximum);
        EXPECT_LE(summary.rms, 0.3);
        EXPECT_LE(summary.maximum, 1.0);
    }
    EXPECT_EQ(stats.markers_extended, 4U);
}

TEST(Markers, TrackDoesNotOverwriteExisting) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    MarkerSequence scene = make_marker_sequence(spec);
    const std::vector<std::int32_t> ids = seed_all(scene, 0);
    Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    const std::optional<Vec2> truth = scene.truth(0, 1);
    ASSERT_TRUE(truth.has_value());
    const Vec2 existing = *truth + Vec2(2.5, -1.5);
    marker->projections.push_back(
        {scene.frames[1].camera_id, existing, true});
    run_track(scene, base_options());
    marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    EXPECT_EQ(marker->projections.size(), static_cast<std::size_t>(kFrames));
    const MarkerProjection* preserved =
        projection_for(*marker, scene.frames[1].camera_id);
    ASSERT_NE(preserved, nullptr);
    EXPECT_LT((preserved->pixel - existing).norm(), 1e-12);
    // An existing projection re-anchors the run, so the next frame is tracked
    // from the preserved pixel rather than from a fresh LK estimate.
    EXPECT_NE(projection_for(*marker, scene.frames[2].camera_id), nullptr);
}

TEST(Markers, TrackIsUndoable) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    MarkerSequence scene = make_marker_sequence(spec);
    seed_all(scene, 0);
    const std::vector<Marker> original = scene.chunk->markers();
    TrackMarkersPlan plan =
        plan_track_markers(scene.document, scene.chunk->id(), base_options());
    ASSERT_TRUE(plan.command);
    UndoStack stack(&scene.document);
    stack.execute(std::move(plan.command));
    EXPECT_EQ(stack.size(), 1U);
    ASSERT_EQ(stack.command_names().size(), 1U);
    EXPECT_EQ(stack.command_names()[0], "Track Markers");
    EXPECT_FALSE(scene.chunk->markers() == original);
    stack.undo();
    EXPECT_TRUE(scene.chunk->markers() == original);
    stack.redo();
    EXPECT_FALSE(scene.chunk->markers() == original);
}

TEST(Markers, TrackStatsAddUp) {
    MarkerSequenceSpec spec;
    spec.frames = kFrames;
    MarkerSequence scene = make_marker_sequence(spec);
    seed_all(scene, 0);
    Marker empty;
    empty.id = scene.chunk->allocate_id();
    empty.label = "empty";
    empty.type = MarkerType::Fiducial;
    scene.chunk->markers().push_back(std::move(empty));
    TrackMarkersOptions options = base_options();
    options.redetect = false;
    const TrackMarkersStats stats = run_track(scene, options);
    std::printf("[track stats] tracked %zu redetected %zu misses %zu "
                "stopped %zu extended %zu unchanged %zu skipped %zu\n",
                stats.tracked, stats.redetected, stats.misses,
                stats.stopped_markers, stats.markers_extended,
                stats.markers_unchanged, stats.markers_skipped);
    EXPECT_EQ(stats.tracked + stats.redetected, 4U * 19U);
    EXPECT_EQ(stats.redetected, 0U);
    EXPECT_EQ(stats.misses, 0U);
    EXPECT_EQ(stats.stopped_markers, 0U);
    EXPECT_EQ(stats.markers_extended, 4U);
    EXPECT_EQ(stats.markers_unchanged, 0U);
    EXPECT_EQ(stats.markers_skipped, 1U);
    EXPECT_EQ(stats.markers_extended + stats.markers_unchanged +
                  stats.markers_skipped,
              scene.chunk->markers().size());
}

}  // namespace
}  // namespace alphashape
