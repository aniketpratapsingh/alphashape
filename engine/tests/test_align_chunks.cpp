// Marker-based chunk alignment (M11 §4.8, WP-11.7): AlignChunks.
//
// The scene is the aerial fixture (12 truth-posed cameras in a local ENU frame).
// Two chunks stand for the same site in different frames: the ACTIVE chunk in
// the truth frame, the REFERENCE chunk in the truth frame mapped through a
// known similarity S = (s = 1.7, R = 15 deg about z, t = (100, -40, 2)).
// Markers with matching labels -- but independent ids, as Metashape semantics
// require -- sit at the same physical points in both.
//
// What is pinned:
//   * the estimated similarity is S itself (scale/rotation/translation bounds);
//   * applying it lands the active chunk's camera centres on the reference's;
//   * fewer than three common markers, or collinear ones, are refused;
//   * one gross outlier is rejected by RANSAC and does not bias the fit;
//   * method=reference is an identity on two chunks with the same CRS;
//   * the applied 4x4 is recorded in both chunks' meta, and the geometry edit is
//     undoable through the command layer (apply -> revert restores it exactly).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/chunk/align_chunks_task.hpp"
#include "common/algorithm/sfm/similarity.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/chunk_geometry.hpp"
#include "common/core/chunk_ops.hpp"
#include "common/core/document.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "aerial_fixture.hpp"

namespace alphashape {
namespace {

/// The known similarity every marker-based case recovers.
sfm::Similarity known_similarity() {
    sfm::Similarity similarity;
    similarity.scale = 1.7;
    similarity.R =
        Eigen::AngleAxisd(15.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    similarity.t = Eigen::Vector3d(100.0, -40.0, 2.0);
    return similarity;
}

/// Six well-spread, non-collinear physical points in the truth (ENU) frame.
std::vector<Eigen::Vector3d> marker_points() {
    return {Eigen::Vector3d(0.0, 0.0, 0.0),   Eigen::Vector3d(30.0, 0.0, 5.0),
            Eigen::Vector3d(0.0, 25.0, 10.0), Eigen::Vector3d(20.0, 20.0, 3.0),
            Eigen::Vector3d(-25.0, 10.0, -4.0), Eigen::Vector3d(-10.0, -20.0, 7.0)};
}

void add_marker(Chunk& chunk, const std::string& label, const Eigen::Vector3d& position) {
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = label;
    marker.position = position;
    marker.valid = true;
    chunk.markers().push_back(std::move(marker));
}

/// Camera centre of a world->camera pose: the world point with X_cam = 0, i.e.
/// -A^-1 b for pose = [A | b]. The general inverse (not -A^T b) because a
/// scaled chunk's poses carry the 1/s factor of the applied similarity.
Eigen::Vector3d camera_centre(const Camera& camera) {
    const Eigen::Matrix3d rotation = camera.pose.block<3, 3>(0, 0);
    const Eigen::Vector3d translation = camera.pose.block<3, 1>(0, 3);
    return -(rotation.inverse() * translation);
}

/// Two chunks of the aerial block with matching markers: `active` in the truth
/// frame, `reference` in the truth frame mapped by `similarity`.
struct AlignmentScene {
    Document document;
    Chunk* active = nullptr;
    Chunk* reference = nullptr;
};

AlignmentScene make_scene(const sfm::Similarity& similarity,
                          const std::vector<Eigen::Vector3d>& active_markers) {
    AlignmentScene scene;
    scene.document = aerial_fixture::make_truth_chunk_document();
    Chunk* truth = scene.document.workspace().chunks().front().get();

    const std::int32_t new_id = scene.document.workspace().next_chunk_id();
    auto active = std::make_shared<Chunk>(duplicate_chunk(*truth, new_id, "Active"));
    scene.document.workspace().chunks().push_back(active);
    scene.document.workspace().set_next_chunk_id(new_id + 1);
    scene.document.set_active_chunk_id(new_id);
    scene.active = active.get();
    scene.reference = truth;

    for (std::size_t i = 0; i < active_markers.size(); ++i) {
        const std::string label = "M" + std::to_string(i + 1);
        add_marker(*scene.active, label, active_markers[i]);
        add_marker(*scene.reference, label, active_markers[i]);
    }

    // Move the reference chunk into the other frame. Its marker positions map
    // to S(p), which is what the estimator must recover from.
    ChunkGeometry geometry = capture_geometry(*scene.reference);
    apply_similarity_to_geometry(geometry, similarity);
    apply_geometry(*scene.reference, geometry);
    return scene;
}

/// Run AlignChunks through the registry exactly as a front end would.
void run_align(AlignmentScene& scene, const std::string& method,
               std::int64_t min_common = 3, double accuracy = 0.0) {
    Params params = make_params(builtin_registry().require("AlignChunks"));
    params.set("reference_chunk_id",
               ParamValue{static_cast<std::int64_t>(scene.reference->id())});
    params.set("method", ParamValue{method});
    params.set("min_common_markers", ParamValue{min_common});
    params.set("accuracy_m", ParamValue{accuracy});
    run_task(builtin_registry(), scene.document, "AlignChunks", params);
}

// -- the estimate ------------------------------------------------------------

TEST(AlignChunks, RecoversKnownSimilarity) {
    const sfm::Similarity expected = known_similarity();
    AlignmentScene scene = make_scene(expected, marker_points());

    const MarkerAlignmentEstimate estimate =
        estimate_marker_alignment(*scene.active, *scene.reference);
    ASSERT_TRUE(estimate.ok) << estimate.reason;
    EXPECT_EQ(estimate.markers_used, marker_points().size());

    const sfm::Similarity& fit = estimate.similarity;
    EXPECT_NEAR(fit.scale / expected.scale - 1.0, 0.0, 1e-6);
    const Eigen::Matrix3d delta = fit.R.transpose() * expected.R;
    const double angle = Eigen::AngleAxisd(delta).angle();
    EXPECT_LE(std::abs(angle), 1e-6);
    EXPECT_LE((fit.t - expected.t).norm(), 1e-5);
    EXPECT_LE(estimate.rms_m, 1e-6);
}

TEST(AlignChunks, MovesCameraCentresOntoTheReference) {
    const sfm::Similarity expected = known_similarity();
    AlignmentScene scene = make_scene(expected, marker_points());
    const std::vector<Eigen::Vector3d> reference_centres = [&scene] {
        std::vector<Eigen::Vector3d> centres;
        for (const Camera& camera : scene.reference->cameras()) {
            centres.push_back(camera_centre(camera));
        }
        return centres;
    }();

    run_align(scene, "markers");

    ASSERT_EQ(scene.active->cameras().size(), reference_centres.size());
    for (std::size_t i = 0; i < reference_centres.size(); ++i) {
        EXPECT_LE((camera_centre(scene.active->cameras()[i]) - reference_centres[i]).norm(),
                  1e-5)
            << "camera " << i;
    }
    // Both chunks record the applied matrix under the other chunk's id.
    const std::string* active_record =
        scene.active->meta().find("AlignChunks/" + std::to_string(scene.reference->id()) +
                                  "/matrix");
    const std::string* reference_record =
        scene.reference->meta().find("AlignChunks/" + std::to_string(scene.active->id()) +
                                     "/matrix");
    ASSERT_NE(active_record, nullptr);
    ASSERT_NE(reference_record, nullptr);
    // Row-major 16 values: the active chunk records the applied similarity,
    // the reference records its inverse.
    std::istringstream applied(*active_record);
    std::vector<double> values;
    double value = 0.0;
    while (applied >> value) {
        values.push_back(value);
    }
    ASSERT_EQ(values.size(), 16U);
    Eigen::Matrix4d matrix;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            matrix(row, col) = values[static_cast<std::size_t>(row * 4 + col)];
        }
    }
    EXPECT_LE((matrix - expected.matrix()).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(AlignChunks, RequiresThreeMarkers) {
    const sfm::Similarity expected = known_similarity();
    const std::vector<Eigen::Vector3d> all = marker_points();
    const std::vector<Eigen::Vector3d> two(all.begin(), all.begin() + 2);
    AlignmentScene scene = make_scene(expected, two);
    std::vector<Transform> before;
    for (const Camera& camera : scene.active->cameras()) {
        before.push_back(camera.pose);
    }

    try {
        run_align(scene, "markers");
        FAIL() << "expected a refusal";
    } catch (const Error& error) {
        EXPECT_EQ(std::string(error.what()).rfind("align_chunks: need at least 3 common "
                                                  "markers", 0),
                  0U)
            << error.what();
    }
    // The refused run left the active chunk exactly where it was.
    ASSERT_EQ(scene.active->cameras().size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(scene.active->cameras()[i].pose, before[i]) << "camera " << i;
    }
}

TEST(AlignChunks, IgnoresDisagreeingMarker) {
    const sfm::Similarity expected = known_similarity();
    std::vector<Eigen::Vector3d> points = marker_points();
    AlignmentScene scene = make_scene(expected, points);

    // One gross outlier: the same label 50 m away in the reference frame.
    add_marker(*scene.active, "OUTLIER", Eigen::Vector3d(5.0, 5.0, 5.0));
    add_marker(*scene.reference, "OUTLIER",
               expected.apply(Eigen::Vector3d(5.0, 5.0, 5.0)) +
                   Eigen::Vector3d(50.0, 0.0, 0.0));

    const MarkerAlignmentEstimate estimate =
        estimate_marker_alignment(*scene.active, *scene.reference);
    ASSERT_TRUE(estimate.ok) << estimate.reason;
    EXPECT_EQ(estimate.markers_matched, points.size() + 1);
    EXPECT_EQ(estimate.markers_used, points.size());
    EXPECT_LE(estimate.rms_m, 1e-3);
    EXPECT_NEAR(estimate.similarity.scale / expected.scale - 1.0, 0.0, 1e-6);
    EXPECT_LE((estimate.similarity.t - expected.t).norm(), 1e-5);
}

TEST(AlignChunks, FailsOnCollinearMarkers) {
    const sfm::Similarity expected = known_similarity();
    std::vector<Eigen::Vector3d> collinear;
    for (int i = 0; i < 5; ++i) {
        const double step = static_cast<double>(i);
        collinear.emplace_back(step, 2.0 * step, 0.5 * step);
    }
    AlignmentScene scene = make_scene(expected, collinear);

    try {
        run_align(scene, "markers");
        FAIL() << "expected a refusal";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("collinear"), std::string::npos)
            << error.what();
    }
}

TEST(AlignChunks, ReferenceMethodNoopWhenSameCrs) {
    const sfm::Similarity expected = known_similarity();
    AlignmentScene scene = make_scene(expected, marker_points());

    // Same CRS on both sides (the fixture's own is 4326, and the similarity
    // leaves the CRS descriptor untouched).
    ASSERT_EQ(scene.active->crs().epsg, scene.reference->crs().epsg);
    std::vector<Transform> before;
    for (const Camera& camera : scene.active->cameras()) {
        before.push_back(camera.pose);
    }

    run_align(scene, "reference");

    ASSERT_EQ(scene.active->cameras().size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(scene.active->cameras()[i].pose, before[i]) << "camera " << i;
    }
    EXPECT_TRUE(scene.active->meta().find(
                    "AlignChunks/" + std::to_string(scene.reference->id()) + "/matrix") !=
                nullptr);
}

TEST(AlignChunks, ReferenceMethodRequiresGeoreferencedChunks) {
    const sfm::Similarity expected = known_similarity();
    AlignmentScene scene = make_scene(expected, marker_points());
    scene.active->crs() = CRS{};

    EXPECT_THROW(run_align(scene, "reference"), Error);
}

// -- the undoable edit -------------------------------------------------------

TEST(AlignChunks, GeometryEditIsUndoable) {
    const sfm::Similarity expected = known_similarity();
    AlignmentScene scene = make_scene(expected, marker_points());

    const ChunkGeometry before = capture_geometry(*scene.active);
    ChunkGeometry after = capture_geometry(*scene.active);
    apply_similarity_to_geometry(after, expected);

    auto command = make_align_chunks_command(scene.active->id(), std::move(after));
    command->apply(scene.document);

    const ChunkGeometry applied = capture_geometry(*scene.active);
    for (std::size_t i = 0; i < applied.camera_poses.size(); ++i) {
        EXPECT_LE((applied.camera_poses[i] - scene.reference->cameras()[i].pose)
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-12)
            << "camera " << i;
    }

    command->revert(scene.document);
    const ChunkGeometry restored = capture_geometry(*scene.active);
    EXPECT_EQ(restored.transform, before.transform);
    EXPECT_EQ(restored.region, before.region);
    EXPECT_EQ(restored.camera_poses.size(), before.camera_poses.size());
    for (std::size_t i = 0; i < before.camera_poses.size(); ++i) {
        EXPECT_EQ(restored.camera_poses[i], before.camera_poses[i]) << "camera " << i;
    }
    for (std::size_t i = 0; i < before.marker_positions.size(); ++i) {
        ASSERT_EQ(restored.marker_positions[i].has_value(),
                  before.marker_positions[i].has_value());
        if (before.marker_positions[i].has_value()) {
            EXPECT_EQ(*restored.marker_positions[i], *before.marker_positions[i]);
        }
    }
    EXPECT_TRUE(scene.active->meta().find(
                    "AlignChunks/" + std::to_string(scene.reference->id()) + "/matrix") ==
                nullptr);
}

// -- payload ownership -------------------------------------------------------

TEST(AlignChunks, PayloadsMoveAndRestoreWithoutAliasing) {
    AlignmentScene scene = make_scene(known_similarity(), marker_points());
    scene.active->add_point_cloud("cloud");
    scene.reference->add_point_cloud("cloud");
    const std::vector<Eigen::Vector3d> original{
        Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector3d(4.0, -5.0, 6.0)};
    scene.active->point_cloud().points = original;
    scene.reference->point_cloud().points = original;

    const ChunkGeometry before = capture_geometry(*scene.active);
    ChunkGeometry after = capture_geometry(*scene.active);
    apply_similarity_to_geometry(after, known_similarity());

    auto command = make_align_chunks_command(scene.active->id(), std::move(after));
    command->apply(scene.document);

    // The chunk's points followed the similarity, and the snapshot it was
    // captured from (the reference's state) was not mutated through aliasing.
    const sfm::Similarity expected = known_similarity();
    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_LE((scene.active->point_cloud().points[i] - expected.apply(original[i]))
                      .norm(),
                  1e-12);
        EXPECT_EQ(scene.reference->point_cloud().points[i], original[i]);
    }

    command->revert(scene.document);
    ASSERT_EQ(scene.active->point_cloud().points.size(), original.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(scene.active->point_cloud().points[i], original[i]);
    }
}

}  // namespace
}  // namespace alphashape
