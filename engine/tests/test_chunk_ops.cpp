// Chunk operations (M11 WP-11.6): duplicate, merge and append with id
// remapping, plus the chunk-subset save.
//
// Every result must pass Chunk::validate(), and every referenced id (camera ->
// sensor, marker -> group, asset -> source asset, mask/depth map/orthomosaic
// label -> camera) must still resolve.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"

#include "base/utils/error.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk_ops.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

using namespace std::chrono;

/// World-space camera centre of a world->camera pose.
Eigen::Vector3d camera_centre(const Camera& camera) {
    const Eigen::Matrix3d rotation = camera.pose.block<3, 3>(0, 0);
    const Eigen::Vector3d translation = camera.pose.block<3, 1>(0, 3);
    return -(rotation.transpose() * translation);
}

/// A rich chunk: every container filled, every reference valid. `seed`
/// shifts the camera poses so two chunks can be told apart after a merge.
void build_rich_chunk(Chunk& chunk, int camera_count, double seed) {
    for (int c = 0; c < camera_count; ++c) {
        if (c == 0) {
            Sensor sensor;
            sensor.label = "sensor";
            sensor.width = 100;
            sensor.height = 80;
            sensor.focal_length_mm = 35.0;
            sensor.make = "ACME";
            sensor.model = "X1";
            sensor.serial = "42";
            sensor.calibration = Calibration::initial(CameraType::Frame, 100, 80, 120.0);
            chunk.sensors().push_back(sensor);
        }
        Camera camera;
        camera.id = c;
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(c);
        camera.enabled = true;
        camera.pose = Transform::Identity();
        camera.pose(0, 3) = seed + c;          // camera centre at -(seed + c)
        camera.pose(2, 3) = 0.25 * c;
        chunk.cameras().push_back(camera);

        auto frame = std::make_shared<Frame>("img" + std::to_string(c) + ".pgm", 100, 80);
        frame->set_id(c);
        frame->set_camera_id(c);
        chunk.frames().push_back(std::move(frame));
    }

    // -- markers, a marker group and a scale bar --------------------------
    MarkerGroup marker_group;
    marker_group.id = chunk.allocate_id();
    marker_group.label = "ground control";
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = "GCP 1";
    marker.group_id = marker_group.id;
    marker.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    for (int c = 0; c < camera_count; ++c) {
        MarkerProjection projection;
        projection.camera_id = c;
        projection.pixel = Eigen::Vector2d(10.0 + c, 20.0);
        marker.projections.push_back(projection);
    }
    marker_group.members = {marker.id};
    chunk.markers().push_back(marker);
    chunk.marker_groups().push_back(marker_group);

    Scalebar bar;
    bar.id = chunk.allocate_id();
    bar.label = "scale";
    bar.from = ScalebarEndpoint{ScalebarEndpoint::Kind::Marker, marker.id};
    bar.to = ScalebarEndpoint{ScalebarEndpoint::Kind::Camera, 0};
    bar.reference_length_m = 1.5;
    chunk.scalebars().push_back(bar);

    CameraGroup camera_group;
    camera_group.id = chunk.allocate_id();
    camera_group.label = "folder";
    camera_group.type = CameraGroupType::Folder;
    for (int c = 0; c < camera_count; ++c) {
        camera_group.members.push_back(c);
    }
    chunk.camera_groups().push_back(camera_group);

    // -- shapes and a trajectory ------------------------------------------
    ShapeGroup shape_group;
    shape_group.id = chunk.allocate_id();
    shape_group.label = "layer group";
    Shape shape;
    shape.id = chunk.allocate_id();
    shape.type = ShapeType::Polygon;
    shape.vertices = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
                      Eigen::Vector3d(0, 1, 0)};
    shape_group.members = {shape.id};
    chunk.shapes().features.push_back(shape);
    chunk.shapes().groups.push_back(shape_group);

    Trajectory trajectory;
    trajectory.id = chunk.allocate_id();
    trajectory.label = "flight";
    trajectory.points.push_back(TrajectoryPoint{});
    chunk.trajectories().push_back(trajectory);

    // -- masks: one per camera --------------------------------------------
    for (int c = 0; c < camera_count; ++c) {
        Mask mask;
        mask.camera_id = c;
        mask.width = 4;
        mask.height = 3;
        mask.data.assign(12, 255);
        chunk.masks().push_back(mask);
    }

    // -- tie points: one keypoint set per camera, one match, one track -----
    for (int c = 0; c < camera_count; ++c) {
        KeypointSet set;
        set.camera_id = c;
        set.keypoints.resize(4);
        set.descriptors.resize(4);
        chunk.tie_points().keypoint_sets.push_back(std::move(set));
    }
    if (camera_count >= 2) {
        PairMatches pair;
        pair.camera_a = 0;
        pair.camera_b = 1;
        pair.matches.push_back(KeypointMatch{0, 1});
        chunk.tie_points().matches.push_back(pair);
        Track track;
        track.observations.push_back(Observation{0, 0});
        track.observations.push_back(Observation{1, 1});
        track.point = Eigen::Vector3d(0.5, 0.5, 1.0);
        chunk.tie_points().tracks.push_back(track);
    }

    // -- three assets: a point cloud, a model linked to it, a DEM ----------
    PointCloudAsset& cloud = chunk.add_point_cloud("cloud");
    cloud.data->points.resize(8);
    const std::int32_t first_cloud_id = cloud.id;
    PointCloudAsset& coarse = chunk.add_point_cloud("coarse cloud");
    coarse.data->points.resize(2);
    // Make the first cloud active again, so a copy's active-id handling is
    // exercised rather than trivially agreeing with "the last one added".
    // (`cloud` is dangling after the push_back above; use the saved id.)
    chunk.set_active_point_cloud_id(first_cloud_id);
    ModelAsset& model = chunk.add_model("model");
    model.data->vertices.resize(4);
    model.data->faces.push_back(Eigen::Vector3i(0, 1, 2));
    model.source_id = first_cloud_id;
    ElevationAsset& elevation = chunk.add_elevation("dem");
    // A real 2x2 grid: an EMPTY elevation cannot carry its source label
    // through the v2 writer (it skips empty rasters), so a payload-less DEM
    // would not round-trip and the subset test would fail for the wrong
    // reason.
    elevation.data->raster.grid.width = 2;
    elevation.data->raster.grid.height = 2;
    elevation.data->raster.grid.pixel_size_x = 1.0;
    elevation.data->raster.grid.pixel_size_y = 1.0;
    elevation.data->raster.channels = 1;
    elevation.data->raster.data.assign(4, 10.0F);
    elevation.data->set_epsg(32644);
    elevation.data->source = "point_cloud";

    DepthMapsAsset& depth = chunk.add_depth_maps("depth");
    DepthMap map;
    map.camera_id = 0;
    map.width = 2;
    map.height = 2;
    map.depths.assign(4, 1.0F);
    depth.data->push_back(map);

    chunk.region().centre = Eigen::Vector3d(seed, 0.0, 0.0);
    chunk.region().size = Eigen::Vector3d(10.0, 10.0, 10.0);
}

/// Assert the chunk is structurally valid and report the first violation.
void expect_valid(const Chunk& chunk, const char* what) {
    const std::vector<std::string> problems = chunk.validate();
    ASSERT_TRUE(problems.empty()) << what << ": " << problems.front();
}

std::vector<std::int32_t> camera_id_set(const Chunk& chunk) {
    std::vector<std::int32_t> ids;
    for (const Camera& camera : chunk.cameras()) {
        ids.push_back(camera.id);
    }
    return ids;
}

std::vector<std::int32_t> local_id_set(const Chunk& chunk) {
    std::vector<std::int32_t> ids;
    for (const auto& slot : chunk.point_clouds()) ids.push_back(slot.id);
    for (const auto& slot : chunk.models()) ids.push_back(slot.id);
    for (const auto& slot : chunk.elevations()) ids.push_back(slot.id);
    for (const auto& slot : chunk.depth_map_sets()) ids.push_back(slot.id);
    for (const Marker& marker : chunk.markers()) ids.push_back(marker.id);
    for (const auto& group : chunk.marker_groups()) ids.push_back(group.id);
    for (const auto& bar : chunk.scalebars()) ids.push_back(bar.id);
    for (const auto& group : chunk.camera_groups()) ids.push_back(group.id);
    for (const Shape& shape : chunk.shapes().features) ids.push_back(shape.id);
    for (const auto& group : chunk.shapes().groups) ids.push_back(group.id);
    for (const auto& trajectory : chunk.trajectories()) ids.push_back(trajectory.id);
    return ids;
}

bool is_sorted_unique(std::vector<std::int32_t> ids) {
    std::sort(ids.begin(), ids.end());
    return std::adjacent_find(ids.begin(), ids.end()) == ids.end();
}

/// Name the first container that differs between two chunks, so a failed
/// round-trip comparison says *what* was lost rather than only that it failed.
std::string first_difference(const Chunk& a, const Chunk& b) {
    if (a.id() != b.id()) return "id";
    if (a.label() != b.label()) return "label";
    if (a.sensors() != b.sensors()) return "sensors";
    if (a.cameras() != b.cameras()) return "cameras";
    if (a.frames().size() != b.frames().size()) return "frame count";
    for (std::size_t i = 0; i < a.frames().size(); ++i) {
        if (!(*a.frames()[i] == *b.frames()[i])) {
            return "frame " + std::to_string(i);
        }
    }
    if (!(a.tie_points() == b.tie_points())) return "tie points";
    if (a.point_clouds() != b.point_clouds()) return "point clouds";
    if (a.models() != b.models()) return "models";
    if (a.elevations() != b.elevations()) return "elevations";
    if (a.orthomosaics() != b.orthomosaics()) return "orthomosaics";
    if (a.depth_map_sets() != b.depth_map_sets()) return "depth maps";
    if (a.next_id() != b.next_id()) return "next_id";
    if (a.markers() != b.markers()) return "markers";
    if (a.marker_groups() != b.marker_groups()) return "marker groups";
    if (a.scalebars() != b.scalebars()) return "scalebars";
    if (a.camera_groups() != b.camera_groups()) return "camera groups";
    if (!(a.masks() == b.masks())) return "masks";
    if (!(a.shapes() == b.shapes())) return "shapes";
    if (!(a.trajectories() == b.trajectories())) return "trajectories";
    if (!(a.region() == b.region())) return "region";
    if (!(a.meta() == b.meta())) return "meta";
    if (!(a.crs() == b.crs())) return "crs";
    if (!(a.transform() == b.transform())) return "transform";
    if (a.active_point_cloud_id() != b.active_point_cloud_id()) return "active point cloud";
    if (a.active_model_id() != b.active_model_id()) return "active model";
    if (a.active_elevation_id() != b.active_elevation_id()) return "active elevation";
    if (a.active_depth_maps_id() != b.active_depth_maps_id()) return "active depth maps";
    return {};
}

}  // namespace

TEST(ChunkOps, DuplicateIsDeepAndIndependent) {
    Chunk source("source");
    source.set_id(5);
    build_rich_chunk(source, 3, 0.0);
    const Chunk before = source;

    Chunk copy = duplicate_chunk(source, 9, "source copy");
    expect_valid(copy, "duplicate");
    EXPECT_EQ(copy.id(), 9);
    EXPECT_EQ(copy.label(), "source copy");

    // Mutate every payload the copy could have shared, then compare the source
    // against its pre-copy snapshot.
    ASSERT_EQ(copy.cameras().size(), 3U);
    copy.cameras()[0].label = "mutated";
    copy.cameras()[0].pose(0, 3) = 99.0;
    copy.point_clouds()[0].data->points.clear();
    copy.models()[0].data->vertices.clear();
    copy.sensors()[0].focal_length_mm = 1.0;
    copy.markers()[0].label = "changed";
    copy.depth_map_sets()[0].data->clear();
    copy.tie_points().keypoint_sets[0].keypoints.clear();
    copy.masks()[0].data.assign(12, 0);

    EXPECT_TRUE(source == before) << "mutating the copy changed the source";
}

TEST(ChunkOps, DuplicateRemapsAllIds) {
    Chunk source("source");
    source.set_id(0);
    build_rich_chunk(source, 12, 0.0);
    expect_valid(source, "source");

    // The acceptance timing bound: a 12-camera / 3-asset chunk duplicates in
    // under 100 ms.
    const auto started = steady_clock::now();
    Chunk copy = duplicate_chunk(source, 1, "copy");
    const auto elapsed = steady_clock::now() - started;
    std::cout << "[duplicate timing] 12 cameras / 4 assets (+ tie points, masks, "
                 "markers) = "
              << duration_cast<microseconds>(elapsed).count() << " us\n";
    EXPECT_LT(duration_cast<milliseconds>(elapsed).count(), 100)
        << "duplicate took " << duration_cast<milliseconds>(elapsed).count() << " ms";

    expect_valid(copy, "copy");
    EXPECT_TRUE(is_sorted_unique(camera_id_set(copy)));
    EXPECT_TRUE(is_sorted_unique(local_id_set(copy)));

    // Every camera points at a sensor the copy owns; every frame at a copied
    // camera; every marker projection, mask and depth map at a copied camera.
    ASSERT_EQ(copy.sensors().size(), 1U);
    for (const Camera& camera : copy.cameras()) {
        ASSERT_GE(camera.sensor_id, 0);
        ASSERT_LT(static_cast<std::size_t>(camera.sensor_id), copy.sensors().size());
    }
    for (const auto& frame : copy.frames()) {
        EXPECT_TRUE(std::any_of(copy.cameras().begin(), copy.cameras().end(),
                            [&](const Camera& c) {
                                return c.id == frame->camera_id();
                            }));
    }
    for (const Marker& marker : copy.markers()) {
        for (const MarkerProjection& projection : marker.projections) {
            EXPECT_TRUE(std::any_of(copy.cameras().begin(), copy.cameras().end(),
                                    [&](const Camera& c) {
                                        return c.id == projection.camera_id;
                                    }));
        }
    }
    for (const Mask& mask : copy.masks()) {
        EXPECT_TRUE(std::any_of(copy.cameras().begin(), copy.cameras().end(),
                                [&](const Camera& c) {
                                    return c.id == mask.camera_id;
                                }));
    }
    for (const DepthMapsAsset& set : copy.depth_map_sets()) {
        for (const DepthMap& map : *set.data) {
            EXPECT_TRUE(std::any_of(copy.cameras().begin(), copy.cameras().end(),
                                    [&](const Camera& c) {
                                        return c.id == map.camera_id;
                                    }));
        }
    }
    // The active asset stays the same one, not merely "the last copied".
    EXPECT_EQ(copy.active_point_cloud_id(), copy.point_clouds()[0].id);
    EXPECT_NE(copy.active_point_cloud_id(), copy.point_clouds()[1].id);

    // Model -> source asset survives the remap and names a copied asset.
    ASSERT_EQ(copy.models().size(), 1U);
    const std::int64_t linked = copy.models()[0].source_id;
    EXPECT_EQ(linked, static_cast<std::int64_t>(copy.point_clouds()[0].id));

    // Tie points: keypoint sets and track observations use copied camera ids.
    for (const KeypointSet& set : copy.tie_points().keypoint_sets) {
        EXPECT_TRUE(std::any_of(copy.cameras().begin(), copy.cameras().end(),
                                [&](const Camera& c) {
                                    return c.id == set.camera_id;
                                }));
    }
    for (const Track& track : copy.tie_points().tracks) {
        for (const Observation& observation : track.observations) {
            EXPECT_TRUE(std::any_of(copy.cameras().begin(), copy.cameras().end(),
                                    [&](const Camera& c) {
                                        return c.id == observation.camera_id;
                                    }));
        }
    }
}

TEST(ChunkOps, MergeConcatenatesAndKeepsPositions) {
    Chunk a("a");
    a.set_id(0);
    build_rich_chunk(a, 3, 0.0);
    Chunk b("b");
    b.set_id(1);
    build_rich_chunk(b, 2, 100.0);
    expect_valid(a, "a");
    expect_valid(b, "b");

    Chunk merged = merge_chunks({&a, &b}, 7, "Merged chunk");
    expect_valid(merged, "merged");
    EXPECT_EQ(merged.id(), 7);
    EXPECT_EQ(merged.label(), "Merged chunk");
    ASSERT_EQ(merged.cameras().size(), 5U);
    ASSERT_EQ(merged.sensors().size(), 1U);  // identical identities dedup

    // Every source camera has a counterpart at the same world position.
    for (const Chunk* source : {&a, &b}) {
        for (const Camera& original : source->cameras()) {
            const Eigen::Vector3d want = camera_centre(original);
            bool found = false;
            for (const Camera& camera : merged.cameras()) {
                if (camera.label != original.label ||
                    camera.sensor_id != 0) {
                    continue;
                }
                if ((camera_centre(camera) - want).norm() <= 1e-12) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "camera " << original.label << " moved in the merge";
        }
    }

    // Camera and chunk-local id sets of the two sources are disjoint.
    const std::vector<std::int32_t> first_cameras = camera_id_set(a);
    const std::vector<std::int32_t> all_cameras = camera_id_set(merged);
    EXPECT_TRUE(is_sorted_unique(all_cameras));
    const std::vector<std::int32_t> local = local_id_set(merged);
    EXPECT_TRUE(is_sorted_unique(local));
    EXPECT_GT(*std::max_element(all_cameras.begin(), all_cameras.end()),
              *std::max_element(first_cameras.begin(), first_cameras.end()));

    // Regions union, so the merged box spans both sources.
    EXPECT_GT(merged.region().size.x(), 100.0);

    // Merged id sets are disjoint: the markers contributed by `a` and by `b`
    // occupy different id ranges (a's objects come first, in source order).
    std::vector<std::int32_t> merged_markers;
    for (const Marker& marker : merged.markers()) {
        merged_markers.push_back(marker.id);
    }
    ASSERT_EQ(merged_markers.size(), a.markers().size() + b.markers().size());
    std::vector<std::int32_t> a_ids, b_ids;
    for (std::size_t i = 0; i < merged_markers.size(); ++i) {
        (i < a.markers().size() ? a_ids : b_ids).push_back(merged_markers[i]);
    }
    for (const std::int32_t id : a_ids) {
        EXPECT_EQ(std::find(b_ids.begin(), b_ids.end(), id), b_ids.end());
    }
    EXPECT_TRUE(is_sorted_unique(a_ids));
    EXPECT_TRUE(is_sorted_unique(b_ids));}

TEST(ChunkOps, AppendIntoPopulatedChunkGrowsIdsAndDedupsSensors) {
    Chunk target("target");
    target.set_id(0);
    build_rich_chunk(target, 2, 0.0);
    Chunk incoming("incoming");
    incoming.set_id(1);
    build_rich_chunk(incoming, 3, 10.0);

    const std::int32_t next_local_before = target.next_id();
    const std::int32_t next_camera_before =
        target.cameras().back().id + 1;
    append_chunk(target, incoming);

    expect_valid(target, "appended");
    EXPECT_EQ(target.cameras().size(), 5U);
    // Exact sensor identity -> one sensor serves every camera.
    ASSERT_EQ(target.sensors().size(), 1U);
    for (const Camera& camera : target.cameras()) {
        EXPECT_EQ(camera.sensor_id, 0);
    }
    // New cameras continue after the target's highest camera id.
    EXPECT_TRUE(is_sorted_unique(camera_id_set(target)));
    EXPECT_GT(target.cameras().back().id, next_camera_before - 1);
    // Chunk-local ids only grow.
    EXPECT_GE(target.next_id(), next_local_before);
    EXPECT_TRUE(is_sorted_unique(local_id_set(target)));
    // The incoming world positions survive at their remapped ids.
    for (const Camera& original : incoming.cameras()) {
        const Eigen::Vector3d want = camera_centre(original);
        EXPECT_TRUE(std::any_of(target.cameras().begin(), target.cameras().end(),
                                [&](const Camera& camera) {
                                    return (camera_centre(camera) - want).norm() <= 1e-12;
                                }));
    }
}

TEST(ChunkOps, MergeRejectsMixedCrs) {
    Chunk local("local");
    local.set_id(0);
    build_rich_chunk(local, 2, 0.0);
    Chunk referenced("referenced");
    referenced.set_id(1);
    build_rich_chunk(referenced, 2, 0.0);
    referenced.crs().epsg = 32644;
    referenced.crs().wkt = "EPSG:32644";
    Chunk other("other");
    other.set_id(2);
    build_rich_chunk(other, 2, 0.0);
    other.crs().epsg = 32645;
    other.crs().wkt = "EPSG:32645";

    // Local + referenced is allowed; two different non-local CRSs are not.
    EXPECT_TRUE(crs_compatible(local, referenced));
    EXPECT_FALSE(crs_compatible(referenced, other));
    EXPECT_THROW(merge_chunks({&referenced, &other}, 3, "mixed"), ValueError);
    EXPECT_NO_THROW(merge_chunks({&local, &referenced}, 3, "ok"));

    // Mixed CRS through the task: the document is left untouched.
    Document document;
    Chunk& first = document.workspace().add_chunk("first");
    build_rich_chunk(first, 2, 0.0);
    first.crs().epsg = 32644;
    Chunk& second = document.workspace().add_chunk("second");
    build_rich_chunk(second, 2, 0.0);
    second.crs().epsg = 32645;
    document.set_active_chunk_id(first.id());

    Params params = make_params(builtin_registry().require("MergeChunks"));
    params.set("chunk_ids", ParamValue{std::vector<std::string>{
                                std::to_string(first.id()), std::to_string(second.id())}});
    EXPECT_THROW(run_task(builtin_registry(), document, "MergeChunks", params), ValueError);
    EXPECT_EQ(document.workspace().chunks().size(), 2U);
}

TEST(ChunkOps, MergeThroughTaskProducesDisjointIds) {
    Document document;
    Chunk& a = document.workspace().add_chunk("a");
    build_rich_chunk(a, 3, 0.0);
    Chunk& b = document.workspace().add_chunk("b");
    build_rich_chunk(b, 2, 50.0);
    document.set_active_chunk_id(a.id());
    const std::int32_t next_before = document.workspace().next_chunk_id();

    Params params = make_params(builtin_registry().require("MergeChunks"));
    params.set("chunk_ids",
               ParamValue{std::vector<std::string>{std::to_string(a.id()),
                                                  std::to_string(b.id())}});
    run_task(builtin_registry(), document, "MergeChunks", params);

    ASSERT_EQ(document.workspace().chunks().size(), 3U);
    Chunk& merged = *document.workspace().chunks().back();
    EXPECT_EQ(merged.id(), next_before);
    EXPECT_EQ(document.active_chunk_id(), merged.id());
    expect_valid(merged, "merged");
    EXPECT_EQ(merged.cameras().size(), 5U);
    EXPECT_TRUE(is_sorted_unique(local_id_set(merged)));
    // The two sources keep their own ids (merge builds a new chunk).
    EXPECT_TRUE(is_sorted_unique(local_id_set(a)));
    EXPECT_TRUE(is_sorted_unique(local_id_set(b)));
    // run_task records the effective parameters into the chunk that was active
    // when the task started (the first source), not the new chunk.
    EXPECT_NE(a.meta().find("MergeChunks/label"), nullptr);
}

TEST(ChunkOps, DuplicateThroughTaskUsesDefaults) {
    Document document;
    Chunk& source = document.workspace().add_chunk("Block 1");
    build_rich_chunk(source, 4, 0.0);
    document.set_active_chunk_id(source.id());

    Params params = make_params(builtin_registry().require("DuplicateChunk"));
    run_task(builtin_registry(), document, "DuplicateChunk", params);

    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    const Chunk& copy = *document.workspace().chunks().back();
    EXPECT_EQ(copy.label(), "Block 1 copy");
    EXPECT_EQ(copy.cameras().size(), 4U);
    EXPECT_EQ(document.active_chunk_id(), copy.id());
    expect_valid(copy, "copy");
}

TEST(ChunkOps, DuplicateHonoursCopyFlags) {
    Chunk source("source");
    build_rich_chunk(source, 3, 0.0);

    ChunkCopyOptions options;
    options.copy_tie_points = false;
    Chunk copy = duplicate_chunk(source, 1, "copy", options);
    EXPECT_TRUE(copy.tie_points().keypoint_sets.empty());
    EXPECT_TRUE(copy.tie_points().matches.empty());
    EXPECT_TRUE(copy.tie_points().tracks.empty());
    EXPECT_EQ(copy.cameras().size(), 3U);
    EXPECT_EQ(copy.point_clouds().size(), 2U);

    options = ChunkCopyOptions{};
    options.copy_assets = false;
    copy = duplicate_chunk(source, 1, "copy", options);
    EXPECT_TRUE(copy.point_clouds().empty());
    EXPECT_TRUE(copy.models().empty());
    EXPECT_TRUE(copy.elevations().empty());
    EXPECT_EQ(copy.active_point_cloud_id(), -1);
    EXPECT_FALSE(copy.tie_points().tracks.empty());
    expect_valid(copy, "copy without assets");

    options = ChunkCopyOptions{};
    options.copy_depth_maps = false;
    copy = duplicate_chunk(source, 1, "copy", options);
    EXPECT_TRUE(copy.depth_map_sets().empty());
    EXPECT_EQ(copy.active_depth_maps_id(), -1);
    expect_valid(copy, "copy without depth maps");
}

TEST(ChunkOps, DuplicateIsUndoable) {
    Document document;
    Chunk& source = document.workspace().add_chunk("source");
    build_rich_chunk(source, 2, 0.0);
    document.set_active_chunk_id(source.id());
    const Chunk original = source;
    const std::int32_t next_before = document.workspace().next_chunk_id();

    UndoStack stack(&document);
    stack.execute(make_duplicate_chunk_command(source.id(), "source copy"));
    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    const Chunk& copy = *document.workspace().chunks().back();
    expect_valid(copy, "copy");
    EXPECT_EQ(copy.cameras().size(), 2U);
    EXPECT_TRUE(copy.point_clouds()[0].data->points.size() == 8U);

    stack.undo();
    ASSERT_EQ(document.workspace().chunks().size(), 1U);
    EXPECT_EQ(document.workspace().next_chunk_id(), next_before);
    EXPECT_TRUE(document.workspace().chunks()[0]->id() == source.id());
    EXPECT_TRUE(*document.workspace().chunks()[0] == original);

    stack.redo();
    ASSERT_EQ(document.workspace().chunks().size(), 2U);
    EXPECT_EQ(document.workspace().chunks()[1]->label(), "source copy");
    expect_valid(*document.workspace().chunks()[1], "redone copy");
}

TEST(ChunkOps, SaveChunksSubsetRoundTrip) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        ("alphashape_chunk_subset_" + std::to_string(platform::process_id()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "subset.psx").string();

    Document document;
    Chunk& first = document.workspace().add_chunk("first");
    build_rich_chunk(first, 2, 0.0);
    Chunk& second = document.workspace().add_chunk("second");
    build_rich_chunk(second, 3, 10.0);
    Chunk& third = document.workspace().add_chunk("third");
    build_rich_chunk(third, 1, 20.0);
    document.set_active_chunk_id(second.id());
    const std::int32_t next_before = document.workspace().next_chunk_id();

    // Save only the second chunk: ids and the chunk counter must be preserved,
    // so a later append cannot collide.
    document.save(path, {second.id()});
    ASSERT_TRUE(std::filesystem::exists(path));

    std::shared_ptr<Document> loaded = ProjectIO::load(path);
    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks()[0];
    EXPECT_EQ(got.id(), second.id());
    EXPECT_TRUE(got == second)
        << "the subset chunk did not round-trip; first difference: "
        << first_difference(got, second);
    EXPECT_EQ(loaded->workspace().next_chunk_id(), next_before);
    EXPECT_EQ(loaded->active_chunk_id(), second.id());

    // An id the document does not hold is refused.
    EXPECT_THROW(document.save(path, {999}), ValueError);
    // The full save still writes every chunk.
    document.save(path);
    loaded = ProjectIO::load(path);
    EXPECT_EQ(loaded->workspace().chunks().size(), 3U);

    std::filesystem::remove_all(dir);
}

TEST(ChunkOps, AppendOffsetsIds) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        ("alphashape_chunk_append_" + std::to_string(platform::process_id()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string source_path = (dir / "source.psx").string();

    Document source;
    Chunk& incoming = source.workspace().add_chunk("incoming");
    build_rich_chunk(incoming, 2, 0.0);
    const std::int32_t incoming_id = incoming.id();
    const Chunk incoming_before = incoming;
    source.save(source_path);

    Document destination;
    Chunk& held = destination.workspace().add_chunk("held");
    build_rich_chunk(held, 2, 100.0);
    destination.set_active_chunk_id(held.id());
    const std::int32_t offset = destination.workspace().next_chunk_id();

    Params params = make_params(builtin_registry().require("AppendProject"));
    params.set("path", ParamValue{source_path});
    run_task(builtin_registry(), destination, "AppendProject", params);

    ASSERT_EQ(destination.workspace().chunks().size(), 2U);
    const Chunk& appended = *destination.workspace().chunks().back();
    EXPECT_EQ(appended.id(), offset + incoming_id);
    EXPECT_EQ(destination.active_chunk_id(), appended.id());
    EXPECT_EQ(appended.cameras().size(), 2U);
    expect_valid(appended, "appended");
    // Chunk-local ids are preserved inside their chunk.
    EXPECT_EQ(appended.next_id(), incoming_before.next_id());
    EXPECT_TRUE(is_sorted_unique(local_id_set(appended)));
    EXPECT_EQ(destination.workspace().next_chunk_id(), offset + incoming_id + 1);

    // Selecting a chunk id the source does not hold fails cleanly.
    Params bad = make_params(builtin_registry().require("AppendProject"));
    bad.set("path", ParamValue{source_path});
    bad.set("chunk_ids", ParamValue{std::vector<std::string>{"99"}});
    EXPECT_THROW(run_task(builtin_registry(), destination, "AppendProject", bad),
                 ValueError);
    EXPECT_EQ(destination.workspace().chunks().size(), 2U);

    std::filesystem::remove_all(dir);
}

}  // namespace alphashape

