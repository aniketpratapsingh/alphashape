// Tests for the camera-group, camera-selection and frame-statistics surfaces
// (WP-11.8).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/photos/camera_groups.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/frame_stats.hpp"
#include "common/io/project_io.hpp"
#include "model/camera/calibration.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "model/crs/coordinate_system.hpp"
#include "model/crs/local_frame.hpp"
#include "model/crs/transform.hpp"
#endif

namespace alphashape {
namespace {

namespace fs = std::filesystem;

std::int32_t add_chunk(Document& document) {
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return chunk.id();
}

Chunk& chunk_of(Document& document, std::int32_t id) {
    return *document.workspace().find_chunk(id);
}

Sensor make_sensor(std::string make, std::string model, std::string serial) {
    Sensor sensor;
    sensor.label = model;
    sensor.type = CameraType::Frame;
    sensor.width = 4;
    sensor.height = 3;
    sensor.make = std::move(make);
    sensor.model = std::move(model);
    sensor.serial = std::move(serial);
    sensor.calibration = Calibration::initial(CameraType::Frame, 4, 3, 5.0);
    return sensor;
}

Camera make_camera(std::int32_t id, std::int32_t sensor_id, std::string label) {
    Camera camera;
    camera.id = id;
    camera.sensor_id = sensor_id;
    camera.label = std::move(label);
    return camera;
}

class ScopedTempDir {
public:
    ScopedTempDir() {
        const std::string name =
            "alphashape_groups_" + std::to_string(platform::process_id()) + "_" +
            std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
            std::to_string(counter_++);
        dir_ = (fs::temp_directory_path() / name).string();
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_, ec);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path(const std::string& relative) const {
        return (fs::path(dir_) / relative).string();
    }

private:
    static int counter_;
    std::string dir_;
};
int ScopedTempDir::counter_ = 0;

}  // namespace

TEST(CameraGroups, AutoStationBySerial) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    chunk.sensors().push_back(make_sensor("ACME", "X2", "SERIAL-B"));
    const std::int32_t a1 = chunk.allocate_id();
    const std::int32_t a2 = chunk.allocate_id();
    const std::int32_t b1 = chunk.allocate_id();
    chunk.cameras().push_back(make_camera(a1, 0, "a1"));
    chunk.cameras().push_back(make_camera(a2, 0, "a2"));
    chunk.cameras().push_back(make_camera(b1, 1, "b1"));

    auto command = make_group_cameras_command(chunk_id, GroupType::Station);
    command->apply(document);

    ASSERT_EQ(chunk.camera_groups().size(), 2U);
    EXPECT_EQ(chunk.camera_groups()[0].type, CameraGroupType::Station);
    EXPECT_EQ(chunk.camera_groups()[0].label, "SERIAL-A");
    EXPECT_EQ(chunk.camera_groups()[0].members, (std::vector<std::int32_t>{0, 1}));
    EXPECT_EQ(chunk.camera_groups()[1].label, "SERIAL-B");
    EXPECT_EQ(chunk.camera_groups()[1].members, (std::vector<std::int32_t>{2}));

    // Undo restores the exact pre-grouping list.
    command->revert(document);
    EXPECT_TRUE(chunk.camera_groups().empty());
    EXPECT_TRUE(chunk.is_valid());
}

TEST(CameraGroups, SelectionPersistsRoundTrip) {
    ScopedTempDir temp;
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    Camera first = make_camera(chunk.allocate_id(), 0, "first");
    first.selected = true;
    Camera second = make_camera(chunk.allocate_id(), 0, "second");
    chunk.cameras().push_back(first);
    chunk.cameras().push_back(second);
    std::shared_ptr<Frame> frame = std::make_shared<Frame>("first.png", 4, 3);
    frame->set_id(chunk.allocate_id());
    frame->set_camera_id(first.id);
    chunk.frames().push_back(frame);

    const std::string project = temp.path("selection.psx");
    document.set_path(project);
    ProjectIO::save(document, project);
    const auto loaded = ProjectIO::load(project);

    ASSERT_NE(loaded->active_chunk(), nullptr);
    const Chunk& got = *loaded->active_chunk();
    ASSERT_EQ(got.cameras().size(), 2U);
    EXPECT_TRUE(got.cameras()[0].selected);
    EXPECT_FALSE(got.cameras()[1].selected);
}

TEST(FrameStats, MatchesKnownAerialBlock) {
    constexpr double kSpacing = 60.0;
    constexpr double kAltitude = 50.0;
    constexpr int kRows = 3;
    constexpr int kCols = 3;

    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const std::int32_t camera_id = chunk.allocate_id();
            Camera camera = make_camera(camera_id, 0, "c" + std::to_string(camera_id));
            // The centre is C = -A^-1 b, so a translation of -C with an
            // identity rotation puts the centre exactly at (x, y, z).
            camera.pose(0, 3) = -kSpacing * static_cast<double>(col);
            camera.pose(1, 3) = -kSpacing * static_cast<double>(row);
            camera.pose(2, 3) = -kAltitude;
            camera.enabled = true;
            chunk.cameras().push_back(camera);
        }
    }

    const FrameStats stats = compute_frame_stats(chunk);
    EXPECT_EQ(stats.camera_count, 9U);
    EXPECT_EQ(stats.enabled_count, 9U);
    EXPECT_EQ(stats.aligned_count, 9U);
    EXPECT_EQ(stats.frame_count, 0U);
    EXPECT_EQ(stats.group_count, 0U);
    ASSERT_TRUE(stats.has_extent);
    EXPECT_NEAR(stats.extent_min.x(), 0.0, 1e-6);
    EXPECT_NEAR(stats.extent_min.y(), 0.0, 1e-6);
    EXPECT_NEAR(stats.extent_min.z(), kAltitude, 1e-6);
    EXPECT_NEAR(stats.extent_max.x(), kSpacing * (kCols - 1), 1e-6);
    EXPECT_NEAR(stats.extent_max.y(), kSpacing * (kRows - 1), 1e-6);
    EXPECT_NEAR(stats.extent_max.z(), kAltitude, 1e-6);
    // Every nearest-neighbour distance in this grid is exactly one spacing, so
    // the estimate is exactly 2 * 60 m up to floating-point rounding (the
    // distances are sqrt((60-0)^2), exact in binary). Pin it rather than
    // leave a band wide enough to hide a real regression.
    constexpr double kExpectedDistance = 2.0 * kSpacing;
    EXPECT_NEAR(stats.mean_capture_distance, kExpectedDistance, 1e-9 * kExpectedDistance);
    EXPECT_FALSE(stats.has_world_extent);
}

TEST(FrameStats, EmptyAndSingleCameraBlocksHaveNoFabricatedValues) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));

    const FrameStats empty = compute_frame_stats(chunk);
    EXPECT_EQ(empty.camera_count, 0U);
    EXPECT_EQ(empty.enabled_count, 0U);
    EXPECT_FALSE(empty.has_extent);
    EXPECT_FALSE(empty.has_world_extent);
    EXPECT_EQ(empty.mean_capture_distance, 0.0);
    EXPECT_TRUE(std::isfinite(empty.mean_capture_distance));

    const std::int32_t id = chunk.allocate_id();
    Camera camera = make_camera(id, 0, "only");
    camera.pose(0, 3) = -5.0;
    camera.pose(1, 3) = -7.0;
    camera.pose(2, 3) = -9.0;
    camera.enabled = true;
    chunk.cameras().push_back(camera);

    const FrameStats one = compute_frame_stats(chunk);
    EXPECT_EQ(one.enabled_count, 1U);
    ASSERT_TRUE(one.has_extent);
    EXPECT_DOUBLE_EQ(one.extent_min.x(), 5.0);
    EXPECT_DOUBLE_EQ(one.extent_min.y(), 7.0);
    EXPECT_DOUBLE_EQ(one.extent_min.z(), 9.0);
    EXPECT_DOUBLE_EQ(one.extent_min.x(), one.extent_max.x());
    EXPECT_DOUBLE_EQ(one.extent_min.y(), one.extent_max.y());
    EXPECT_DOUBLE_EQ(one.extent_min.z(), one.extent_max.z());
    // One camera has no neighbour: documented as 0, never NaN or infinity.
    EXPECT_EQ(one.mean_capture_distance, 0.0);
    EXPECT_TRUE(std::isfinite(one.mean_capture_distance));
    EXPECT_FALSE(one.has_world_extent);
}

TEST(CameraGroups, StationKeyTrimsTextAndSplitsUnknownSensors) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", " SN-1 "));
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SN-1"));
    chunk.sensors().push_back(make_sensor("", "", ""));
    chunk.sensors().push_back(make_sensor("", "", ""));
    for (std::int32_t s = 0; s < 4; ++s) {
        chunk.cameras().push_back(make_camera(chunk.allocate_id(), s, "c"));
    }

    make_group_cameras_command(chunk_id, GroupType::Station)->apply(document);

    // The padded and unpadded serials are the same body; the two identity-less
    // sensors are NOT merged into one "Unknown" bucket.
    ASSERT_EQ(chunk.camera_groups().size(), 3U);
    for (const CameraGroup& group : chunk.camera_groups()) {
        if (group.members.size() == 2U) {
            EXPECT_EQ(group.label, "SN-1");
            EXPECT_EQ(group.members, (std::vector<std::int32_t>{0, 1}));
        } else {
            EXPECT_EQ(group.label, "Unknown");
            EXPECT_EQ(group.members.size(), 1U);
        }
    }
}

TEST(CameraGroups, StationLabelJoinsKnownPartsWithoutLeadingSpace) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("AlphaOptics", "", ""));
    chunk.sensors().push_back(make_sensor("", "AO-100", ""));
    chunk.sensors().push_back(make_sensor("AlphaOptics", "AO-100", ""));
    for (std::int32_t s = 0; s < 3; ++s) {
        chunk.cameras().push_back(make_camera(chunk.allocate_id(), s, "c"));
    }

    make_group_cameras_command(chunk_id, GroupType::Station)->apply(document);

    ASSERT_EQ(chunk.camera_groups().size(), 3U);
    std::vector<std::string> labels;
    for (const CameraGroup& group : chunk.camera_groups()) {
        labels.push_back(group.label);
    }
    std::sort(labels.begin(), labels.end());
    EXPECT_EQ(labels, (std::vector<std::string>{"AO-100", "AlphaOptics",
                                                "AlphaOptics AO-100"}));
}

TEST(CameraGroups, FolderLabelsUseTheShortestUniqueSuffix) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    for (const char* folder : {"cardA/DCIM/100MEDIA", "cardB/DCIM/100MEDIA"}) {
        Camera camera = make_camera(chunk.allocate_id(), 0, "c");
        std::shared_ptr<Frame> frame =
            std::make_shared<Frame>(std::string(folder) + "/shot.jpg", 4, 3);
        frame->set_id(chunk.allocate_id());
        frame->set_camera_id(camera.id);
        chunk.cameras().push_back(camera);
        chunk.frames().push_back(frame);
    }

    make_group_cameras_command(chunk_id, GroupType::Folder)->apply(document);

    ASSERT_EQ(chunk.camera_groups().size(), 2U);
    std::vector<std::string> labels;
    for (const CameraGroup& group : chunk.camera_groups()) {
        labels.push_back(group.label);
    }
    std::sort(labels.begin(), labels.end());
    EXPECT_EQ(labels,
              (std::vector<std::string>{"cardA/DCIM/100MEDIA", "cardB/DCIM/100MEDIA"}));
}

TEST(CameraGroups, FolderGroupingLeavesFrameLessCamerasUngrouped) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    Camera camera = make_camera(chunk.allocate_id(), 0, "orphan");
    chunk.cameras().push_back(camera);  // deliberately no frame

    EXPECT_NO_THROW(make_group_cameras_command(chunk_id, GroupType::Folder)->apply(document));
    EXPECT_TRUE(chunk.camera_groups().empty());
}

TEST(CameraGroups, NoneKeepsUserGroupsAndUndoRestoresTheIdAllocator) {
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    chunk.sensors().push_back(make_sensor("ACME", "X2", "SERIAL-B"));
    chunk.cameras().push_back(make_camera(chunk.allocate_id(), 0, "a"));
    chunk.cameras().push_back(make_camera(chunk.allocate_id(), 1, "b"));
    CameraGroup user;
    user.id = chunk.allocate_id();
    user.label = "My selection";
    user.type = CameraGroupType::Fixed;
    user.members = {0};
    chunk.camera_groups().push_back(user);

    auto station = make_group_cameras_command(chunk_id, GroupType::Station);
    station->apply(document);
    ASSERT_EQ(chunk.camera_groups().size(), 3U);
    std::vector<std::int32_t> first_ids;
    for (const CameraGroup& group : chunk.camera_groups()) {
        first_ids.push_back(group.id);
    }

    // Undo puts the allocator back, so redo must reuse the same group ids.
    station->revert(document);
    ASSERT_EQ(chunk.camera_groups().size(), 1U);
    station->apply(document);
    std::vector<std::int32_t> second_ids;
    for (const CameraGroup& group : chunk.camera_groups()) {
        second_ids.push_back(group.id);
    }
    EXPECT_EQ(first_ids, second_ids);

    auto none = make_group_cameras_command(chunk_id, GroupType::None);
    none->apply(document);
    ASSERT_EQ(chunk.camera_groups().size(), 1U);
    EXPECT_EQ(chunk.camera_groups()[0].label, "My selection");
    EXPECT_EQ(chunk.camera_groups()[0].type, CameraGroupType::Fixed);

    none->revert(document);
    EXPECT_EQ(chunk.camera_groups().size(), 3U);
}

#if defined(ALPHASHAPE_USE_PROJ)

namespace {

/// A chunk whose two cameras sit at local (0,0,0) and (100,0,0) in an ENU frame
/// anchored at `anchor_geodetic`, georeferenced into `target`.
FrameStats stats_for_enu_chunk(const crs::CoordinateSystem& target,
                               const Eigen::Vector3d& anchor_geodetic,
                               crs::LocalFrame& frame) {
    const crs::CoordinateSystem geographic = crs::CoordinateSystem::from_epsg(4326);
    const crs::CoordinateSystem ecef = crs::CoordinateSystem::from_epsg(4978);
    const crs::CrsTransform to_ecef = crs::CrsTransform::create(geographic, ecef);
    const auto anchor = to_ecef.forward(
        crs::CrsPoint{anchor_geodetic.x(), anchor_geodetic.y(), anchor_geodetic.z()});
    frame = crs::LocalFrame::enu_at(
        target, Eigen::Vector3d(anchor->x, anchor->y, anchor->z));

    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    chunk.set_coordinate_system(target);
    // The chunk transform is local -> ECEF (origin + local axes in ECEF), not
    // LocalFrame::transform(), which is the linearised local -> CRS matrix.
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 1>(0, 3) = frame.origin_ecef();
    transform.block<3, 3>(0, 0) = frame.rotation();
    chunk.transform() = transform;
    for (int i = 0; i < 2; ++i) {
        Camera camera = make_camera(chunk.allocate_id(), 0, "c" + std::to_string(i));
        camera.pose(0, 3) = -100.0 * static_cast<double>(i);
        camera.enabled = true;
        chunk.cameras().push_back(camera);
    }
    return compute_frame_stats(chunk);
}

}  // namespace

TEST(FrameStats, WorldExtentIsInProjectedCrsUnits) {
    const crs::CoordinateSystem utm = crs::CoordinateSystem::from_epsg(32633);
    crs::LocalFrame frame;
    const FrameStats stats = stats_for_enu_chunk(utm, Eigen::Vector3d(15.0, 60.0, 100.0), frame);
    ASSERT_TRUE(stats.has_world_extent);
    const crs::CrsPoint first = frame.to_crs(Eigen::Vector3d(0.0, 0.0, 0.0));
    const crs::CrsPoint second = frame.to_crs(Eigen::Vector3d(100.0, 0.0, 0.0));
    // UTM 33N metres (easting around 5e5), not ECEF metres (x around 3.4e6).
    EXPECT_NEAR(stats.world_extent_min.x(), std::min(first.x, second.x), 1e-6);
    EXPECT_NEAR(stats.world_extent_max.x(), std::max(first.x, second.x), 1e-6);
    EXPECT_NEAR(stats.world_extent_min.y(), first.y, 1e-6);
    EXPECT_GT(stats.world_extent_min.x(), 1.0e5);
    EXPECT_LT(stats.world_extent_max.x(), 1.0e6);
    // The chunk-local extent stays metres.
    EXPECT_DOUBLE_EQ(stats.extent_max.x(), 100.0);
}

TEST(FrameStats, WorldExtentIsInDegreesForAGeographicCrs) {
    const crs::CoordinateSystem geographic = crs::CoordinateSystem::from_epsg(4326);
    crs::LocalFrame frame;
    const FrameStats stats =
        stats_for_enu_chunk(geographic, Eigen::Vector3d(15.0, 60.0, 100.0), frame);
    ASSERT_TRUE(stats.has_world_extent);
    const crs::CrsPoint first = frame.to_crs(Eigen::Vector3d(0.0, 0.0, 0.0));
    const crs::CrsPoint second = frame.to_crs(Eigen::Vector3d(100.0, 0.0, 0.0));
    EXPECT_NEAR(stats.world_extent_min.x(), std::min(first.x, second.x), 1e-6);
    EXPECT_NEAR(stats.world_extent_max.x(), std::max(first.x, second.x), 1e-6);
    EXPECT_NEAR(stats.world_extent_min.y(), first.y, 1e-6);
    // Longitude/latitude degrees, never ECEF metres.
    EXPECT_GT(stats.world_extent_min.x(), 14.0);
    EXPECT_LT(stats.world_extent_max.x(), 16.0);
    EXPECT_GT(stats.world_extent_min.y(), 59.0);
    EXPECT_LT(stats.world_extent_max.y(), 61.0);
}

TEST(FrameStats, ACrsWithAnIdentityTransformIsNotWorldExtent) {
    const crs::CoordinateSystem utm = crs::CoordinateSystem::from_epsg(32633);
    Document document;
    const std::int32_t chunk_id = add_chunk(document);
    Chunk& chunk = chunk_of(document, chunk_id);
    chunk.sensors().push_back(make_sensor("ACME", "X1", "SERIAL-A"));
    chunk.set_coordinate_system(utm);  // label only; geometry is still local
    Camera camera = make_camera(chunk.allocate_id(), 0, "c");
    camera.enabled = true;
    chunk.cameras().push_back(camera);

    const FrameStats stats = compute_frame_stats(chunk);
    EXPECT_TRUE(stats.has_extent);
    EXPECT_FALSE(stats.has_world_extent);
}

#endif  // ALPHASHAPE_USE_PROJ

}  // namespace alphashape
