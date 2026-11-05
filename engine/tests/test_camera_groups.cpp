// Tests for the camera-group, camera-selection and frame-statistics surfaces
// (WP-11.8).

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
    std::unique_ptr<Document> loaded = ProjectIO::load(project);

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
    // Every nearest-neighbour distance in this grid is one spacing, so the
    // estimate is 2 * 60 m; the 5% band allows for a future tie-break change.
    EXPECT_NEAR(stats.mean_capture_distance, 2.0 * kSpacing, 0.05 * 2.0 * kSpacing);
    EXPECT_FALSE(stats.has_world_extent);
}

}  // namespace alphashape
