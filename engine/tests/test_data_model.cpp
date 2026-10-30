// Tests for the core data model: ids, containers, and structural equality.
//
// The `Assets` suite at the end covers the M11 multi-instance asset containers:
// ids and labels per kind, the active-asset rule the legacy single-slot
// accessors follow, deep-copy semantics, validation, and the producing and
// exporting tasks that append and select assets.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/build_dem_task.hpp"
#include "common/algorithm/elevation/export_dem_task.hpp"
#include "common/core/assets.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "dense_fixture.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape {
namespace {

Keypoint make_keypoint(float x, float y) {
    Keypoint kp;
    kp.x = x;
    kp.y = y;
    kp.scale = 2.0F;
    kp.orientation = 0.5F;
    return kp;
}

Descriptor make_descriptor(std::uint8_t fill) {
    Descriptor d;
    d.bytes.fill(fill);
    return d;
}

/// A fully-populated chunk used as the equality baseline.
Chunk build_baseline_chunk() {
    Chunk chunk("baseline");
    chunk.set_id(3);

    Sensor sensor;
    sensor.label = "s0";
    sensor.type = CameraType::Frame;
    sensor.width = 100;
    sensor.height = 80;
    sensor.pixel_size_um = 3.5;
    sensor.focal_length_mm = 35.0;
    sensor.make = "ACME";
    sensor.model = "X1";
    sensor.serial = "s/n 42";
    sensor.calibration = Calibration::initial(CameraType::Frame, 100, 80, 120.0);
    sensor.fixed_params = {"f", "cy"};
    chunk.sensors().push_back(sensor);

    Camera camera;
    camera.id = 0;
    camera.sensor_id = 0;
    camera.label = "cam0";
    camera.pose = Transform::Identity();
    camera.pose(0, 3) = 0.1;
    camera.enabled = true;
    camera.orientation = 6;
    camera.reference.location = Eigen::Vector3d(77.2, 28.6, 213.0);
    camera.reference.enabled = false;
    chunk.cameras().push_back(camera);

    chunk.frames().push_back(std::make_shared<Frame>("a.pgm", 100, 80));

    KeypointSet set;
    set.camera_id = 0;
    set.keypoints = {make_keypoint(1.0F, 2.0F)};
    set.descriptors = {make_descriptor(0x01)};
    chunk.tie_points().keypoint_sets.push_back(set);

    return chunk;
}

TEST(DataModel, ChunkIdsIncrementFromZero) {
    Document document;
    Chunk& first = document.workspace().add_chunk("a");
    Chunk& second = document.workspace().add_chunk("b");
    EXPECT_EQ(first.id(), 0);
    EXPECT_EQ(second.id(), 1);
    EXPECT_EQ(document.workspace().next_chunk_id(), 2);
}

TEST(DataModel, RestoreChunkWithExistingIdThrowsAndKeepsCounterAhead) {
    Document document;
    document.workspace().add_chunk("a");  // id 0, next 1
    EXPECT_THROW(document.workspace().restore_chunk(0, "dup"), Error);

    // Counter must still be ahead of the max id after the failed restore.
    document.workspace().restore_chunk(5, "later");
    EXPECT_EQ(document.workspace().next_chunk_id(), 6);
}

TEST(DataModel, FindChunkUnknownReturnsNullptr) {
    Document document;
    document.workspace().add_chunk("a");
    EXPECT_EQ(document.workspace().find_chunk(999), nullptr);
}

TEST(DataModel, ActiveChunkNullWhenUnset) {
    Document document;
    document.workspace().add_chunk("a");
    // active id is -1 by default.
    EXPECT_EQ(document.active_chunk(), nullptr);
}

TEST(DataModel, ActiveChunkResolvesById) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("a");
    document.set_active_chunk_id(chunk.id());
    ASSERT_NE(document.active_chunk(), nullptr);
    EXPECT_EQ(document.active_chunk()->label(), "a");
}

TEST(DataModel, MetaSetReplacesInPlace) {
    Meta meta;
    meta.set("k", "one");
    meta.set("k", "two");
    ASSERT_EQ(meta.items.size(), 1U);
    ASSERT_NE(meta.find("k"), nullptr);
    EXPECT_EQ(*meta.find("k"), "two");
    EXPECT_EQ(meta.find("absent"), nullptr);
}

TEST(DataModel, KeypointSetConsistentTracksKeypointCount) {
    KeypointSet set;
    EXPECT_TRUE(set.consistent());  // empty == empty
    set.keypoints = {make_keypoint(0.0F, 0.0F)};
    EXPECT_FALSE(set.consistent());
    set.descriptors = {make_descriptor(0x00)};
    EXPECT_TRUE(set.consistent());
}

TEST(DataModel, ChunkEqualityTrueForIdenticalBuilds) {
    EXPECT_TRUE(build_baseline_chunk() == build_baseline_chunk());
}

TEST(DataModel, ChunkEqualityFalseWhenCameraPoseElementDiffers) {
    Chunk a = build_baseline_chunk();
    Chunk b = build_baseline_chunk();
    b.cameras()[0].pose(0, 3) += 1e-9;
    EXPECT_FALSE(a == b);
}

TEST(DataModel, ChunkEqualityFalseWhenOneKeypointDiffers) {
    Chunk a = build_baseline_chunk();
    Chunk b = build_baseline_chunk();
    b.tie_points().keypoint_sets[0].keypoints[0].x += 0.5F;
    EXPECT_FALSE(a == b);
}

TEST(DataModel, ChunkEqualityFalseWhenOneDescriptorByteDiffers) {
    Chunk a = build_baseline_chunk();
    Chunk b = build_baseline_chunk();
    b.tie_points().keypoint_sets[0].descriptors[0].bytes[10] ^= 0x01U;
    EXPECT_FALSE(a == b);
}

TEST(DataModel, DescriptorIs63Bytes) {
    EXPECT_EQ(Descriptor::kBytes, 63U);
    EXPECT_EQ(sizeof(Descriptor::bytes), 63U);
}

// -- CameraReference ---------------------------------------------------------

TEST(DataModel, ReferenceWithoutLocationEquality) {
    CameraReference a;
    CameraReference b;
    EXPECT_FALSE(a.location.has_value());
    EXPECT_TRUE(a == b);
    b.enabled = false;
    EXPECT_FALSE(a == b);
}

TEST(DataModel, ReferenceLocationEquality) {
    CameraReference a;
    a.location = Eigen::Vector3d(77.2, 28.6, 213.0);
    CameraReference b = a;
    EXPECT_TRUE(a == b);
    (*b.location).z() += 1.0;
    EXPECT_FALSE(a == b);
}

TEST(DataModel, ReferenceNaNAltitudeComparesEqual) {
    // No GNSS altitude: z is NaN, and NaN must compare equal to NaN so the
    // reference survives a save/load round-trip and compares to itself.
    CameraReference a;
    a.location = Eigen::Vector3d(77.2, 28.6,
                                 std::numeric_limits<double>::quiet_NaN());
    CameraReference b = a;
    EXPECT_TRUE(a == b);
    // A finite altitude still differs from a NaN altitude.
    CameraReference c;
    c.location = Eigen::Vector3d(77.2, 28.6, 0.0);
    EXPECT_FALSE(a == c);
}

TEST(DataModel, ReferenceAccuracyDefaultsToTenMetres) {
    CameraReference a;
    EXPECT_EQ(a.accuracy, Eigen::Vector3d(10.0, 10.0, 10.0));
    CameraReference b;
    EXPECT_TRUE(a == b);
}

TEST(DataModel, ReferenceEqualityTracksAccuracy) {
    CameraReference a;
    a.location = Eigen::Vector3d(77.2, 28.6, 213.0);
    CameraReference b = a;
    EXPECT_TRUE(a == b);
    b.accuracy.x() += 1.0;
    EXPECT_FALSE(a == b);
    // A different component is a different reference too.
    CameraReference c = a;
    c.accuracy.z() -= 0.5;
    EXPECT_FALSE(a == c);
}

TEST(DataModel, SensorEqualityTracksCalibrationAndFixedParams) {
    Sensor a;
    a.label = "s";
    a.calibration = Calibration::initial(CameraType::Frame, 100, 80, 120.0);
    Sensor b = a;
    EXPECT_TRUE(a == b);
    b.calibration.k1 = 1e-9;
    EXPECT_FALSE(a == b);
    Sensor c = a;
    c.fixed_params = {"k1"};
    EXPECT_FALSE(a == c);
}

// -- M11: chunk-local ids, new containers, validation ------------------------

namespace {

/// A chunk carrying one populated instance of every new M11 container, used
/// as the equality baseline and as the "valid chunk" fixture.
Chunk build_m11_chunk() {
    Chunk chunk("m11");
    chunk.set_id(7);

    Camera camera;
    camera.id = 0;
    camera.sensor_id = 0;
    camera.label = "cam0";
    chunk.cameras().push_back(camera);
    Camera camera1 = camera;
    camera1.id = 1;
    camera1.label = "cam1";
    chunk.cameras().push_back(camera1);

    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = "cp0";
    marker.type = MarkerType::Fiducial;
    MarkerProjection projection;
    projection.camera_id = 0;
    projection.pixel = Eigen::Vector2d(12.5, 8.25);
    marker.projections.push_back(projection);
    projection.camera_id = 1;
    marker.projections.push_back(projection);
    marker.reference = Eigen::Vector3d(77.2, 28.6, 213.0);
    marker.reference_accuracy = Eigen::Vector3d(0.01, 0.01, 0.02);
    marker.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    marker.position_covariance = Eigen::Matrix3d::Identity() * 0.25;
    marker.pinned = true;
    marker.valid = true;
    chunk.markers().push_back(marker);

    MarkerGroup marker_group;
    marker_group.id = chunk.allocate_id();
    marker_group.label = "control";
    marker_group.members = {marker.id};
    chunk.marker_groups().push_back(marker_group);
    chunk.markers()[0].group_id = marker_group.id;

    Scalebar bar;
    bar.id = chunk.allocate_id();
    bar.label = "bar0";
    bar.from.kind = ScalebarEndpoint::Kind::Marker;
    bar.from.id = marker.id;
    bar.to.kind = ScalebarEndpoint::Kind::Camera;
    bar.to.id = 1;
    bar.reference_length_m = 2.5;
    bar.reference_accuracy_m = 0.002;
    bar.estimated_length_m = 2.51;
    chunk.scalebars().push_back(bar);

    CameraGroup camera_group;
    camera_group.id = chunk.allocate_id();
    camera_group.label = "flight";
    camera_group.members = {0, 1};
    camera_group.type = CameraGroupType::Folder;
    camera_group.folder = "/data/flight";
    chunk.camera_groups().push_back(camera_group);

    Mask mask;
    mask.camera_id = 0;
    mask.width = 2;
    mask.height = 2;
    mask.data = {0, 1, 1, 0};
    chunk.masks().push_back(mask);
    mask.camera_id = 1;
    chunk.masks().push_back(mask);

    Shape shape;
    shape.id = chunk.allocate_id();
    shape.type = ShapeType::Polygon;
    shape.label = "roof";
    shape.vertices = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
                      Eigen::Vector3d(0, 1, 0)};
    shape.closed = true;
    shape.attributes["kind"] = "building";
    shape.layer = "vectors";
    shape.has_altitude = true;
    chunk.shapes().features.push_back(shape);

    ShapeGroup shape_group;
    shape_group.id = chunk.allocate_id();
    shape_group.label = "buildings";
    shape_group.members = {shape.id};
    chunk.shapes().groups.push_back(shape_group);

    Trajectory trajectory;
    trajectory.id = chunk.allocate_id();
    trajectory.label = "gnss";
    TrajectoryPoint point;
    point.position = Eigen::Vector3d(4.0, 5.0, 6.0);
    point.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
    point.timestamp_s = 1.5;
    point.has_orientation = true;
    trajectory.points.push_back(point);
    chunk.trajectories().push_back(trajectory);

    PanoramaAsset panorama;
    panorama.id = chunk.allocate_id();
    panorama.label = "pano";
    panorama.data = std::make_shared<Image>(2, 2, 3);
    chunk.panorama() = panorama;

    TiledModelAsset tiled;
    tiled.id = chunk.allocate_id();
    tiled.label = "tiles";
    tiled.data = std::make_shared<TiledModel>();
    tiled.data->payload = {1, 2, 3, 4};
    chunk.tiled_models().push_back(tiled);
    chunk.set_active_tiled_model_id(tiled.id);

    return chunk;
}

}  // namespace

TEST(DataModel, IdsAreNeverReused) {
    Chunk chunk;
    std::int32_t previous = -1;
    for (int i = 0; i < 10000; ++i) {
        const std::int32_t id = chunk.allocate_id();
        EXPECT_EQ(chunk.next_id(), id + 1);
        EXPECT_GT(id, previous);
        previous = id;

        // Add then remove: the id must not come back.
        Marker marker;
        marker.id = id;
        chunk.markers().push_back(marker);
        chunk.markers().clear();
    }
    EXPECT_EQ(previous, 9999);
    EXPECT_EQ(chunk.next_id(), 10000);
}

TEST(DataModel, ValidateRejectsDanglingActiveId) {
    Chunk chunk = build_m11_chunk();
    EXPECT_TRUE(chunk.validate().empty()) << chunk.validate().front();

    chunk.set_active_tiled_model_id(4242);
    const std::vector<std::string> problems = chunk.validate();
    EXPECT_FALSE(problems.empty());
    EXPECT_FALSE(chunk.is_valid());

    // An empty chunk with a non -1 active id is also a violation.
    Chunk empty;
    empty.set_active_tiled_model_id(0);
    EXPECT_FALSE(empty.validate().empty());
}

TEST(DataModel, ValidateRejectsUnsortedProjections) {
    Chunk chunk = build_m11_chunk();
    ASSERT_EQ(chunk.markers()[0].projections.size(), 2U);

    // Swap the two projections: camera ids become 1 then 0.
    std::swap(chunk.markers()[0].projections[0], chunk.markers()[0].projections[1]);
    EXPECT_FALSE(chunk.validate().empty());

    // Duplicate camera id is rejected too.
    Chunk duplicate = build_m11_chunk();
    duplicate.markers()[0].projections[1].camera_id = 0;
    EXPECT_FALSE(duplicate.validate().empty());

    // A projection into a camera that does not exist is rejected.
    Chunk missing = build_m11_chunk();
    missing.markers()[0].projections[1].camera_id = 99;
    EXPECT_FALSE(missing.validate().empty());
}

TEST(DataModel, EqualityCoversEveryContainer) {
    const Chunk baseline = build_m11_chunk();

    // Sanity: two independent builds compare equal.
    EXPECT_TRUE(baseline == build_m11_chunk());

    const auto differs = [&baseline](const Chunk& mutated) {
        return baseline != mutated;
    };

    {
        Chunk c = baseline;
        c.allocate_id();
        EXPECT_TRUE(differs(c)) << "next_id";
    }
    {
        Chunk c = baseline;
        c.markers()[0].label = "changed";
        EXPECT_TRUE(differs(c)) << "markers";
    }
    {
        Chunk c = baseline;
        c.markers()[0].projections[0].pixel.x() += 1.0;
        EXPECT_TRUE(differs(c)) << "marker projections";
    }
    {
        Chunk c = baseline;
        (*c.markers()[0].reference).z() += 1.0;
        EXPECT_TRUE(differs(c)) << "marker reference";
    }
    {
        Chunk c = baseline;
        c.markers()[0].position.reset();
        EXPECT_TRUE(differs(c)) << "marker position";
    }
    {
        Chunk c = baseline;
        c.markers()[0].pinned = false;
        EXPECT_TRUE(differs(c)) << "marker pinned";
    }
    {
        Chunk c = baseline;
        c.marker_groups()[0].label = "changed";
        EXPECT_TRUE(differs(c)) << "marker_groups";
    }
    {
        Chunk c = baseline;
        c.scalebars()[0].reference_length_m += 0.5;
        EXPECT_TRUE(differs(c)) << "scalebars";
    }
    {
        Chunk c = baseline;
        c.scalebars()[0].to.kind = ScalebarEndpoint::Kind::Marker;
        EXPECT_TRUE(differs(c)) << "scalebar endpoint kind";
    }
    {
        Chunk c = baseline;
        c.camera_groups()[0].type = CameraGroupType::Station;
        EXPECT_TRUE(differs(c)) << "camera_groups type";
    }
    {
        Chunk c = baseline;
        c.camera_groups()[0].folder = "/other";
        EXPECT_TRUE(differs(c)) << "camera_groups folder";
    }
    {
        Chunk c = baseline;
        c.masks()[0].data[0] = 1;
        EXPECT_TRUE(differs(c)) << "masks";
    }
    {
        Chunk c = baseline;
        c.masks()[1].invert = true;
        EXPECT_TRUE(differs(c)) << "mask invert";
    }
    {
        Chunk c = baseline;
        c.shapes().features[0].vertices[0].z() = 9.0;
        EXPECT_TRUE(differs(c)) << "shapes features";
    }
    {
        Chunk c = baseline;
        c.shapes().features[0].attributes["kind"] = "road";
        EXPECT_TRUE(differs(c)) << "shape attributes";
    }
    {
        Chunk c = baseline;
        c.shapes().features[0].layer = "other";
        EXPECT_TRUE(differs(c)) << "shape layer";
    }
    {
        Chunk c = baseline;
        c.shapes().groups[0].label = "changed";
        EXPECT_TRUE(differs(c)) << "shapes groups";
    }
    {
        Chunk c = baseline;
        c.trajectories()[0].points[0].timestamp_s += 1.0;
        EXPECT_TRUE(differs(c)) << "trajectories";
    }
    {
        Chunk c = baseline;
        c.trajectories()[0].points[0].has_orientation = false;
        EXPECT_TRUE(differs(c)) << "trajectory orientation flag";
    }
    {
        Chunk c = baseline;
        c.panorama().reset();
        EXPECT_TRUE(differs(c)) << "panorama presence";
    }
    {
        Chunk c = baseline;
        c.panorama()->label = "changed";
        EXPECT_TRUE(differs(c)) << "panorama label";
    }
    {
        Chunk c = baseline;
        c.panorama()->data = std::make_shared<Image>(3, 3, 1);
        EXPECT_TRUE(differs(c)) << "panorama data";
    }
    {
        Chunk c = baseline;
        // Replace rather than mutate in place: AssetSlot copies share `data`,
        // so in-place mutation would alias the baseline too.
        auto replacement = std::make_shared<TiledModel>();
        replacement->payload = {9};
        c.tiled_models()[0].data = replacement;
        EXPECT_TRUE(differs(c)) << "tiled_models data";
    }
    {
        Chunk c = baseline;
        c.set_active_tiled_model_id(-1);
        EXPECT_TRUE(differs(c)) << "active tiled model id";
    }
    {
        // Per-object CRS slots (WP-12.6): every slot is part of chunk identity,
        // so a chunk that carries an explicit slot must never compare equal to
        // one that follows its chunk CRS.
        Chunk c = baseline;
        CRS slot;
        slot.epsg = 4326;
        slot.wkt = "EPSG:4326";
        c.set_camera_crs(slot);
        EXPECT_TRUE(differs(c)) << "camera_crs slot";
    }
    {
        Chunk c = baseline;
        CRS slot;
        slot.epsg = 4326;
        slot.wkt = "EPSG:4326";
        c.set_marker_crs(slot);
        EXPECT_TRUE(differs(c)) << "marker_crs slot";
    }
    {
        Chunk c = baseline;
        CRS slot;
        slot.epsg = 4326;
        slot.wkt = "EPSG:4326";
        c.set_world_crs(slot);
        EXPECT_TRUE(differs(c)) << "world_crs slot";
    }
}

TEST(DataModel, DefaultChunkHasNoAssets) {
    const Chunk chunk;

    EXPECT_TRUE(chunk.markers().empty());
    EXPECT_TRUE(chunk.marker_groups().empty());
    EXPECT_TRUE(chunk.scalebars().empty());
    EXPECT_TRUE(chunk.camera_groups().empty());
    EXPECT_TRUE(chunk.masks().empty());
    EXPECT_TRUE(chunk.shapes().features.empty());
    EXPECT_TRUE(chunk.shapes().groups.empty());
    EXPECT_TRUE(chunk.trajectories().empty());
    EXPECT_TRUE(chunk.tiled_models().empty());

    EXPECT_FALSE(chunk.panorama().has_value());
    EXPECT_EQ(chunk.active_tiled_model_id(), -1);
    EXPECT_EQ(chunk.next_id(), 0);

    EXPECT_TRUE(chunk.validate().empty());
    EXPECT_TRUE(chunk.is_valid());
}

TEST(DataModel, MaskIsValidRequiresMatchingDimensions) {
    Mask mask;
    EXPECT_FALSE(mask.is_valid());  // zero dimensions
    mask.width = 2;
    mask.height = 2;
    EXPECT_FALSE(mask.is_valid());  // empty data
    mask.data.resize(3);
    EXPECT_FALSE(mask.is_valid());  // too short
    mask.data.resize(4);
    EXPECT_TRUE(mask.is_valid());
    mask.data.resize(5);
    EXPECT_FALSE(mask.is_valid());  // too long
}

// ---------------------------------------------------------------------------
// M11 multi-instance assets
// ---------------------------------------------------------------------------

namespace {

namespace fs = std::filesystem;

using dense_fixture::aligned_document;
using dense_fixture::parse_truth_images;
using dense_fixture::seq_files;
using dense_fixture::set_truth_calibration;
using dense_fixture::set_truth_poses;
using dense_fixture::TruthImage;

/// A scratch directory removed on destruction (the same scheme the pipeline
/// fixtures use, kept local so this file owns no shared state).
struct ScratchDir {
    fs::path path;
    explicit ScratchDir(const std::string& tag) {
        path = fs::temp_directory_path() /
               (tag + "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path, ec);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string file(const std::string& name) const { return (path / name).string(); }
};

/// A camera-less chunk in the local frame: enough for BuildDem's point-cloud
/// source and for the exports.
Document make_local_chunk_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

/// An analytic height field: a tilted plane with a raised block, so two DEMs
/// built at different resolutions differ in more than their metadata.
double test_height(double x, double y) {
    const double plane = 100.0 + 0.1 * x - 0.05 * y;
    const bool on_block = x > -4.0 && x < 4.0 && y > -4.0 && y < 4.0;
    return plane + (on_block ? 7.0 : 0.0);
}

/// A point cloud sampled on a 0.25 m lattice over [-10, 10]^2.
void fill_height_cloud(Chunk& chunk) {
    PointCloud& cloud = chunk.point_cloud();
    cloud.points.clear();
    for (double x = -10.0; x <= 10.0 + 1e-9; x += 0.25) {
        for (double y = -10.0; y <= 10.0 + 1e-9; y += 0.25) {
            cloud.points.push_back(Eigen::Vector3d(x, y, test_height(x, y)));
        }
    }
}

Params dem_params(double resolution) {
    Params params = make_params(builtin_registry().require("BuildDem"));
    params.set("source_data", ParamValue{std::string("point_cloud")});
    params.set("interpolation", ParamValue{std::string("enabled")});
    params.set("resolution", ParamValue{resolution});
    params.set("fill_method", ParamValue{std::string("natural_neighbour")});
    params.set("smoothing", ParamValue{std::string("none")});
    return params;
}

Params export_dem_params(const std::string& path) {
    Params params = make_params(builtin_registry().require("ExportDem"));
    params.set("path", ParamValue{path});
    params.set("tiled", ParamValue{false});
    return params;
}

/// A small hand-built ground grid, as test_build_texture.cpp uses for its
/// non-meshing UV/texture tests.
Mesh make_ground_grid(int nx, int ny, double size = 12.0) {
    Mesh mesh;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const double x = -size * 0.5 + size * static_cast<double>(i) / static_cast<double>(nx);
            const double y = -size + size * static_cast<double>(j) / static_cast<double>(ny);
            mesh.vertices.emplace_back(x, y, 0.0);
        }
    }
    const auto at = [&](int i, int j) { return j * (nx + 1) + i; };
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            mesh.faces.emplace_back(at(i, j), at(i + 1, j), at(i, j + 1));
            mesh.faces.emplace_back(at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
        }
    }
    return mesh;
}

}  // namespace

TEST(Assets, AddAllocatesIdsAndLabels) {
    Chunk chunk;
    const std::int32_t e1 = chunk.add_elevation().id;
    const std::int32_t e2 = chunk.add_elevation().id;
    const std::int32_t o1 = chunk.add_orthomosaic().id;
    const std::int32_t m1 = chunk.add_model().id;
    const std::int32_t p1 = chunk.add_point_cloud().id;
    const std::int32_t d1 = chunk.add_depth_maps().id;

    EXPECT_EQ(e1, 0);
    EXPECT_EQ(e2, 1);
    EXPECT_EQ(o1, 2);
    EXPECT_EQ(m1, 3);
    EXPECT_EQ(p1, 4);
    EXPECT_EQ(d1, 5);
    EXPECT_EQ(chunk.next_id(), 6);

    ASSERT_EQ(chunk.elevations().size(), 2U);
    EXPECT_EQ(chunk.elevations()[0].label, "Elevation 1");
    EXPECT_EQ(chunk.elevations()[1].label, "Elevation 2");
    EXPECT_EQ(chunk.orthomosaics()[0].label, "Orthomosaic 1");
    EXPECT_EQ(chunk.models()[0].label, "Model 1");
    EXPECT_EQ(chunk.point_clouds()[0].label, "Point Cloud 1");
    EXPECT_EQ(chunk.depth_map_sets()[0].label, "Depth Maps 1");

    // Assets are kept sorted by id and every id is unique.
    for (std::size_t i = 1; i < chunk.elevations().size(); ++i) {
        EXPECT_LT(chunk.elevations()[i - 1].id, chunk.elevations()[i].id);
    }
    EXPECT_TRUE(chunk.validate().empty());

    // The last added asset of each kind is the active one.
    EXPECT_EQ(chunk.active_elevation_id(), e2);
    EXPECT_EQ(chunk.active_orthomosaic_id(), o1);
    EXPECT_EQ(chunk.active_model_id(), m1);
    EXPECT_EQ(chunk.active_point_cloud_id(), p1);
    EXPECT_EQ(chunk.active_depth_maps_id(), d1);
}

TEST(Assets, ExplicitLabelIsKeptAndUnusedNumbersAreReused) {
    Chunk chunk;
    chunk.add_elevation("custom dem");
    EXPECT_EQ(chunk.elevations()[0].label, "custom dem");
    // "Elevation 1" is still free, so the next default label takes it.
    chunk.add_elevation();
    EXPECT_EQ(chunk.elevations()[1].label, "Elevation 1");
    chunk.add_elevation();
    EXPECT_EQ(chunk.elevations()[2].label, "Elevation 2");
    EXPECT_TRUE(chunk.validate().empty());
}

TEST(Assets, LegacyAccessorsReturnActive) {
    Chunk chunk;
    // The non-const overload lazily creates and activates the first asset.
    PointCloud& first = chunk.point_cloud();
    ASSERT_EQ(chunk.point_clouds().size(), 1U);
    EXPECT_EQ(chunk.active_point_cloud_id(), 0);
    EXPECT_EQ(&first, chunk.point_clouds()[0].data.get());
    first.points.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));

    ModelAsset& first_model = chunk.add_model();
    first_model.data->vertices = {Eigen::Vector3d(0.0, 0.0, 0.0)};
    ModelAsset& second_model = chunk.add_model();
    second_model.data->vertices = {Eigen::Vector3d(1.0, 1.0, 1.0)};
    second_model.textures.push_back(Image(2, 2, 3));
    EXPECT_EQ(&chunk.model(), second_model.data.get());
    EXPECT_EQ(chunk.model_textures().size(), 1U);
    ASSERT_EQ(chunk.models().size(), 2U);
    EXPECT_NE(chunk.models()[0].data.get(), second_model.data.get());
    // The legacy texture accessor never mints a model of its own.
    EXPECT_EQ(&chunk.model(), second_model.data.get());

    // The const overloads read the same slot and never mint one.
    chunk.add_elevation();
    const Chunk& view = chunk;
    EXPECT_EQ(&view.model(), second_model.data.get());
    EXPECT_EQ(view.model_textures().size(), 1U);
    EXPECT_EQ(view.point_cloud().points.size(), 1U);
    EXPECT_EQ(&view.elevation(), view.elevations().front().data.get());

    // Writing through the legacy accessors writes the active asset.
    chunk.elevation().set_epsg(32632);
    EXPECT_EQ(chunk.elevations().back().data->epsg(), 32632);

    // depth_maps() behaves the same way.
    EXPECT_TRUE(chunk.depth_maps().empty());
    EXPECT_EQ(chunk.depth_map_sets().size(), 1U);
    EXPECT_EQ(chunk.active_depth_maps_id(), chunk.depth_map_sets().front().id);
}

TEST(Assets, ConstAccessorsOnEmptyChunkAreEmptySingleton) {
    const Chunk chunk;
    ASSERT_TRUE(chunk.point_clouds().empty());
    EXPECT_TRUE(chunk.point_cloud().points.empty());
    EXPECT_TRUE(chunk.model().empty());
    EXPECT_TRUE(chunk.model_textures().empty());
    EXPECT_TRUE(chunk.depth_maps().empty());
    EXPECT_TRUE(chunk.elevation().empty());
    EXPECT_TRUE(chunk.orthomosaic().empty());
    // Reading never mutates.
    EXPECT_TRUE(chunk.point_clouds().empty());
    EXPECT_TRUE(chunk.elevations().empty());
    EXPECT_EQ(chunk.active_point_cloud_id(), -1);
    EXPECT_EQ(chunk.active_model_id(), -1);
    EXPECT_EQ(chunk.active_depth_maps_id(), -1);
    EXPECT_EQ(chunk.active_elevation_id(), -1);
    EXPECT_EQ(chunk.active_orthomosaic_id(), -1);
}

TEST(Assets, NullPayloadsReadAsEmptyAndAreRecreatedOnMutableAccess) {
    Chunk chunk;
    chunk.add_point_cloud();
    chunk.add_model();
    chunk.add_elevation();
    chunk.add_orthomosaic();
    chunk.add_depth_maps();

    // Simulate an unloaded payload (what validate() reports as "null payload").
    chunk.point_clouds().front().data.reset();
    chunk.models().front().data.reset();
    chunk.elevations().front().data.reset();
    chunk.orthomosaics().front().data.reset();
    chunk.depth_map_sets().front().data.reset();

    // The const accessors never dereference the cleared pointer.
    const Chunk& view = chunk;
    EXPECT_TRUE(view.point_cloud().points.empty());
    EXPECT_TRUE(view.model().empty());
    EXPECT_TRUE(view.elevation().empty());
    EXPECT_TRUE(view.orthomosaic().empty());
    EXPECT_TRUE(view.depth_maps().empty());
    EXPECT_FALSE(chunk.validate().empty()) << "the cleared payloads are reported";

    // The mutable accessors create a payload on demand instead of dereferencing null, so a
    // caller that writes to the active asset gets a live object rather than undefined behaviour.
    chunk.point_cloud().points.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    chunk.model().vertices.push_back(Eigen::Vector3d::Zero());
    chunk.elevation().set_epsg(32632);
    chunk.orthomosaic().set_epsg(3857);
    chunk.depth_maps().push_back(DepthMap{});

    ASSERT_NE(chunk.point_clouds().front().data, nullptr);
    ASSERT_NE(chunk.models().front().data, nullptr);
    ASSERT_NE(chunk.elevations().front().data, nullptr);
    ASSERT_NE(chunk.orthomosaics().front().data, nullptr);
    ASSERT_NE(chunk.depth_map_sets().front().data, nullptr);
    EXPECT_EQ(chunk.point_cloud().points.size(), 1U);
    EXPECT_EQ(chunk.model().vertices.size(), 1U);
    EXPECT_EQ(chunk.elevation().epsg(), 32632);
    EXPECT_EQ(chunk.depth_maps().size(), 1U);
    // The write landed in the slot the accessors hand out, so validation passes again.
    EXPECT_TRUE(chunk.validate().empty()) << chunk.validate().front();}

TEST(Assets, ConstTexturesSurviveAnUnloadedMeshPayload) {
    Chunk chunk;
    chunk.add_model();
    chunk.models().front().textures.push_back(Image(2, 2, 3));
    // The pages live in the slot, independently of the mesh payload.
    chunk.models().front().data.reset();

    const Chunk& view = chunk;
    EXPECT_TRUE(view.model().empty());
    ASSERT_EQ(view.model_textures().size(), 1U);
    EXPECT_EQ(view.model_textures().front().width(), 2);
}

TEST(Assets, ValidateChecksActiveIdsAndSorting) {
    Chunk chunk;
    chunk.add_elevation();
    chunk.add_elevation();
    EXPECT_TRUE(chunk.validate().empty());

    // An active id that names nothing.
    Chunk dangling = chunk;
    dangling.set_active_elevation_id(99);
    EXPECT_FALSE(dangling.validate().empty());

    // A non -1 active id on an empty container.
    Chunk spare;
    spare.set_active_orthomosaic_id(0);
    EXPECT_FALSE(spare.validate().empty());

    // Duplicate ids.
    Chunk duplicate = chunk;
    duplicate.elevations()[1].id = duplicate.elevations()[0].id;
    EXPECT_FALSE(duplicate.validate().empty());

    // An id at or past next_id.
    Chunk future = chunk;
    future.elevations()[0].id = future.next_id();
    EXPECT_FALSE(future.validate().empty());

    // A null payload.
    Chunk nulled = chunk;
    nulled.elevations()[0].data.reset();
    EXPECT_FALSE(nulled.validate().empty());
}

TEST(Assets, EqualityCoversAssetsAndActiveIds) {
    const auto build = [] {
        Chunk chunk;
        chunk.add_elevation();
        chunk.elevation().raster.data = {1.0F, 2.0F};
        chunk.add_point_cloud();
        chunk.point_cloud().points.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
        chunk.add_model();
        chunk.models().front().textures.push_back(Image(2, 2, 3));
        return chunk;
    };
    const Chunk baseline = build();
    EXPECT_TRUE(baseline == build());

    const auto differs = [&baseline](const Chunk& mutated) { return baseline != mutated; };

    {
        Chunk c = baseline;
        c.elevation().raster.data[0] = 42.0F;
        EXPECT_TRUE(differs(c)) << "payload";
    }
    {
        Chunk c = baseline;
        c.set_active_elevation_id(-1);
        EXPECT_TRUE(differs(c)) << "active id";
    }
    {
        Chunk c = baseline;
        c.models().front().label = "renamed";
        EXPECT_TRUE(differs(c)) << "label";
    }
    {
        Chunk c = baseline;
        c.models().front().textures[0].at(0, 0, 0) = 255;
        EXPECT_TRUE(differs(c)) << "texture pixel";
    }
}

TEST(Assets, CopyDeepClonesPayloads) {
    Chunk original;
    original.add_point_cloud();
    original.point_cloud().points.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    original.add_model();
    original.models().front().textures.push_back(Image(2, 2, 3));
    original.add_elevation();
    original.elevation().set_epsg(32632);

    Chunk copy = original;
    EXPECT_TRUE(copy == original);
    // Payloads are distinct objects, so mutating one cannot touch the other.
    EXPECT_NE(copy.point_clouds().front().data.get(),
              original.point_clouds().front().data.get());
    EXPECT_NE(copy.models().front().data.get(), original.models().front().data.get());
    EXPECT_NE(&copy.models().front().textures.front(),
              &original.models().front().textures.front());

    copy.point_cloud().points.push_back(Eigen::Vector3d(9.0, 9.0, 9.0));
    copy.elevation().set_epsg(4326);
    copy.models().front().textures.front().at(0, 0, 0) = 7;
    EXPECT_EQ(original.point_cloud().points.size(), 1U);
    EXPECT_EQ(original.elevation().epsg(), 32632);
    EXPECT_EQ(original.models().front().textures.front().at(0, 0, 0), 0);

    // Assignment is deep too.
    Chunk assigned;
    assigned = original;
    assigned.point_cloud().points.clear();
    EXPECT_EQ(original.point_cloud().points.size(), 1U);
}

TEST(Assets, BuildDemTwiceCreatesTwoElevations) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_height_cloud(chunk);

    Params first = dem_params(0.5);
    run_task(builtin_registry(), document, "BuildDem", first);
    ASSERT_EQ(chunk.elevations().size(), 1U);
    EXPECT_EQ(chunk.elevations()[0].label, "Elevation 1");

    Params second = dem_params(0.25);
    second.set("label", ParamValue{std::string("fine")});
    run_task(builtin_registry(), document, "BuildDem", second);

    ASSERT_EQ(chunk.elevations().size(), 2U);
    EXPECT_EQ(chunk.elevations()[0].label, "Elevation 1");
    EXPECT_EQ(chunk.elevations()[1].label, "fine");
    EXPECT_NE(chunk.elevations()[0].id, chunk.elevations()[1].id);
    // The coarser DEM survives untouched.
    EXPECT_NEAR(chunk.elevations()[0].data->raster.grid.pixel_size_x, 0.5, 1e-9);
    EXPECT_NEAR(chunk.elevations()[1].data->raster.grid.pixel_size_x, 0.25, 1e-9);
    EXPECT_TRUE(chunk.validate().empty());
}

TEST(Assets, ActiveFollowsLastBuild) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_height_cloud(chunk);

    Params first = dem_params(0.5);
    run_task(builtin_registry(), document, "BuildDem", first);
    const std::int32_t first_id = chunk.elevations().front().id;
    EXPECT_EQ(chunk.active_elevation_id(), first_id);

    Params second = dem_params(0.5);
    run_task(builtin_registry(), document, "BuildDem", second);
    const std::int32_t second_id = chunk.elevations().back().id;
    EXPECT_NE(second_id, first_id);
    EXPECT_EQ(chunk.active_elevation_id(), second_id);
    // The legacy accessor follows the active id.
    EXPECT_EQ(&chunk.elevation(), chunk.elevations().back().data.get());
}

/// Read a DEM GeoTIFF and compare it cell by cell with an elevation's raster.
void expect_raster_matches(const GeoRaster<float>& written, const Elevation& expected) {
    ASSERT_EQ(written.grid.width, expected.raster.grid.width);
    ASSERT_EQ(written.grid.height, expected.raster.grid.height);
    ASSERT_EQ(written.data.size(), expected.raster.data.size());
    for (std::size_t i = 0; i < written.data.size(); ++i) {
        ASSERT_EQ(written.data[i], expected.raster.data[i]) << "cell " << i;
    }
}

TEST(Assets, ExportSelectsById) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_height_cloud(chunk);

    Params coarse = dem_params(0.5);
    run_task(builtin_registry(), document, "BuildDem", coarse);
    const std::int32_t coarse_id = chunk.elevations().front().id;
    Params fine = dem_params(0.25);
    run_task(builtin_registry(), document, "BuildDem", fine);
    const std::int32_t fine_id = chunk.elevations().back().id;
    ASSERT_NE(coarse_id, fine_id);
    ASSERT_EQ(chunk.active_elevation_id(), fine_id);

    ScratchDir dir("alphashape_assets_export");

    // Export the FIRST (coarse) DEM while the second is active.
    const std::string coarse_path = dir.file("coarse.tif");
    Params by_id = export_dem_params(coarse_path);
    by_id.set("asset_id", ParamValue{std::int64_t{coarse_id}});
    run_task(builtin_registry(), document, "ExportDem", by_id);
    expect_raster_matches(read_geotiff_float(coarse_path), *chunk.elevations()[0].data);

    // -1 selects the active asset, and must not equal the coarse export.
    const std::string active_path = dir.file("active.tif");
    Params active = export_dem_params(active_path);
    active.set("asset_id", ParamValue{std::int64_t{-1}});
    run_task(builtin_registry(), document, "ExportDem", active);
    expect_raster_matches(read_geotiff_float(active_path), *chunk.elevations()[1].data);
    EXPECT_NE(read_geotiff_float(active_path).grid.width,
              read_geotiff_float(coarse_path).grid.width);

    // An unknown id is a bad parameter, not a crash.
    Params bad = export_dem_params(dir.file("bad.tif"));
    bad.set("asset_id", ParamValue{std::int64_t{424242}});
    EXPECT_THROW(run_task(builtin_registry(), document, "ExportDem", bad), ValueError);
    EXPECT_FALSE(fs::exists(dir.file("bad.tif")));

    // The export never mutates the chunk.
    EXPECT_EQ(chunk.elevations().size(), 2U);
    EXPECT_EQ(chunk.active_elevation_id(), fine_id);
}

/// A saved chunk restores the ACTIVE asset's id, label and the chunk's id
/// allocator, so the identity the export selection relies on survives a reload.
TEST(Assets, AssetIdentitySurvivesRoundTrip) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_height_cloud(chunk);
    Params first = dem_params(0.5);
    run_task(builtin_registry(), document, "BuildDem", first);
    Params second = dem_params(0.5);
    run_task(builtin_registry(), document, "BuildDem", second);

    // Format v2 persists every asset of every kind, so both DEMs come back with
    // their own ids and labels, and the ACTIVE one stays active.
    const std::int32_t active_id = chunk.active_elevation_id();
    const std::string active_label = chunk.elevations().back().label;
    const std::int32_t next_id = chunk.next_id();
    ASSERT_EQ(chunk.elevations().size(), 2U);

    ScratchDir dir("alphashape_assets_roundtrip");
    const std::string project = dir.file("proj.psx");
    ProjectIO::save(document, project);
    std::shared_ptr<Document> loaded = ProjectIO::load(project);
    ASSERT_NE(loaded->active_chunk(), nullptr);
    const Chunk& got = *loaded->active_chunk();
    ASSERT_EQ(got.elevations().size(), 2U);
    EXPECT_EQ(got.elevations().front().id, chunk.elevations().front().id);
    EXPECT_EQ(got.elevations().front().label, chunk.elevations().front().label);
    EXPECT_EQ(got.elevations().back().id, active_id);
    EXPECT_EQ(got.elevations().back().label, active_label);
    EXPECT_EQ(got.active_elevation_id(), active_id);
    EXPECT_EQ(got.next_id(), next_id);
    EXPECT_TRUE(got.validate().empty());
    EXPECT_TRUE(got == chunk);
}

TEST(Assets, TexturesStayWithTheirModel) {
    const std::vector<TruthImage> truth = parse_truth_images("seq_truth.json");
    Document document = aligned_document(seq_files(4));
    set_truth_poses(document, truth);
    set_truth_calibration(document, 480, 360, 500.0);
    Chunk& chunk = *document.active_chunk();

    chunk.model() = make_ground_grid(6, 6);
    Params uv = make_params(builtin_registry().require("BuildUV"));
    uv.set("texture_size", ParamValue{std::int64_t{256}});
    run_task(builtin_registry(), document, "BuildUV", uv);

    Params texture = make_params(builtin_registry().require("BuildTexture"));
    texture.set("texture_size", ParamValue{std::int64_t{256}});
    run_task(builtin_registry(), document, "BuildTexture", texture);

    ASSERT_EQ(chunk.models().size(), 1U);
    const std::int32_t first_id = chunk.models().front().id;
    ASSERT_EQ(chunk.models().front().textures.size(), 1U);
    const int first_width = chunk.models().front().textures.front().width();

    // A second model, added the way BuildModel does it: a fresh slot carrying
    // the mesh it produced, with no texture pages attached.
    ModelAsset& second = chunk.add_model("second");
    *second.data = make_ground_grid(4, 4, 8.0);

    ASSERT_EQ(chunk.models().size(), 2U);
    const ModelAsset* stored = find_asset(chunk.models(), first_id);
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->textures.size(), 1U);
    EXPECT_EQ(stored->textures.front().width(), first_width);
    // The new, active model carries no texture yet.
    EXPECT_TRUE(chunk.model_textures().empty());
    EXPECT_EQ(chunk.active_model_id(), second.id);
}

}  // namespace
}  // namespace alphashape

