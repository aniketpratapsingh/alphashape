// doc.xml format v2: writer layout, determinism, full-container round-trip,
// empty-container omission and sidecar magics (M11 WP-11.3).
//
// The v2 writer emits every container of every chunk and every asset of every
// kind, with bulk data under "chunks/<chunk_id>/". The reader for v2 rebuilds
// exactly that. These tests assert the documented §4.4 layout, that a load/save
// of an untouched project reproduces identical bytes, and that a chunk holding
// two of each asset plus every M11 container survives ProjectIO::save/load with
// Chunk::operator==.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"

#include "base/io/zip.hpp"
#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/io/sidecars.hpp"
#include "image/image.hpp"
#include "aerial_fixture.hpp"
#include "v1_fixture.hpp"
#include "v1_writer.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// Fuzz iterations. The sanitized build is much slower per attempt, so it runs
/// the reduced corpus WP-11.4 specifies.
constexpr int kFuzzCases =
#ifdef ALPHASHAPE_SANITIZED_BUILD
    2000;
#else
    10000;
#endif

class ProjectV2 : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base = fs::temp_directory_path() /
                          ("alphashape_v2_" + std::to_string(platform::process_id()) + "_" +
                           std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
        dir_ = base.string() + "_" + std::to_string(counter_++);
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_, ec);
        ASSERT_FALSE(ec) << ec.message();
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    std::string path(const std::string& name) const {
        return (fs::path(dir_) / name).string();
    }
    /// The archive a project at `psx` keeps its content in.
    static std::string zip_of(const std::string& psx) {
        fs::path base = psx;
        base.replace_extension();
        return base.string() + ".files/project.zip";
    }
    static std::vector<std::uint8_t> zip_bytes(const std::string& psx) {
        std::ifstream in(zip_of(psx), std::ios::binary);
        EXPECT_TRUE(in);
        return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                         std::istreambuf_iterator<char>());
    }
    static std::string doc_text(const std::string& psx) {
        return ZipArchive::parse(zip_bytes(psx)).get_text("doc.xml");
    }

    static int counter_;
    std::string dir_;
};
int ProjectV2::counter_ = 0;

// -- fixture builders -------------------------------------------------------

Keypoint make_keypoint(float x, float y) {
    Keypoint kp;
    kp.x = x;
    kp.y = y;
    kp.scale = 1.5F;
    kp.orientation = 0.25F;
    return kp;
}

Descriptor make_descriptor(std::uint8_t seed) {
    Descriptor d;
    for (std::size_t i = 0; i < Descriptor::kBytes; ++i) {
        d.bytes[i] = static_cast<std::uint8_t>(seed + i * 3U);
    }
    return d;
}

Image make_texture(int width, int height, int salt) {
    Image image(width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at(x, y, 0) = static_cast<std::uint8_t>((x + y + salt) % 256);
            image.at(x, y, 1) = static_cast<std::uint8_t>((x * 3 + salt) % 256);
            image.at(x, y, 2) = static_cast<std::uint8_t>((y * 5 + salt) % 256);
        }
    }
    return image;
}

/// A 2 x 2 DEM so the GeoTIFF side is real without making the fixture big.
Elevation make_elevation(float base) {
    Elevation dem;
    dem.raster.grid.width = 2;
    dem.raster.grid.height = 2;
    dem.raster.grid.origin_x = 500000.0;
    dem.raster.grid.origin_y = 5410000.0;
    dem.raster.grid.pixel_size_x = 0.5;
    dem.raster.grid.pixel_size_y = 0.5;
    dem.raster.channels = 1;
    dem.raster.data = {base, base + 1.0F, base + 2.0F, base + 3.0F};
    dem.set_epsg(32633);
    dem.nodata = -9999.0F;
    dem.source = "depth_maps";
    return dem;
}

Orthomosaic make_ortho(std::uint8_t salt) {
    Orthomosaic ortho;
    ortho.raster.grid.width = 2;
    ortho.raster.grid.height = 2;
    ortho.raster.grid.origin_x = 500000.0;
    ortho.raster.grid.origin_y = 5410000.0;
    ortho.raster.grid.pixel_size_x = 0.5;
    ortho.raster.grid.pixel_size_y = 0.5;
    ortho.raster.channels = 4;
    ortho.raster.data.resize(16);
    for (std::size_t i = 0; i < ortho.raster.data.size(); ++i) {
        ortho.raster.data[i] = static_cast<std::uint8_t>((i * 7U + salt) % 256U);
    }
    ortho.set_epsg(32633);
    ortho.surface = "elevation";
    ortho.labels = {0, 1, 1, -1};
    return ortho;
}

Mesh make_mesh(double offset, int integer_salt) {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(offset, 0.0, 0.0),
                     Eigen::Vector3d(1.0 + offset, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0 + offset)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
    const std::uint8_t s = static_cast<std::uint8_t>(integer_salt);
    mesh.colours = {s, s, s, s, s, s, s, s, s};
    return mesh;
}

/// Populate ONE chunk with two of every asset kind and every M11 container.
/// The fixture is deliberately small (all payloads are a handful of elements)
/// so the whole document stays far below the 20 MB budget.
Document build_full_document() {
    Document document;
    Workspace& workspace = document.workspace();
    Chunk& chunk = workspace.add_chunk("full <&> \"chunk\"");
    document.set_active_chunk_id(chunk.id());

    Sensor sensor;
    sensor.label = "sensor";
    sensor.type = CameraType::Frame;
    sensor.width = 64;
    sensor.height = 48;
    sensor.pixel_size_um = 3.0;
    sensor.focal_length_mm = 20.0;
    sensor.make = "ACME";
    sensor.model = "M";
    sensor.serial = "sn";
    sensor.calibration = Calibration::initial(CameraType::Frame, 64, 48, 60.0);
    sensor.fixed_params = {"f", "k1"};
    chunk.sensors().push_back(sensor);

    for (std::int32_t id = 0; id < 2; ++id) {
        Camera camera;
        camera.id = id;
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(id);
        camera.enabled = true;
        camera.pose(0, 3) = static_cast<double>(id) + 0.5;
        chunk.cameras().push_back(camera);
    }

    for (std::int32_t id = 0; id < 2; ++id) {
        auto frame = std::make_shared<Frame>("images/" + std::to_string(id) + ".jpg", 64,
                                            48);
        frame->set_id(id);
        frame->set_camera_id(id);
        chunk.frames().push_back(frame);
    }

    // Keypoints for both cameras, so matches and tracks validate.
    for (std::int32_t camera_id = 0; camera_id < 2; ++camera_id) {
        KeypointSet set;
        set.camera_id = camera_id;
        set.keypoints = {make_keypoint(1.0F + static_cast<float>(camera_id), 2.0F),
                         make_keypoint(3.0F, 4.0F)};
        set.descriptors = {make_descriptor(static_cast<std::uint8_t>(camera_id)),
                           make_descriptor(static_cast<std::uint8_t>(camera_id + 9))};
        chunk.tie_points().keypoint_sets.push_back(set);
    }
    PairMatches pair;
    pair.camera_a = 0;
    pair.camera_b = 1;
    pair.matches = {KeypointMatch{0, 0}, KeypointMatch{1, 1}};
    chunk.tie_points().matches.push_back(pair);
    Track track;
    track.observations = {Observation{0, 0}, Observation{1, 0}};
    track.point = Eigen::Vector3d(1.5, -2.5, 3.5);
    chunk.tie_points().tracks.push_back(track);

    // Two point clouds.
    for (int i = 0; i < 2; ++i) {
        PointCloudAsset& asset = chunk.add_point_cloud("cloud " + std::to_string(i));
        asset.data->points = {Eigen::Vector3d(i, 0.0, 0.0), Eigen::Vector3d(0.0, i, 1.0)};
        asset.data->colours = {1, 2, 3, 4, 5, 6};
        asset.group_id = -1;
    }
    chunk.set_active_point_cloud_id(chunk.point_clouds().front().id);

    // Two models, each with a texture page, and a distinct source link.
    for (int i = 0; i < 2; ++i) {
        ModelAsset& asset = chunk.add_model("model " + std::to_string(i));
        *asset.data = make_mesh(static_cast<double>(i), i * 20);
        asset.textures.push_back(make_texture(3, 2, i * 11));
        asset.source_id = 100 + i;
    }
    chunk.set_active_model_id(chunk.models().back().id);

    // Two depth-map sets.
    for (int i = 0; i < 2; ++i) {
        DepthMapsAsset& asset = chunk.add_depth_maps("depth " + std::to_string(i));
        DepthMap map;
        map.camera_id = i;
        map.width = 2;
        map.height = 1;
        map.downscale = 1;
        map.depths = {1.0F + static_cast<float>(i), -1.0F};
        asset.data->push_back(map);
    }
    chunk.set_active_depth_maps_id(chunk.depth_map_sets().front().id);

    // Two elevations and two orthomosaics.
    for (int i = 0; i < 2; ++i) {
        ElevationAsset& asset = chunk.add_elevation("dem " + std::to_string(i));
        *asset.data = make_elevation(10.0F * static_cast<float>(i));
    }
    for (int i = 0; i < 2; ++i) {
        OrthomosaicAsset& asset = chunk.add_orthomosaic("ortho " + std::to_string(i));
        *asset.data = make_ortho(static_cast<std::uint8_t>(i * 31));
    }
    chunk.set_active_elevation_id(chunk.elevations().front().id);
    chunk.set_active_orthomosaic_id(chunk.orthomosaics().front().id);

    // Three markers, each with projections, one with a reference and a
    // position; a marker group holding all three.
    MarkerGroup marker_group;
    marker_group.id = chunk.allocate_id();
    marker_group.label = "control";
    for (int i = 0; i < 3; ++i) {
        Marker marker;
        marker.id = chunk.allocate_id();
        marker.label = "marker " + std::to_string(i);
        marker.group_id = marker_group.id;
        marker.type = i == 0 ? MarkerType::Fiducial : MarkerType::Regular;
        for (std::int32_t camera_id = 0; camera_id < 2; ++camera_id) {
            MarkerProjection projection;
            projection.camera_id = camera_id;
            projection.pixel = Eigen::Vector2d(1.5 * i + camera_id, 2.25);
            marker.projections.push_back(projection);
        }
        if (i == 0) {
            marker.reference = Eigen::Vector3d(77.2, 28.6, 213.0);
            marker.reference_accuracy = Eigen::Vector3d(0.01, 0.02, 0.03);
            marker.position = Eigen::Vector3d(1.0, 2.0, 3.0);
            marker.position_covariance = Eigen::Matrix3d::Identity() * 0.5;
            marker.pinned = true;
            marker.valid = true;
        }
        marker_group.members.push_back(marker.id);
        chunk.markers().push_back(marker);
    }
    chunk.marker_groups().push_back(marker_group);

    // One scale bar, marker -> camera.
    Scalebar bar;
    bar.id = chunk.allocate_id();
    bar.label = "bar";
    bar.from.kind = ScalebarEndpoint::Kind::Marker;
    bar.from.id = chunk.markers().front().id;
    bar.to.kind = ScalebarEndpoint::Kind::Camera;
    bar.to.id = 1;
    bar.reference_length_m = 2.5;
    bar.reference_accuracy_m = 0.004;
    bar.estimated_length_m = 2.49;
    chunk.scalebars().push_back(bar);

    // Two camera groups (one folder, one station).
    CameraGroup folder_group;
    folder_group.id = chunk.allocate_id();
    folder_group.label = "flight";
    folder_group.members = {0, 1};
    folder_group.type = CameraGroupType::Folder;
    folder_group.folder = "/data/flight";
    chunk.camera_groups().push_back(folder_group);
    CameraGroup station_group;
    station_group.id = chunk.allocate_id();
    station_group.label = "station";
    station_group.members = {1};
    station_group.type = CameraGroupType::Station;
    chunk.camera_groups().push_back(station_group);

    // One mask per camera; the second inverted.
    for (int camera_id = 0; camera_id < 2; ++camera_id) {
        Mask mask;
        mask.camera_id = camera_id;
        mask.width = 3;
        mask.height = 2;
        mask.invert = camera_id == 1;
        mask.data = {0, 0, 1, 1, 0, 0};
        chunk.masks().push_back(mask);
    }

    // Two shapes on two layers, with attributes, and a shape group.
    ShapeGroup shape_group;
    shape_group.id = chunk.allocate_id();
    shape_group.label = "buildings";
    Shape polygon;
    polygon.id = chunk.allocate_id();
    polygon.type = ShapeType::Polygon;
    polygon.label = "roof";
    polygon.vertices = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
                        Eigen::Vector3d(1, 1, 0)};
    polygon.closed = true;
    polygon.attributes["kind"] = "building";
    polygon.attributes["storeys"] = "2";
    polygon.layer = "boundary";
    polygon.has_altitude = true;
    chunk.shapes().features.push_back(polygon);
    Shape point;
    point.id = chunk.allocate_id();
    point.type = ShapeType::Point;
    point.label = "control point";
    point.vertices = {Eigen::Vector3d(0.5, 0.5, 0.5)};
    point.layer = "seamlines";
    chunk.shapes().features.push_back(point);
    shape_group.members = {polygon.id};
    chunk.shapes().groups.push_back(shape_group);

    // One trajectory with two timed poses.
    Trajectory trajectory;
    trajectory.id = chunk.allocate_id();
    trajectory.label = "gnss";
    TrajectoryPoint first;
    first.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    first.timestamp_s = 0.0;
    TrajectoryPoint second;
    second.position = Eigen::Vector3d(2.0, 3.0, 4.0);
    second.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(
        0.5, Eigen::Vector3d::UnitZ()));
    second.timestamp_s = 1.5;
    second.has_orientation = true;
    trajectory.points = {first, second};
    chunk.trajectories().push_back(trajectory);

    // A panorama.
    PanoramaAsset panorama;
    panorama.id = chunk.allocate_id();
    panorama.label = "pano";
    panorama.data = std::make_shared<Image>(make_texture(4, 2, 99));
    chunk.panorama() = panorama;

    // A tiled-model stub carrying opaque bytes.
    TiledModelAsset tiled;
    tiled.id = chunk.allocate_id();
    tiled.label = "tiles";
    tiled.data->payload = {1, 2, 3, 4, 5};
    chunk.tiled_models().push_back(tiled);
    chunk.set_active_tiled_model_id(tiled.id);

    chunk.region().centre = Eigen::Vector3d(1.0, 2.0, 3.0);
    chunk.region().size = Eigen::Vector3d(4.0, 5.0, 6.0);
    chunk.transform()(0, 3) = 0.25;
    chunk.crs().epsg = 32633;
    chunk.crs().wkt = "PROJCRS[\"WGS 84 / UTM zone 33N\"]";
    chunk.meta().set("note", "<&>\"'");

    workspace.set_next_chunk_id(11);
    document.meta().set("author", "a & b");
    document.global_crs().epsg = 4326;
    document.global_crs().wkt = "GEOGCRS[\"WGS 84\"]";
    return document;
}

// -- tests ------------------------------------------------------------------

/// A project written before M11 has version="1" and the flat entry names. The
/// v1 reader must still load it exactly as it did, and a re-save must produce
/// v2 (reading v1 and writing v2 is the documented migration path).
TEST_F(ProjectV2, ReadsVersionOneAndResavesAsVersionTwo) {
    // A minimal, hand-written v1 project: one chunk with a sensor, two cameras,
    // keypoints for both and one match pair, stored under the flat v1 names.
    const std::string psx = path("old.psx");
    fs::path base = psx;
    base.replace_extension();
    const std::string files_dir = base.string() + ".files";

    // The sidecars are produced by the shared encoders, so this test exercises
    // the v1 *document structure*, which is what the split isolated.
    ZipArchive archive;
    {
        KeypointSet first;
        first.camera_id = 0;
        first.keypoints = {make_keypoint(1.0F, 2.0F)};
        first.descriptors = {make_descriptor(1)};
        KeypointSet second;
        second.camera_id = 1;
        second.keypoints = {make_keypoint(3.0F, 4.0F)};
        second.descriptors = {make_descriptor(2)};
        archive.add("tie_points/0.kpt", docxml::sidecar::encode_keypoints(first));
        archive.add("tie_points/1.kpt", docxml::sidecar::encode_keypoints(second));
        PairMatches pair;
        pair.camera_a = 0;
        pair.camera_b = 1;
        pair.matches = {KeypointMatch{0, 0}};
        archive.add("tie_points/matches.bin",
                    docxml::sidecar::encode_matches({pair}));
    }
    const std::string xml =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<document version=\"1\" next_id=\"4\" active_id=\"0\">\n"
        "  <global_crs epsg=\"4326\" wkt=\"WGS84\"/>\n"
        "  <meta/>\n"
        "  <chunks>\n"
        "    <chunk id=\"0\" label=\"legacy\" next_id=\"4\">\n"
        "      <sensors>\n"
        "        <sensor id=\"0\" label=\"s\" type=\"frame\" width=\"64\" "
        "height=\"48\" pixel_size=\"3\" focal_length=\"20\" make=\"ACME\" "
        "model=\"M\" serial=\"sn\" fixed_params=\"\">\n"
        "          <calibration type=\"frame\" width=\"64\" height=\"48\" f=\"60\" "
        "cx=\"32\" cy=\"24\" b1=\"0\" b2=\"0\" k1=\"0\" k2=\"0\" k3=\"0\" "
        "k4=\"0\" p1=\"0\" p2=\"0\"/>\n"
        "        </sensor>\n"
        "      </sensors>\n"
        "      <cameras>\n"
        "        <camera id=\"0\" sensor_id=\"0\" label=\"cam0\" enabled=\"true\" "
        "orientation=\"1\" transform=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n"
        "        <camera id=\"1\" sensor_id=\"0\" label=\"cam1\" enabled=\"true\" "
        "orientation=\"1\" transform=\"1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n"
        "      </cameras>\n"
        "      <frames/>\n"
        "      <tie_points>\n"
        "        <keypoint_set camera_id=\"0\" count=\"1\" path=\"tie_points/0.kpt\"/>\n"
        "        <keypoint_set camera_id=\"1\" count=\"1\" path=\"tie_points/1.kpt\"/>\n"
        "        <matches pairs=\"1\" path=\"tie_points/matches.bin\"/>\n"
        "      </tie_points>\n"
        "      <region centre=\"0 0 0\" size=\"10 10 10\"/>\n"
        "      <transform matrix=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n"
        "      <crs epsg=\"32633\" wkt=\"w\"/>\n"
        "      <meta><property name=\"k\" value=\"v\"/></meta>\n"
        "    </chunk>\n"
        "  </chunks>\n"
        "</document>\n";
    archive.add("doc.xml", xml);

    std::error_code ec;
    fs::create_directories(files_dir, ec);
    ASSERT_FALSE(ec) << ec.message();
    {
        const std::vector<std::uint8_t> out = archive.serialise();
        std::ofstream file(files_dir + "/project.zip", std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(file);
        file.write(reinterpret_cast<const char*>(out.data()),
                   static_cast<std::streamsize>(out.size()));
    }
    {
        const std::string manifest =
            "alphashape-project 1\nfiles=" + fs::path(files_dir).filename().string() +
            "/project.zip\n";
        std::ofstream file(psx, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(file);
        file.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    }

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_EQ(got.id(), 0);
    EXPECT_EQ(got.label(), "legacy");
    EXPECT_EQ(got.sensors().size(), 1U);
    EXPECT_EQ(got.cameras().size(), 2U);
    EXPECT_EQ(got.tie_points().keypoint_sets.size(), 2U);
    ASSERT_EQ(got.tie_points().matches.size(), 1U);
    EXPECT_EQ(got.tie_points().matches[0].camera_b, 1);
    EXPECT_EQ(got.crs().epsg, 32633);
    EXPECT_EQ(got.next_id(), 4);
    // v1 held no markers, masks, shapes or models; they read back empty.
    EXPECT_TRUE(got.markers().empty());
    EXPECT_TRUE(got.masks().empty());
    EXPECT_TRUE(got.shapes().features.empty());
    EXPECT_TRUE(got.models().empty());

    // Saving the loaded document writes the current version under the per-chunk
    // entry names.
    const std::string resaved = path("resaved.psx");
    ProjectIO::save(*loaded, resaved);
    EXPECT_NE(doc_text(resaved).find("<document version=\"" +
                                     std::to_string(Document::kFormatVersion) + "\""),
              std::string::npos);
    EXPECT_TRUE(ZipArchive::parse(zip_bytes(resaved))
                    .contains("chunks/0/tie_points/matches.bin"));
}

TEST_F(ProjectV2, WritesVersionTwo) {
    Document document = build_full_document();
    const std::string psx = path("full.psx");
    ProjectIO::save(document, psx);

    // The layout tests below are the M11 (v2) vocabulary; the CURRENT version
    // number is higher because the depth-map sidecar moved to ADM2 (v3) and
    // then ADM3 (v4, repair provenance). Pin both facts: the number, and that
    // the XML it writes is the v2 vocabulary.
    ASSERT_EQ(Document::kFormatVersion, 4);
    const std::string xml = doc_text(psx);
    EXPECT_NE(xml.find("<document version=\"4\""), std::string::npos);
    EXPECT_NE(xml.find("<chunks>"), std::string::npos);
    // The archive is partitioned per chunk: every bulk entry sits under
    // chunks/<chunk_id>/ and the flat v1 names are gone.
    const ZipArchive archive = ZipArchive::parse(zip_bytes(psx));
    const std::string prefix =
        "chunks/" + std::to_string(document.workspace().chunks().front()->id()) + "/";
    std::size_t bulk = 0;
    for (const ZipEntry& entry : archive.entries()) {
        if (entry.name == "doc.xml") {
            continue;
        }
        EXPECT_EQ(entry.name.rfind(prefix, 0), 0U) << entry.name;
        ++bulk;
    }
    EXPECT_GT(bulk, 0U);
    for (const char* legacy :
         {"tie_points/matches.bin", "model/mesh.bin", "depth_maps/depth_maps.bin",
          "point_cloud/points.bin", "elevation/dem.tif", "orthomosaic/ortho.tif"}) {
        EXPECT_FALSE(archive.contains(legacy)) << legacy;
    }
    // Line endings are LF and indentation two spaces, as §4.4 requires.
    EXPECT_EQ(xml.find('\r'), std::string::npos);
    EXPECT_NE(xml.find("\n  <chunks>\n"), std::string::npos);
}

TEST_F(ProjectV2, Calibration13ParamsRoundTrip) {
    // Non-zero p3/p4 must survive save/load...
    Document document = build_full_document();
    Calibration& cal = document.active_chunk()->sensors()[0].calibration;
    cal.p3 = 2.5e-4;
    cal.p4 = -1.2e-4;
    const std::string psx = path("p3p4.psx");
    ProjectIO::save(document, psx);
    const std::string xml = doc_text(psx);
    EXPECT_NE(xml.find("p3=\""), std::string::npos);
    EXPECT_NE(xml.find("p4=\""), std::string::npos);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_NE(loaded->active_chunk(), nullptr);
    const Calibration& got = loaded->active_chunk()->sensors()[0].calibration;
    EXPECT_DOUBLE_EQ(got.p3, 2.5e-4);
    EXPECT_DOUBLE_EQ(got.p4, -1.2e-4);
    EXPECT_TRUE(*loaded->active_chunk() == *document.active_chunk());

    // ...and the pre-13 layout stays byte-identical: zero terms are omitted.
    Document plain = build_full_document();
    const std::string plain_psx = path("plain.psx");
    ProjectIO::save(plain, plain_psx);
    const std::string plain_xml = doc_text(plain_psx);
    EXPECT_EQ(plain_xml.find("p3=\""), std::string::npos);
    EXPECT_EQ(plain_xml.find("p4=\""), std::string::npos);
}

TEST_F(ProjectV2, UntouchedChunkResaveIsByteStable) {
    Document document = build_full_document();
    const std::string first = path("a.psx");
    ProjectIO::save(document, first);

    // Load what the writer produced, then save it again untouched.
    std::shared_ptr<Document> loaded = ProjectIO::load(first);
    const std::string second = path("b.psx");
    ProjectIO::save(*loaded, second);

    EXPECT_EQ(zip_bytes(first), zip_bytes(second));
    // And a third pass adds nothing.
    std::shared_ptr<Document> again = ProjectIO::load(second);
    const std::string third = path("c.psx");
    ProjectIO::save(*again, third);
    EXPECT_EQ(zip_bytes(first), zip_bytes(third));
}

TEST_F(ProjectV2, RoundTripsAllContainers) {
    Document original = build_full_document();
    const std::string psx = path("full.psx");
    ProjectIO::save(original, psx);
    std::shared_ptr<Document> loaded = ProjectIO::load(psx);

    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    const Chunk& want = *original.workspace().chunks().front();

    EXPECT_EQ(got.point_clouds().size(), 2U);
    EXPECT_EQ(got.models().size(), 2U);
    EXPECT_EQ(got.depth_map_sets().size(), 2U);
    EXPECT_EQ(got.elevations().size(), 2U);
    EXPECT_EQ(got.orthomosaics().size(), 2U);
    EXPECT_EQ(got.markers().size(), 3U);
    EXPECT_EQ(got.marker_groups().size(), 1U);
    EXPECT_EQ(got.scalebars().size(), 1U);
    EXPECT_EQ(got.camera_groups().size(), 2U);
    EXPECT_EQ(got.masks().size(), 2U);
    EXPECT_EQ(got.shapes().features.size(), 2U);
    EXPECT_EQ(got.shapes().groups.size(), 1U);
    EXPECT_EQ(got.trajectories().size(), 1U);
    EXPECT_TRUE(got.panorama().has_value());
    EXPECT_EQ(got.tiled_models().size(), 1U);
    EXPECT_EQ(got.frames().size(), 2U);

    EXPECT_TRUE(got == want);
    EXPECT_TRUE(got.validate().empty());
    EXPECT_EQ(got.next_id(), want.next_id());

    // Spot-check the details the container count would not catch.
    EXPECT_EQ(got.models()[1].textures.size(), 1U);
    EXPECT_EQ(got.models()[1].textures.front().width(), 3);
    EXPECT_EQ(got.models()[0].source_id, 100);
    EXPECT_EQ(got.masks()[1].invert, true);
    EXPECT_EQ(got.masks()[0].data.size(), 6U);
    EXPECT_EQ(got.shapes().features[0].attributes.at("kind"), "building");
    EXPECT_EQ(got.shapes().features[1].layer, "seamlines");
    EXPECT_EQ(got.markers()[0].projections.size(), 2U);
    EXPECT_EQ(got.markers()[0].type, MarkerType::Fiducial);
    EXPECT_EQ(got.camera_groups()[0].type, CameraGroupType::Folder);
    EXPECT_EQ(got.camera_groups()[1].type, CameraGroupType::Station);
    EXPECT_TRUE(got.trajectories()[0].points[1].has_orientation);

    // The active ids survive, which is what a later export selects by.
    EXPECT_EQ(got.active_point_cloud_id(), want.active_point_cloud_id());
    EXPECT_EQ(got.active_model_id(), want.active_model_id());
    EXPECT_EQ(got.active_depth_maps_id(), want.active_depth_maps_id());
    EXPECT_EQ(got.active_elevation_id(), want.active_elevation_id());
    EXPECT_EQ(got.active_orthomosaic_id(), want.active_orthomosaic_id());
    EXPECT_EQ(got.active_tiled_model_id(), want.active_tiled_model_id());
}

TEST_F(ProjectV2, EmptyContainersAreOmitted) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("empty");
    chunk.sensors().push_back(Sensor{});
    document.set_active_chunk_id(chunk.id());
    const std::string psx = path("empty.psx");
    ProjectIO::save(document, psx);

    const std::string xml = doc_text(psx);
    for (const char* element :
         {"<frames", "<markers", "<scalebars", "<camera_groups", "<tie_points",
          "<mask_sets", "<depth_map_sets", "<dense_clouds", "<models", "<tiled_models",
          "<elevations", "<orthomosaics", "<trajectories", "<shapes", "<panorama"}) {
        EXPECT_EQ(xml.find(element), std::string::npos) << element;
    }
    // The containers that always exist are still there.
    EXPECT_NE(xml.find("<sensors>"), std::string::npos);
    EXPECT_NE(xml.find("<cameras/>"), std::string::npos);
    EXPECT_NE(xml.find("<region "), std::string::npos);
    EXPECT_NE(xml.find("<transform "), std::string::npos);

    EXPECT_EQ(ZipArchive::parse(zip_bytes(psx)).entries().size(), 1U);  // doc.xml only
    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    EXPECT_TRUE(loaded->workspace().chunks().front()->markers().empty());
    EXPECT_TRUE(loaded->workspace().chunks().front()->shapes().features.empty());
    EXPECT_FALSE(loaded->workspace().chunks().front()->panorama().has_value());
}

TEST_F(ProjectV2, SidecarMagicsAreDistinct) {
    Document document = build_full_document();
    const std::string psx = path("magic.psx");
    ProjectIO::save(document, psx);

    const ZipArchive archive = ZipArchive::parse(zip_bytes(psx));
    // Every entry that is not doc.xml is either a GeoTIFF, a PNG, or a
    // magic-tagged sidecar whose four-byte magic is one of the documented ones.
    // A mislabelled entry (a decoder reading a different layout) cannot pass.
    const std::set<std::string> known = {"AKP1", "AMT1", "ATR2", "ADM1", "ADM2", "ADM3",
                                         "APC1", "AMS2", "AOL1", "AMK1", "ASH1",
                                         "ATJ1", "AFR1", "ATM1"};
    std::set<std::string> seen;
    std::size_t sidecars = 0;
    for (const ZipEntry& entry : archive.entries()) {
        if (entry.name == "doc.xml") {
            continue;
        }
        ASSERT_GE(entry.data.size(), 4U) << entry.name;
        const char* data = reinterpret_cast<const char*>(entry.data.data());
        const bool is_tiff = std::memcmp(data, "II*\0", 4) == 0 ||
                             std::memcmp(data, "MM\0*", 4) == 0;
        const bool is_png = std::memcmp(data, "\x89PNG", 4) == 0;
        if (is_tiff || is_png) {
            continue;
        }
        const std::string magic(data, 4);
        EXPECT_TRUE(known.count(magic) == 1U) << entry.name << " has magic " << magic;
        seen.insert(magic);
        ++sidecars;
    }
    EXPECT_GT(sidecars, 6U);
    // The M11 containers all appear, i.e. every new magic is exercised.
    for (const char* expected : {"AMK1", "ASH1", "ATJ1", "AFR1", "ATM1"}) {
        EXPECT_TRUE(seen.count(expected) == 1U) << expected;
    }
}


// ---------------------------------------------------------------------------
// WP-11.4: the versioned reader, v1 compatibility and fuzzing (§4.5).
//
// The v1 fixture is produced by tests/v1_writer.hpp, which reproduces the
// pre-M11 writer (version="1", flat sidecar names, one asset per kind) and by
// tests/v1_fixture.hpp, the document behind the checked-in
// tests/fixtures/v1_project.zip. Reading either back has to yield the document
// it was saved from: that is what makes every project written before M11 still
// openable.
// ---------------------------------------------------------------------------

namespace {

/// Deterministic draws for the fuzzer. rng::uniform_below is inclusive of its
/// argument; every use here is an index or offset into a container, so this
/// wrapper returns [0, bound).
std::uint32_t fuzz_draw(std::mt19937_64& engine, std::uint32_t bound) {
    if (bound == 0) {
        return 0;
    }
    return rng::uniform_below(engine, bound - 1);
}

/// Sidecar entry indices (everything but doc.xml) with a payload worth mutating.
std::vector<std::size_t> sidecar_indices(const std::vector<ZipEntry>& entries) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].name != "doc.xml" && entries[i].data.size() > 8) {
            out.push_back(i);
        }
    }
    return out;
}

}  // namespace

/// tests/fixtures/v1_project.zip is checked in. Regenerating it in memory must
/// reproduce the same bytes, so the checked-in file and the builder in
/// tests/v1_fixture.hpp cannot drift apart unnoticed.
TEST_F(ProjectV2, V1FixtureBytesMatchTheGenerator) {
    const std::vector<std::uint8_t> regenerated =
        test_v1::write_project(test_v1::make_v1_fixture_document());
    std::ifstream in(std::string(ALPHASHAPE_TEST_FIXTURE_DIR) + "/v1_project.zip",
                     std::ios::binary);
    ASSERT_TRUE(in) << "tests/fixtures/v1_project.zip is missing";
    const std::vector<std::uint8_t> checked_in((std::istreambuf_iterator<char>(in)),
                                               std::istreambuf_iterator<char>());
    // Compare decoded entries, not archive bytes: deflate output differs between
    // zlib builds (zlib, zlib-ng, Apple's), so raw bytes are platform-dependent.
    // The .tif sidecars are deflate-compressed inside too; their decoded content is
    // checked by the load test below, so here only their presence is compared.
    const std::vector<ZipEntry> want = ZipArchive::parse(regenerated).entries();
    const std::vector<ZipEntry> have = ZipArchive::parse(checked_in).entries();
    ASSERT_EQ(want.size(), have.size())
        << "regenerate with: make_v1_fixture > tests/fixtures/v1_project.zip";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(want[i].name, have[i].name) << "entry " << i;
        const std::string& name = want[i].name;
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tif") == 0) {
            continue;
        }
        EXPECT_EQ(want[i].data, have[i].data)
            << want[i].name
            << " differs; regenerate with: make_v1_fixture > tests/fixtures/v1_project.zip";
    }
}

/// The checked-in fixture loads to exactly the document it was built from:
/// chunk/camera/keypoint/track counts and the DEM grid and heights, plus the
/// ortho labels and the mesh texture. This is the §4.5 compatibility promise.
TEST_F(ProjectV2, LoadsV1Fixture) {
    const Document original = test_v1::make_v1_fixture_document();
    std::ifstream in(std::string(ALPHASHAPE_TEST_FIXTURE_DIR) + "/v1_project.zip",
                     std::ios::binary);
    ASSERT_TRUE(in) << "tests/fixtures/v1_project.zip is missing";
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);

    // The bytes really are the pre-M11 layout: version 1, flat entry names.
    const std::string xml = archive.get_text("doc.xml");
    EXPECT_NE(xml.find("<document version=\"1\""), std::string::npos);
    for (const char* entry : {"tie_points/0.kpt", "tie_points/matches.bin",
                              "tie_points/tracks.bin", "model/mesh.bin",
                              "model/texture_0.png", "elevation/dem.tif",
                              "orthomosaic/ortho.tif", "orthomosaic/labels.bin"}) {
        EXPECT_TRUE(archive.contains(entry)) << entry;
    }
    EXPECT_FALSE(archive.contains("chunks/0/tie_points/matches.bin"));

    ReadReport report;
    const std::shared_ptr<Document> loaded = read_document(archive, &report);
    EXPECT_EQ(report.version, 1);
    EXPECT_TRUE(report.warnings.empty())
        << (report.warnings.empty() ? std::string() : report.warnings.front());

    const Chunk& want = *original.workspace().chunks().front();
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_EQ(got.id(), want.id());
    EXPECT_EQ(got.label(), want.label());
    EXPECT_EQ(got.sensors().size(), want.sensors().size());
    EXPECT_EQ(got.cameras().size(), 3U);
    EXPECT_EQ(got.frames().size(), 3U);
    ASSERT_EQ(got.tie_points().keypoint_sets.size(), 3U);
    for (std::size_t c = 0; c < got.tie_points().keypoint_sets.size(); ++c) {
        EXPECT_EQ(got.tie_points().keypoint_sets[c].keypoints.size(), 150U);
        EXPECT_EQ(got.tie_points().keypoint_sets[c].keypoints,
                  want.tie_points().keypoint_sets[c].keypoints);
    }
    EXPECT_EQ(got.tie_points().matches.size(), 2U);
    EXPECT_EQ(got.tie_points().tracks.size(), 8U);

    ASSERT_EQ(got.elevations().size(), 1U);
    const Elevation& dem = *got.elevations().front().data;
    EXPECT_EQ(dem.raster.grid.width, 4);
    EXPECT_EQ(dem.raster.grid.height, 3);
    EXPECT_EQ(dem.raster.grid.origin_x, 500000.0);
    EXPECT_EQ(dem.epsg(), 32633);
    EXPECT_EQ(dem.nodata, -32767.0F);
    EXPECT_EQ(dem.source, "depth_maps");
    EXPECT_EQ(dem.raster.data, want.elevations().front().data->raster.data);
    ASSERT_EQ(got.orthomosaics().size(), 1U);
    EXPECT_EQ(got.orthomosaics().front().data->labels,
              want.orthomosaics().front().data->labels);
    ASSERT_EQ(got.models().size(), 1U);
    EXPECT_EQ(got.models().front().data->faces.size(), 2U);
    ASSERT_EQ(got.models().front().textures.size(), 1U);
    EXPECT_EQ(got.models().front().textures.front().width(), 4);
    EXPECT_TRUE(got.validate().empty());
}

/// The same compatibility promise on a document this build produces: every
/// container the v1 *writer* could emit (build_full_document), minus the M11
/// ones, round-trips through the v1 writer and back.
TEST_F(ProjectV2, LoadsV1RichProject) {
    Document original = build_full_document();
    // Drop the v1-unrepresentable containers so v1 and v2 agree. The document is
    // still rich: every asset kind, both tie-point containers, DEM, ortho.
    Chunk& chunk = *original.workspace().chunks().front();
    chunk.markers().clear();
    chunk.marker_groups().clear();
    chunk.scalebars().clear();
    chunk.camera_groups().clear();
    chunk.masks().clear();
    chunk.shapes().features.clear();
    chunk.shapes().groups.clear();
    chunk.trajectories().clear();
    chunk.panorama().reset();
    chunk.tiled_models().clear();
    // v1 stored the active asset only, so keep exactly those.
    std::erase_if(chunk.point_clouds(),
                  [&](const PointCloudAsset& a) { return a.id != chunk.active_point_cloud_id(); });
    std::erase_if(chunk.models(),
                  [&](const ModelAsset& a) { return a.id != chunk.active_model_id(); });
    std::erase_if(chunk.depth_map_sets(),
                  [&](const DepthMapsAsset& a) { return a.id != chunk.active_depth_maps_id(); });
    std::erase_if(chunk.elevations(),
                  [&](const ElevationAsset& a) { return a.id != chunk.active_elevation_id(); });
    std::erase_if(chunk.orthomosaics(),
                  [&](const OrthomosaicAsset& a) { return a.id != chunk.active_orthomosaic_id(); });

    const ZipArchive archive = ZipArchive::parse(test_v1::write_project(original));
    // The bytes really are the pre-M11 layout.
    const std::string xml = archive.get_text("doc.xml");
    EXPECT_NE(xml.find("<document version=\"1\""), std::string::npos);
    EXPECT_TRUE(archive.contains("tie_points/matches.bin"));
    EXPECT_TRUE(archive.contains("tie_points/tracks.bin"));
    EXPECT_TRUE(archive.contains("model/mesh.bin"));
    EXPECT_TRUE(archive.contains("model/texture_0.png"));
    EXPECT_TRUE(archive.contains("elevation/dem.tif"));
    EXPECT_TRUE(archive.contains("orthomosaic/ortho.tif"));
    EXPECT_FALSE(archive.contains("chunks/0/tie_points/matches.bin"));

    ReadReport report;
    const std::shared_ptr<Document> loaded = read_document(archive, &report);
    EXPECT_EQ(report.version, 1);
    EXPECT_TRUE(report.warnings.empty())
        << (report.warnings.empty() ? std::string() : report.warnings.front());

    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_EQ(got.id(), chunk.id());
    EXPECT_EQ(got.label(), chunk.label());
    EXPECT_EQ(got.sensors().size(), chunk.sensors().size());
    EXPECT_EQ(got.cameras().size(), chunk.cameras().size());
    EXPECT_EQ(got.frames().size(), chunk.frames().size());
    ASSERT_EQ(got.tie_points().keypoint_sets.size(),
              chunk.tie_points().keypoint_sets.size());
    for (std::size_t i = 0; i < got.tie_points().keypoint_sets.size(); ++i) {
        EXPECT_EQ(got.tie_points().keypoint_sets[i].keypoints.size(),
                  chunk.tie_points().keypoint_sets[i].keypoints.size());
    }
    EXPECT_EQ(got.tie_points().matches.size(), chunk.tie_points().matches.size());
    EXPECT_EQ(got.tie_points().tracks.size(), chunk.tie_points().tracks.size());

    // DEM grid and heights, and the ortho's seamline labels, survive exactly.
    ASSERT_EQ(got.elevations().size(), 1U);
    ASSERT_EQ(chunk.elevations().size(), 1U);
    const Elevation& dem = *got.elevations().front().data;
    const Elevation& want_dem = *chunk.elevations().front().data;
    EXPECT_EQ(dem.raster.grid.width, want_dem.raster.grid.width);
    EXPECT_EQ(dem.raster.grid.height, want_dem.raster.grid.height);
    EXPECT_EQ(dem.raster.grid.origin_x, want_dem.raster.grid.origin_x);
    EXPECT_EQ(dem.epsg(), want_dem.epsg());
    EXPECT_EQ(dem.nodata, want_dem.nodata);
    EXPECT_EQ(dem.source, want_dem.source);
    EXPECT_EQ(dem.raster.data, want_dem.raster.data);
    ASSERT_EQ(got.orthomosaics().size(), 1U);
    EXPECT_EQ(got.orthomosaics().front().data->labels,
              chunk.orthomosaics().front().data->labels);

    // One asset per kind: v1 could store no more, and M11's containers were
    // not in the format at all.
    EXPECT_EQ(got.depth_map_sets().size(), 1U);
    EXPECT_EQ(got.point_clouds().size(), 1U);
    EXPECT_EQ(got.models().size(), 1U);
    EXPECT_TRUE(got.markers().empty());
    EXPECT_TRUE(got.masks().empty());
    EXPECT_TRUE(got.shapes().features.empty());
    EXPECT_FALSE(got.panorama().has_value());
    EXPECT_TRUE(got.validate().empty());
}

/// The M9 product shape at real fixture scale: a georeferenced chunk whose DEM
/// and orthomosaic come from data/aerial_truth_*. That is what an M9-era
/// project on disk looks like (depth maps, DEM, ortho, all in UTM 33N), and the
/// v1 reader must return it unchanged. Depth maps are written directly from the
/// truth surface rather than run through the stereo pipeline, which takes
/// minutes per image on the CI VM and is covered by its own tests.
TEST_F(ProjectV2, LoadsV1M9Project) {
    using namespace alphashape::aerial_fixture;
    const TruthGrid truth = load_truth_dem();

    Document original = make_truth_chunk_document();
    Chunk& chunk = *original.active_chunk();
    // Downscale the truth grid 8x: the geometry is the same surface, and the
    // fixture stays small enough to keep the test at a fraction of a second.
    constexpr int kStep = 8;
    Elevation dem;
    dem.raster.grid.width = truth.width / kStep;
    dem.raster.grid.height = truth.height / kStep;
    dem.raster.grid.pixel_size_x = truth.cell * kStep;
    dem.raster.grid.pixel_size_y = truth.cell * kStep;
    dem.raster.grid.origin_x = truth.origin_x;
    dem.raster.grid.origin_y = truth.origin_y;
    dem.raster.channels = 1;
    dem.raster.data.resize(static_cast<std::size_t>(dem.raster.grid.width) *
                           static_cast<std::size_t>(dem.raster.grid.height));
    for (int row = 0; row < dem.raster.grid.height; ++row) {
        for (int col = 0; col < dem.raster.grid.width; ++col) {
            dem.raster.at(col, row) = truth.at(col * kStep, row * kStep);
        }
    }
    // The DEM is expressed in UTM 33N through the chunk's ENU placement.
    dem.set_epsg(32633);
    dem.nodata = -32767.0F;
    dem.source = "depth_maps";
    chunk.elevation() = dem;

    // One depth map per camera, at the DEM's cell size: a constant range to the
    // truth surface, which is enough to exercise the encoder and the reader.
    for (std::size_t c = 0; c < chunk.cameras().size(); ++c) {
        DepthMap map;
        map.camera_id = static_cast<std::int32_t>(c);
        map.width = 8;
        map.height = 6;
        map.downscale = 8;
        map.depths.assign(48, 60.0F + static_cast<float>(c));
        chunk.depth_maps().push_back(std::move(map));
    }

    Orthomosaic ortho;
    ortho.raster.grid.width = truth.width / (kStep * 2);
    ortho.raster.grid.height = truth.height / (kStep * 2);
    ortho.raster.grid.pixel_size_x = truth.cell * kStep * 2;
    ortho.raster.grid.pixel_size_y = truth.cell * kStep * 2;
    ortho.raster.grid.origin_x = truth.origin_x;
    ortho.raster.grid.origin_y = truth.origin_y;
    ortho.raster.channels = 4;
    const std::size_t cells = static_cast<std::size_t>(ortho.raster.grid.width) *
                              static_cast<std::size_t>(ortho.raster.grid.height);
    ortho.raster.data.resize(cells * 4);
    ortho.labels.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        ortho.raster.data[i * 4 + 0] = static_cast<std::uint8_t>(i % 256);
        ortho.raster.data[i * 4 + 1] = static_cast<std::uint8_t>((i * 3) % 256);
        ortho.raster.data[i * 4 + 2] = static_cast<std::uint8_t>((i * 7) % 256);
        ortho.raster.data[i * 4 + 3] = 255;
        ortho.labels[i] = static_cast<std::int32_t>(i % 12);
    }
    ortho.set_epsg(32633);
    ortho.surface = "elevation";
    chunk.orthomosaic() = ortho;

    // Round-trip through the v1 writer, then read it back.
    const ZipArchive v1 = ZipArchive::parse(test_v1::write_project(original));
    ReadReport report;
    const std::shared_ptr<Document> loaded = read_document(v1, &report);
    EXPECT_EQ(report.version, 1);
    EXPECT_TRUE(report.warnings.empty())
        << (report.warnings.empty() ? std::string() : report.warnings.front());

    const Chunk& got = *loaded->active_chunk();
    EXPECT_EQ(got.cameras().size(), 12U);
    EXPECT_EQ(got.sensors().size(), 1U);
    EXPECT_EQ(got.depth_map_sets().front().data->size(), 12U);
    ASSERT_EQ(got.elevations().size(), 1U);
    const Elevation& got_dem = *got.elevations().front().data;
    EXPECT_EQ(got_dem.raster.grid.width, dem.raster.grid.width);
    EXPECT_EQ(got_dem.raster.grid.height, dem.raster.grid.height);
    EXPECT_EQ(got_dem.epsg(), 32633);
    EXPECT_EQ(got_dem.raster.data, dem.raster.data);
    // The truth heights survive the GeoTIFF round-trip to float precision.
    EXPECT_EQ(got_dem.raster.at(0, 0), truth.at(0, 0));
    EXPECT_EQ(got_dem.raster.at(10, 5), truth.at(10 * kStep, 5 * kStep));
    ASSERT_EQ(got.orthomosaics().size(), 1U);
    const Orthomosaic& got_ortho = *got.orthomosaics().front().data;
    EXPECT_EQ(got_ortho.raster.data, ortho.raster.data);
    EXPECT_EQ(got_ortho.labels, ortho.labels);
    EXPECT_TRUE(got.validate().empty());
}

/// v1 in, v2 out, v2 in again: the migration path leaves nothing behind that
/// the v1 reader could express.
TEST_F(ProjectV2, MigratesV1ToV2) {
    const Document original = build_full_document();
    const ZipArchive v1 = ZipArchive::parse(test_v1::write_project(original));
    const std::shared_ptr<Document> loaded = read_document(v1);

    ZipArchive out;
    write_document(*loaded, out);
    EXPECT_NE(out.get_text("doc.xml").find("<document version=\"4\""),
              std::string::npos);
    EXPECT_TRUE(out.contains("chunks/" +
                             std::to_string(loaded->workspace().chunks().front()->id()) +
                             "/tie_points/matches.bin"));

    ReadReport report;
    const std::shared_ptr<Document> again =
        read_document(ZipArchive::parse(out.serialise()), &report);
    EXPECT_EQ(report.version, Document::kFormatVersion);
    ASSERT_EQ(again->workspace().chunks().size(), 1U);
    EXPECT_TRUE(*again->workspace().chunks().front() ==
                *loaded->workspace().chunks().front());
}

TEST_F(ProjectV2, ReadsVersionTwoWithAdm1DepthMaps) {
    // A v2 project is still readable: its depth-map sidecars are ADM1, which
    // decode_depth_maps dispatches on its own magic. The version bump to 3 is
    // about not handing an ADM2 sidecar to an older build; it must not cost us
    // the ability to open files we wrote yesterday.
    ZipArchive archive;
    write_document(build_full_document(), archive);

    // Downgrade every depth-map sidecar from ADM2 to ADM1 (the pre-fingerprint
    // record: no u64 fingerprint, two flags rather than three, no filter mode),
    // keeping the map count -- and therefore doc.xml's count attribute -- the
    // same. Relabel the document as v2: exactly what the previous release wrote.
    std::size_t downgraded = 0;
    // Collect first: add() mutates the entry list this loop would iterate.
    std::vector<std::string> sidecars;
    for (const ZipEntry& entry : archive.entries()) {
        if (entry.name.find("/depth_maps/") != std::string::npos &&
            entry.name.size() >= 4 &&
            entry.name.substr(entry.name.size() - 4) == ".bin") {
            sidecars.push_back(entry.name);
        }
    }
    for (const std::string& name : sidecars) {
        const std::vector<DepthMap> maps = docxml::sidecar::decode_depth_maps(archive.get(name));
        std::vector<std::uint8_t> adm1 = {'A', 'D', 'M', '1'};
        docxml::sidecar::put_u32(adm1, static_cast<std::uint32_t>(maps.size()));
        for (const DepthMap& map : maps) {
            docxml::sidecar::put_i32(adm1, map.camera_id);
            docxml::sidecar::put_i32(adm1, map.width);
            docxml::sidecar::put_i32(adm1, map.height);
            docxml::sidecar::put_i32(adm1, map.downscale);
            adm1.push_back(map.normals.empty() ? 0 : 1);
            adm1.push_back(map.confidence.empty() ? 0 : 1);
            for (const float d : map.depths) {
                docxml::sidecar::put_f32(adm1, d);
            }
            for (const float n : map.normals) {
                docxml::sidecar::put_f32(adm1, n);
            }
            for (const float c : map.confidence) {
                docxml::sidecar::put_f32(adm1, c);
            }
        }
        archive.add(name, adm1);
        ++downgraded;
    }
    ASSERT_GT(downgraded, 0U) << "the fixture has no depth-map sidecar to downgrade";

    std::string xml = archive.get_text("doc.xml");
    const std::string current = "<document version=\"" +
                                std::to_string(Document::kFormatVersion) + "\"";
    const std::size_t at = xml.find(current);
    ASSERT_NE(at, std::string::npos);
    xml.replace(at, current.size(), "<document version=\"2\"");
    archive.add("doc.xml", xml);

    ReadReport report;
    const std::shared_ptr<Document> loaded = read_document(archive, &report);
    EXPECT_EQ(report.version, 2);
    ASSERT_NE(loaded->active_chunk(), nullptr);
    // ADM1 carries no fingerprint, so every map reads as "unknown" (0, never
    // reused) and as the mild filter a pre-ADM2 project could only have run.
    ASSERT_FALSE(loaded->active_chunk()->depth_map_sets().empty());
    std::size_t maps_seen = 0;
    for (const DepthMapsAsset& asset : loaded->active_chunk()->depth_map_sets()) {
        for (const DepthMap& map : *asset.data) {
            EXPECT_EQ(map.alignment_fingerprint, 0U) << "camera " << map.camera_id;
            EXPECT_EQ(map.filter_mode, 1) << "camera " << map.camera_id;
            ++maps_seen;
        }
    }
    EXPECT_GT(maps_seen, 0U);

    // Reading it back does not silently upgrade the file: saving writes the
    // CURRENT version.
    ZipArchive again;
    write_document(*loaded, again);
    EXPECT_NE(again.get_text("doc.xml").find(
                  "<document version=\"" + std::to_string(Document::kFormatVersion) + "\""),
              std::string::npos);
}

TEST_F(ProjectV2, RejectsNewerVersion) {
    ZipArchive archive;
    write_document(build_full_document(), archive);
    std::string xml = archive.get_text("doc.xml");
    const std::string from = "<document version=\"" +
                             std::to_string(Document::kFormatVersion) + "\"";
    const std::size_t at = xml.find(from);
    ASSERT_NE(at, std::string::npos);
    xml.replace(at, from.size(),
                "<document version=\"" + std::to_string(Document::kFormatVersion + 1) + "\"");
    archive.add("doc.xml", xml);
    EXPECT_THROW(read_document(archive), ProjectIOError);

    // No version attribute, or version 0: neither is a readable project.
    ZipArchive bare;
    bare.add("doc.xml", "<document next_id=\"1\" active_id=\"0\"/>");
    EXPECT_THROW(read_document(bare), ProjectIOError);
    ZipArchive zero;
    zero.add("doc.xml", "<document version=\"0\" next_id=\"1\" active_id=\"0\"/>");
    EXPECT_THROW(read_document(zero), ProjectIOError);

    // Writing anything but the current version is refused, not silently
    // upgraded. v2 and v1 are read-only formats now.
    EXPECT_THROW(write_document(build_full_document(), archive, 1), ProjectIOError);
    EXPECT_THROW(write_document(build_full_document(), archive, 2), ProjectIOError);
    EXPECT_THROW(write_document(build_full_document(), archive, 3), ProjectIOError);
}

TEST_F(ProjectV2, RejectsUnknownV2Element) {
    ZipArchive archive;
    write_document(build_full_document(), archive);
    std::string xml = archive.get_text("doc.xml");
    // Insert an element the v2 vocabulary does not list immediately before the
    // first <region>, i.e. inside <chunk>. v2 is strict there.
    const std::size_t region = xml.find("<region ");
    ASSERT_NE(region, std::string::npos);
    xml.insert(xml.rfind('\n', region) + 1, "      <warp_grid id=\"0\"/>\n");
    archive.add("doc.xml", xml);
    EXPECT_THROW(read_document(archive), ProjectIOError);
}

TEST_F(ProjectV2, ReadReportListsWarnings) {
    Document document = build_full_document();
    Chunk& chunk = *document.workspace().chunks().front();
    // An absolute frame path that does not exist: unresolvable but not fatal.
    const std::string missing = path("definitely_absent.jpg");
    chunk.frames().front()->set_path(missing);

    ZipArchive archive;
    write_document(document, archive);
    std::string xml = archive.get_text("doc.xml");
    // An attribute on <chunk> a later revision could have added.
    const std::size_t chunk_at = xml.find("<chunk id=");
    ASSERT_NE(chunk_at, std::string::npos);
    xml.insert(xml.find('>', chunk_at), " future_revision=\"7\"");
    // An element at the document root: v2 is strict only inside <chunk>, so a
    // root-level newcomer is skipped with a warning.
    const std::size_t chunks = xml.find("<chunks>");
    ASSERT_NE(chunks, std::string::npos);
    xml.insert(xml.rfind('\n', chunks) + 1, "  <lighting mode=\"hdri\"/>\n");
    archive.add("doc.xml", xml);

    ReadReport report;
    const std::shared_ptr<Document> loaded = read_document(archive, &report);
    EXPECT_EQ(report.version, Document::kFormatVersion);
    ASSERT_FALSE(report.warnings.empty());
    const auto mentions = [&report](const std::string& text) {
        return std::any_of(report.warnings.begin(), report.warnings.end(),
                           [&text](const std::string& w) {
                               return w.find(text) != std::string::npos;
                           });
    };
    EXPECT_TRUE(mentions("future_revision"));
    EXPECT_TRUE(mentions("lighting"));
    EXPECT_TRUE(mentions(missing));

    // Warnings are advisory: everything else still loaded.
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_EQ(loaded->workspace().chunks().front()->frames().size(),
              chunk.frames().size());
    EXPECT_EQ(loaded->workspace().chunks().front()->models().size(), 2U);

    // Omitting the report changes nothing, and a reused report is reset.
    EXPECT_NO_THROW(read_document(archive));
    ReadReport reused;
    reused.version = 42;
    reused.warnings.emplace_back("stale");
    (void)read_document(archive, &reused);
    EXPECT_EQ(reused.version, Document::kFormatVersion);
    EXPECT_TRUE(std::find(reused.warnings.begin(), reused.warnings.end(), "stale") ==
                reused.warnings.end());
}

/// Deterministic mutation fuzzing (§4.5 / WP-11.4).
///
/// Each case rebuilds the fixture, applies one mutation, and asserts the whole
/// error contract: a document or a ProjectIOError -- never another exception
/// type, no crash, no hang, no runaway allocation. Every decoder bounds a
/// declared count against its remaining bytes, and the zip layer rejects a
/// declared entry length the compressed data cannot produce.
///
/// Families: archive truncation, doc.xml bit flips, random attribute values, a
/// renamed sidecar entry, an oversized declared count, bit flips inside a
/// sidecar payload. Seed and iteration count are fixed, so a failure reproduces.
TEST_F(ProjectV2, FuzzCorruptArchives) {
    const std::vector<std::uint8_t> valid = test_v1::write_project(build_full_document());
    ASSERT_GT(valid.size(), 1000U);
    std::mt19937_64 engine(0x5eedULL);

    int loaded = 0;
    int rejected = 0;
    std::string unexpected;

    for (int case_index = 0; case_index < kFuzzCases && unexpected.empty(); ++case_index) {
        const std::uint32_t kind = fuzz_draw(engine, 6);
        try {
            switch (kind) {
                case 0: {  // truncate the archive
                    const auto cut = static_cast<std::ptrdiff_t>(
                        fuzz_draw(engine, static_cast<std::uint32_t>(valid.size())));
                    const std::vector<std::uint8_t> bytes(valid.begin(), valid.begin() + cut);
                    (void)read_document(ZipArchive::parse(bytes));
                    break;
                }
                case 1: {  // bit flips in the doc.xml text
                    ZipArchive archive = ZipArchive::parse(valid);
                    std::string xml = archive.get_text("doc.xml");
                    const int flips = 1 + static_cast<int>(fuzz_draw(engine, 16));
                    for (int f = 0; f < flips && !xml.empty(); ++f) {
                        const std::size_t at =
                            fuzz_draw(engine, static_cast<std::uint32_t>(xml.size()));
                        xml[at] = static_cast<char>(
                            static_cast<unsigned char>(xml[at]) ^
                            static_cast<unsigned char>(1U << fuzz_draw(engine, 8)));
                    }
                    archive.add("doc.xml", xml);
                    (void)read_document(ZipArchive::parse(archive.serialise()));
                    break;
                }
                case 2: {  // random attribute values in doc.xml
                    ZipArchive archive = ZipArchive::parse(valid);
                    std::string xml = archive.get_text("doc.xml");
                    static const char* const kValues[] = {
                        "-1",                   "0",    "4294967295",
                        "999999999999999999999", "nan",  "inf",
                        "&bogus;",              "<",    "1e400",
                        "-2147483648",          "true", "",
                    };
                    const int edits = 1 + static_cast<int>(fuzz_draw(engine, 8));
                    for (int e = 0; e < edits; ++e) {
                        const std::size_t open = xml.find('"');
                        if (open == std::string::npos) {
                            break;
                        }
                        const std::size_t close = xml.find('"', open + 1);
                        if (close == std::string::npos) {
                            break;
                        }
                        const std::size_t at =
                            open + 1 +
                            fuzz_draw(engine, static_cast<std::uint32_t>(close - open));
                        const std::string value =
                            kValues[fuzz_draw(engine, static_cast<std::uint32_t>(
                                                          sizeof kValues / sizeof kValues[0]))];
                        xml.insert(at, value);
                        if (xml.size() > (2U << 20)) {
                            break;
                        }
                    }
                    archive.add("doc.xml", xml);
                    (void)read_document(ZipArchive::parse(archive.serialise()));
                    break;
                }
                case 3: {  // rename a sidecar so a doc.xml reference dangles
                    ZipArchive archive = ZipArchive::parse(valid);
                    std::vector<ZipEntry> entries = archive.entries();
                    const std::vector<std::size_t> sidecars = sidecar_indices(entries);
                    if (!sidecars.empty()) {
                        entries[sidecars[fuzz_draw(engine, static_cast<std::uint32_t>(
                                                       sidecars.size()))]]
                            .name += ".renamed";
                    }
                    ZipArchive broken;
                    for (const ZipEntry& entry : entries) {
                        broken.add(entry.name, entry.data);
                    }
                    (void)read_document(ZipArchive::parse(broken.serialise()));
                    break;
                }
                case 4: {  // an oversized declared count or size
                    ZipArchive archive = ZipArchive::parse(valid);
                    std::string xml = archive.get_text("doc.xml");
                    static const char* const kCounts[] = {"count", "pairs", "faces",
                                                          "vertices", "textures"};
                    std::vector<std::size_t> sites;
                    for (const char* name : kCounts) {
                        std::size_t at = 0;
                        const std::string needle = std::string(" ") + name + "=\"";
                        while ((at = xml.find(needle, at)) != std::string::npos) {
                            sites.push_back(at + needle.size());
                            at += needle.size();
                        }
                    }
                    if (!sites.empty()) {
                        const std::size_t at =
                            sites[fuzz_draw(engine, static_cast<std::uint32_t>(sites.size()))];
                        const std::size_t end = xml.find('"', at);
                        xml.replace(at, end - at,
                                    fuzz_draw(engine, 2) == 0 ? "4294967295"
                                                              : "-4294967295");
                    }
                    archive.add("doc.xml", xml);
                    (void)read_document(ZipArchive::parse(archive.serialise()));
                    break;
                }
                default: {  // bit flips inside a sidecar payload
                    ZipArchive archive = ZipArchive::parse(valid);
                    std::vector<ZipEntry> entries = archive.entries();
                    const std::vector<std::size_t> sidecars = sidecar_indices(entries);
                    if (!sidecars.empty()) {
                        ZipEntry& entry =
                            entries[sidecars[fuzz_draw(engine, static_cast<std::uint32_t>(
                                                         sidecars.size()))]];
                        const int flips = 1 + static_cast<int>(fuzz_draw(engine, 8));
                        for (int f = 0; f < flips; ++f) {
                            const std::size_t at =
                                fuzz_draw(engine, static_cast<std::uint32_t>(
                                                              entry.data.size()));
                            entry.data[at] = static_cast<std::uint8_t>(
                                entry.data[at] ^ (1U << fuzz_draw(engine, 8)));
                        }
                    }
                    ZipArchive broken;
                    for (const ZipEntry& entry : entries) {
                        broken.add(entry.name, entry.data);
                    }
                    (void)read_document(ZipArchive::parse(broken.serialise()));
                    break;
                }
            }
            ++loaded;
        } catch (const ProjectIOError&) {
            ++rejected;
        } catch (const std::exception& e) {
            unexpected = "case " + std::to_string(case_index) + " (kind " +
                         std::to_string(kind) + "): " + e.what();
        } catch (...) {
            unexpected = "case " + std::to_string(case_index) + " (kind " +
                         std::to_string(kind) + "): non-standard exception";
        }
    }

    EXPECT_TRUE(unexpected.empty()) << unexpected;
    // Both outcomes must occur: an all-reject corpus says nothing about the
    // accept path, and an all-accept corpus says nothing about the guards.
    EXPECT_GT(loaded, 0);
    EXPECT_GT(rejected, 0);
    EXPECT_EQ(loaded + rejected, kFuzzCases);
}

/// A declared entry length the compressed data cannot produce must be refused
/// before anything is allocated -- the zip layer's side of the same contract.
TEST_F(ProjectV2, RejectsImpossibleDeclaredEntrySize) {
    ZipArchive small;
    small.add("doc.xml", std::string(64, 'x'));
    const std::vector<std::uint8_t> bytes = small.serialise();

    // Rewrite the central-directory "uncompressed size" of doc.xml to 2 GiB.
    std::vector<std::uint8_t> tampered = bytes;
    const std::uint32_t central_sig = 0x02014b50;
    std::size_t central = std::string::npos;
    for (std::size_t i = 0; i + 4 <= tampered.size(); ++i) {
        std::uint32_t value = 0;
        std::memcpy(&value, tampered.data() + i, 4);
        if (value == central_sig) {
            central = i;
            break;
        }
    }
    ASSERT_NE(central, std::string::npos);
    const std::uint32_t huge = 0x80000000U;
    std::memcpy(tampered.data() + central + 24, &huge, 4);
    EXPECT_THROW(ZipArchive::parse(tampered), ProjectIOError);
}

}  // namespace
}  // namespace alphashape
