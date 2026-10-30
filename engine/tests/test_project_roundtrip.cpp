// Project save/load round-trip and container-corruption tests.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "base/utils/portable_rng.hpp"

#include "base/platform/temp.hpp"
#include "base/io/zip.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "image/image.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// A unique scratch directory under the system temp dir, removed on teardown.
///
/// The name must be unique across *processes*, not just within one: ctest -jN
/// runs each test in its own process, and gtest's default random_seed() is
/// GetTimeInMillis() % 99999 with counter_ restarting at 0 in every process, so
/// two tests launched in the same millisecond compute the same path and clobber
/// each other's files. The pid disambiguates them.
class TempDir : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base = fs::temp_directory_path() /
                          ("alphashape_test_" + std::to_string(platform::process_id()) + "_" +
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

    std::string path(const std::string& name) const { return (fs::path(dir_) / name).string(); }

    static int counter_;
    std::string dir_;
};
int TempDir::counter_ = 0;

/// Format v2 (M11) stores each chunk's bulk data under "chunks/<chunk_id>/".
/// These helpers name the entries a single-chunk document writes, so the
/// corruption tests below can reach the right entry without hard-coding an id.
std::string v2_chunk_prefix(const Document& document) {
    return "chunks/" + std::to_string(document.workspace().chunks().front()->id());
}

/// "chunks/<id>/<tail>".
std::string v2_entry(const Document& document, const std::string& tail) {
    return v2_chunk_prefix(document) + "/" + tail;
}

Keypoint make_keypoint(float x, float y, float scale, float orientation) {
    Keypoint kp;
    kp.x = x;
    kp.y = y;
    kp.scale = scale;
    kp.orientation = orientation;
    return kp;
}

Descriptor make_descriptor(std::uint8_t seed) {
    Descriptor d;
    for (std::size_t i = 0; i < Descriptor::kBytes; ++i) {
        d.bytes[i] = static_cast<std::uint8_t>(seed + i * 7U);
    }
    return d;
}

/// A document with every v1-representable feature populated, including
/// awkward doubles and XML-hostile text.
Document build_rich_document() {
    Document document;
    Workspace& workspace = document.workspace();

    Chunk& chunk = workspace.add_chunk("chunk <&> \"quoted\"");
    document.set_active_chunk_id(chunk.id());

    Sensor s0;
    s0.label = "sensor<0>";
    s0.type = CameraType::Frame;
    s0.width = 4000;
    s0.height = 3000;
    s0.pixel_size_um = 3.76;
    s0.focal_length_mm = 24.0;
    s0.make = "ACME";
    s0.model = "Model &\"Quoted\"";
    s0.serial = "sn<1>";
    s0.calibration = Calibration::initial(CameraType::Frame, 4000, 3000, 3000.5);
    s0.calibration.cx = 12.25;
    s0.calibration.cy = -7.5;
    s0.calibration.b1 = 1.0 / 3.0;
    s0.calibration.b2 = -1e-12;
    s0.calibration.k1 = -0.1;
    s0.calibration.k2 = 0.05;
    s0.calibration.k3 = 1e-300;
    s0.calibration.k4 = -0.0001;
    s0.calibration.p1 = 1e-4;
    s0.calibration.p2 = -3.5e-5;
    s0.fixed_params = {"f", "cy", "k3"};
    chunk.sensors().push_back(s0);

    // A fisheye sensor with awkward doubles, so both branches of the type
    // attribute and the calibration children are exercised.
    Sensor s1;
    s1.label = "sensor1";
    s1.type = CameraType::Fisheye;
    s1.width = 4000;
    s1.height = 3000;
    s1.pixel_size_um = 1.0 / 3.0;
    s1.focal_length_mm = -1e-300;
    s1.make = "";
    s1.model = "";
    s1.serial = "";
    s1.calibration = Calibration::initial(CameraType::Fisheye, 4000, 3000, 1000.0);
    s1.calibration.k1 = 0.2;
    s1.calibration.k2 = 1e-300;
    s1.calibration.k3 = 0.0;
    s1.calibration.k4 = 2.0 / 3.0;
    s1.calibration.cx = -0.5;
    s1.fixed_params = {};
    chunk.sensors().push_back(s1);

    Camera c0;
    c0.id = 0;
    c0.sensor_id = 0;
    c0.label = "cam&0";
    c0.pose = Transform::Identity();
    c0.pose(0, 0) = 0.1;
    c0.pose(1, 1) = -1e-12;
    c0.pose(2, 2) = 1.0 / 3.0;
    c0.pose(0, 3) = 123.456789;
    c0.pose(1, 3) = -987.654321;
    c0.enabled = true;
    c0.orientation = 6;
    // A reference with no altitude: z is NaN and must round-trip as NaN.
    c0.reference.location = Eigen::Vector3d(77.209, 28.6139, nan(""));
    c0.reference.enabled = true;
    chunk.cameras().push_back(c0);

    Camera c1 = c0;
    c1.id = 1;
    c1.sensor_id = 1;
    c1.label = "cam<1>";
    c1.pose(0, 3) = 1e-300;
    c1.enabled = false;
    c1.orientation = 1;
    // A full reference, with a non-default per-axis accuracy that must
    // round-trip bit-exactly.
    c1.reference.location = Eigen::Vector3d(-122.4194, 37.7749, 12.25);
    c1.reference.accuracy = Eigen::Vector3d(0.03, 0.03, 0.07);
    c1.reference.enabled = false;
    chunk.cameras().push_back(c1);

    Camera c2 = c0;
    c2.id = 2;
    c2.sensor_id = 0;
    c2.label = "cam2";
    // No reference at all: the <reference> element must be absent.
    c2.reference.location.reset();
    chunk.cameras().push_back(c2);

    chunk.frames().push_back(std::make_shared<Frame>("images/a.pgm", 4000, 3000));
    chunk.frames().back()->set_id(0);
    chunk.frames().back()->set_camera_id(0);
    chunk.frames().push_back(std::make_shared<Frame>("images/b.pgm", 4000, 3000));
    chunk.frames().back()->set_id(1);
    chunk.frames().back()->set_camera_id(1);

    KeypointSet set0;
    set0.camera_id = 0;
    set0.keypoints = {make_keypoint(0.1F, -1e-12F, 2.5F, 0.7853981F),
                      make_keypoint(1234.5F, 999.25F, 1.0F, 3.14159F)};
    set0.descriptors = {make_descriptor(3), make_descriptor(200)};
    chunk.tie_points().keypoint_sets.push_back(set0);
    KeypointSet set1;
    set1.camera_id = 1;
    set1.keypoints = {make_keypoint(50.5F, 60.5F, 4.0F, 1.5F)};
    set1.descriptors = {make_descriptor(17)};
    chunk.tie_points().keypoint_sets.push_back(set1);
    // A third keypoint set, so two *distinct* match pairs are valid against the
    // container (a pair needs both cameras to have keypoints) and the strict
    // pair ordering is exercised by more than one element.
    KeypointSet set2;
    set2.camera_id = 2;
    set2.keypoints = {make_keypoint(1.0F, 2.0F, 1.0F, 0.0F), make_keypoint(3.0F, 4.0F, 1.0F, 0.5F),
                      make_keypoint(5.0F, 6.0F, 1.0F, 1.0F)};
    set2.descriptors = {make_descriptor(40), make_descriptor(80), make_descriptor(120)};
    chunk.tie_points().keypoint_sets.push_back(set2);

    // Two verified pairs, strictly sorted by (camera_a, camera_b): (0,1) then
    // (1,2). Every index is in range of its camera's set.
    PairMatches pair01;
    pair01.camera_a = 0;
    pair01.camera_b = 1;
    pair01.matches = {KeypointMatch{0, 0}, KeypointMatch{1, 0}};
    chunk.tie_points().matches.push_back(pair01);
    PairMatches pair12;
    pair12.camera_a = 1;
    pair12.camera_b = 2;
    pair12.matches = {KeypointMatch{0, 0}, KeypointMatch{0, 2}};
    chunk.tie_points().matches.push_back(pair12);

    // Two tracks, each with at least two observations, sorted by camera. The
    // first carries a triangulated point with awkward doubles (a repeating
    // fraction, a denormal-range magnitude and a long decimal) that must
    // round-trip bit-exactly; the second has none, so both branches of the
    // has_point flag are exercised.
    Track track0;
    track0.observations = {Observation{0, 0}, Observation{1, 0}};
    track0.point = Eigen::Vector3d(1.0 / 3.0, -1e-300, 12345.678901234567);
    chunk.tie_points().tracks.push_back(track0);
    Track track1;
    track1.observations = {Observation{0, 1}, Observation{1, 0}, Observation{2, 2}};
    chunk.tie_points().tracks.push_back(track1);

    chunk.meta().set("note", "<&>\"'");
    chunk.meta().set("DetectFeatures/keypoint_limit", "50");

    chunk.crs().epsg = 32633;
    chunk.crs().wkt = "PROJCRS[\"WGS 84 / UTM zone 33N\",\n  BASEGEOGCRS[\"WGS 84\"],\n  "
                      "CONVERSION[\"UTM zone 33N\"]\n]";

    chunk.region().centre = Eigen::Vector3d(1.0 / 3.0, -1e-12, 0.1);
    chunk.region().size = Eigen::Vector3d(10.0, 20.0, 30.0);
    chunk.transform() = Transform::Identity();
    chunk.transform()(0, 3) = 0.1;
    chunk.transform()(1, 1) = 1.0 / 3.0;

    workspace.set_next_chunk_id(7);  // ahead of any id: must survive the round-trip

    document.meta().set("author", "a & b <c>");
    document.global_crs().epsg = 4326;
    document.global_crs().wkt = "GEOGCRS[\"WGS 84\"]";
    return document;
}

TEST_F(TempDir, SaveWritesPsxAndProjectZip) {
    const Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);
    EXPECT_TRUE(fs::exists(psx));
    EXPECT_TRUE(fs::exists(path("x.files/project.zip")));
}

TEST_F(TempDir, RoundTripPreservesEverything) {
    Document original = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(original, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);

    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_TRUE(*loaded->workspace().chunks().front() == *original.workspace().chunks().front());
    EXPECT_EQ(loaded->active_chunk_id(), original.active_chunk_id());
    EXPECT_EQ(loaded->workspace().next_chunk_id(), 7);
    EXPECT_TRUE(loaded->meta() == original.meta());
    EXPECT_EQ(loaded->global_crs(), original.global_crs());
}

TEST_F(TempDir, LoadDirectlyFromProjectZipPath) {
    Document original = build_rich_document();
    ProjectIO::save(original, path("x.psx"));

    std::shared_ptr<Document> loaded = ProjectIO::load(path("x.files/project.zip"));
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_TRUE(*loaded->workspace().chunks().front() == *original.workspace().chunks().front());
    EXPECT_EQ(loaded->path(), path("x.files/project.zip"));
}

TEST_F(TempDir, MissingFilesDirectoryThrowsProjectIOError) {
    Document original = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(original, psx);
    std::error_code ec;
    fs::remove_all(path("x.files"), ec);
    ASSERT_FALSE(ec);
    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// A document whose chunk carries a single-triangle-pair mesh with every
/// optional attribute present, so every branch of the "AMS2" flags byte and both
/// the vertex and face arrays are exercised on the round-trip. The uv layout is
/// the pooled form: a shared coordinate pool plus a per-face corner map.
Document build_mesh_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("mesh");
    document.set_active_chunk_id(chunk.id());

    Mesh& mesh = chunk.model();
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector3d(2.0, 3.0, 4.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(1, 2, 3)};
    // A 4-entry pool, with the two faces indexing it. The pool is deliberately
    // larger than the face count times three would allow, to prove the reader
    // keys the pool off its own stored count rather than the vertex count.
    mesh.uvs = {Eigen::Vector2f(0.0F, 0.0F), Eigen::Vector2f(1.0F, 0.0F),
                Eigen::Vector2f(0.0F, 1.0F), Eigen::Vector2f(0.5F, 0.5F)};
    mesh.uv_faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(1, 2, 3)};
    mesh.colours = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    mesh.confidence = {0.5F, 0.25F, 0.125F, 1.0F};
    return document;
}

TEST_F(TempDir, RoundTripPreservesMesh) {
    Document original = build_mesh_document();
    const std::string psx = path("mesh.psx");
    ProjectIO::save(original, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_TRUE(got == *original.workspace().chunks().front());
    EXPECT_EQ(got.model().vertices.size(), 4U);
    EXPECT_EQ(got.model().faces.size(), 2U);
    EXPECT_EQ(got.model().uvs.size(), 4U);
    EXPECT_EQ(got.model().uv_faces.size(), 2U);
    EXPECT_EQ(got.model().colours.size(), 12U);
    EXPECT_EQ(got.model().confidence.size(), 4U);
    EXPECT_EQ(got.model().faces[1], Eigen::Vector3i(1, 2, 3));
    EXPECT_EQ(got.model().uv_faces[1], Eigen::Vector3i(1, 2, 3));
}

/// The mesh sidecar carries the "AMS2" magic and an <model> element appears.
TEST_F(TempDir, MeshStoredAsSidecar) {
    Document original = build_mesh_document();
    const std::string psx = path("mesh.psx");
    ProjectIO::save(original, psx);

    std::ifstream in(path("mesh.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    ASSERT_TRUE(archive.contains(v2_entry(original, "models/" +
        std::to_string(original.workspace().chunks().front()->active_model_id()) +
        "/mesh.bin")));
    const std::vector<std::uint8_t> sidecar = archive.get(
        v2_entry(original, "models/" +
        std::to_string(original.workspace().chunks().front()->active_model_id()) +
        "/mesh.bin"));
    ASSERT_GE(sidecar.size(), 4U);
    EXPECT_EQ(sidecar[0], 'A');
    EXPECT_EQ(sidecar[1], 'M');
    EXPECT_EQ(sidecar[2], 'S');
    EXPECT_EQ(sidecar[3], '2');
    EXPECT_NE(archive.get_text("doc.xml").find("<model "), std::string::npos);
}

/// A chunk with no mesh must not emit the sidecar or the <model> element, so a
/// pre-milestone-7 project stays byte-identical.
TEST_F(TempDir, EmptyMeshEmitsNoSidecar) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("c");
    document.set_active_chunk_id(chunk.id());
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);

    std::ifstream in(path("x.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    EXPECT_FALSE(archive.contains(v2_entry(document, "models/0/mesh.bin")));
    EXPECT_EQ(archive.get_text("doc.xml").find("<model"), std::string::npos);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    EXPECT_TRUE(loaded->workspace().chunks().front()->model().empty());
}

/// An out-of-range face index in the sidecar is rejected; the reader validates
/// face indices against the vertex count on load. (Defined with the other
/// sidecar-corruption tests at the end of this file, after the byte helpers.)

/// Writing a mesh whose faces reference a vertex past the end must throw before
/// any byte reaches the archive.
TEST_F(TempDir, WritingAnOutOfRangeFaceIndexThrows) {
    Document document = build_mesh_document();
    document.workspace().chunks().front()->model().faces.push_back(Eigen::Vector3i(0, 1, 99));

    const std::string psx = path("x.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
    EXPECT_FALSE(fs::exists(path("x.files")));
}

/// Replace an entry inside a project's zip in place, so a load() sees the edited
/// doc.xml. Uses ZipArchive for both the parse and the re-serialise.
void rewrite_zip_entry(const std::string& zip_path, const std::string& entry,
                       const std::string& text) {
    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    ZipArchive archive = ZipArchive::parse(bytes);
    archive.add(entry, text);
    const std::vector<std::uint8_t> out = archive.serialise();
    std::ofstream out_file(zip_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out_file);
    out_file.write(reinterpret_cast<const char*>(out.data()),
                   static_cast<std::streamsize>(out.size()));
}

/// A small RGB texture page whose pixels are a deterministic function of their
/// coordinates, so a decode that transposes, flips or shifts the image is
/// detected rather than passing on uniform data.
Image make_test_texture(int width, int height, int salt) {
    Image image(width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at(x, y, 0) = static_cast<std::uint8_t>((x * 3 + y + salt) % 256);
            image.at(x, y, 1) = static_cast<std::uint8_t>((x + y * 5 + salt * 2) % 256);
            image.at(x, y, 2) = static_cast<std::uint8_t>((x * 7 + y * 11 + salt) % 256);
        }
    }
    return image;
}

/// Build a document with the standard mesh document plus one texture page.
Document build_textured_document() {
    Document document = build_mesh_document();
    document.workspace().chunks().front()->model_textures().push_back(
        make_test_texture(6, 4, 17));
    return document;
}

/// Save -> load a mesh carrying uvs, uv_faces and a texture page; the reloaded
/// chunk must compare equal, and the texture pixels byte-exact.
TEST_F(TempDir, RoundTripPreservesTexturedMesh) {
    Document original = build_textured_document();
    const std::string psx = path("textured.psx");
    ProjectIO::save(original, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_TRUE(got == *original.workspace().chunks().front());
    ASSERT_EQ(got.model_textures().size(), 1U);
    const Image& page = got.model_textures().front();
    const Image& reference = original.workspace().chunks().front()->model_textures().front();
    ASSERT_EQ(page.width(), reference.width());
    ASSERT_EQ(page.height(), reference.height());
    ASSERT_EQ(page.channels(), 3);
    ASSERT_EQ(page.size_bytes(), reference.size_bytes());
    for (std::size_t i = 0; i < page.size_bytes(); ++i) {
        ASSERT_EQ(page.data()[i], reference.data()[i]) << "byte " << i << " differs";
    }
}

/// The texture page is stored at the documented entry name and announced by a
/// textures attribute on <model>.
TEST_F(TempDir, TextureStoredAsPngEntry) {
    Document original = build_textured_document();
    const std::string psx = path("textured.psx");
    ProjectIO::save(original, psx);

    std::ifstream in(path("textured.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    const std::string texture_path = v2_entry(
        original, "models/" +
        std::to_string(original.workspace().chunks().front()->active_model_id()) +
        "/texture_0.png");
    ASSERT_TRUE(archive.contains(texture_path));
    const std::vector<std::uint8_t>& png = archive.get(texture_path);
    ASSERT_GE(png.size(), 8U);
    // The PNG signature, proving the entry really is a PNG byte stream.
    const std::uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(png[static_cast<std::size_t>(i)], signature[i]);
    }
    EXPECT_NE(archive.get_text("doc.xml").find("pages=\"1\""), std::string::npos);
}

/// A project written before milestone 8 stores per-vertex uvs under the "AMS1"
/// magic. The reader must still accept it, discard the legacy coordinates and
/// leave uv_faces empty so the mesh is a valid (if untextured) pooled layout.
TEST_F(TempDir, LegacyAms1MeshStillLoads) {
    // Hand-build an AMS1 sidecar: 4 vertices, 1 face, uvs present, no colours
    // or confidence. This is the exact milestone-7 layout the reader supports.
    std::vector<std::uint8_t> sidecar;
    const auto put_u32 = [&](std::uint32_t v) {
        for (unsigned s = 0; s < 32; s += 8) {
            sidecar.push_back(static_cast<std::uint8_t>((v >> s) & 0xFFU));
        }
    };
    const auto put_i32 = [&](std::int32_t v) { put_u32(static_cast<std::uint32_t>(v)); };
    const auto put_f64 = [&](double d) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &d, sizeof bits);
        for (unsigned s = 0; s < 64; s += 8) {
            sidecar.push_back(static_cast<std::uint8_t>((bits >> s) & 0xFFULL));
        }
    };
    const auto put_f32 = [&](float f) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof bits);
        put_u32(bits);
    };

    sidecar.insert(sidecar.end(), {'A', 'M', 'S', '1'});
    put_u32(4);  // vertex count
    put_u32(1);  // face count
    sidecar.push_back(1U << 0U);  // has_uvs only
    const double verts[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 3, 4};
    for (const double d : verts) put_f64(d);
    put_i32(0);
    put_i32(1);
    put_i32(2);
    const float uvs[8] = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.5F};
    for (const float f : uvs) put_f32(f);

    // A minimal project with this sidecar installed at model/mesh.bin.
    Document document;
    Chunk& chunk = document.workspace().add_chunk("legacy");
    document.set_active_chunk_id(chunk.id());
    const std::string psx = path("legacy.psx");
    ProjectIO::save(document, psx);

    ZipArchive archive;
    {
        std::ifstream in(path("legacy.files/project.zip"), std::ios::binary);
        ASSERT_TRUE(in);
        const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                              std::istreambuf_iterator<char>());
        archive = ZipArchive::parse(bytes);
    }
    // Install the AMS1 mesh and a matching <models> element. The project this
    // is installed into is v2 (it was just written by the v2 writer); only the
    // MESH bytes are the legacy milestone-7 layout, which is the point.
    const std::string mesh_path = v2_entry(document, "models/0/mesh.bin");
    archive.add(mesh_path, sidecar);
    std::string doc_xml = archive.get_text("doc.xml");
    const std::size_t close = doc_xml.rfind("</chunk>");
    ASSERT_NE(close, std::string::npos);
    const std::string model_element =
        "<models><model id=\"0\" label=\"Model 1\" active=\"true\" "
        "faceCount=\"1\" vertexCount=\"4\" path=\"" + mesh_path +
        "\"/></models>\n  ";
    doc_xml.insert(close, model_element);
    archive.add("doc.xml", doc_xml);
    {
        const std::vector<std::uint8_t> out = archive.serialise();
        std::ofstream out_file(path("legacy.files/project.zip"),
                               std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out_file);
        out_file.write(reinterpret_cast<const char*>(out.data()),
                       static_cast<std::streamsize>(out.size()));
    }

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    const Mesh& mesh = loaded->workspace().chunks().front()->model();
    EXPECT_EQ(mesh.vertices.size(), 4U);
    EXPECT_EQ(mesh.faces.size(), 1U);
    // Legacy per-vertex uvs cannot be represented as a pooled layout, so they
    // are dropped and the mesh comes back with no texture coordinates.
    EXPECT_TRUE(mesh.uvs.empty());
    EXPECT_TRUE(mesh.uv_faces.empty());
}

/// An AMS2 sidecar whose uv_faces index past the end of the uv pool must be
/// rejected on load rather than producing a mesh with dangling coordinates.
TEST_F(TempDir, BadUvIndexInSidecarThrows) {
    Document original = build_textured_document();
    const std::string psx = path("baduv.psx");
    ProjectIO::save(original, psx);

    ZipArchive archive;
    {
        std::ifstream in(path("baduv.files/project.zip"), std::ios::binary);
        ASSERT_TRUE(in);
        const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                              std::istreambuf_iterator<char>());
        archive = ZipArchive::parse(bytes);
    }
    // Corrupt one uv_faces index to point one past the 4-entry pool. The
    // trailing uv_faces block is the last 24 bytes of the sidecar (2 faces x
    // 3 i32), so the final index sits in the last 4 bytes.
    const std::string mesh_path = v2_entry(
        original, "models/" +
        std::to_string(original.workspace().chunks().front()->active_model_id()) +
        "/mesh.bin");
    std::vector<std::uint8_t> sidecar = archive.get(mesh_path);
    ASSERT_GE(sidecar.size(), 4U);
    const std::size_t last_index = sidecar.size() - 4U;
    const std::uint32_t bad = 99U;
    for (unsigned s = 0; s < 32; s += 8) {
        sidecar[last_index + s / 8] = static_cast<std::uint8_t>((bad >> s) & 0xFFU);
    }
    archive.add(mesh_path, sidecar);
    {
        const std::vector<std::uint8_t> out = archive.serialise();
        std::ofstream out_file(path("baduv.files/project.zip"),
                               std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out_file);
        out_file.write(reinterpret_cast<const char*>(out.data()),
                       static_cast<std::streamsize>(out.size()));
    }

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// The doc-level invariant validator rejects a partial uv_faces array (one
/// entry for a two-face mesh) before any byte is written.
TEST_F(TempDir, WritingAPartialUvFacesThrows) {
    Document document = build_mesh_document();
    document.workspace().chunks().front()->model().uv_faces.pop_back();
    const std::string psx = path("partial.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
}

/// Replace a binary entry inside a project's zip in place. The string overload
/// of rewrite_zip_entry cannot carry the bytes a binary sidecar needs.
void rewrite_zip_entry_bytes(const std::string& zip_path, const std::string& entry,
                             const std::vector<std::uint8_t>& data) {
    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    ZipArchive archive = ZipArchive::parse(bytes);
    archive.add(entry, data);
    const std::vector<std::uint8_t> out = archive.serialise();
    std::ofstream out_file(zip_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out_file);
    out_file.write(reinterpret_cast<const char*>(out.data()),
                   static_cast<std::streamsize>(out.size()));
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}
void put_i32(std::vector<std::uint8_t>& out, std::int32_t v) {
    put_u32(out, static_cast<std::uint32_t>(v));
}

/// Hand-build an "AMT1" sidecar from (camera_a, camera_b, matches) triples, so a
/// corruption test can produce bytes the writer would never emit.
std::vector<std::uint8_t> mat_magic() { return {'A', 'M', 'T', '1'}; }
std::vector<std::uint8_t> trk_magic() { return {'A', 'T', 'R', '2'}; }

void append_match(std::vector<std::uint8_t>& out, std::int32_t a, std::int32_t b,
                  const std::vector<std::pair<std::uint32_t, std::uint32_t>>& pairs) {
    put_i32(out, a);
    put_i32(out, b);
    put_u32(out, static_cast<std::uint32_t>(pairs.size()));
    for (const auto& [ia, ib] : pairs) {
        put_u32(out, ia);
        put_u32(out, ib);
    }
}

void append_track(std::vector<std::uint8_t>& out,
                  const std::vector<std::pair<std::int32_t, std::uint32_t>>& obs,
                  std::uint8_t has_point = 0) {
    put_u32(out, static_cast<std::uint32_t>(obs.size()));
    for (const auto& [camera, index] : obs) {
        put_i32(out, camera);
        put_u32(out, index);
    }
    out.push_back(has_point);
}

void put_f64(std::vector<std::uint8_t>& out, double d) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &d, sizeof bits);
    for (unsigned shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xFFULL));
    }
}

/// An "AMT1" file with the given pairs and a header count matching their number.
std::vector<std::uint8_t> matches_file(
    const std::vector<std::tuple<std::int32_t, std::int32_t,
                                 std::vector<std::pair<std::uint32_t, std::uint32_t>>>>& pairs) {
    std::vector<std::uint8_t> out = mat_magic();
    put_u32(out, static_cast<std::uint32_t>(pairs.size()));
    for (const auto& [a, b, m] : pairs) append_match(out, a, b, m);
    return out;
}

/// An "ATR2" file with the given tracks and a header count matching their number.
std::vector<std::uint8_t> tracks_file(
    const std::vector<std::vector<std::pair<std::int32_t, std::uint32_t>>>& tracks) {
    std::vector<std::uint8_t> out = trk_magic();
    put_u32(out, static_cast<std::uint32_t>(tracks.size()));
    for (const auto& t : tracks) append_track(out, t);
    return out;
}

/// The rich document's only chunk is the first one added, so its id — and with
/// it the v2 entry prefix — is always 0.
std::string rich_entry(const std::string& tail) { return "chunks/0/" + tail; }

/// Save the rich document (whose keypoint sets define cameras 0, 1 and 2 with
/// 2, 1 and 3 keypoints) to `psx` and return the path to its project zip.
std::string save_rich_zip(const std::string& psx) {
    Document document = build_rich_document();
    ProjectIO::save(document, psx);
    // ProjectIO::save writes "<base>.files/project.zip" beside the manifest.
    fs::path base = fs::path(psx);
    base.replace_extension();
    return (base.string() + ".files/project.zip");
}

TEST_F(TempDir, TruncatedMatchesSidecarFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // The valid sidecar written for the rich document, minus its final 4 bytes.
    std::ifstream in(zip, std::ios::binary);
    ASSERT_TRUE(in);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    std::vector<std::uint8_t> sidecar = ZipArchive::parse(bytes).get(rich_entry("tie_points/matches.bin"));
    ASSERT_GT(sidecar.size(), 4U);
    sidecar.resize(sidecar.size() - 4);
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/matches.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, MatchCountBeyondFileLengthFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // One pair claimed, but only 4 bytes follow (a count that runs off the end).
    std::vector<std::uint8_t> sidecar = mat_magic();
    put_u32(sidecar, 1);          // pair count = 1
    put_i32(sidecar, 0);
    put_i32(sidecar, 1);          // ids present, but the match count is missing
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/matches.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, OutOfRangeMatchIndexFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // Index 1000 is far past camera 1's single keypoint.
    rewrite_zip_entry_bytes(
        zip, rich_entry("tie_points/matches.bin"),
        matches_file({{0, 1, {{0, 1000}}}}));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, UnsortedMatchPairsFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // (1,2) before (0,1): valid individually, but not strictly sorted.
    rewrite_zip_entry_bytes(
        zip, rich_entry("tie_points/matches.bin"),
        matches_file({{1, 2, {{0, 0}}}, {0, 1, {{0, 0}}}}));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, TrackWithOneObservationFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"), tracks_file({{{0, 0}}}));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, TrackWithUnsortedCamerasFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // Two observations, but the second camera id is lower than the first.
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"),
                            tracks_file({{{1, 0}, {0, 0}}}));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, BadMatchesMagicFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    std::vector<std::uint8_t> sidecar = matches_file({{0, 1, {{0, 0}}}});
    sidecar[0] = 'X';  // "XMT1"
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/matches.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, MatchReferencingAnUnknownCameraFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // Camera 7 does not exist in the chunk (which has cameras 0, 1 and 2).
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/matches.bin"),
                            matches_file({{0, 7, {{0, 0}}}}));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, TrackReferencingAnUnknownCameraFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"),
                            tracks_file({{{0, 0}, {9, 0}}}));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, BadTracksMagicFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    std::vector<std::uint8_t> sidecar = tracks_file({{{0, 0}, {1, 0}}});
    sidecar[3] = 'X';  // "ATRX"
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, OldAtr1TracksMagicFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // The pre-milestone-4 magic. There are no shipped files in the old format,
    // so the reader must reject it like any other bad magic.
    std::vector<std::uint8_t> sidecar = {'A', 'T', 'R', '1'};
    put_u32(sidecar, 1);
    append_track(sidecar, {{0, 0}, {1, 0}});
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, BadHasPointFlagFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    std::vector<std::uint8_t> sidecar = trk_magic();
    put_u32(sidecar, 1);
    append_track(sidecar, {{0, 0}, {1, 0}}, 2);  // has_point must be 0 or 1
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, TruncatedTrackPointFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // has_point = 1, but the three f64 that must follow are cut short.
    std::vector<std::uint8_t> sidecar = trk_magic();
    put_u32(sidecar, 1);
    append_track(sidecar, {{0, 0}, {1, 0}}, 1);
    put_f64(sidecar, 1.0);  // only one of the three coordinates
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, NonFiniteTrackPointFailsOnLoad) {
    const std::string psx = path("x.psx");
    const std::string zip = save_rich_zip(psx);
    // has_point = 1 with a NaN x: structurally valid, semantically not.
    std::vector<std::uint8_t> sidecar = trk_magic();
    put_u32(sidecar, 1);
    append_track(sidecar, {{0, 0}, {1, 0}}, 1);
    put_f64(sidecar, nan(""));
    put_f64(sidecar, 0.0);
    put_f64(sidecar, 0.0);
    rewrite_zip_entry_bytes(zip, rich_entry("tie_points/tracks.bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, XmlPairCountMismatchFailsOnLoad) {
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);
    const std::string zip = path("x.files/project.zip");

    std::ifstream in(zip, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::string xml = ZipArchive::parse(bytes).get_text("doc.xml");
    const std::string needle = "<matches pairs=\"2\"";
    ASSERT_NE(xml.find(needle), std::string::npos);
    xml.replace(xml.find(needle), needle.size(), "<matches pairs=\"3\"");
    rewrite_zip_entry(zip, "doc.xml", xml);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, XmlTrackCountMismatchFailsOnLoad) {
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);
    const std::string zip = path("x.files/project.zip");

    std::ifstream in(zip, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::string xml = ZipArchive::parse(bytes).get_text("doc.xml");
    const std::string needle = "<tracks count=\"2\"";
    ASSERT_NE(xml.find(needle), std::string::npos);
    xml.replace(xml.find(needle), needle.size(), "<tracks count=\"5\"");
    rewrite_zip_entry(zip, "doc.xml", xml);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// A non-positive accuracy component is not a valid sigma, so a hand-edited
/// file that carries one must be rejected on load.
TEST_F(TempDir, ZeroReferenceAccuracyFailsOnLoad) {
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);
    const std::string zip = path("x.files/project.zip");

    std::ifstream in(zip, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::string xml = ZipArchive::parse(bytes).get_text("doc.xml");
    const std::string needle = "accuracy=\"10 10 10\"";
    const std::size_t at = xml.find(needle);
    ASSERT_NE(at, std::string::npos);
    xml.replace(at, needle.size(), "accuracy=\"0 1 1\"");
    rewrite_zip_entry(zip, "doc.xml", xml);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// The accuracy attribute is required whenever <reference> is present.
TEST_F(TempDir, MissingReferenceAccuracyFailsOnLoad) {
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);
    const std::string zip = path("x.files/project.zip");

    std::ifstream in(zip, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::string xml = ZipArchive::parse(bytes).get_text("doc.xml");
    const std::string needle = " accuracy=\"10 10 10\"";
    const std::size_t at = xml.find(needle);
    ASSERT_NE(at, std::string::npos);
    xml.erase(at, needle.size());
    rewrite_zip_entry(zip, "doc.xml", xml);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// Writing a reference with a non-positive accuracy must throw before any byte
/// reaches the archive.
TEST_F(TempDir, WritingNonPositiveReferenceAccuracyThrows) {
    Document document = build_rich_document();
    document.workspace().chunks().front()->cameras().front().reference.accuracy =
        Eigen::Vector3d(10.0, 0.0, 10.0);

    const std::string psx = path("x.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
}

/// The rich document's two references carry distinct accuracies (the default
/// 10/10/10 and the 0.03/0.03/0.07 set on camera 1), and both must survive the
/// round-trip bit-exactly.
TEST_F(TempDir, ReferenceAccuracySurvivesRoundTrip) {
    Document original = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(original, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    const std::vector<Camera>& cameras = loaded->workspace().chunks().front()->cameras();
    ASSERT_EQ(cameras.size(), 3U);
    EXPECT_EQ(cameras[0].reference.accuracy, Eigen::Vector3d(10.0, 10.0, 10.0));
    EXPECT_EQ(cameras[1].reference.accuracy, Eigen::Vector3d(0.03, 0.03, 0.07));
}

TEST_F(TempDir, MissingMatchesSidecarFailsOnLoad) {
    // The XML references a sidecar that is not in the archive.
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);
    const std::string zip = path("x.files/project.zip");

    std::ifstream in(zip, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    ZipArchive archive = ZipArchive::parse(bytes);
    // Rebuild the archive without the matches entry.
    ZipArchive reduced;
    for (const ZipEntry& entry : archive.entries()) {
        if (entry.name != rich_entry("tie_points/matches.bin")) {
            reduced.add(entry.name, entry.data);
        }
    }
    const std::vector<std::uint8_t> out = reduced.serialise();
    std::ofstream out_file(zip, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out_file);
    out_file.write(reinterpret_cast<const char*>(out.data()),
                   static_cast<std::streamsize>(out.size()));

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, WritingAReversedMatchPairThrowsAndWritesNothing) {
    Document document = build_rich_document();
    // camera_a > camera_b violates the PairMatches invariant.
    PairMatches bad;
    bad.camera_a = 2;
    bad.camera_b = 0;
    bad.matches = {KeypointMatch{0, 0}};
    document.workspace().chunks().front()->tie_points().matches.push_back(bad);

    const std::string psx = path("x.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
    EXPECT_FALSE(fs::exists(path("x.files")));
}

TEST_F(TempDir, WritingAMatchIndexOutOfRangeThrows) {
    Document document = build_rich_document();
    PairMatches bad;
    bad.camera_a = 0;
    bad.camera_b = 1;
    bad.matches = {KeypointMatch{99, 0}};  // camera 0 has only 2 keypoints
    document.workspace().chunks().front()->tie_points().matches.push_back(bad);

    EXPECT_THROW(ProjectIO::save(document, path("x.psx")), ProjectIOError);
    EXPECT_FALSE(fs::exists(path("x.psx")));
}

TEST_F(TempDir, WritingASingleObservationTrackThrows) {
    Document document = build_rich_document();
    Track bad;
    bad.observations = {Observation{0, 0}};  // needs at least two
    document.workspace().chunks().front()->tie_points().tracks.push_back(bad);

    EXPECT_THROW(ProjectIO::save(document, path("x.psx")), ProjectIOError);
    EXPECT_FALSE(fs::exists(path("x.psx")));
}

TEST_F(TempDir, MatchesAndTracksRoundTripAreStoredAsSidecars) {
    Document original = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(original, psx);

    // The sidecars exist and are non-empty.
    std::ifstream in(path("x.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    EXPECT_TRUE(archive.contains(rich_entry("tie_points/matches.bin")));
    EXPECT_TRUE(archive.contains(rich_entry("tie_points/tracks.bin")));

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    const TiePoints& tp = loaded->workspace().chunks().front()->tie_points();
    EXPECT_EQ(tp.matches.size(), 2U);
    EXPECT_EQ(tp.tracks.size(), 2U);
    EXPECT_EQ(tp.matches[0].camera_a, 0);
    EXPECT_EQ(tp.matches[0].camera_b, 1);
    EXPECT_EQ(tp.matches[1].camera_a, 1);
    EXPECT_EQ(tp.matches[1].camera_b, 2);
    EXPECT_EQ(tp.tracks[1].observations.size(), 3U);
    // The point survived bit-exactly, and the track without one stayed nullopt.
    ASSERT_TRUE(tp.tracks[0].point.has_value());
    EXPECT_EQ((*tp.tracks[0].point).x(), 1.0 / 3.0);
    EXPECT_EQ((*tp.tracks[0].point).y(), -1e-300);
    EXPECT_EQ((*tp.tracks[0].point).z(), 12345.678901234567);
    EXPECT_FALSE(tp.tracks[1].point.has_value());
}

TEST_F(TempDir, WritingANonFiniteTrackPointThrows) {
    Document document = build_rich_document();
    document.workspace().chunks().front()->tie_points().tracks.front().point =
        Eigen::Vector3d(0.0, std::numeric_limits<double>::infinity(), 0.0);

    const std::string psx = path("x.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
    EXPECT_FALSE(fs::exists(path("x.files")));
}

/// A document whose tie points carry no matches or tracks must not emit the
/// two sidecar entries at all: an old project stays readable both ways.
TEST_F(TempDir, EmptyMatchesAndTracksEmitNoSidecars) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("c");
    document.set_active_chunk_id(chunk.id());
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);

    std::ifstream in(path("x.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    EXPECT_FALSE(archive.contains(rich_entry("tie_points/matches.bin")));
    EXPECT_FALSE(archive.contains(rich_entry("tie_points/tracks.bin")));
    const std::string xml = archive.get_text("doc.xml");
    EXPECT_EQ(xml.find("<matches"), std::string::npos);
    EXPECT_EQ(xml.find("<tracks"), std::string::npos);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    EXPECT_TRUE(loaded->workspace().chunks().front()->tie_points().matches.empty());
    EXPECT_TRUE(loaded->workspace().chunks().front()->tie_points().tracks.empty());
}

TEST_F(TempDir, UnknownSensorFixedParamFailsOnLoad) {
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);

    const std::string zip_path = path("x.files/project.zip");
    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::string xml = ZipArchive::parse(bytes).get_text("doc.xml");
    const std::string needle = "fixed_params=\"f cy k3\"";
    const std::size_t at = xml.find(needle);
    ASSERT_NE(at, std::string::npos);
    xml.replace(at, needle.size(), "fixed_params=\"f cy bogus\"");
    rewrite_zip_entry(zip_path, "doc.xml", xml);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

TEST_F(TempDir, UnknownSensorTypeFailsOnLoad) {
    Document document = build_rich_document();
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);

    const std::string zip_path = path("x.files/project.zip");
    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::string xml = ZipArchive::parse(bytes).get_text("doc.xml");
    const std::size_t at = xml.find("type=\"fisheye\"");
    ASSERT_NE(at, std::string::npos);
    xml.replace(at, std::string("type=\"fisheye\"").size(), "type=\"spherical\"");
    rewrite_zip_entry(zip_path, "doc.xml", xml);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

// ---------------------------------------------------------------------------
// ZipArchive round-trips and corruption.
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> random_bytes(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    // Portable byte draw: std::uniform_int_distribution is specified only by
    // its output range, so libc++ and libstdc++ pick different values from one
    // seed (TESTING.md, "Numeric consistency").
    std::vector<std::uint8_t> out(n);
    for (auto& b : out) b = static_cast<std::uint8_t>(alphashape::rng::uniform_below(rng, 255u));
    return out;
}

TEST(ZipArchive, RoundTripEmptyLargeAndIncompressibleEntries) {
    ZipArchive archive;
    archive.add("empty", std::vector<std::uint8_t>{});
    archive.add("large", std::string(100000, 'a'));  // highly compressible
    const auto random = random_bytes(4096, 12345);
    archive.add("random", random);

    const std::vector<std::uint8_t> bytes = archive.serialise();
    const ZipArchive parsed = ZipArchive::parse(bytes);

    ASSERT_EQ(parsed.entries().size(), 3U);
    EXPECT_TRUE(parsed.get("empty").empty());
    EXPECT_EQ(parsed.get_text("large"), std::string(100000, 'a'));
    EXPECT_EQ(parsed.get("random"), random);
}

TEST(ZipArchive, CorruptedStoredEntryThrowsProjectIOError) {
    ZipArchive archive;
    const auto random = random_bytes(256, 999);
    archive.add("f.bin", random);  // incompressible -> stored

    std::vector<std::uint8_t> bytes = archive.serialise();
    // Local header is 30 bytes + name length; flip a byte inside the stored data.
    const std::size_t name_len = bytes[26] | (static_cast<std::size_t>(bytes[27]) << 8U);
    const std::size_t data_at = 30 + name_len;
    ASSERT_LT(data_at + 1, bytes.size());
    bytes[data_at + 1] ^= 0xFFU;

    EXPECT_THROW(ZipArchive::parse(bytes), ProjectIOError);
}

TEST(ZipArchive, MalformedBytesThrowProjectIOError) {
    const std::vector<std::uint8_t> garbage = random_bytes(16, 1);
    EXPECT_THROW(ZipArchive::parse(garbage), ProjectIOError);
}

// ---------------------------------------------------------------------------
// Depth maps and the dense point cloud (milestone 6).
// ---------------------------------------------------------------------------

/// A document whose chunk carries two depth maps — one with normals and
/// confidence, one without either — and a point cloud with every attribute, so
/// both branches of every presence flag are exercised on the round-trip.
Document build_dense_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("dense");
    document.set_active_chunk_id(chunk.id());

    Sensor s;
    s.label = "s";
    s.type = CameraType::Frame;
    s.width = 160;
    s.height = 120;
    s.calibration = Calibration::initial(CameraType::Frame, 160, 120, 200.0);
    chunk.sensors().push_back(s);

    for (std::int32_t id = 0; id < 2; ++id) {
        Camera c;
        c.id = id;
        c.sensor_id = 0;
        c.label = "cam" + std::to_string(id);
        c.enabled = true;
        chunk.cameras().push_back(c);
    }

    DepthMap with_extras;
    with_extras.camera_id = 0;
    with_extras.width = 2;
    with_extras.height = 3;
    with_extras.downscale = 2;
    with_extras.depths = {1.0F, 2.0F, -1.0F, 3.5F, 4.0F, 5.0F};
    with_extras.normals.resize(18);
    for (std::size_t i = 0; i < with_extras.normals.size(); ++i) {
        with_extras.normals[i] = static_cast<float>(i) * 0.125F;
    }
    with_extras.confidence = {0.5F, 0.25F, 0.0F, 1.0F, 0.75F, 0.125F};
    chunk.depth_maps().push_back(with_extras);

    DepthMap plain;
    plain.camera_id = 1;
    plain.width = 1;
    plain.height = 2;
    plain.downscale = 1;
    plain.depths = {10.0F, -1.0F};
    chunk.depth_maps().push_back(plain);

    PointCloud& cloud = chunk.point_cloud();
    cloud.points = {Eigen::Vector3d(1.0 / 3.0, -1e-300, 12345.678901234567),
                    Eigen::Vector3d(-0.5, 2.25, 0.0)};
    cloud.normals = {Eigen::Vector3f(0.0F, 0.0F, 1.0F), Eigen::Vector3f(1.0F, 0.0F, 0.0F)};
    cloud.colours = {10, 20, 30, 40, 50, 60};
    cloud.confidence = {0.5F, 0.25F};
    return document;
}

TEST_F(TempDir, RoundTripPreservesDepthMapsAndPointCloud) {
    Document original = build_dense_document();
    const std::string psx = path("dense.psx");
    ProjectIO::save(original, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_TRUE(got == *original.workspace().chunks().front());

    ASSERT_EQ(got.depth_maps().size(), 2U);
    EXPECT_EQ(got.depth_maps()[0].downscale, 2);
    EXPECT_EQ(got.depth_maps()[0].normals.size(), 18U);
    EXPECT_EQ(got.depth_maps()[0].confidence.size(), 6U);
    EXPECT_EQ(got.depth_maps()[0].depths[2], -1.0F);
    EXPECT_TRUE(got.depth_maps()[1].normals.empty());
    EXPECT_TRUE(got.depth_maps()[1].confidence.empty());

    ASSERT_EQ(got.point_cloud().points.size(), 2U);
    EXPECT_EQ(got.point_cloud().colours.size(), 6U);
    EXPECT_EQ(got.point_cloud().confidence.size(), 2U);
    // The awkward double survived bit-exactly.
    EXPECT_EQ(got.point_cloud().points[0].x(), 1.0 / 3.0);
    EXPECT_EQ(got.point_cloud().points[0].z(), 12345.678901234567);
}

/// A chunk with no depth maps or point cloud must not emit the two sidecar
/// entries at all, so a pre-milestone-6 project stays readable both ways.
TEST_F(TempDir, EmptyDenseProductsEmitNoSidecars) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("c");
    document.set_active_chunk_id(chunk.id());
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);

    std::ifstream in(path("x.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    EXPECT_FALSE(archive.contains(v2_entry(document, "depth_maps/0.bin")));
    EXPECT_FALSE(archive.contains(v2_entry(document, "point_clouds/0.bin")));
    const std::string xml = archive.get_text("doc.xml");
    EXPECT_EQ(xml.find("<depth_maps"), std::string::npos);
    EXPECT_EQ(xml.find("<dense_cloud"), std::string::npos);
}

/// Truncating the depth-map sidecar must be rejected on load, not silently
/// accepted as a short map.
TEST_F(TempDir, TruncatedDepthMapSidecarThrows) {
    Document document = build_dense_document();
    const std::string psx = path("dense.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("dense.files/project.zip");

    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::vector<std::uint8_t> sidecar =
        ZipArchive::parse(bytes).get(v2_entry(document, "depth_maps/" +
        std::to_string(document.workspace().chunks().front()->active_depth_maps_id()) + ".bin"));
    ASSERT_GT(sidecar.size(), 4U);
    sidecar.resize(sidecar.size() - 4U);  // drop the final f32 -> truncated array
    rewrite_zip_entry_bytes(zip_path, v2_entry(document, "depth_maps/" +
        std::to_string(document.workspace().chunks().front()->active_depth_maps_id()) + ".bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// A depth-map sidecar whose width*height*4 overflows a 64-bit product must be
/// rejected before any allocation. 0x7fffffff x 0x7fffffff pixels is the smallest
/// case that used to wrap `pixels * 4` to 0 and slip past the length check,
/// driving a ~2^62-float resize (a crash / OOM, not a clean error).
TEST_F(TempDir, DepthMapDimensionOverflowFailsOnLoad) {
    Document document = build_dense_document();
    const std::string psx = path("dense.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("dense.files/project.zip");

    std::vector<std::uint8_t> sidecar = {'A', 'D', 'M', '1'};
    put_u32(sidecar, 1);                     // one map
    put_i32(sidecar, 0);                     // camera_id
    put_i32(sidecar, 0x7fffffff);            // width
    put_i32(sidecar, 0x7fffffff);            // height
    put_i32(sidecar, 1);                     // downscale
    sidecar.push_back(0);                    // has_normals
    sidecar.push_back(0);                    // has_confidence
    rewrite_zip_entry_bytes(zip_path, v2_entry(document, "depth_maps/" +
        std::to_string(document.workspace().chunks().front()->active_depth_maps_id()) + ".bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// Truncating the point-cloud sidecar must likewise be rejected on load.
TEST_F(TempDir, TruncatedPointCloudSidecarThrows) {
    Document document = build_dense_document();
    const std::string psx = path("dense.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("dense.files/project.zip");

    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    std::vector<std::uint8_t> sidecar =
        ZipArchive::parse(bytes).get(v2_entry(document, "point_clouds/" +
        std::to_string(document.workspace().chunks().front()->active_point_cloud_id()) + ".bin"));
    ASSERT_GT(sidecar.size(), 4U);
    sidecar.resize(sidecar.size() - 3U);  // drop a colour byte -> truncated
    rewrite_zip_entry_bytes(zip_path, v2_entry(document, "point_clouds/" +
        std::to_string(document.workspace().chunks().front()->active_point_cloud_id()) + ".bin"), sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

// ---------------------------------------------------------------------------
// Mesh sidecar (milestone 7). These follow the byte helpers defined above.
// ---------------------------------------------------------------------------

/// Truncating the mesh sidecar must be rejected on load, not silently accepted.
TEST_F(TempDir, TruncatedMeshSidecarThrows) {
    Document document = build_mesh_document();
    const std::string psx = path("mesh.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("mesh.files/project.zip");

    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const std::string mesh_path = v2_entry(
        document, "models/" +
        std::to_string(document.workspace().chunks().front()->active_model_id()) +
        "/mesh.bin");
    std::vector<std::uint8_t> sidecar = ZipArchive::parse(bytes).get(mesh_path);
    ASSERT_GT(sidecar.size(), 3U);
    sidecar.resize(sidecar.size() - 3U);  // drop a colour byte -> truncated
    rewrite_zip_entry_bytes(zip_path, mesh_path, sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// An out-of-range face index in the sidecar is rejected: the reader validates
/// face indices against the vertex count on load. The sidecar is hand built so
/// its own counts agree with the XML — only the index is invalid.
TEST_F(TempDir, OutOfRangeMeshFaceIndexThrows) {
    Document document = build_mesh_document();
    const std::string psx = path("mesh.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("mesh.files/project.zip");

    // "AMS1", 4 vertices, 2 faces, flags = 0, then 4 * 3 f64 and 2 faces.
    std::vector<std::uint8_t> sidecar = {'A', 'M', 'S', '1'};
    put_u32(sidecar, 4);
    put_u32(sidecar, 2);
    sidecar.push_back(0);
    for (int i = 0; i < 12; ++i) put_f64(sidecar, static_cast<double>(i));
    put_i32(sidecar, 0);
    put_i32(sidecar, 1);
    put_i32(sidecar, 2);
    put_i32(sidecar, 1);
    put_i32(sidecar, 2);
    put_i32(sidecar, 99);  // out of range for 4 vertices
    rewrite_zip_entry_bytes(
        zip_path,
        v2_entry(document, "models/" +
            std::to_string(document.workspace().chunks().front()->active_model_id()) +
            "/mesh.bin"),
        sidecar);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}


// ---------------------------------------------------------------------------
// DEM and orthomosaic (milestone 9). These follow the byte helpers above.
// ---------------------------------------------------------------------------

/// A chunk carrying a DEM whose grid deliberately does not divide evenly into
/// the 256-pixel GeoTIFF tile size, with a nodata sentinel that is NOT the
/// default, and an orthomosaic whose label grid includes -1 cells.
Document build_products_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("products");
    document.set_active_chunk_id(chunk.id());

    Elevation& dem = chunk.elevation();
    dem.raster.grid.width = 37;   // 37 = 256 % 37 != 0: partial edge tiles
    dem.raster.grid.height = 23;
    dem.raster.grid.origin_x = 500000.0 + 1.0 / 3.0;
    dem.raster.grid.origin_y = 5410000.0 - 7.0 / 9.0;
    dem.raster.grid.pixel_size_x = 0.25;
    dem.raster.grid.pixel_size_y = 0.5;
    dem.raster.channels = 1;
    dem.raster.data.resize(37U * 23U);
    for (std::size_t i = 0; i < dem.raster.data.size(); ++i) {
        // A deterministic non-uniform surface; every 17th cell is nodata.
        dem.raster.data[i] = (i % 17 == 3) ? -9999.5F
                                          : static_cast<float>(100.0 + std::sin(static_cast<double>(i) * 0.1));
    }
    dem.set_epsg(32632);
    dem.nodata = -9999.5F;
    dem.source = "depth_maps";

    Orthomosaic& ortho = chunk.orthomosaic();
    ortho.raster.grid.width = 19;  // also not a tile multiple
    ortho.raster.grid.height = 13;
    ortho.raster.grid.origin_x = 500000.0 + 1.0 / 3.0;
    ortho.raster.grid.origin_y = 5410000.0 - 7.0 / 9.0;
    ortho.raster.grid.pixel_size_x = 0.25;
    ortho.raster.grid.pixel_size_y = 0.5;
    ortho.raster.channels = 4;
    ortho.raster.data.resize(19U * 13U * 4U);
    for (std::size_t i = 0; i < ortho.raster.data.size(); ++i) {
        ortho.raster.data[i] = static_cast<std::uint8_t>((i * 37U + 11U) % 256U);
    }
    ortho.set_epsg(32632);
    ortho.surface = "elevation";
    const std::size_t cells = 19U * 13U;
    ortho.labels.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        ortho.labels[i] = (i % 5U == 0) ? -1 : static_cast<std::int32_t>(i % 7U);
    }
    return document;
}

TEST_F(TempDir, RoundTripPreservesElevationAndOrthomosaic) {
    Document original = build_products_document();
    const std::string psx = path("products.psx");
    ProjectIO::save(original, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_TRUE(got == *original.workspace().chunks().front());

    const Elevation& dem = got.elevation();
    EXPECT_EQ(dem.raster.grid.width, 37);
    EXPECT_EQ(dem.raster.grid.height, 23);
    EXPECT_DOUBLE_EQ(dem.raster.grid.origin_x, 500000.0 + 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(dem.raster.grid.origin_y, 5410000.0 - 7.0 / 9.0);
    EXPECT_DOUBLE_EQ(dem.raster.grid.pixel_size_x, 0.25);
    EXPECT_DOUBLE_EQ(dem.raster.grid.pixel_size_y, 0.5);
    EXPECT_EQ(dem.epsg(), 32632);
    EXPECT_FLOAT_EQ(dem.nodata, -9999.5F);
    EXPECT_EQ(dem.source, "depth_maps");

    const Orthomosaic& ortho = got.orthomosaic();
    EXPECT_EQ(ortho.raster.grid.width, 19);
    EXPECT_EQ(ortho.raster.grid.height, 13);
    EXPECT_EQ(ortho.raster.channels, 4);
    EXPECT_EQ(ortho.epsg(), 32632);
    EXPECT_EQ(ortho.surface, "elevation");
    ASSERT_EQ(ortho.labels.size(), 19U * 13U);
    EXPECT_EQ(ortho.labels[0], -1);  // i % 5 == 0
}

/// Both products live inside project.zip as GeoTIFF entries (and one labels
/// sidecar), each announced by its doc.xml element.
TEST_F(TempDir, ProductsStoredAsGeoTiffSidecars) {
    Document original = build_products_document();
    const std::string psx = path("products.psx");
    ProjectIO::save(original, psx);

    std::ifstream in(path("products.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    ASSERT_TRUE(archive.contains(v2_entry(original, "elevations/" +
        std::to_string(original.workspace().chunks().front()->active_elevation_id()) + ".tif")));
    ASSERT_TRUE(archive.contains(v2_entry(original, "orthomosaics/" +
        std::to_string(original.workspace().chunks().front()->active_orthomosaic_id()) + ".tif")));
    ASSERT_TRUE(archive.contains(v2_entry(original, "orthomosaics/" +
        std::to_string(original.workspace().chunks().front()->active_orthomosaic_id()) + ".labels.bin")));

    // A TIFF starts with "II*\0" (little-endian) or "MM\0*" (big-endian).
    const std::vector<std::uint8_t>& dem = archive.get(v2_entry(original, "elevations/" +
        std::to_string(original.workspace().chunks().front()->active_elevation_id()) + ".tif"));
    ASSERT_GE(dem.size(), 4U);
    EXPECT_TRUE((dem[0] == 'I' && dem[1] == 'I') || (dem[0] == 'M' && dem[1] == 'M'));
    const std::vector<std::uint8_t>& labels = archive.get(v2_entry(original, "orthomosaics/" +
        std::to_string(original.workspace().chunks().front()->active_orthomosaic_id()) + ".labels.bin"));
    ASSERT_GE(labels.size(), 4U);
    EXPECT_EQ(std::memcmp(labels.data(), "AOL1", 4), 0);

    const std::string xml = archive.get_text("doc.xml");
    EXPECT_NE(xml.find("<elevation "), std::string::npos);
    EXPECT_NE(xml.find("<orthomosaic "), std::string::npos);
    EXPECT_NE(xml.find("epsg=\"32632\""), std::string::npos);
    EXPECT_NE(xml.find("nodataValue=\"-9999.5\""), std::string::npos);
    EXPECT_NE(xml.find("source_data=\"depth_maps\""), std::string::npos);
    EXPECT_NE(xml.find("blendingMode=\"elevation\""), std::string::npos);
}

/// A chunk with neither product must emit no sidecar and no element, so a
/// pre-milestone-9 project keeps its exact bytes.
TEST_F(TempDir, EmptyProductsEmitNoSidecars) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("c");
    document.set_active_chunk_id(chunk.id());
    const std::string psx = path("x.psx");
    ProjectIO::save(document, psx);

    std::ifstream in(path("x.files/project.zip"), std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const ZipArchive archive = ZipArchive::parse(bytes);
    EXPECT_FALSE(archive.contains(v2_entry(document, "elevations/0.tif")));
    EXPECT_FALSE(archive.contains(v2_entry(document, "orthomosaics/0.tif")));
    const std::string xml = archive.get_text("doc.xml");
    EXPECT_EQ(xml.find("<elevation"), std::string::npos);
    EXPECT_EQ(xml.find("<orthomosaic"), std::string::npos);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    EXPECT_TRUE(loaded->workspace().chunks().front()->elevation().empty());
    EXPECT_TRUE(loaded->workspace().chunks().front()->orthomosaic().empty());
}

/// A truncated DEM GeoTIFF is a corrupt project, not a silently short raster.
TEST_F(TempDir, TruncatedDemGeoTiffThrows) {
    Document document = build_products_document();
    const std::string psx = path("products.psx");
    ProjectIO::save(document, psx);

    const std::string zip_path = path("products.files/project.zip");
    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    const std::string dem_path = v2_entry(
        document, "elevations/" +
        std::to_string(document.workspace().chunks().front()->active_elevation_id()) +
        ".tif");
    std::vector<std::uint8_t> dem = ZipArchive::parse(bytes).get(dem_path);
    ASSERT_GT(dem.size(), 64U);
    dem.resize(dem.size() / 2U);  // cut the pixel data in half
    rewrite_zip_entry_bytes(zip_path, dem_path, dem);

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// A missing labels sidecar, or one whose header contradicts the GeoTIFF, is
/// rejected.
TEST_F(TempDir, MissingOrMismatchedOrthoLabelsThrow) {
    Document document = build_products_document();
    const std::string psx = path("products.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("products.files/project.zip");

    const std::string labels_path = v2_entry(
        document, "orthomosaics/" +
        std::to_string(document.workspace().chunks().front()->active_orthomosaic_id()) +
        ".labels.bin");

    // 1. Wrong dimensions in an otherwise valid AOL1 header.
    std::vector<std::uint8_t> labels = {'A', 'O', 'L', '1'};
    put_u32(labels, 19U * 13U);
    put_i32(labels, 19);
    put_i32(labels, 12);  // the GeoTIFF says 13
    for (std::size_t i = 0; i < 19U * 13U; ++i) put_i32(labels, -1);
    rewrite_zip_entry_bytes(zip_path, labels_path, labels);
    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);

    // 2. A label count that does not match width*height.
    labels = {'A', 'O', 'L', '1'};
    put_u32(labels, 10);
    put_i32(labels, 19);
    put_i32(labels, 13);
    for (int i = 0; i < 10; ++i) put_i32(labels, 0);
    rewrite_zip_entry_bytes(zip_path, labels_path, labels);
    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// The doc.xml <elevation> element is authoritative for the grid: a GeoTIFF
/// whose dimensions disagree with it is rejected rather than trusted.
TEST_F(TempDir, DemGridMismatchWithDocXmlThrows) {
    Document document = build_products_document();
    const std::string psx = path("products.psx");
    ProjectIO::save(document, psx);
    const std::string zip_path = path("products.files/project.zip");

    std::ifstream in(zip_path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    ZipArchive archive = ZipArchive::parse(bytes);
    std::string xml = archive.get_text("doc.xml");
    const std::string needle = "width=\"37\"";
    const std::size_t at = xml.find(needle);
    ASSERT_NE(at, std::string::npos);
    xml.replace(at, needle.size(), "width=\"36\"");
    archive.add("doc.xml", xml);
    const std::vector<std::uint8_t> out = archive.serialise();
    std::ofstream out_file(zip_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out_file);
    out_file.write(reinterpret_cast<const char*>(out.data()),
                   static_cast<std::streamsize>(out.size()));
    out_file.close();

    EXPECT_THROW(ProjectIO::load(psx), ProjectIOError);
}

/// Writing a DEM whose data does not match its grid throws before any byte
/// reaches the archive — the writer and reader share one validator.
TEST_F(TempDir, WritingInconsistentDemThrows) {
    Document document = build_products_document();
    document.workspace().chunks().front()->elevation().raster.data.pop_back();

    const std::string psx = path("x.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
}

/// Writing an orthomosaic whose labels are the wrong length throws.
TEST_F(TempDir, WritingInconsistentOrthoLabelsThrows) {
    Document document = build_products_document();
    document.workspace().chunks().front()->orthomosaic().labels.pop_back();

    const std::string psx = path("x.psx");
    EXPECT_THROW(ProjectIO::save(document, psx), ProjectIOError);
    EXPECT_FALSE(fs::exists(psx));
}

/// A project written before the products existed (no <elevation> /
/// <orthomosaic> elements and no sidecars) still loads, and the products read
/// back empty rather than absent.
TEST_F(TempDir, OlderProjectWithoutProductElementsStillLoads) {
    Document document = build_mesh_document();
    const std::string psx = path("old.psx");
    ProjectIO::save(document, psx);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_TRUE(got.elevation().empty());
    EXPECT_TRUE(got.orthomosaic().empty());
    EXPECT_TRUE(got == *document.workspace().chunks().front());
}

}  // namespace
}  // namespace alphashape
