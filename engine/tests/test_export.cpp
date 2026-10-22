// test_export.cpp -- tests for the ExportModel / ExportPointCloud tasks (milestone 7).
//
// The tasks are thin adapters over model/mesh/ply.hpp. What these tests pin:
//
//   * parameter validation: an empty path and an unknown format are ValueErrors,
//     and an unknown keyword never reaches the task (the registry binder rejects
//     it) -- but here we exercise the task-level checks directly.
//   * the header the task produces (element counts and the property names
//     present/absent per save_colors / save_confidence).
//   * world_coordinates: a translation on the chunk transform shifts positions.
//   * the empty-container errors: "No model" / "No point cloud".
//   * that an export does NOT mutate the chunk (stripping happens on a copy).

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "image/io/image_io.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

class ExportTempDir : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base = fs::temp_directory_path() /
                          ("alphashape_export_" + std::to_string(platform::process_id()) + "_" +
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
int ExportTempDir::counter_ = 0;

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::size_t body_offset(const std::vector<std::uint8_t>& bytes) {
    const std::string needle = "end_header\n";
    const std::string text(bytes.begin(), bytes.end());
    const std::size_t at = text.find(needle);
    EXPECT_NE(at, std::string::npos);
    return at + needle.size();
}

std::string header_text(const std::vector<std::uint8_t>& bytes) {
    return std::string(bytes.begin(),
                       bytes.begin() + static_cast<std::ptrdiff_t>(body_offset(bytes)));
}

double read_f64(const std::vector<std::uint8_t>& b, std::size_t at) {
    std::uint64_t bits = 0;
    for (unsigned i = 0; i < 8; ++i) bits |= static_cast<std::uint64_t>(b[at + i]) << (8U * i);
    double d = 0.0;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

/// A one-chunk document with no geometry, plus the chunk handle.
Document empty_document(Chunk** out_chunk = nullptr) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    if (out_chunk != nullptr) {
        *out_chunk = &chunk;
    }
    return document;
}

/// A tiny hand-built mesh: a single triangle with colours and confidence.
void install_tiny_mesh(Chunk& chunk) {
    Mesh& mesh = chunk.model();
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
    mesh.colours = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    mesh.confidence = {0.25F, 0.5F, 0.75F};
}

/// A tiny hand-built cloud: two points with colours and confidence.
void install_tiny_cloud(Chunk& chunk) {
    PointCloud& cloud = chunk.point_cloud();
    cloud.points = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 2.0, 3.0)};
    cloud.normals = {Eigen::Vector3f(0.0F, 0.0F, 1.0F), Eigen::Vector3f(1.0F, 0.0F, 0.0F)};
    cloud.colours = {1, 2, 3, 4, 5, 6};
    cloud.confidence = {0.5F, 1.0F};
}

Params export_params(const std::string& task, const std::string& path,
                     const std::string& format = "ply", bool save_colors = true,
                     bool save_confidence = true, bool world_coordinates = false) {
    Params params = make_params(builtin_registry().require(task));
    params.set("path", ParamValue{path});
    params.set("format", ParamValue{format});
    params.set("save_colors", ParamValue{save_colors});
    params.set("save_confidence", ParamValue{save_confidence});
    params.set("world_coordinates", ParamValue{world_coordinates});
    return params;
}

void run_export(Document& document, const std::string& task, Params& params) {
    run_task(builtin_registry(), document, task, params);
}

}  // namespace

// ---------------------------------------------------------------------------
// ExportModel
// ---------------------------------------------------------------------------

TEST_F(ExportTempDir, ModelHeaderAndCounts) {
    Document document = empty_document();
    install_tiny_mesh(*document.active_chunk());
    Params params = export_params("ExportModel", path("mesh.ply"));
    run_export(document, "ExportModel", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("mesh.ply"));
    ASSERT_FALSE(bytes.empty());
    const std::string header = header_text(bytes);
    EXPECT_NE(header.find("format binary_little_endian 1.0"), std::string::npos);
    EXPECT_NE(header.find("element vertex 3"), std::string::npos);
    EXPECT_NE(header.find("element face 1"), std::string::npos);
    EXPECT_NE(header.find("property double x"), std::string::npos);
    EXPECT_NE(header.find("property list uchar int vertex_indices"), std::string::npos);
    // Colours and confidence were both requested and both present.
    EXPECT_NE(header.find("property uchar red"), std::string::npos);
    EXPECT_NE(header.find("property float confidence"), std::string::npos);
}

TEST_F(ExportTempDir, ModelSaveFlagsDropAttributes) {
    Document document = empty_document();
    install_tiny_mesh(*document.active_chunk());
    Params params =
        export_params("ExportModel", path("no_attrs.ply"), "ply", /*save_colors=*/false,
                      /*save_confidence=*/false);
    run_export(document, "ExportModel", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("no_attrs.ply"));
    ASSERT_FALSE(bytes.empty());
    const std::string header = header_text(bytes);
    EXPECT_EQ(header.find("red"), std::string::npos);
    EXPECT_EQ(header.find("confidence"), std::string::npos);
    // 3 vertices * (3 f64) + 1 face * (uchar + 3 i32): no attribute bytes.
    EXPECT_EQ(bytes.size() - body_offset(bytes), 3U * 24U + 1U + 12U);
}

TEST_F(ExportTempDir, ModelExportDoesNotMutateChunk) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    install_tiny_mesh(chunk);
    const Mesh before = chunk.model();

    Params params = export_params("ExportModel", path("m.ply"), "ply", /*save_colors=*/false,
                                  /*save_confidence=*/false);
    run_export(document, "ExportModel", params);

    // Stripping happens on the copy, so the chunk keeps colours and confidence.
    EXPECT_EQ(chunk.model(), before);
    EXPECT_EQ(chunk.model().colours.size(), 9U);
    EXPECT_EQ(chunk.model().confidence.size(), 3U);
}

TEST_F(ExportTempDir, ModelWorldCoordinatesShiftVertices) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    install_tiny_mesh(chunk);
    Transform t = Transform::Identity();
    t(0, 3) = 100.0;
    t(1, 3) = -5.0;
    t(2, 3) = 0.5;
    chunk.transform() = t;

    Params params = export_params("ExportModel", path("world.ply"), "ply", true, true,
                                  /*world_coordinates=*/true);
    run_export(document, "ExportModel", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("world.ply"));
    const std::size_t body = body_offset(bytes);
    // First vertex (0,0,0) is lifted to the translation column.
    EXPECT_EQ(read_f64(bytes, body + 0), 100.0);
    EXPECT_EQ(read_f64(bytes, body + 8), -5.0);
    EXPECT_EQ(read_f64(bytes, body + 16), 0.5);
    // The chunk's own mesh is untouched: still in chunk-local coordinates.
    EXPECT_EQ(chunk.model().vertices[0], Eigen::Vector3d(0.0, 0.0, 0.0));
}

TEST_F(ExportTempDir, ModelLocalCoordinatesIgnoreTransform) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    install_tiny_mesh(chunk);
    Transform t = Transform::Identity();
    t(0, 3) = 100.0;
    chunk.transform() = t;

    Params params = export_params("ExportModel", path("local.ply"));
    run_export(document, "ExportModel", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("local.ply"));
    const std::size_t body = body_offset(bytes);
    EXPECT_EQ(read_f64(bytes, body + 0), 0.0);
}

TEST_F(ExportTempDir, ModelEmptyPathIsRejected) {
    Document document = empty_document();
    install_tiny_mesh(*document.active_chunk());
    Params params = export_params("ExportModel", "");
    EXPECT_THROW(run_export(document, "ExportModel", params), ValueError);
}

TEST_F(ExportTempDir, ModelUnknownFormatIsRejected) {
    Document document = empty_document();
    install_tiny_mesh(*document.active_chunk());
    Params params = export_params("ExportModel", path("m.gltf"), "gltf");
    EXPECT_THROW(run_export(document, "ExportModel", params), ValueError);
}

TEST_F(ExportTempDir, ModelWithNoMeshFails) {
    Document document = empty_document();
    ASSERT_TRUE(document.active_chunk()->model().empty());
    Params params = export_params("ExportModel", path("none.ply"));
    EXPECT_THROW(run_export(document, "ExportModel", params), Error);
}

// ---------------------------------------------------------------------------
// ExportModel: OBJ + companion PNG texture (milestone 8)
// ---------------------------------------------------------------------------

namespace {

/// A 4x4 RGB page with a distinctive gradient so the decoded PNG can be checked
/// against what was exported.
Image make_page(int size = 4) {
    Image page(size, size, 3);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            page.at(x, y, 0) = static_cast<std::uint8_t>(x * 60);
            page.at(x, y, 1) = static_cast<std::uint8_t>(y * 60);
            page.at(x, y, 2) = static_cast<std::uint8_t>(40 + x + y);
        }
    }
    return page;
}

/// A tiny mesh carrying a UV layout: one triangle whose three corners are
/// distinct pool entries, so the OBJ must emit vt lines and an mtl.
Mesh make_textured_triangle() {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
    mesh.uvs = {Eigen::Vector2f(0.1F, 0.2F), Eigen::Vector2f(0.9F, 0.2F),
                Eigen::Vector2f(0.2F, 0.8F)};
    mesh.uv_faces = {Eigen::Vector3i(0, 1, 2)};
    return mesh;
}

std::string read_text(const std::string& path) {
    const std::vector<std::uint8_t> bytes = read_bytes(path);
    return std::string(bytes.begin(), bytes.end());
}

}  // namespace

TEST_F(ExportTempDir, ModelObjWritesVertexTexturesAndMaterialLibrary) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    chunk.model() = make_textured_triangle();
    chunk.model_textures().push_back(make_page());

    Params params = export_params("ExportModel", path("model.obj"), "obj");
    run_export(document, "ExportModel", params);

    const std::string obj = read_text(path("model.obj"));
    ASSERT_FALSE(obj.empty());
    // Three texture coordinates and a face referencing them as a/ta.
    std::size_t vt_count = 0;
    const std::string vt = "\nvt ";
    for (std::size_t at = obj.find(vt); at != std::string::npos; at = obj.find(vt, at + 1)) {
        ++vt_count;
    }
    EXPECT_EQ(vt_count, 3U);
    EXPECT_NE(obj.find("v 0 0 0"), std::string::npos);
    EXPECT_NE(obj.find("f 1/1 2/2 3/3"), std::string::npos);
    EXPECT_NE(obj.find("mtllib model.mtl"), std::string::npos);

    // The material library names the companion PNG and uses it as diffuse map.
    const std::string mtl = read_text(path("model.mtl"));
    ASSERT_FALSE(mtl.empty());
    EXPECT_NE(mtl.find("map_Kd model.png"), std::string::npos);

    // The PNG exists and decodes back to the page that was exported.
    const auto info = read_image_info(path("model.png"));
    EXPECT_EQ(info.format, ImageFormat::Png);
    EXPECT_EQ(info.width, 4);
    EXPECT_EQ(info.height, 4);
    const Image decoded = read_image(path("model.png"));
    const Image page = make_page();
    ASSERT_EQ(decoded.width(), page.width());
    ASSERT_EQ(decoded.height(), page.height());
    for (int y = 0; y < page.height(); ++y) {
        for (int x = 0; x < page.width(); ++x) {
            for (int c = 0; c < 3; ++c) {
                EXPECT_EQ(decoded.at(x, y, c), page.at(x, y, c)) << "at " << x << "," << y
                                                                  << " c" << c;
            }
        }
    }
}

TEST_F(ExportTempDir, ModelObjSaveTextureFalseWritesNoPng) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    chunk.model() = make_textured_triangle();
    chunk.model_textures().push_back(make_page());

    Params params = export_params("ExportModel", path("bare.obj"), "obj");
    params.set("save_texture", ParamValue{false});
    run_export(document, "ExportModel", params);

    // The OBJ is written with its uvs (the geometry is still mapped -- only the
    // texture image is omitted), but no material library and no PNG appear.
    EXPECT_FALSE(fs::exists(path("bare.png")));
    EXPECT_FALSE(fs::exists(path("bare.mtl")));
    const std::string obj = read_text(path("bare.obj"));
    ASSERT_FALSE(obj.empty());
    EXPECT_EQ(obj.find("mtllib"), std::string::npos);
    EXPECT_NE(obj.find("vt "), std::string::npos);
    EXPECT_NE(obj.find("f 1/1 2/2 3/3"), std::string::npos);
}

TEST_F(ExportTempDir, ModelObjWithoutTexturePageWritesPlainObj) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    chunk.model() = make_textured_triangle();  // uvs present, but no baked page

    Params params = export_params("ExportModel", path("plain.obj"), "obj");
    run_export(document, "ExportModel", params);

    EXPECT_FALSE(fs::exists(path("plain.png")));
    const std::string obj = read_text(path("plain.obj"));
    ASSERT_FALSE(obj.empty());
    EXPECT_EQ(obj.find("mtllib"), std::string::npos);
}

TEST_F(ExportTempDir, ModelPlyStillWritesAFile) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    install_tiny_mesh(chunk);
    chunk.model_textures().push_back(make_page());

    Params params = export_params("ExportModel", path("mesh.ply"));
    run_export(document, "ExportModel", params);
    EXPECT_FALSE(read_bytes(path("mesh.ply")).empty());
    // The PLY path never writes a texture even when one is present.
    EXPECT_FALSE(fs::exists(path("mesh.png")));
}

TEST_F(ExportTempDir, CloudRejectsObjFormat) {
    Document document = empty_document();
    install_tiny_cloud(*document.active_chunk());
    Params params = export_params("ExportPointCloud", path("cloud.obj"), "obj");
    EXPECT_THROW(run_export(document, "ExportPointCloud", params), ValueError);
}

// ---------------------------------------------------------------------------
// ExportPointCloud
// ---------------------------------------------------------------------------

TEST_F(ExportTempDir, CloudHeaderAndCounts) {
    Document document = empty_document();
    install_tiny_cloud(*document.active_chunk());
    Params params = export_params("ExportPointCloud", path("cloud.ply"));
    run_export(document, "ExportPointCloud", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("cloud.ply"));
    ASSERT_FALSE(bytes.empty());
    const std::string header = header_text(bytes);
    EXPECT_NE(header.find("element vertex 2"), std::string::npos);
    // A cloud has no faces.
    EXPECT_EQ(header.find("element face"), std::string::npos);
    EXPECT_NE(header.find("property float nx"), std::string::npos);
    EXPECT_NE(header.find("property uchar red"), std::string::npos);
    EXPECT_NE(header.find("property float confidence"), std::string::npos);
}

TEST_F(ExportTempDir, CloudSaveFlagsDropAttributes) {
    Document document = empty_document();
    install_tiny_cloud(*document.active_chunk());
    Params params =
        export_params("ExportPointCloud", path("cloud_plain.ply"), "ply", /*save_colors=*/false,
                      /*save_confidence=*/false);
    run_export(document, "ExportPointCloud", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("cloud_plain.ply"));
    ASSERT_FALSE(bytes.empty());
    const std::string header = header_text(bytes);
    EXPECT_EQ(header.find("red"), std::string::npos);
    EXPECT_EQ(header.find("confidence"), std::string::npos);
    // 2 points * (3 f64 positions + 3 f32 normals): no attribute bytes.
    EXPECT_EQ(bytes.size() - body_offset(bytes), 2U * (24U + 12U));
}

TEST_F(ExportTempDir, CloudExportDoesNotMutateChunk) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    install_tiny_cloud(chunk);
    const PointCloud before = chunk.point_cloud();

    Params params = export_params("ExportPointCloud", path("c.ply"), "ply", false, false);
    run_export(document, "ExportPointCloud", params);

    EXPECT_EQ(chunk.point_cloud(), before);
    EXPECT_EQ(chunk.point_cloud().colours.size(), 6U);
    EXPECT_EQ(chunk.point_cloud().confidence.size(), 2U);
}

TEST_F(ExportTempDir, CloudWorldCoordinatesShiftPoints) {
    Document document = empty_document();
    Chunk& chunk = *document.active_chunk();
    install_tiny_cloud(chunk);
    Transform t = Transform::Identity();
    t(0, 3) = 7.0;
    t(1, 3) = 8.0;
    chunk.transform() = t;

    Params params = export_params("ExportPointCloud", path("cloud_world.ply"), "ply", true, true,
                                  /*world_coordinates=*/true);
    run_export(document, "ExportPointCloud", params);

    const std::vector<std::uint8_t> bytes = read_bytes(path("cloud_world.ply"));
    const std::size_t body = body_offset(bytes);
    // Point 0 was (0,0,0) -> (7,8,0).
    EXPECT_EQ(read_f64(bytes, body + 0), 7.0);
    EXPECT_EQ(read_f64(bytes, body + 8), 8.0);
}

TEST_F(ExportTempDir, CloudEmptyPathIsRejected) {
    Document document = empty_document();
    install_tiny_cloud(*document.active_chunk());
    Params params = export_params("ExportPointCloud", "");
    EXPECT_THROW(run_export(document, "ExportPointCloud", params), ValueError);
}

TEST_F(ExportTempDir, CloudUnknownFormatIsRejected) {
    Document document = empty_document();
    install_tiny_cloud(*document.active_chunk());
    Params params = export_params("ExportPointCloud", path("c.las"), "las");
    EXPECT_THROW(run_export(document, "ExportPointCloud", params), ValueError);
}

TEST_F(ExportTempDir, CloudWithNoCloudFails) {
    Document document = empty_document();
    ASSERT_TRUE(document.active_chunk()->point_cloud().points.empty());
    Params params = export_params("ExportPointCloud", path("none.ply"));
    EXPECT_THROW(run_export(document, "ExportPointCloud", params), Error);
}

TEST_F(ExportTempDir, ExportRecordsParamsInMeta) {
    Document document = empty_document();
    install_tiny_mesh(*document.active_chunk());
    Params params = export_params("ExportModel", path("meta.ply"));
    run_export(document, "ExportModel", params);
    const Meta& meta = document.active_chunk()->meta();
    ASSERT_NE(meta.find("ExportModel/format"), nullptr);
    EXPECT_EQ(*meta.find("ExportModel/format"), "ply");
    ASSERT_NE(meta.find("ExportModel/path"), nullptr);
    EXPECT_EQ(*meta.find("ExportModel/path"), path("meta.ply"));
}

}  // namespace alphashape
