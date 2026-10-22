// PLY export: header shape, body encoding and the optional transform.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/ply.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

class PlyTempDir : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base = fs::temp_directory_path() /
                          ("alphashape_ply_" + std::to_string(platform::process_id()) + "_" +
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
int PlyTempDir::counter_ = 0;

/// The raw bytes of a file, or empty when it does not exist.
std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// Offset of the first body byte (just past "end_header\n").
std::size_t body_offset(const std::vector<std::uint8_t>& bytes) {
    const std::string needle = "end_header\n";
    const std::string text(bytes.begin(), bytes.end());
    const std::size_t at = text.find(needle);
    EXPECT_NE(at, std::string::npos);
    return at + needle.size();
}

std::string header_text(const std::vector<std::uint8_t>& bytes) {
    return std::string(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(body_offset(bytes)));
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& b, std::size_t at) {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(b[at + i]) << (8U * i);
    return v;
}

double read_f64(const std::vector<std::uint8_t>& b, std::size_t at) {
    std::uint64_t bits = 0;
    for (unsigned i = 0; i < 8; ++i) bits |= static_cast<std::uint64_t>(b[at + i]) << (8U * i);
    double d = 0.0;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

float read_f32(const std::vector<std::uint8_t>& b, std::size_t at) {
    const std::uint32_t bits = read_u32(b, at);
    float f = 0.0F;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

TEST_F(PlyTempDir, MeshHeaderAndCounts) {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
    write_ply(path("mesh.ply"), mesh);

    const std::vector<std::uint8_t> bytes = read_bytes(path("mesh.ply"));
    ASSERT_FALSE(bytes.empty());
    const std::string header = header_text(bytes);
    EXPECT_NE(header.find("format binary_little_endian 1.0"), std::string::npos);
    EXPECT_NE(header.find("element vertex 3"), std::string::npos);
    EXPECT_NE(header.find("element face 1"), std::string::npos);
    EXPECT_NE(header.find("property double x"), std::string::npos);
    EXPECT_NE(header.find("property list uchar int vertex_indices"), std::string::npos);
    // No colours and no confidence were provided, so those properties are absent.
    EXPECT_EQ(header.find("red"), std::string::npos);
    EXPECT_EQ(header.find("confidence"), std::string::npos);

    // 3 vertices * (3 f64) + 1 face * (uchar + 3 i32).
    const std::size_t expected = 3U * 24U + 1U + 12U;
    EXPECT_EQ(bytes.size() - body_offset(bytes), expected);

    // The second vertex's x is 1.0, bit-exact via the f64 reader.
    EXPECT_EQ(read_f64(bytes, body_offset(bytes) + 24), 1.0);
}

TEST_F(PlyTempDir, MeshWithColoursAndConfidence) {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(2.5, -3.5, 4.5), Eigen::Vector3d(1.0, 1.0, 1.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 0)};
    mesh.colours = {10, 20, 30, 200, 210, 220};
    mesh.confidence = {0.25F, 0.75F};
    write_ply(path("m.ply"), mesh);

    const std::vector<std::uint8_t> bytes = read_bytes(path("m.ply"));
    const std::string header = header_text(bytes);
    EXPECT_NE(header.find("property uchar red"), std::string::npos);
    EXPECT_NE(header.find("property float confidence"), std::string::npos);

    const std::size_t body = body_offset(bytes);
    // Each vertex is 3 f64 (24) + RGB (3) + confidence (4) = 31 bytes.
    EXPECT_EQ(bytes[body + 24], 10);
    EXPECT_EQ(bytes[body + 25], 20);
    EXPECT_EQ(bytes[body + 26], 30);
    EXPECT_FLOAT_EQ(read_f32(bytes, body + 27), 0.25F);
    // Vertex 1 starts at 31; its red byte is 24 bytes in.
    EXPECT_EQ(bytes[body + 31 + 24], 200);
    // Total: 2 vertices * 31 + 1 face * 13.
    EXPECT_EQ(bytes.size() - body, 2U * 31U + 13U);
}

TEST_F(PlyTempDir, MeshTransformMovesVertices) {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(1.0, 2.0, 3.0)};
    mesh.faces = {Eigen::Vector3i(0, 0, 0)};
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = 10.0;
    t(1, 3) = -1.0;
    t(2, 3) = 0.5;
    write_ply(path("t.ply"), mesh, &t);

    const std::vector<std::uint8_t> bytes = read_bytes(path("t.ply"));
    const std::size_t body = body_offset(bytes);
    EXPECT_EQ(read_f64(bytes, body + 0), 11.0);
    EXPECT_EQ(read_f64(bytes, body + 8), 1.0);
    EXPECT_EQ(read_f64(bytes, body + 16), 3.5);
}

TEST_F(PlyTempDir, PointCloudHeaderNormalsAndTransform) {
    PointCloud cloud;
    cloud.points = {Eigen::Vector3d(1.0, 0.0, 0.0)};
    cloud.normals = {Eigen::Vector3f(0.0F, 0.0F, 1.0F)};
    cloud.colours = {1, 2, 3};
    cloud.confidence = {0.5F};

    // A 90-degree rotation about ẑ plus a translation: the normal must rotate
    // to (-1, 0, 0) and must NOT pick up the translation.
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 0) = 0.0;
    t(0, 1) = -1.0;
    t(1, 0) = 1.0;
    t(1, 1) = 0.0;
    t(0, 3) = 100.0;
    t(1, 3) = 200.0;
    write_ply(path("c.ply"), cloud, &t);

    const std::vector<std::uint8_t> bytes = read_bytes(path("c.ply"));
    const std::string header = header_text(bytes);
    EXPECT_NE(header.find("element vertex 1"), std::string::npos);
    EXPECT_EQ(header.find("element face"), std::string::npos);  // no faces
    EXPECT_NE(header.find("property float nx"), std::string::npos);

    const std::size_t body = body_offset(bytes);
    EXPECT_EQ(read_f64(bytes, body + 0), 100.0);
    EXPECT_EQ(read_f64(bytes, body + 8), 201.0);
    // Normal: rotation only, so (-0, 0, 1).
    const std::size_t n_at = body + 24;
    EXPECT_FLOAT_EQ(read_f32(bytes, n_at + 0), 0.0F);
    EXPECT_FLOAT_EQ(read_f32(bytes, n_at + 4), 0.0F);
    EXPECT_FLOAT_EQ(read_f32(bytes, n_at + 8), 1.0F);
    // Colour + confidence follow the normal.
    EXPECT_EQ(bytes[n_at + 12], 1);
    EXPECT_FLOAT_EQ(read_f32(bytes, n_at + 15), 0.5F);
}

TEST_F(PlyTempDir, MismatchedAttributeSizeThrows) {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0)};
    mesh.colours = {1, 2, 3};  // needs 3 bytes per vertex
    EXPECT_THROW(write_ply(path("bad.ply"), mesh), ValueError);
}

TEST_F(PlyTempDir, UnopenablePathThrows) {
    const Mesh mesh;
    EXPECT_THROW(write_ply(path("does_not_exist_dir/m.ply"), mesh), ProjectIOError);
}

}  // namespace
}  // namespace alphashape
