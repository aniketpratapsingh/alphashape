// test_obj.cpp -- tests for the Wavefront OBJ writer (milestone 8).
//
// What these tests pin:
//   * the element counts a parser reads back (v, vt, f),
//   * the v-flip: our v grows down, OBJ's grows up, so vt is 1 - v,
//   * the face reference form (a/ta b/tb c/tc) with 1-based indices,
//   * the companion .mtl and its map_Kd line when a texture file is named,
//   * the no-uv and no-texture forms (plain `f a b c`, no mtllib),
//   * vertex colours appended to the `v` line and normalised to [0,1].

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/obj.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// A unique scratch directory removed on destruction. A plain helper (not a
/// gtest fixture) because these tests construct it directly inside TEST bodies.
class ObjTempDir {
public:
    ObjTempDir() {
        const auto base = fs::temp_directory_path() /
                          ("alphashape_obj_" + std::to_string(platform::process_id()) + "_" +
                           std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
        dir_ = base.string() + "_" + std::to_string(counter_++);
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_, ec);
        EXPECT_FALSE(ec) << ec.message();
    }
    ~ObjTempDir() {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    ObjTempDir(const ObjTempDir&) = delete;
    ObjTempDir& operator=(const ObjTempDir&) = delete;

    std::string path(const std::string& name) const { return (fs::path(dir_) / name).string(); }

    static int counter_;
    std::string dir_;
};
int ObjTempDir::counter_ = 0;

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// A single textured triangle with a distinct v on each corner, so the flip is
/// visible in the results.
Mesh textured_triangle() {
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
    mesh.uvs = {Eigen::Vector2f(0.0F, 0.0F), Eigen::Vector2f(1.0F, 0.25F),
                Eigen::Vector2f(0.25F, 1.0F)};
    mesh.uv_faces = {Eigen::Vector3i(0, 1, 2)};
    return mesh;
}

}  // namespace

// ---------------------------------------------------------------------------
// Textured triangle
// ---------------------------------------------------------------------------

TEST(Obj, WritesVertexTextureAndFaceCounts) {
    ObjTempDir dir;
    const std::string out = dir.path("tri.obj");
    write_obj(out, textured_triangle(), "tri_texture.png");

    const std::string text = read_text(out);
    ASSERT_FALSE(text.empty());
    // The header comment may precede the first vertex line.
    EXPECT_NE(text.find("v "), std::string::npos);

    // Count the element lines a parser would key on. "v " also prefixes "vt ",
    // so each line's leading token is compared rather than searched for.
    std::istringstream lines(text);
    std::string line;
    int v = 0;
    int vt = 0;
    int f = 0;
    while (std::getline(lines, line)) {
        if (line.rfind("v ", 0) == 0) ++v;
        else if (line.rfind("vt ", 0) == 0) ++vt;
        else if (line.rfind("f ", 0) == 0) ++f;
    }
    EXPECT_EQ(v, 3);
    EXPECT_EQ(vt, 3);
    EXPECT_EQ(f, 1);
}

/// The face line must be the 1-based a/ta b/tb c/tc form.
TEST(Obj, FaceUsesOneBasedSlashIndices) {
    ObjTempDir dir;
    const std::string out = dir.path("tri.obj");
    write_obj(out, textured_triangle(), "tri_texture.png");
    const std::string text = read_text(out);
    EXPECT_NE(text.find("f 1/1 2/2 3/3"), std::string::npos);
}

/// Our v grows down; OBJ's grows up. A stored v of 0 must be written as vt v=1,
/// and a stored 1 as vt v=0.
TEST(Obj, VCoordinateIsFlipped) {
    ObjTempDir dir;
    const std::string out = dir.path("tri.obj");
    write_obj(out, textured_triangle(), "tri_texture.png");
    const std::string text = read_text(out);

    // Corner 0: u=0, v=0 -> "vt 0 1".
    EXPECT_NE(text.find("vt 0 1"), std::string::npos);
    // Corner 1: u=1, v=0.25 -> "vt 1 0.75".
    EXPECT_NE(text.find("vt 1 0.75"), std::string::npos);
    // Corner 2: u=0.25, v=1 -> "vt 0.25 0".
    EXPECT_NE(text.find("vt 0.25 0"), std::string::npos);
}

/// A texture file name produces the companion .mtl, a matching mtllib line and
/// the map_Kd entry.
TEST(Obj, WritesMaterialLibraryWhenTextured) {
    ObjTempDir dir;
    const std::string out = dir.path("model.obj");
    write_obj(out, textured_triangle(), "atlas.png");

    const std::string obj_text = read_text(out);
    EXPECT_NE(obj_text.find("mtllib model.mtl"), std::string::npos);
    EXPECT_NE(obj_text.find("usemtl "), std::string::npos);

    const std::string mtl_text = read_text(dir.path("model.mtl"));
    ASSERT_FALSE(mtl_text.empty());
    EXPECT_NE(mtl_text.find("newmtl "), std::string::npos);
    EXPECT_NE(mtl_text.find("map_Kd atlas.png"), std::string::npos);
}

/// With no texture file, no material library is written and no mtllib appears.
TEST(Obj, NoMaterialLibraryWithoutTextureFile) {
    ObjTempDir dir;
    const std::string out = dir.path("plain.obj");
    write_obj(out, textured_triangle(), "");

    const std::string text = read_text(out);
    EXPECT_EQ(text.find("mtllib"), std::string::npos);
    EXPECT_FALSE(fs::exists(dir.path("plain.mtl")));
    // The vt block is still written, because the uv layout exists; only the
    // material reference is dropped.
    EXPECT_NE(text.find("vt "), std::string::npos);
}

/// A mesh with no uv layout at all writes plain `f a b c` and no vt block.
TEST(Obj, NoUvLayoutWritesPlainFaces) {
    ObjTempDir dir;
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};

    const std::string out = dir.path("noUV.obj");
    write_obj(out, mesh, "ignored.png");  // texture ignored: no uv_faces

    const std::string text = read_text(out);
    EXPECT_EQ(text.find("vt "), std::string::npos);
    EXPECT_NE(text.find("f 1 2 3"), std::string::npos);
    EXPECT_EQ(text.find("mtllib"), std::string::npos);
    EXPECT_FALSE(fs::exists(dir.path("noUV.mtl")));
}

/// Vertex colours are appended to the `v` line, normalised to [0,1].
TEST(Obj, VertexColoursAreAppendedAndNormalised) {
    ObjTempDir dir;
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(1.0, 2.0, 3.0)};
    mesh.faces = {Eigen::Vector3i(0, 0, 0)};
    mesh.colours = {255, 128, 0};

    const std::string out = dir.path("colours.obj");
    write_obj(out, mesh, "");

    const std::string text = read_text(out);
    // "v 1 2 3 1 0.501961 0": 255/255 = 1, 128/255 as float, 0/255 = 0.
    EXPECT_NE(text.find("v 1 2 3 1 "), std::string::npos) << text;
    EXPECT_NE(text.find(" 0.501960814 0"), std::string::npos) << text;
}

/// Positions round-trip: a value with many significant digits is written at
/// full precision and reads back to the same double.
TEST(Obj, PositionsRoundTripExactly) {
    ObjTempDir dir;
    Mesh mesh;
    const Eigen::Vector3d tricky(1.0 / 3.0, -2.0 / 7.0, 123456.789012345);
    mesh.vertices = {tricky};
    mesh.faces = {Eigen::Vector3i(0, 0, 0)};

    const std::string out = dir.path("precise.obj");
    write_obj(out, mesh, "");

    const std::string text = read_text(out);
    const std::size_t at = text.find("v ");
    ASSERT_NE(at, std::string::npos);
    // Parse the three fields back and compare exactly.
    std::istringstream line(text.substr(at + 2));
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    line >> x >> y >> z;
    EXPECT_EQ(x, tricky.x());
    EXPECT_EQ(y, tricky.y());
    EXPECT_EQ(z, tricky.z());
}

/// A transform is applied to positions.
TEST(Obj, TransformShiftsPositions) {
    ObjTempDir dir;
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(1.0, 2.0, 3.0)};
    mesh.faces = {Eigen::Vector3i(0, 0, 0)};

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = 10.0;

    const std::string out = dir.path("moved.obj");
    write_obj(out, mesh, "", &t);
    const std::string text = read_text(out);
    EXPECT_NE(text.find("v 11 2 3"), std::string::npos) << text;
}

/// A face index past the end of the vertex array is a ValueError.
TEST(Obj, RejectsOutOfRangeFaceIndex) {
    ObjTempDir dir;
    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 0)};
    EXPECT_THROW(write_obj(dir.path("bad.obj"), mesh, ""), ValueError);
}

/// A uv_faces entry past the end of the uv pool is a ValueError.
TEST(Obj, RejectsOutOfRangeUvIndex) {
    ObjTempDir dir;
    Mesh mesh = textured_triangle();
    mesh.uv_faces[0][2] = 9;
    EXPECT_THROW(write_obj(dir.path("bad.obj"), mesh, "t.png"), ValueError);
}

/// Writing into a directory that does not exist is a ProjectIOError, not a
/// silent no-op.
TEST(Obj, MissingDirectoryIsAnIoError) {
    ObjTempDir dir;
    const Mesh mesh = textured_triangle();
    EXPECT_THROW(write_obj(dir.path("nope/x.obj"), mesh, ""), ProjectIOError);
}

}  // namespace alphashape
