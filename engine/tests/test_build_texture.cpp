// test_build_texture.cpp -- tests for the BuildUV and BuildTexture tasks
// (milestone 8).
//
// Unit level: parameter validation for both tasks, the validation gates (no
// model for BuildUV, no UV layout for BuildTexture), and a project save/load
// round-trip after a full BuildUV + BuildTexture run.
//
// End to end: BuildModel on the truth-pose sequence fixture (dense_fixture.hpp),
// then BuildUV and BuildTexture, measuring:
//
//   * coverage      -- fraction of in-chart texels that are non-black.
//   * photo-consistency -- for random in-chart texels, the difference between
//                      the baked grey and the mean grey of every camera that
//                      sees the texel's 3-D point by a depth test.
//   * an OBJ export of the textured model.
//
// The end-to-end scenario is ONE test on purpose: gtest_discover_tests runs each
// case in its own process, so a split into several cases would pay the whole
// dense + BuildModel pass per case.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/mesh/build_model_task.hpp"
#include "common/algorithm/texture/bake.hpp"
#include "common/algorithm/texture/face_views.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "dense_fixture.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/uv_charts.hpp"

namespace alphashape {
namespace {

using dense_fixture::aligned_document;
using dense_fixture::parse_truth_images;
using dense_fixture::seq_files;
using dense_fixture::set_truth_calibration;
using dense_fixture::set_truth_poses;
using dense_fixture::TempDir;
using dense_fixture::TruthImage;

// -- task runners ------------------------------------------------------------

Params uv_params(std::int64_t texture_size = 1024, const std::string& mapping_mode = "generic",
                 std::int64_t page_count = 1) {
    Params params = make_params(builtin_registry().require("BuildUV"));
    params.set("mapping_mode", ParamValue{mapping_mode});
    params.set("page_count", ParamValue{page_count});
    params.set("texture_size", ParamValue{texture_size});
    return params;
}

Params texture_params(std::int64_t texture_size = 1024,
                      const std::string& blending_mode = "mosaic", bool fill_holes = true,
                      bool ghosting_filter = true, bool color_correction = true) {
    Params params = make_params(builtin_registry().require("BuildTexture"));
    params.set("blending_mode", ParamValue{blending_mode});
    params.set("texture_size", ParamValue{texture_size});
    params.set("fill_holes", ParamValue{fill_holes});
    params.set("ghosting_filter", ParamValue{ghosting_filter});
    params.set("color_correction", ParamValue{color_correction});
    return params;
}

void run_task_named(Document& document, const std::string& name, Params& params,
                    const Progress& progress = Progress{}) {
    run_task(builtin_registry(), document, name, params, progress);
}

/// A one-chunk document with no geometry, so a task's parameter validation runs
/// (run_task rejects a chunk-targeted task on a document with no active chunk
/// before apply() is reached).
Document empty_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

/// A flat grid mesh on the truth scene's ground plane (z = 0), covering the area
/// the arc of cameras looks at. CCW wound so its normal points up (+z).
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
            // CCW seen from +z (above): (i,j) -> (i+1,j) -> (i,j+1).
            mesh.faces.emplace_back(at(i, j), at(i + 1, j), at(i, j + 1));
            mesh.faces.emplace_back(at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
        }
    }
    return mesh;
}

bool message_contains(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}

// -- texel map (re-derived here so the test never trusts the baker's own map) --

/// Which face covers each texel and the texel centre's barycentric coordinates
/// in that face. face == -1 means no chart covers the texel.
struct TexelRef {
    int face = -1;
    Eigen::Vector3d bary = Eigen::Vector3d::Zero();
};

Eigen::Vector3d barycentric(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
                            const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    const Eigen::Vector2d v0 = b - a;
    const Eigen::Vector2d v1 = c - a;
    const Eigen::Vector2d v2 = p - a;
    const double d00 = v0.dot(v0);
    const double d01 = v0.dot(v1);
    const double d11 = v1.dot(v1);
    const double d20 = v2.dot(v0);
    const double d21 = v2.dot(v1);
    const double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-18) {
        return Eigen::Vector3d(-1.0, 0.0, 0.0);  // degenerate: never inside
    }
    const double v = (d11 * d20 - d01 * d21) / denom;
    const double w = (d00 * d21 - d01 * d20) / denom;
    return Eigen::Vector3d(1.0 - v - w, v, w);
}

std::vector<TexelRef> build_texel_map(const Mesh& mesh, int size) {
    std::vector<TexelRef> map(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
    const double inv = 1.0 / static_cast<double>(size);
    for (std::size_t f = 0; f < mesh.uv_faces.size(); ++f) {
        const Eigen::Vector3i& uvf = mesh.uv_faces[f];
        const Eigen::Vector2d a(mesh.uvs[static_cast<std::size_t>(uvf[0])].x(),
                                mesh.uvs[static_cast<std::size_t>(uvf[0])].y());
        const Eigen::Vector2d b(mesh.uvs[static_cast<std::size_t>(uvf[1])].x(),
                                mesh.uvs[static_cast<std::size_t>(uvf[1])].y());
        const Eigen::Vector2d c(mesh.uvs[static_cast<std::size_t>(uvf[2])].x(),
                                mesh.uvs[static_cast<std::size_t>(uvf[2])].y());
        const int i0 = std::max(0, static_cast<int>(std::floor(std::min({a.x(), b.x(), c.x()}) *
                                                              size - 1.0)));
        const int i1 = std::min(size - 1, static_cast<int>(std::ceil(std::max({a.x(), b.x(), c.x()}) *
                                                                     size)));
        const int j0 = std::max(0, static_cast<int>(std::floor(std::min({a.y(), b.y(), c.y()}) *
                                                              size - 1.0)));
        const int j1 = std::min(size - 1, static_cast<int>(std::ceil(std::max({a.y(), b.y(), c.y()}) *
                                                                     size)));
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                const std::size_t at = static_cast<std::size_t>(j) * static_cast<std::size_t>(size) +
                                       static_cast<std::size_t>(i);
                if (map[at].face >= 0) {
                    continue;  // an earlier face already claimed this texel
                }
                const Eigen::Vector2d p((static_cast<double>(i) + 0.5) * inv,
                                        (static_cast<double>(j) + 0.5) * inv);
                const Eigen::Vector3d bary = barycentric(p, a, b, c);
                constexpr double kFillTolerance = 1e-6;
                if (bary.x() < -kFillTolerance || bary.y() < -kFillTolerance ||
                    bary.z() < -kFillTolerance) {
                    continue;
                }
                map[at].face = static_cast<int>(f);
                map[at].bary = bary;
            }
        }
    }
    return map;
}

double grey(const Image& image, int x, int y) {
    if (image.channels() >= 3) {
        return 0.2126 * image.at(x, y, 0) + 0.7152 * image.at(x, y, 1) + 0.0722 * image.at(x, y, 2);
    }
    return image.at(x, y, 0);
}

constexpr double kBlackThreshold = 4.0;  // luminance at or below this counts as black

bool is_black(const Image& page, int x, int y) { return grey(page, x, y) <= kBlackThreshold; }

}  // namespace

// ---------------------------------------------------------------------------
// 1. Parameter validation
// ---------------------------------------------------------------------------

TEST(BuildUvParams, RejectsUnknownMappingMode) {
    Document document = empty_document();
    Params params = uv_params(1024, "bogus");
    try {
        run_task_named(document, "BuildUV", params);
        FAIL() << "expected ValueError";
    } catch (const ValueError& error) {
        EXPECT_TRUE(message_contains(error.what(), "mapping_mode"));
    }
}

TEST(BuildUvParams, RejectsUnsupportedMappingMode) {
    // A value Metashape's API names but we do not implement yet is an
    // "unsupported" Error, not a ValueError (it is a known, understood mode).
    Document document = empty_document();
    Params params = uv_params(1024, "orthophoto");
    try {
        run_task_named(document, "BuildUV", params);
        FAIL() << "expected Error";
    } catch (const ValueError& error) {
        FAIL() << "unexpected ValueError: " << error.what();
    } catch (const Error& error) {
        EXPECT_TRUE(message_contains(error.what(), "orthophoto mapping is not supported yet"));
    }
}

TEST(BuildUvParams, RejectsInvalidPageCount) {
    Document document = empty_document();
    Params zero = uv_params(1024, "generic", 0);
    EXPECT_THROW(run_task_named(document, "BuildUV", zero), ValueError);

    Params many = uv_params(1024, "generic", 2);
    try {
        run_task_named(document, "BuildUV", many);
        FAIL() << "expected Error";
    } catch (const ValueError& error) {
        FAIL() << "unexpected ValueError: " << error.what();
    } catch (const Error& error) {
        EXPECT_TRUE(message_contains(error.what(), "page_count > 1 is not supported yet"));
    }
}

TEST(BuildUvParams, RejectsOutOfRangeTextureSize) {
    Document document = empty_document();
    for (const std::int64_t size : {std::int64_t{100}, std::int64_t{20000}}) {
        Params params = uv_params(size);
        EXPECT_THROW(run_task_named(document, "BuildUV", params), ValueError) << "size " << size;
    }
}

TEST(BuildUvValidate, RequiresAModel) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    ASSERT_TRUE(chunk.model().empty());
    Params params = uv_params(512);
    try {
        run_task_named(document, "BuildUV", params);
        FAIL() << "expected ValueError";
    } catch (const ValueError& error) {
        EXPECT_TRUE(message_contains(error.what(), "build a model first"));
    }
}

TEST(BuildTextureParams, RejectsUnknownBlendingMode) {
    Document document = empty_document();
    Params params = texture_params(1024, "bogus");
    try {
        run_task_named(document, "BuildTexture", params);
        FAIL() << "expected ValueError";
    } catch (const ValueError& error) {
        EXPECT_TRUE(message_contains(error.what(), "blending"));
    }
}

TEST(BuildTextureParams, RejectsOutOfRangeTextureSize) {
    Document document = empty_document();
    for (const std::int64_t size : {std::int64_t{100}, std::int64_t{20000}}) {
        Params params = texture_params(size);
        EXPECT_THROW(run_task_named(document, "BuildTexture", params), ValueError) << "size " << size;
    }
}

TEST(BuildTextureValidate, RequiresAUvLayout) {
    // A model with no uv_faces must be rejected before any camera work: the
    // message points the caller at BuildUV.
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    chunk.model() = make_ground_grid(3, 3);
    ASSERT_FALSE(chunk.model().empty());
    ASSERT_TRUE(chunk.model().uv_faces.empty());
    Params params = texture_params(512);
    try {
        run_task_named(document, "BuildTexture", params);
        FAIL() << "expected ValueError";
    } catch (const ValueError& error) {
        EXPECT_TRUE(message_contains(error.what(), "BuildUV"));
    }
    // The failed run must not have installed a page.
    EXPECT_TRUE(chunk.model_textures().empty());
}

// ---------------------------------------------------------------------------
// 2. Round-trip after BuildUV + BuildTexture
// ---------------------------------------------------------------------------

TEST(BuildTextureRoundTrip, RoundTripsAfterBuildUvAndTexture) {
    // A truth-pose document (so the cameras are enabled and their poses are
    // meaningful) carrying a hand-built ground grid instead of a BuildModel mesh:
    // the round-trip only needs a textured model, not the meshing cost.
    const std::vector<TruthImage> truth = parse_truth_images("seq_truth.json");
    Document document = aligned_document(seq_files(4));
    set_truth_poses(document, truth);
    set_truth_calibration(document, 480, 360, 500.0);

    Chunk& chunk = *document.active_chunk();
    chunk.model() = make_ground_grid(6, 6);

    Params uv = uv_params(/*texture_size=*/256);
    run_task_named(document, "BuildUV", uv);
    Params texture = texture_params(/*texture_size=*/256);
    run_task_named(document, "BuildTexture", texture);

    ASSERT_EQ(chunk.model_textures().size(), 1U);
    const Image baked = chunk.model_textures()[0];
    ASSERT_EQ(baked.width(), 256);
    ASSERT_EQ(baked.height(), 256);

    const Chunk before = *document.active_chunk();
    TempDir dir("alphashape_build_texture_roundtrip");
    const std::string project = dir.file("proj.psx");
    document.set_path(project);
    ProjectIO::save(document, project);
    auto reloaded = ProjectIO::load(project);

    ASSERT_NE(reloaded->active_chunk(), nullptr);
    // The whole chunk, texture pixels included, must compare equal.
    EXPECT_TRUE(*reloaded->active_chunk() == before);
    ASSERT_EQ(reloaded->active_chunk()->model_textures().size(), 1U);
    const Image& reloaded_page = reloaded->active_chunk()->model_textures()[0];
    ASSERT_EQ(reloaded_page.size_bytes(), baked.size_bytes());
    EXPECT_EQ(std::memcmp(reloaded_page.data(), baked.data(), baked.size_bytes()), 0);
}

// ---------------------------------------------------------------------------
// 3. End to end on the analytic sequence fixture, on the TRUE poses
//
// AddPhotos + MatchPhotos + AlignCameras build the tracks, the camera poses are
// overwritten with data/seq_truth.json, the tracks are re-triangulated and the
// sensor is set to the true pinhole calibration -- exactly the setup
// test_build_model.cpp uses (dense_fixture.hpp). BuildModel then runs with
// depth maps, BuildUV lays the mesh out, and BuildTexture bakes one page per
// blending mode.
// ---------------------------------------------------------------------------

namespace {

constexpr int kTruthWidth = 480;
constexpr int kTruthHeight = 360;
constexpr double kTruthFocal = 500.0;

/// One input photograph plus everything a consistency test needs to decide
/// whether it sees a surface point.
struct View {
    const Camera* camera = nullptr;
    const Image* image = nullptr;
    Calibration full;          ///< full-resolution calibration
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

Eigen::Matrix3d pose_R(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_t(const Transform& pose) { return pose.block<3, 1>(0, 3); }

/// One view per enabled camera, in camera order, with the photograph decoded
/// into `storage` (the caller keeps `storage` alive: View borrows it).
std::vector<View> collect_views(const Document& document, const Chunk& chunk,
                                std::vector<Image>& storage) {
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk.frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());
        }
    }
    std::vector<View> views;
    storage.reserve(chunk.cameras().size());  // so &storage.back() stays stable
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;
        }
        const auto frame_it = frame_of.find(camera.id);
        if (frame_it == frame_of.end()) {
            continue;
        }
        // .string() is explicit: libc++/libstdc++ give path an implicit
        // conversion to the native string type, MSVC's does not, and the
        // engine's file API takes a std::string path throughout.
        storage.push_back(
            read_image((std::filesystem::path(document.path()).parent_path() /
                        frame_it->second->path())
                           .string()));
        View view;
        view.camera = &camera;
        view.image = &storage.back();
        view.full = chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        view.R = pose_R(camera.pose);
        view.t = pose_t(camera.pose);
        views.push_back(view);
    }
    return views;
}

struct TextureMetrics {
    std::size_t chart_texels = 0;
    std::size_t chart_non_black = 0;
    double coverage = 0.0;
    std::size_t seen_texels = 0;      ///< in-chart texels at least one view sees
    std::size_t covered_seen = 0;     ///< of those, how many are non-black
    double seen_fraction = 0.0;       ///< seen_texels / chart_texels
    double covered_seen_fraction = 0.0;  ///< covered_seen / seen_texels

    // Why the black in-chart texels stay black. Each such texel is bucketed by
    // asking every view whether the texel's surface point projects inside that
    // image and, if so, whether the image shows a non-black pixel there:
    //   bb_outside_image - the point projects inside NO image (surface outside
    //                      every camera's frame).
    //   bb_no_photo      - the point is inside some image(s) but every one is
    //                      background there: no photograph recorded this surface,
    //                      so there is no colour to sample. (This is where the
    //                      mesh-bbox margin past the camera arc lands.)
    //   bb_has_photo     - some photograph shows a non-black pixel at the point
    //                      yet the baker still wrote black: an occlusion-picked
    //                      view, a grazing view, or a genuine miss.
    // The three are disjoint and sum to (chart_texels - chart_non_black). The
    // (a)/(b)/(c) split by view-assignment geometry is measured in the
    // BuildTextureDiagnostics test.
    std::size_t bb_outside_image = 0;
    std::size_t bb_no_photo = 0;
    std::size_t bb_has_photo = 0;

    std::size_t consistency_samples = 0;
    double consistency_mean = 0.0;
    double consistency_median = 0.0;
    double consistency_p90 = 0.0;
};

/// The 3-D point a texel's centre maps to, from the covering face's barycentrics.
Eigen::Vector3d texel_point(const Mesh& mesh, const TexelRef& ref) {
    const Eigen::Vector3i& face = mesh.faces[static_cast<std::size_t>(ref.face)];
    return mesh.vertices[static_cast<std::size_t>(face[0])] * ref.bary.x() +
           mesh.vertices[static_cast<std::size_t>(face[1])] * ref.bary.y() +
           mesh.vertices[static_cast<std::size_t>(face[2])] * ref.bary.z();
}

/// Does `view` see the surface point `X`? `count_only` drops the image sample:
/// the depth test alone decides visibility. `out_grey` receives the sampled grey
/// when the view sees the point and count_only is false.
bool view_samples(const Eigen::Vector3d& X, const View& view,
                  const std::unordered_map<std::int32_t, const DepthMap*>& map_of,
                  const std::unordered_map<std::int32_t, Calibration>& scaled_of,
                  double depth_tolerance, bool count_only, double* out_grey) {
    const auto map_it = map_of.find(view.camera->id);
    if (map_it == map_of.end()) {
        return false;
    }
    const DepthMap& depth_map = *map_it->second;
    const auto scaled_it = scaled_of.find(view.camera->id);
    if (scaled_it == scaled_of.end()) {
        return false;
    }
    const Eigen::Vector3d cam = view.R * X + view.t;
    if (!(cam.z() > 1e-9)) {
        return false;
    }
    const Eigen::Vector2d pixel = scaled_it->second.project(cam);
    const int mu = static_cast<int>(std::floor(pixel.x()));
    const int mv = static_cast<int>(std::floor(pixel.y()));
    if (mu < 0 || mv < 0 || mu >= depth_map.width || mv >= depth_map.height) {
        return false;
    }
    const double depth = static_cast<double>(
        depth_map.depths[static_cast<std::size_t>(mv) * static_cast<std::size_t>(depth_map.width) +
                         static_cast<std::size_t>(mu)]);
    if (!(depth > 0.0)) {
        return false;
    }
    if (std::abs(cam.norm() - depth) / depth >= depth_tolerance) {
        return false;  // occluded here
    }
    if (count_only) {
        return true;
    }
    // Full-resolution pixel: the depth-map pixel covers a ds x ds block.
    const int fx = static_cast<int>(pixel.x() * static_cast<double>(depth_map.downscale));
    const int fy = static_cast<int>(pixel.y() * static_cast<double>(depth_map.downscale));
    if (fx < 0 || fy < 0 || fx >= view.image->width() || fy >= view.image->height()) {
        return false;
    }
    *out_grey = grey(*view.image, fx, fy);
    return true;
}

/// Measure coverage and photo-consistency of `page` against the chunk.
///
/// coverage: of the texels some uv triangle covers, the fraction whose baked
/// grey is above kBlackThreshold. (Unobserved surface is genuinely black, so
/// this mixes "did the baker paint" with "was there anything to paint".)
///
/// seen/covered-seen: the same quantities restricted to texels at least one view
/// sees by the depth test. covered_seen_fraction is the honest "did the baker
/// paint every texel it could" number.
///
/// consistency: for up to kSampleCount random in-chart texels, the texel's 3-D
/// point (barycentric of its face) is projected into every view whose depth map
/// sees it by a relative depth test (|distance - depth| / depth < kDepthTol at
/// the corresponding depth-map pixel). The absolute difference between the baked
/// grey and the mean grey of those samples is collected; the mean, median and
/// p90 are reported. A texel no view sees is skipped.
TextureMetrics measure_texture(const Chunk& chunk, const Image& page,
                               const std::vector<View>& views, int size) {
    TextureMetrics metrics;
    const Mesh& mesh = chunk.model();
    const std::vector<TexelRef> map = build_texel_map(mesh, size);
    for (const TexelRef& ref : map) {
        if (ref.face >= 0) {
            ++metrics.chart_texels;
        }
    }

    // Per-view depth-map lookup keyed by camera id.
    std::unordered_map<std::int32_t, const DepthMap*> map_of;
    for (const DepthMap& depth_map : chunk.depth_maps()) {
        map_of.emplace(depth_map.camera_id, &depth_map);
    }
    // The scaled calibration each depth map's pixels were produced with.
    std::unordered_map<std::int32_t, Calibration> scaled_of;
    for (const DepthMap& depth_map : chunk.depth_maps()) {
        const auto camera = std::find_if(chunk.cameras().begin(), chunk.cameras().end(),
                                         [&](const Camera& c) { return c.id == depth_map.camera_id; });
        if (camera == chunk.cameras().end() || camera->sensor_id < 0) {
            continue;
        }
        scaled_of.emplace(
            depth_map.camera_id,
            scaled_calibration(chunk.sensors()[static_cast<std::size_t>(camera->sensor_id)]
                                   .calibration,
                               depth_map.downscale));
    }

    constexpr double kDepthTolerance = 0.02;
    constexpr std::size_t kSampleCount = 5000;

    std::vector<std::size_t> in_chart;
    in_chart.reserve(metrics.chart_texels);
    std::vector<std::size_t> black_in_chart;  // texel indices, for the buckets below
    for (std::size_t at = 0; at < map.size(); ++at) {
        if (map[at].face >= 0) {
            if (!is_black(page, static_cast<int>(at % static_cast<std::size_t>(size)),
                          static_cast<int>(at / static_cast<std::size_t>(size)))) {
                ++metrics.chart_non_black;
            } else {
                black_in_chart.push_back(at);
            }
            in_chart.push_back(at);
        }
    }
    metrics.coverage =
        metrics.chart_texels == 0
            ? 0.0
            : static_cast<double>(metrics.chart_non_black) / static_cast<double>(metrics.chart_texels);

    if (in_chart.empty()) {
        return metrics;
    }

    // Count, over every in-chart texel, how many a view sees. This distinguishes
    // "the baker left a texel black" from "no camera ever saw that surface".
    for (const std::size_t at : in_chart) {
        const Eigen::Vector3d X = texel_point(mesh, map[at]);
        bool any = false;
        for (const View& view : views) {
            if (view_samples(X, view, map_of, scaled_of, kDepthTolerance, /*count_only=*/true,
                             nullptr)) {
                any = true;
                break;
            }
        }
        if (any) {
            ++metrics.seen_texels;
            const int i = static_cast<int>(at % static_cast<std::size_t>(size));
            const int j = static_cast<int>(at / static_cast<std::size_t>(size));
            if (!is_black(page, i, j)) {
                ++metrics.covered_seen;
            }
        }
    }
    metrics.seen_fraction = static_cast<double>(metrics.seen_texels) /
                            static_cast<double>(metrics.chart_texels);
    metrics.covered_seen_fraction =
        metrics.seen_texels == 0
            ? 0.0
            : static_cast<double>(metrics.covered_seen) / static_cast<double>(metrics.seen_texels);

    // Bucket the black in-chart texels (see the counter comment above): for each,
    // ask every view whether the point lands inside its image and whether that
    // image shows a non-black pixel there.
    for (const std::size_t at : black_in_chart) {
        const Eigen::Vector3d X = texel_point(mesh, map[at]);
        bool inside_any = false;
        bool nonblack_any = false;
        for (const View& view : views) {
            const Eigen::Vector3d cam = view.R * X + view.t;
            if (!(cam.z() > 1e-9)) {
                continue;
            }
            const Eigen::Vector2d pixel = view.full.project(cam);
            const int fx = static_cast<int>(std::floor(pixel.x()));
            const int fy = static_cast<int>(std::floor(pixel.y()));
            if (fx < 0 || fy < 0 || fx >= view.image->width() || fy >= view.image->height()) {
                continue;
            }
            inside_any = true;
            if (grey(*view.image, fx, fy) > kBlackThreshold) {
                nonblack_any = true;
                break;
            }
        }
        if (!inside_any) {
            ++metrics.bb_outside_image;
        } else if (nonblack_any) {
            ++metrics.bb_has_photo;
        } else {
            ++metrics.bb_no_photo;
        }
    }

    // Portable index draw: std::uniform_int_distribution is specified only by
    // its output range, so libc++ and libstdc++ select different texels from the
    // same seed and the sample statistics are not comparable across platforms
    // (the same reason every sampler goes through portable_rng).
    std::mt19937 rng(20260818U);
    std::vector<double> diffs;
    diffs.reserve(kSampleCount);
    for (std::size_t sample = 0; sample < kSampleCount; ++sample) {
        const std::size_t at = in_chart[rng::index_below(rng, in_chart.size())];
        const int i = static_cast<int>(at % static_cast<std::size_t>(size));
        const int j = static_cast<int>(at / static_cast<std::size_t>(size));
        const TexelRef& ref = map[at];
        const Eigen::Vector3d X = texel_point(mesh, ref);
        const double baked = grey(page, i, j);

        double sum = 0.0;
        std::size_t count = 0;
        for (const View& view : views) {
            double sampled = 0.0;
            if (view_samples(X, view, map_of, scaled_of, kDepthTolerance, /*count_only=*/false,
                             &sampled)) {
                sum += sampled;
                ++count;
            }
        }
        if (count == 0) {
            continue;
        }
        diffs.push_back(std::abs(baked - sum / static_cast<double>(count)));
    }
    metrics.consistency_samples = diffs.size();
    if (!diffs.empty()) {
        double total = 0.0;
        for (const double d : diffs) {
            total += d;
        }
        metrics.consistency_mean = total / static_cast<double>(diffs.size());
        std::sort(diffs.begin(), diffs.end());
        metrics.consistency_median = dense_fixture::percentile(diffs, 0.50);
        metrics.consistency_p90 = dense_fixture::percentile(diffs, 0.90);
    }
    return metrics;
}

/// BuildModel on the truth-pose sequence, then BuildUV, then BuildTexture in
/// `blending_mode`, returning the measured page. `tag` labels the printout.
TextureMetrics texture_pass(Document& document, const std::vector<View>& views,
                            const std::string& blending_mode, const char* tag) {
    Params uv = uv_params(/*texture_size=*/2048);
    run_task_named(document, "BuildUV", uv);

    Params texture = texture_params(/*texture_size=*/2048, blending_mode);
    run_task_named(document, "BuildTexture", texture);

    const Chunk& chunk = *document.active_chunk();
    EXPECT_EQ(chunk.model_textures().size(), 1U);
    const Image& page = chunk.model_textures()[0];
    const TextureMetrics metrics =
        measure_texture(chunk, page, views, /*size=*/page.width());
    std::printf("[build_texture %s] coverage=%.4f (seen=%.4f covered_seen=%.4f) "
                "chart_texels=%zu non_black=%zu seen=%zu seen_covered=%zu | "
                "consistency n=%zu mean=%.2f median=%.2f p90=%.2f\n",
                tag, metrics.coverage, metrics.seen_fraction, metrics.covered_seen_fraction,
                metrics.chart_texels, metrics.chart_non_black, metrics.seen_texels,
                metrics.covered_seen, metrics.consistency_samples, metrics.consistency_mean,
                metrics.consistency_median, metrics.consistency_p90);
    std::printf("[build_texture %s] black buckets: outside_image=%zu no_photo=%zu has_photo=%zu "
                "(sum=%zu of %zu black)\n",
                tag, metrics.bb_outside_image, metrics.bb_no_photo, metrics.bb_has_photo,
                metrics.bb_outside_image + metrics.bb_no_photo + metrics.bb_has_photo,
                metrics.chart_texels - metrics.chart_non_black);
    return metrics;
}

}  // namespace

TEST(BuildTextureEndToEnd, BakesTheSceneOnTruthPoses) {
    using clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    const std::vector<TruthImage> truth = parse_truth_images("seq_truth.json");
    ASSERT_EQ(truth.size(), 8U);

    auto t0 = clock::now();
    Document document = aligned_document(seq_files());
    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.cameras().size(), 8U);
    set_truth_poses(document, truth);
    set_truth_calibration(document, kTruthWidth, kTruthHeight, kTruthFocal);
    const std::size_t kept = dense_fixture::retriangulate_tracks(chunk);
    ASSERT_GT(kept, 100U) << "too few tracks triangulated with the truth poses";
    dense_fixture::run_build_depth_maps(document, /*downscale=*/2, "mild");
    const auto t1 = clock::now();
    std::printf("[build_texture e2e] dense setup %.0f ms\n", ms(t0, t1));

    Params model = make_params(builtin_registry().require("BuildModel"));
    model.set("source_data", ParamValue{std::string("depth_maps")});
    model.set("face_count", ParamValue{std::string("custom")});
    model.set("face_count_custom", ParamValue{std::int64_t{20000}});
    model.set("volume_resolution", ParamValue{std::int64_t{128}});
    run_task_named(document, "BuildModel", model);
    const auto t2 = clock::now();
    std::printf("[build_texture e2e] BuildModel %.0f ms, %zu faces\n", ms(t1, t2),
                chunk.model().faces.size());
    ASSERT_FALSE(chunk.model().empty());

    // Decode every enabled camera's photograph once: all three passes share them.
    std::vector<Image> storage;
    const std::vector<View> views = collect_views(document, chunk, storage);
    ASSERT_EQ(views.size(), 8U);

    // One page per blending mode. mosaic, average and disabled all sample the
    // photographs and must agree with them.
    TextureMetrics mosaic = texture_pass(document, views, "mosaic", "mosaic");
    const auto t3 = clock::now();
    std::printf("[build_texture e2e] first UV+bake (incl. UV) %.0f ms\n", ms(t2, t3));
    // Coverage and consistency bounds.
    //
    // consistency is the strong assertion: it compares the baked grey with the
    // mean grey of the views the depth test says see the point, and lands at a
    // mean ~4.4 / median ~2.1 on this fixture.
    //
    // seen_fraction ~0.83: the volume reconstruction's bbox margin extends past
    // the camera arc, so ~17% of in-chart texels lie on surface no camera sees;
    // that surface is genuinely black and cannot be textured.
    //
    // covered_seen_fraction is the honest "did the baker paint every texel it
    // could". It reaches ~0.935: of the seen texels 0.8275 -> covered 0.9347 in
    // mosaic. The residual black texels were bucketed (see the printout and the
    // BuildTextureDiagnostics test): ZERO are outside every image, and of the
    // black texels 88% have no non-black pixel in ANY photograph -- they are
    // surface the cameras only ever recorded as background (a ridge/lip the
    // depth maps captured as geometry but never saw the colour of). The "seen"
    // count over-includes these because the depth test accepts a depth-map pixel
    // even where the photograph there is background, so this is an artifact of
    // the probe, not a baker miss. The remaining ~12% of black texels do have a
    // photograph yet bake black (an occluded best view or a grazing one).
    //
    // So this is NOT a min_view_cos / grazing-store problem: (a)+(b) below-arc
    // and grazing-only faces are under 1.5% of the black texels. The bound is set
    // on the measured value with a small margin.
    EXPECT_GT(mosaic.seen_fraction, 0.80);
    EXPECT_GE(mosaic.covered_seen_fraction, 0.925);
    EXPECT_GT(mosaic.consistency_samples, 1000U);
    EXPECT_LT(mosaic.consistency_mean, 10.0);

    const auto t4 = clock::now();
    const TextureMetrics average = texture_pass(document, views, "average", "average");
    std::printf("[build_texture e2e] average bake %.0f ms\n", ms(t4, clock::now()));
    EXPECT_GE(average.seen_fraction, 0.80);
    EXPECT_GE(average.covered_seen_fraction, 0.925);
    EXPECT_LT(average.consistency_mean, 5.0);  // average mixes views, so it agrees better

    // "disabled" is the same best-view / no-correction path as mosaic, so it must
    // paint (and agree with) the photographs exactly as mosaic does. It used to
    // come back as an all-black 0-coverage control page; that was a bug.
    const TextureMetrics disabled = texture_pass(document, views, "disabled", "disabled");
    EXPECT_GT(disabled.chart_texels, 0U) << "the atlas must still cover texels";
    EXPECT_GE(disabled.seen_fraction, 0.80);
    EXPECT_GE(disabled.covered_seen_fraction, 0.925);
    EXPECT_GT(disabled.consistency_samples, 1000U);
    EXPECT_LT(disabled.consistency_mean, 10.0);
    // Same view assignment and same samples as mosaic (which differs only by the
    // gain solve), so the two pages must be within a small tolerance.
    EXPECT_NEAR(disabled.coverage, mosaic.coverage, 0.02);
    EXPECT_NEAR(disabled.consistency_mean, mosaic.consistency_mean, 1.0);

    // -- export the textured model -------------------------------------------
    TempDir dir("alphashape_build_texture_export");
    const std::string obj_path = dir.file("textured.obj");
    Params export_params = make_params(builtin_registry().require("ExportModel"));
    export_params.set("path", ParamValue{obj_path});
    export_params.set("format", ParamValue{std::string("obj")});
    run_task_named(document, "ExportModel", export_params);
    EXPECT_TRUE(std::filesystem::exists(obj_path));
    EXPECT_TRUE(std::filesystem::exists(dir.file("textured.mtl")));
    EXPECT_TRUE(std::filesystem::exists(dir.file("textured.png")));
}

// ---------------------------------------------------------------------------
// Grounding for the coverage gap: WHY do in-chart texels that a camera sees
// (by the depth test) still come out black? Buckets every black in-chart texel
// by the geometry of its covering face:
//   (a) best available viewing cosine <= 0      -> no camera on the front side
//   (b) 0 < best cosine < min_view_cos          -> grazing only, rejected by the
//       gate (split by whether the face projects inside the image at all, and
//       whether such a grazing view would even pass the depth test)
//   (c) the face has an accepted view but the baker left the texel black.
// The test asserts nothing about the split; it prints it so the e2e bounds can
// be set on measured values. It is one process like the e2e test, for the same
// gtest_discover_tests reason.
// ---------------------------------------------------------------------------
TEST(BuildTextureDiagnostics, GroundsTheSeenTexelCoverageGap) {
    const std::vector<TruthImage> truth = parse_truth_images("seq_truth.json");
    ASSERT_EQ(truth.size(), 8U);
    Document document = aligned_document(seq_files());
    Chunk& chunk = *document.active_chunk();
    set_truth_poses(document, truth);
    set_truth_calibration(document, kTruthWidth, kTruthHeight, kTruthFocal);
    ASSERT_GT(dense_fixture::retriangulate_tracks(chunk), 100U);
    dense_fixture::run_build_depth_maps(document, /*downscale=*/2, "mild");

    Params model = make_params(builtin_registry().require("BuildModel"));
    model.set("source_data", ParamValue{std::string("depth_maps")});
    model.set("face_count", ParamValue{std::string("custom")});
    model.set("face_count_custom", ParamValue{std::int64_t{20000}});
    model.set("volume_resolution", ParamValue{std::int64_t{128}});
    run_task_named(document, "BuildModel", model);

    std::vector<Image> storage;
    const std::vector<View> views = collect_views(document, chunk, storage);
    ASSERT_EQ(views.size(), 8U);

    Params uv = uv_params(/*texture_size=*/2048);
    run_task_named(document, "BuildUV", uv);
    Params texture = texture_params(/*texture_size=*/2048, "mosaic");
    run_task_named(document, "BuildTexture", texture);

    const Mesh& mesh = chunk.model();
    const Image& page = chunk.model_textures()[0];
    const int size = page.width();
    const std::vector<TexelRef> map = build_texel_map(mesh, size);

    // Depth-map lookup, as in measure_texture.
    std::unordered_map<std::int32_t, const DepthMap*> map_of;
    for (const DepthMap& depth_map : chunk.depth_maps()) {
        map_of.emplace(depth_map.camera_id, &depth_map);
    }
    std::unordered_map<std::int32_t, Calibration> scaled_of;
    for (const DepthMap& depth_map : chunk.depth_maps()) {
        const auto camera = std::find_if(chunk.cameras().begin(), chunk.cameras().end(),
                                         [&](const Camera& c) { return c.id == depth_map.camera_id; });
        if (camera == chunk.cameras().end() || camera->sensor_id < 0) {
            continue;
        }
        scaled_of.emplace(
            depth_map.camera_id,
            scaled_calibration(chunk.sensors()[static_cast<std::size_t>(camera->sensor_id)]
                                   .calibration,
                               depth_map.downscale));
    }
    constexpr double kDepthTolerance = 0.02;

    // Black in-chart texels per face.
    std::vector<std::size_t> black_per_face(mesh.faces.size(), 0);
    std::size_t chart_texels = 0;
    std::size_t black_total = 0;
    for (std::size_t at = 0; at < map.size(); ++at) {
        if (map[at].face < 0) {
            continue;
        }
        ++chart_texels;
        const int i = static_cast<int>(at % static_cast<std::size_t>(size));
        const int j = static_cast<int>(at / static_cast<std::size_t>(size));
        if (is_black(page, i, j)) {
            ++black_total;
            ++black_per_face[static_cast<std::size_t>(map[at].face)];
        }
    }

    // Reproduce the task's face-view assignment with the documented defaults.
    std::vector<texturing::TexView> tv;
    tv.reserve(views.size());
    for (const View& v : views) {
        texturing::TexView t;
        t.camera_id = v.camera->id;
        t.image = v.image;
        t.calibration = v.full;
        t.R = v.R;
        t.t = v.t;
        tv.push_back(t);
    }
    texturing::FaceViewOptions fo;  // defaults: downscale 2, cos 0.15, smoothness 1, 6 passes
    const texturing::FaceViews fv = texturing::select_face_views(mesh, tv, fo);
    const double gate = fo.min_view_cos;

    std::vector<Eigen::Vector3d> centres;
    centres.reserve(views.size());
    for (const View& v : views) {
        centres.push_back(-v.R.transpose() * v.t);
    }

    std::size_t faces_black = 0;
    std::size_t a_no_front = 0;      // black texels on faces with best_cos <= 0
    std::size_t b_grazing = 0;       // 0 < best_cos < gate, no accepted view
    std::size_t b_grazing_inside = 0;
    std::size_t b_grazing_depth_ok = 0;
    std::size_t b_grazing_outside = 0;
    std::size_t b_occluded = 0;      // best_cos >= gate but no accepted view (occlusion)
    std::size_t c_has_view = 0;      // an accepted view exists yet texels black
    std::size_t c_seen_depth_ok = 0;
    std::size_t overlap_face = 0;    // faces with a view inside the frame yet none accepted
    std::vector<double> best_cos_black;  // per black face, its best viewing cosine

    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        if (black_per_face[f] == 0) {
            continue;
        }
        ++faces_black;
        const Eigen::Vector3i& tri = mesh.faces[f];
        const Eigen::Vector3d& a = mesh.vertices[static_cast<std::size_t>(tri[0])];
        const Eigen::Vector3d& b = mesh.vertices[static_cast<std::size_t>(tri[1])];
        const Eigen::Vector3d& c = mesh.vertices[static_cast<std::size_t>(tri[2])];
        const Eigen::Vector3d centroid = (a + b + c) / 3.0;
        const Eigen::Vector3d n = (b - a).cross(c - a).normalized();

        double best_cos = -2.0;
        bool grazing_inside = false;
        bool grazing_depth_ok = false;
        bool any_inside = false;
        bool any_inside_depth_ok = false;
        for (std::size_t v = 0; v < views.size(); ++v) {
            const Eigen::Vector3d d = centres[v] - centroid;
            const double dist = d.norm();
            const double cos = dist > 0.0 ? n.dot(d) / dist : -1.0;
            best_cos = std::max(best_cos, cos);
            // Does the centroid project inside the full frame?
            const Eigen::Vector3d cam = views[v].R * centroid + views[v].t;
            bool inside = false;
            if (cam.z() > 1e-9) {
                const Eigen::Vector2d px = views[v].full.project(cam);
                inside = px.x() >= 0.0 && px.x() < static_cast<double>(views[v].image->width()) &&
                         px.y() >= 0.0 && px.y() < static_cast<double>(views[v].image->height());
            }
            if (inside) {
                any_inside = true;
            }
            const bool depth_ok =
                view_samples(centroid, views[v], map_of, scaled_of, kDepthTolerance,
                             /*count_only=*/true, nullptr);
            if (inside && depth_ok) {
                any_inside_depth_ok = true;
            }
            if (cos > 0.0 && cos < gate) {
                if (inside) {
                    grazing_inside = true;
                }
                if (depth_ok) {
                    grazing_depth_ok = true;
                }
            }
        }
        best_cos_black.push_back(best_cos);
        const std::size_t count = black_per_face[f];
        const bool has_accepted = fv.best[f] >= 0;
        if (has_accepted) {
            c_has_view += count;
            if (any_inside_depth_ok) {
                c_seen_depth_ok += count;
            }
        } else if (best_cos <= 0.0) {
            a_no_front += count;
        } else if (best_cos < gate) {  // 0 < best_cos < gate: grazing only
            b_grazing += count;
            if (grazing_inside) {
                b_grazing_inside += count;
            } else {
                b_grazing_outside += count;
            }
            if (grazing_depth_ok) {
                b_grazing_depth_ok += count;
            }
        } else {  // best_cos >= gate but still no view: the occlusion test rejected it
            b_occluded += count;
        }
        if (!has_accepted && any_inside) {
            ++overlap_face;
        }
    }

    std::sort(best_cos_black.begin(), best_cos_black.end());
    auto pct = [&](double p) {
        if (best_cos_black.empty()) {
            return 0.0;
        }
        return dense_fixture::percentile(best_cos_black, p);
    };
    std::printf("[gap diag] chart_texels=%zu black=%zu (%.4f) on %zu faces | "
                "faces with any view inside frame but none accepted=%zu\n",
                chart_texels, black_total,
                chart_texels == 0 ? 0.0
                                  : static_cast<double>(black_total) /
                                        static_cast<double>(chart_texels),
                faces_black, overlap_face);
    std::printf("[gap diag] (a) best_cos<=0 no front view=%zu | (b) 0<best_cos<gate=%zu "
                "(projects inside=%zu, outside=%zu; grazing depth-ok=%zu) | "
                "(b') best_cos>=gate but rejected by occlusion=%zu | "
                "(c) accepted view yet black=%zu (of which test-depth-seen=%zu)\n",
                a_no_front, b_grazing, b_grazing_inside, b_grazing_outside, b_grazing_depth_ok,
                b_occluded, c_has_view, c_seen_depth_ok);
    std::printf("[gap diag] best_cos over black faces: min=%.3f p10=%.3f median=%.3f p90=%.3f "
                "max=%.3f\n",
                best_cos_black.empty() ? 0.0 : best_cos_black.front(), pct(0.10), pct(0.50),
                pct(0.90), best_cos_black.empty() ? 0.0 : best_cos_black.back());

    // ---- per-black-texel: is there a photograph of the point at all? -------
    // Reconstruct every black in-chart texel's surface point and ask, over all
    // views, (i) does it project inside some image and (ii) does some such image
    // show a non-black pixel there. A black texel with no non-black sample is a
    // point no photograph recorded (background in every frame); one that has a
    // non-black sample yet baked black is a genuine baker miss. Also check the
    // face's accepted best view specifically.
    std::size_t proj_inside_none = 0;
    std::size_t proj_inside_any = 0;
    std::size_t sample_nonblack_none = 0;
    std::size_t sample_nonblack_any = 0;
    std::size_t bestview_off = 0;
    std::size_t bestview_inside = 0;
    for (std::size_t at = 0; at < map.size(); ++at) {
        if (map[at].face < 0) {
            continue;
        }
        const int i = static_cast<int>(at % static_cast<std::size_t>(size));
        const int j = static_cast<int>(at / static_cast<std::size_t>(size));
        if (!is_black(page, i, j)) {
            continue;
        }
        const Eigen::Vector3d X = texel_point(mesh, map[at]);
        bool inside_any = false;
        bool nonblack_any = false;
        for (const View& v : views) {
            const Eigen::Vector3d cam = v.R * X + v.t;
            if (!(cam.z() > 1e-9)) {
                continue;
            }
            const Eigen::Vector2d px = v.full.project(cam);
            const int fx = static_cast<int>(std::floor(px.x()));
            const int fy = static_cast<int>(std::floor(px.y()));
            if (fx < 0 || fy < 0 || fx >= v.image->width() || fy >= v.image->height()) {
                continue;
            }
            inside_any = true;
            if (grey(*v.image, fx, fy) > kBlackThreshold) {
                nonblack_any = true;
            }
        }
        if (inside_any) {
            ++proj_inside_any;
        } else {
            ++proj_inside_none;
        }
        if (nonblack_any) {
            ++sample_nonblack_any;
        } else {
            ++sample_nonblack_none;
        }

        const int bv = fv.best[static_cast<std::size_t>(map[at].face)];
        if (bv >= 0) {
            const View& v = views[static_cast<std::size_t>(bv)];
            const Eigen::Vector3d cam = v.R * X + v.t;
            const Eigen::Vector2d px = v.full.project(cam);
            if (cam.z() > 1e-9 && px.x() >= 0.0 && px.y() >= 0.0 &&
                px.x() < static_cast<double>(v.image->width()) &&
                px.y() < static_cast<double>(v.image->height())) {
                ++bestview_inside;
            } else {
                ++bestview_off;
            }
        }
    }
    std::printf("[gap diag] black texels: projects inside no image=%zu (some image=%zu) | "
                "no non-black sample in any image=%zu (has one=%zu)\n",
                proj_inside_none, proj_inside_any, sample_nonblack_none, sample_nonblack_any);
    std::printf("[gap diag] black texels on a face with an accepted best view: off-image/behind=%zu "
                "inside=%zu\n",
                bestview_off, bestview_inside);
}

}  // namespace alphashape
