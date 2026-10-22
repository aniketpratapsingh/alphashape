// test_build_model.cpp -- tests for the BuildModel meshing task (milestone 7).
//
// Unit level: parameter validation (source_data / surface_type / interpolation /
// face_count / face_count_custom / volume_resolution), the validation gates
// (no depth maps, no point cloud, point cloud without normals) and a project
// save/load round-trip after BuildModel.
//
// End to end: the same truth-pose setup test_build_dense.cpp uses (see
// dense_fixture.hpp) -- AddPhotos + MatchPhotos + AlignCameras build the tracks,
// then the camera poses are overwritten with data/seq_truth.json, the track
// points are re-triangulated and the sensor is set to the true pinhole
// calibration. BuildDepthMaps + BuildPointCloud then run, and BuildModel is
// measured against the analytic ground (z = 0) and wall (y = 0) planes directly
// in the truth frame.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/mesh/build_model_task.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "dense_fixture.hpp"
#include "model/mesh/mesh_ops.hpp"

namespace alphashape {
namespace {

using dense_fixture::aligned_document;
using dense_fixture::fixture;
using dense_fixture::parse_truth_images;
using dense_fixture::percentile;
using dense_fixture::retriangulate_tracks;
using dense_fixture::seq_files;
using dense_fixture::set_truth_calibration;
using dense_fixture::set_truth_poses;
using dense_fixture::TempDir;
using dense_fixture::TruthImage;

// -- task runner -------------------------------------------------------------

/// Build a BuildModel parameter set from the registry schema with overrides.
Params build_model_params(const std::string& source_data = "depth_maps",
                          const std::string& surface_type = "arbitrary",
                          const std::string& interpolation = "enabled",
                          const std::string& face_count = "medium",
                          std::int64_t face_count_custom = 200000,
                          bool vertex_colors = true, bool vertex_confidence = true,
                          std::int64_t volume_resolution = 128) {
    Params params = make_params(builtin_registry().require("BuildModel"));
    params.set("source_data", ParamValue{source_data});
    params.set("surface_type", ParamValue{surface_type});
    params.set("interpolation", ParamValue{interpolation});
    params.set("face_count", ParamValue{face_count});
    params.set("face_count_custom", ParamValue{face_count_custom});
    params.set("vertex_colors", ParamValue{vertex_colors});
    params.set("vertex_confidence", ParamValue{vertex_confidence});
    params.set("volume_resolution", ParamValue{volume_resolution});
    return params;
}

void run_build_model(Document& document, Params& params) {
    run_task(builtin_registry(), document, "BuildModel", params);
}

/// Depth maps at downscale 2 + point cloud, matching test_build_dense's dense
/// path (normals are needed for the point_cloud source).
void run_dense_stage(Document& document, int downscale = 2) {
    dense_fixture::run_build_depth_maps(document, downscale, "mild");
    dense_fixture::run_build_point_cloud(document);
}

/// A filtered point cloud on top of an already-built set of depth maps. The
/// depth maps are NOT rebuilt: the filter (min_confidence) is applied at fusion
/// time, so running BuildPointCloud with a minimum confidence on the shared
/// fixture's maps is exactly what run_dense_stage_filtered used to do — minus
/// the redundant second BuildDepthMaps pass, which dominated the point-cloud
/// scenario's runtime.
///
/// min_confidence drops the low-support depth outliers screened Poisson would
/// otherwise fold into the surface. At 0 the raw cloud carries ~33k low-support
/// samples Poisson integrates into behind-the-wall floaters (see the report);
/// the point_cloud accuracy scenario uses 6 to isolate Poisson's own accuracy.
void run_filtered_point_cloud(Document& document, double min_confidence) {
    Params pc = make_params(builtin_registry().require("BuildPointCloud"));
    pc.set("point_colors", ParamValue{true});
    pc.set("point_confidence", ParamValue{true});
    pc.set("min_confidence", ParamValue{min_confidence});
    run_task(builtin_registry(), document, "BuildPointCloud", pc);
}

/// Nearest of the two truth planes for a point: distance plus which plane.
double plane_distance(const Eigen::Vector3d& p) {
    return std::min(std::abs(p.z()), std::abs(p.y()));
}

struct DistanceStats {
    std::size_t n = 0;
    double median = 0.0, p90 = 0.0, p99 = 0.0;
};

DistanceStats vertex_plane_stats(const Mesh& mesh) {
    std::vector<double> distances;
    distances.reserve(mesh.vertices.size());
    for (const Eigen::Vector3d& v : mesh.vertices) {
        distances.push_back(plane_distance(v));
    }
    DistanceStats stats;
    stats.n = distances.size();
    std::sort(distances.begin(), distances.end());
    stats.median = percentile(distances, 0.50);
    stats.p90 = percentile(distances, 0.90);
    stats.p99 = percentile(distances, 0.99);
    return stats;
}

/// Number of connected components before remove_small_components: not exposed
/// by the mesh API, so count faces in one component via analyse_topology on a
/// scratch copy (the production stages always run the cleanup).
MeshTopology topology_of(const Mesh& mesh) { return analyse_topology(mesh); }

}  // namespace

// ---------------------------------------------------------------------------
// 1. Parameter validation
// ---------------------------------------------------------------------------

TEST(BuildModelParams, RejectsUnknownSourceData) {
    Document document = aligned_document(seq_files(4));
    Params params = build_model_params("bogus");
    const Chunk before = *document.active_chunk();
    EXPECT_THROW(run_build_model(document, params), ValueError);
    EXPECT_TRUE(*document.active_chunk() == before);
}

TEST(BuildModelParams, RejectsHeightFieldSurface) {
    Document document = aligned_document(seq_files(4));
    Params params = build_model_params("depth_maps", "height_field");
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

TEST(BuildModelParams, RejectsUnknownSurfaceType) {
    Document document = aligned_document(seq_files(4));
    Params params = build_model_params("depth_maps", "bogus");
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

TEST(BuildModelParams, RejectsUnknownInterpolation) {
    Document document = aligned_document(seq_files(4));
    Params params = build_model_params("depth_maps", "arbitrary", "bogus");
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

TEST(BuildModelParams, RejectsUnknownFaceCount) {
    Document document = aligned_document(seq_files(4));
    Params params = build_model_params("depth_maps", "arbitrary", "enabled", "bogus");
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

TEST(BuildModelParams, RejectsZeroFaceCountCustom) {
    Document document = aligned_document(seq_files(4));
    Params params = build_model_params("depth_maps", "arbitrary", "enabled", "custom", 0);
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

TEST(BuildModelParams, RejectsOutOfRangeVolumeResolution) {
    Document document = aligned_document(seq_files(4));
    for (const std::int64_t resolution : {std::int64_t{8}, std::int64_t{1024}}) {
        Params params = build_model_params("depth_maps", "arbitrary", "enabled", "medium", 200000,
                                           true, true, resolution);
        EXPECT_THROW(run_build_model(document, params), ValueError) << "resolution " << resolution;
    }
}

// ---------------------------------------------------------------------------
// 2. Validation gates
// ---------------------------------------------------------------------------

TEST(BuildModelValidate, DepthMapsSourceNeedsDepthMaps) {
    Document document = aligned_document(seq_files(4));
    ASSERT_TRUE(document.active_chunk()->depth_maps().empty());
    const Chunk before = *document.active_chunk();
    Params params = build_model_params("depth_maps");
    EXPECT_THROW(run_build_model(document, params), ValueError);
    EXPECT_TRUE(*document.active_chunk() == before);
    EXPECT_TRUE(document.active_chunk()->model().empty());
}

TEST(BuildModelValidate, PointCloudSourceNeedsPointCloud) {
    Document document = aligned_document(seq_files(4));
    ASSERT_TRUE(document.active_chunk()->point_cloud().points.empty());
    Params params = build_model_params("point_cloud");
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

TEST(BuildModelValidate, PointCloudSourceRejectsCloudWithoutNormals) {
    // A cloud with points but no normals cannot seed screened Poisson.
    Document document = aligned_document(seq_files(4));
    PointCloud& cloud = document.active_chunk()->point_cloud();
    cloud.points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
    cloud.points.push_back(Eigen::Vector3d(1.0, 1.0, 1.0));
    // No normals: the size mismatch is exactly what the task rejects.
    Params params = build_model_params("point_cloud");
    EXPECT_THROW(run_build_model(document, params), ValueError);
}

// ---------------------------------------------------------------------------
// 3. Round-trip after BuildModel
// ---------------------------------------------------------------------------

TEST(BuildModelRoundTrip, RoundTripsAfterBuildModel) {
    Document document = aligned_document(seq_files(4));
    run_dense_stage(document, /*downscale=*/4);
    Params params = build_model_params("depth_maps", "arbitrary", "enabled", "custom", 2000, true,
                                       true, 32);
    run_build_model(document, params);
    const Chunk before = *document.active_chunk();
    ASSERT_FALSE(before.model().empty());

    TempDir dir("alphashape_build_model_roundtrip");
    const std::string project = dir.file("proj.psx");
    document.set_path(project);
    ProjectIO::save(document, project);
    auto reloaded = ProjectIO::load(project);

    ASSERT_NE(reloaded->active_chunk(), nullptr);
    EXPECT_TRUE(*reloaded->active_chunk() == before);
}

// ---------------------------------------------------------------------------
// 4. End to end on the analytic sequence fixture, on the TRUE poses.
//
// Reuses the truth-pose setup test_build_dense.cpp uses (dense_fixture.hpp):
// AddPhotos + MatchPhotos + AlignCameras build the tracks (only their
// connectivity is reused), then every camera pose is OVERWRITTEN with the truth
// pose, the track points are re-triangulated and the sensor is set to the true
// pinhole calibration (f = 500, no distortion). The dense path then runs exactly
// as in production, and the mesh is measured against the two analytic planes
// directly in the truth frame -- no similarity fit at all.
//
// RUN-TIME NOTE. These scenarios used to be four separate TESTs. gtest_discover_
// tests runs each case in its own process, so each paid the full (~14 s)
// BuildDepthMaps pass, and the point-cloud scenario paid it twice; together that
// was ~85 s of the suite. The scenarios are merged into ONE test here so the
// expensive truth-pose dense setup runs once and every scenario reuses it. All
// assertions and bounds are carried over unchanged (scenario 2 now reuses the
// shared depth maps for its filtered fusion rather than recomputing them, which
// cannot change its result because the min_confidence filter is applied at
// fusion time, not at depth estimation).
// ---------------------------------------------------------------------------

namespace {

constexpr int kTruthWidth = 480;
constexpr int kTruthHeight = 360;
constexpr double kTruthFocal = 500.0;

/// Everything the e2e measurement needs, once per (source_data, run).
struct MeshingMeasurement {
    std::size_t faces = 0;
    std::size_t vertices = 0;
    DistanceStats stats_absolute;
    DistanceStats stats_voxels;
    std::size_t nonmanifold_edges = 0;
    std::size_t components = 0;
    double voxel = 0.0;
    bool has_colours = false;
    bool has_confidence = false;
    double confidence_coverage = 0.0;  // fraction of vertices with confidence >= 1
    double red_stddev = 0.0;
    double floater_fraction = 0.0;     // fraction of vertices > 3 voxels from the planes
};

/// Run the dense + BuildModel pipeline on the true poses and measure the mesh.
MeshingMeasurement run_and_measure(Document& document, Params& params, const char* tag,
                                   const std::string& source_data) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    run_build_model(document, params);
    const auto t1 = clock::now();

    const Chunk& chunk = *document.active_chunk();
    const Mesh& mesh = chunk.model();
    MeshingMeasurement m;
    m.faces = mesh.faces.size();
    m.vertices = mesh.vertices.size();
    const MeshTopology topo = topology_of(mesh);
    m.nonmanifold_edges = topo.nonmanifold_edges;
    m.components = topo.components;

    // The grid voxel size is the longest bbox edge over volume_resolution. The
    // task and this test derive the same box from the same helper, so the voxel
    // unit quoted here is exactly the grid the mesh was extracted from.
    const std::pair<Eigen::Vector3d, Eigen::Vector3d> box = build_model_bbox(chunk, source_data);
    const double longest = (box.second - box.first).maxCoeff();
    const std::int64_t resolution = params.get_int("volume_resolution");
    m.voxel = longest / static_cast<double>(resolution);

    m.stats_absolute = vertex_plane_stats(mesh);
    {
        std::vector<double> distances;
        distances.reserve(mesh.vertices.size());
        for (const Eigen::Vector3d& v : mesh.vertices) {
            distances.push_back(plane_distance(v) / m.voxel);
        }
        std::sort(distances.begin(), distances.end());
        m.stats_voxels.n = distances.size();
        m.stats_voxels.median = percentile(distances, 0.50);
        m.stats_voxels.p90 = percentile(distances, 0.90);
        m.stats_voxels.p99 = percentile(distances, 0.99);
        std::size_t floaters = 0;
        for (const double d : distances) {
            if (d > 3.0) {
                ++floaters;
            }
        }
        m.floater_fraction =
            distances.empty() ? 0.0
                              : static_cast<double>(floaters) / static_cast<double>(distances.size());
    }

    // -- colours -------------------------------------------------------------
    m.has_colours = mesh.colours.size() == mesh.vertices.size() * 3;
    m.has_confidence = mesh.confidence.size() == mesh.vertices.size();
    if (m.has_confidence && !mesh.confidence.empty()) {
        std::size_t supported = 0;
        for (const float c : mesh.confidence) {
            if (c >= 1.0F) {
                ++supported;
            }
        }
        m.confidence_coverage =
            static_cast<double>(supported) / static_cast<double>(mesh.confidence.size());
    }
    if (m.has_colours && !mesh.colours.empty()) {
        double mean = 0.0;
        for (std::size_t i = 0; i < mesh.colours.size(); i += 3) {
            mean += static_cast<double>(mesh.colours[i]);
        }
        mean /= static_cast<double>(mesh.colours.size() / 3);
        double variance = 0.0;
        for (std::size_t i = 0; i < mesh.colours.size(); i += 3) {
            const double delta = static_cast<double>(mesh.colours[i]) - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(mesh.colours.size() / 3);
        m.red_stddev = std::sqrt(variance);
    }

    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("[build_model %s] faces=%zu vertices=%zu voxel=%.4f build=%.0f ms\n", tag,
                m.faces, m.vertices, m.voxel, ms(t0, t1));
    std::printf("[build_model %s] distance median=%.4f p90=%.4f p99=%.4f (abs) | "
                "median=%.3f p90=%.3f p99=%.3f (voxels)\n",
                tag, m.stats_absolute.median, m.stats_absolute.p90, m.stats_absolute.p99,
                m.stats_voxels.median, m.stats_voxels.p90, m.stats_voxels.p99);
    std::printf("[build_model %s] components=%zu non-manifold edges=%zu "
                "colours=%d confidence=%d coverage=%.1f%% red_stddev=%.1f floaters=%.2f%%\n",
                tag, m.components, m.nonmanifold_edges, m.has_colours ? 1 : 0,
                m.has_confidence ? 1 : 0, 100.0 * m.confidence_coverage, m.red_stddev,
                100.0 * m.floater_fraction);
    return m;
}

/// A document with the truth poses and the depth maps, but no fused cloud yet.
/// Each end-to-end scenario runs the BuildPointCloud variant it needs on top
/// (the filtered cloud for the point_cloud source, the plain one otherwise), so
/// the expensive BuildDepthMaps pass happens exactly once per process.
Document truth_pose_depth_document(const std::vector<TruthImage>& truth, int downscale = 2) {
    Document document = aligned_document(seq_files());
    Chunk& chunk = *document.active_chunk();
    EXPECT_EQ(chunk.cameras().size(), 8U);
    set_truth_poses(document, truth);
    set_truth_calibration(document, kTruthWidth, kTruthHeight, kTruthFocal);
    const std::size_t kept = retriangulate_tracks(chunk);
    EXPECT_GT(kept, 100U) << "too few tracks triangulated with the truth poses";
    dense_fixture::run_build_depth_maps(document, downscale, "mild");
    return document;
}

/// Run BuildModel with a progress callback that times each stage, and print the
/// per-stage breakdown.
std::size_t timed_build_model(Document& document, Params& params, const char* tag) {
    using clock = std::chrono::steady_clock;
    std::vector<std::pair<std::string, double>> stage_times;
    std::string current;
    auto last = clock::now();
    Progress progress;
    progress.report = [&](const std::string& stage, double fraction) {
        if (stage != current) {
            if (!current.empty()) {
                stage_times.emplace_back(
                    current,
                    std::chrono::duration<double, std::milli>(clock::now() - last).count());
            }
            current = stage;
            last = clock::now();
        }
        if (fraction >= 1.0 && current == stage) {
            stage_times.emplace_back(
                stage, std::chrono::duration<double, std::milli>(clock::now() - last).count());
            current.clear();
            last = clock::now();
        }
    };
    run_task(builtin_registry(), document, "BuildModel", params, progress);

    std::printf("[build_model %s] stages:", tag);
    double total = 0.0;
    for (const auto& [stage, ms] : stage_times) {
        std::printf(" %s %.0f ms", stage.c_str(), ms);
        total += ms;
    }
    std::printf(" | total %.0f ms\n", total);
    return document.active_chunk()->model().faces.size();
}

}  // namespace

TEST(BuildModelEndToEnd, MatchesTheSceneOnTruthPoses) {
    const std::vector<TruthImage> truth = parse_truth_images("seq_truth.json");
    ASSERT_EQ(truth.size(), 8U);
    // ONE dense setup for all four scenarios: the truth-pose depth maps are the
    // expensive part, and every scenario only reads them plus the chunk's
    // triangulated tracks. Each BuildModel run below replaces chunk.model() in
    // place, so the depth maps stay valid across scenarios.
    Document document = truth_pose_depth_document(truth, /*downscale=*/2);

    // -- scenario 1: depth_maps source accuracy ------------------------------
    // "disabled" is the honest accuracy test: no extrapolation into unmeasured
    // space. The "enabled" mask deliberately bridges small gaps and adds
    // unsupported surface (see the interpolation scenario), which would inflate
    // the percentile distance here. "custom" with a target above the extracted
    // face count means no decimation, so the measurement is of the extracted
    // surface.
    {
        Params params = build_model_params("depth_maps", "arbitrary", "disabled", "custom", 1000000,
                                           true, true, 192);
        const MeshingMeasurement m = run_and_measure(document, params, "depth_maps", "depth_maps");

        EXPECT_GT(m.faces, 0U);
        EXPECT_LT(m.stats_voxels.median, 0.5) << "median " << m.stats_voxels.median << " voxels";
        EXPECT_LT(m.stats_voxels.p90, 1.5) << "p90 " << m.stats_voxels.p90 << " voxels";
        EXPECT_EQ(m.nonmanifold_edges, 0U);

        // Colours: present, and not a constant image.
        EXPECT_TRUE(m.has_colours);
        EXPECT_TRUE(m.has_confidence);
        EXPECT_GT(m.confidence_coverage, 0.90);
        EXPECT_GT(m.red_stddev, 5.0);
    }

    // -- scenario 2: point_cloud source accuracy -----------------------------
    // min_confidence 6 drops the low-support depth samples before Poisson; the
    // raw cloud's outliers otherwise dominate the tail (see the report). Fusion
    // runs on the shared depth maps -- the filter is applied at fusion time, so
    // this is exactly the cloud run_dense_stage_filtered would have produced.
    {
        run_filtered_point_cloud(document, /*min_confidence=*/6.0);
        Params params = build_model_params("point_cloud", "arbitrary", "disabled", "custom", 1000000,
                                           true, true, 128);
        const MeshingMeasurement m =
            run_and_measure(document, params, "point_cloud", "point_cloud");

        EXPECT_GT(m.faces, 0U);
        EXPECT_LT(m.stats_voxels.median, 0.5) << "median " << m.stats_voxels.median << " voxels";
        EXPECT_LT(m.stats_voxels.p90, 1.5) << "p90 " << m.stats_voxels.p90 << " voxels";
        EXPECT_EQ(m.nonmanifold_edges, 0U);

        // Colours come from the depth maps even for the point_cloud source.
        EXPECT_TRUE(m.has_colours);
        EXPECT_TRUE(m.has_confidence);
        EXPECT_GT(m.confidence_coverage, 0.90);
        EXPECT_GT(m.red_stddev, 5.0);
    }

    // -- scenario 3: face_count presets are ordered --------------------------
    {
        Params low = build_model_params("depth_maps", "arbitrary", "enabled", "low", 200000, true,
                                        true, 96);
        Params medium = build_model_params("depth_maps", "arbitrary", "enabled", "medium", 200000,
                                           true, true, 96);
        Params high = build_model_params("depth_maps", "arbitrary", "enabled", "high", 200000, true,
                                         true, 96);
        const std::size_t faces_low = timed_build_model(document, low, "low");
        const std::size_t faces_medium = timed_build_model(document, medium, "medium");
        const std::size_t faces_high = timed_build_model(document, high, "high");
        std::printf("[build_model presets] low=%zu medium=%zu high=%zu\n", faces_low, faces_medium,
                    faces_high);

        EXPECT_LT(faces_low, faces_medium);
        EXPECT_LT(faces_medium, faces_high);

        // custom = 5000 -> at most 5000 faces, and the decimator gets close to it.
        Params custom = build_model_params("depth_maps", "arbitrary", "enabled", "custom", 5000,
                                           true, true, 96);
        const std::size_t faces_custom = timed_build_model(document, custom, "custom5000");
        std::printf("[build_model custom] target=5000 actual=%zu\n", faces_custom);
        EXPECT_LE(faces_custom, 5000U);
        EXPECT_GE(faces_custom, 4500U);
    }

    // -- scenario 4: interpolation is monotone -------------------------------
    {
        Params disabled = build_model_params("depth_maps", "arbitrary", "disabled", "medium", 200000,
                                             true, true, 96);
        Params enabled = build_model_params("depth_maps", "arbitrary", "enabled", "medium", 200000,
                                            true, true, 96);
        Params extrapolated = build_model_params("depth_maps", "arbitrary", "extrapolated", "medium",
                                                 200000, true, true, 96);
        const std::size_t faces_disabled = timed_build_model(document, disabled, "disabled");
        const std::size_t faces_enabled = timed_build_model(document, enabled, "enabled");
        const std::size_t faces_extrapolated =
            timed_build_model(document, extrapolated, "extrapolated");
        std::printf("[build_model interpolation] disabled=%zu enabled=%zu extrapolated=%zu\n",
                    faces_disabled, faces_enabled, faces_extrapolated);

        EXPECT_LE(faces_disabled, faces_enabled);
        EXPECT_LE(faces_enabled, faces_extrapolated);
    }
}

}  // namespace alphashape
