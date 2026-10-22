// test_match_photos.cpp -- end-to-end tests for the MatchPhotos task.
//
// These are the real check of the pixel convention: they run AddPhotos and
// MatchPhotos on the three rendered scene fixtures, then compare the relative
// pose recovered from the STORED matches against the ground truth in
// data/scene_truth.json. If keypoint_to_pixel's +0.5 shift or the camera axes
// were wrong, the recovered rotation or translation direction would drift.
//
// The fixtures and their truth file are produced by data/make_scene_fixtures.py;
// the truth file has a fixed structure for which a tiny hand parser is enough.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"
#include "common/algorithm/sfm/essential.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "image/io/image_io.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

/// Our threshold is 2.0 px; on the square 500 px focal length fixture this is
/// 2/500 rad. The test uses the same value so it re-verifies the stored inliers.
double scene_threshold_rad() { return 2.0 / 500.0; }

// ---------------------------------------------------------------------------
// Minimal parser for data/scene_truth.json's "pairs" array. Each object holds
// i, j (ints) then R (9 doubles, row-major) then t (3 doubles), in that key
// order, so per object there are exactly 14 numbers. No nested braces occur, so
// an object ends at the first '}' after its '{'.
// ---------------------------------------------------------------------------
struct TruthPair {
    std::int32_t i = -1;
    std::int32_t j = -1;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::vector<double> numbers_in(const std::string& text) {
    std::vector<double> numbers;
    std::size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '-' || (c >= '0' && c <= '9')) {
            char* end = nullptr;
            const double value = std::strtod(text.c_str() + i, &end);
            if (end != text.c_str() + i) {
                numbers.push_back(value);
                i = static_cast<std::size_t>(end - text.c_str());
                continue;
            }
        }
        ++i;
    }
    return numbers;
}

std::vector<TruthPair> parse_truth_pairs() {
    const std::string json = read_text_file(fixture("scene_truth.json"));
    const std::size_t pairs_at = json.find("\"pairs\"");
    if (pairs_at == std::string::npos) {
        throw std::runtime_error("scene_truth.json has no 'pairs' array");
    }
    std::vector<TruthPair> pairs;
    std::size_t pos = json.find('[', pairs_at);
    while (pos != std::string::npos) {
        const std::size_t open = json.find('{', pos);
        if (open == std::string::npos) break;
        const std::size_t close = json.find('}', open);
        if (close == std::string::npos) break;
        const std::vector<double> n = numbers_in(json.substr(open, close - open));
        if (n.size() != 14) {
            throw std::runtime_error("unexpected number count in a truth pair");
        }
        TruthPair pair;
        pair.i = static_cast<std::int32_t>(std::lround(n[0]));
        pair.j = static_cast<std::int32_t>(std::lround(n[1]));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                pair.R(r, c) = n[2 + static_cast<std::size_t>(r * 3 + c)];
            }
        }
        pair.t = Eigen::Vector3d(n[11], n[12], n[13]);
        pairs.push_back(pair);
        pos = json.find('{', close);
    }
    return pairs;
}

/// Rotation angle of R in degrees.
double rotation_error_deg(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    const double tr = (a.transpose() * b).trace();
    return std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) * 0.5))) * 180.0 / 3.14159265358979323846;
}

/// Angle between the two (unit) translation directions, in degrees.
double direction_error_deg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const double d = std::abs(a.normalized().dot(b.normalized()));
    return std::acos(std::max(-1.0, std::min(1.0, d))) * 180.0 / 3.14159265358979323846;
}

const std::vector<std::string>& scene_files() {
    static const std::vector<std::string> files = {
        fixture("scene_0.jpg"), fixture("scene_1.jpg"), fixture("scene_2.jpg")};
    return files;
}

/// A fresh document with the three scene photos imported through AddPhotos.
Document make_scene_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{scene_files()});
    run_task(builtin_registry(), document, "AddPhotos", params);
    return document;
}

/// Run MatchPhotos with the fixture-appropriate keypoint cap. Extra key/value
/// overrides are applied on top of the defaults.
void run_match_photos(Document& document, std::int64_t keypoint_limit_per_mpx = 10000) {
    Params params = make_params(builtin_registry().require("MatchPhotos"));
    params.set("keypoint_limit_per_mpx", keypoint_limit_per_mpx);
    run_task(builtin_registry(), document, "MatchPhotos", params);
}

const PairMatches* find_pair(const Chunk& chunk, std::int32_t a, std::int32_t b) {
    for (const PairMatches& pair : chunk.tie_points().matches) {
        if (pair.camera_a == a && pair.camera_b == b) return &pair;
    }
    return nullptr;
}

const KeypointSet* set_for(const Chunk& chunk, std::int32_t camera_id) {
    return chunk.tie_points().find_by_camera(camera_id);
}

/// The calibration of the sensor a camera uses.
const Calibration& calibration_for(const Chunk& chunk, std::int32_t camera_id) {
    for (const Camera& camera : chunk.cameras()) {
        if (camera.id == camera_id) {
            return chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        }
    }
    throw std::runtime_error("no camera " + std::to_string(camera_id));
}

/// Rebuild the bearings of one stored pair from the chunk's keypoint sets and
/// run the same RANSAC the task ran, returning the recovered pose.
sfm::EssentialResult recover_pose(const Chunk& chunk, const PairMatches& pair) {
    const KeypointSet& set_a = *set_for(chunk, pair.camera_a);
    const KeypointSet& set_b = *set_for(chunk, pair.camera_b);
    const Calibration& cal_a = calibration_for(chunk, pair.camera_a);
    const Calibration& cal_b = calibration_for(chunk, pair.camera_b);

    std::vector<Eigen::Vector3d> b_a;
    std::vector<Eigen::Vector3d> b_b;
    for (const KeypointMatch& m : pair.matches) {
        b_a.push_back(cal_a.unproject(keypoint_to_pixel(set_a.keypoints[m.index_a])));
        b_b.push_back(cal_b.unproject(keypoint_to_pixel(set_b.keypoints[m.index_b])));
    }
    sfm::RansacOptions options;
    options.threshold_rad = scene_threshold_rad();
    return sfm::estimate_essential_ransac(b_a, b_b, options);
}

/// A scratch directory removed on destruction.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& tag) {
        path = fs::temp_directory_path() /
               (tag + "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path, ec);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string file(const std::string& name) const { return (path / name).string(); }
};

TEST(MatchPhotos, ProducesAllThreePairsWithEnoughInliers) {
    Document document = make_scene_document();
    run_match_photos(document);
    const Chunk& chunk = *document.active_chunk();

    ASSERT_EQ(chunk.tie_points().matches.size(), 3U);
    const std::vector<std::pair<std::int32_t, std::int32_t>> expected = {{0, 1}, {0, 2}, {1, 2}};
    for (const auto& [a, b] : expected) {
        const PairMatches* pair = find_pair(chunk, a, b);
        ASSERT_NE(pair, nullptr) << "missing pair " << a << "," << b;
        EXPECT_GE(pair->matches.size(), 50U) << "pair " << a << "," << b << " has too few inliers";
        for (std::size_t k = 1; k < pair->matches.size(); ++k) {
            EXPECT_LT(pair->matches[k - 1].index_a, pair->matches[k].index_a)
                << "pair matches are not sorted by index_a";
        }
    }
}

TEST(MatchPhotos, RecoveredPoseMatchesGroundTruth) {
    Document document = make_scene_document();
    run_match_photos(document);
    const Chunk& chunk = *document.active_chunk();

    const std::vector<TruthPair> truth = parse_truth_pairs();
    ASSERT_EQ(truth.size(), 3U);

    for (const TruthPair& gt : truth) {
        const PairMatches* pair = find_pair(chunk, gt.i, gt.j);
        ASSERT_NE(pair, nullptr) << "missing pair " << gt.i << "," << gt.j;
        const sfm::EssentialResult res = recover_pose(chunk, *pair);
        ASSERT_TRUE(res.success) << "RANSAC failed on pair " << gt.i << "," << gt.j;
        const double rot = rotation_error_deg(res.pose.R, gt.R);
        const double dir = direction_error_deg(res.pose.t, gt.t);
        EXPECT_LT(rot, 1.0) << "pair " << gt.i << "," << gt.j << " rotation error " << rot << " deg";
        EXPECT_LT(dir, 3.0) << "pair " << gt.i << "," << gt.j << " translation error " << dir
                            << " deg";
    }
}

TEST(MatchPhotos, TracksAreConsistentWithThreeViewCoverage) {
    Document document = make_scene_document();
    run_match_photos(document);
    const Chunk& chunk = *document.active_chunk();
    const auto& tracks = chunk.tie_points().tracks;
    ASSERT_FALSE(tracks.empty());

    std::size_t three_view = 0;
    for (const Track& track : tracks) {
        EXPECT_GE(track.observations.size(), 2U);
        for (std::size_t k = 0; k < track.observations.size(); ++k) {
            const Observation& obs = track.observations[k];
            const KeypointSet* set = set_for(chunk, obs.camera_id);
            ASSERT_NE(set, nullptr) << "track observes a camera with no keypoint set";
            EXPECT_LT(obs.keypoint_index, set->keypoints.size());
            if (k > 0) {
                EXPECT_LT(track.observations[k - 1].camera_id, obs.camera_id)
                    << "track observations are not sorted by camera";
            }
        }
        if (track.observations.size() == 3) {
            ++three_view;
        }
        // Every observation must be corroborated by a stored pair match with
        // some other observation of the same track.
        for (std::size_t k = 0; k < track.observations.size(); ++k) {
            bool linked = false;
            for (std::size_t l = 0; l < track.observations.size() && !linked; ++l) {
                if (l == k) continue;
                const Observation& o1 = track.observations[k];
                const Observation& o2 = track.observations[l];
                const std::int32_t lo = std::min(o1.camera_id, o2.camera_id);
                const std::int32_t hi = std::max(o1.camera_id, o2.camera_id);
                const PairMatches* pair = find_pair(chunk, lo, hi);
                if (pair == nullptr) continue;
                const std::uint32_t ia =
                    o1.camera_id == lo ? o1.keypoint_index : o2.keypoint_index;
                const std::uint32_t ib =
                    o1.camera_id == lo ? o2.keypoint_index : o1.keypoint_index;
                for (const KeypointMatch& m : pair->matches) {
                    if (m.index_a == ia && m.index_b == ib) {
                        linked = true;
                        break;
                    }
                }
            }
            EXPECT_TRUE(linked) << "a track observation is not backed by any stored pair match";
        }
    }
    EXPECT_GE(three_view, 20U) << "only " << three_view << " tracks have three observations";
}

TEST(MatchPhotos, IsDeterministicAcrossRuns) {
    Document first = make_scene_document();
    run_match_photos(first);
    Document second = make_scene_document();
    run_match_photos(second);
    EXPECT_TRUE(first.active_chunk()->tie_points() == second.active_chunk()->tie_points());
}

// The eight-image sequence exercises many candidate pairs and a wide spread of
// RANSAC inlier counts, which is where the JD/refit path in
// fit_essential_linear used to read uninitialised memory (an A sized to the
// input but filled only up to the inlier row count). That made each pair's
// inlier set depend on the process, so running the whole pipeline under one
// thread and then eight threads must now agree exactly.
TEST(MatchPhotos, SequenceIsDeterministicAcrossThreadCounts) {
#if defined(ALPHASHAPE_HAVE_OPENMP)
    const std::vector<std::string> files = {
        fixture("seq_0.jpg"), fixture("seq_1.jpg"), fixture("seq_2.jpg"), fixture("seq_3.jpg"),
        fixture("seq_4.jpg"), fixture("seq_5.jpg"), fixture("seq_6.jpg"), fixture("seq_7.jpg")};
    const auto run = [&files]() {
        Document document;
        Chunk& chunk = document.workspace().add_chunk("Chunk 1");
        document.set_active_chunk_id(chunk.id());
        Params add = make_params(builtin_registry().require("AddPhotos"));
        add.set("filenames", ParamValue{files});
        run_task(builtin_registry(), document, "AddPhotos", add);
        run_match_photos(document, 10000);
        return document.active_chunk()->tie_points();
    };

    omp_set_num_threads(1);
    const TiePoints single = run();
    omp_set_num_threads(8);
    const TiePoints multi = run();
    omp_set_num_threads(omp_get_max_threads());

    EXPECT_FALSE(single.matches.empty());
    EXPECT_TRUE(single == multi) << "tie points differ between 1-thread and 8-thread runs";
#else
    GTEST_SKIP() << "OpenMP not available";
#endif
}

TEST(MatchPhotos, ResetMatchesControlsReuseAndRecomputation) {
    Document document = make_scene_document();
    run_match_photos(document);
    const TiePoints first = document.active_chunk()->tie_points();

    // reset_matches=false with no new pairs: nothing changes.
    Params again = make_params(builtin_registry().require("MatchPhotos"));
    again.set("keypoint_limit_per_mpx", std::int64_t{10000});
    run_task(builtin_registry(), document, "MatchPhotos", again);
    EXPECT_TRUE(document.active_chunk()->tie_points() == first);

    // reset_matches=true recomputes from scratch: the same result as the first run.
    Params reset = make_params(builtin_registry().require("MatchPhotos"));
    reset.set("keypoint_limit_per_mpx", std::int64_t{10000});
    reset.set("reset_matches", true);
    run_task(builtin_registry(), document, "MatchPhotos", reset);
    EXPECT_TRUE(document.active_chunk()->tie_points() == first);
}

TEST(MatchPhotos, InvalidParamsThrowAndLeaveTheChunkUnchanged) {
    Document document = make_scene_document();
    const Chunk before = *document.active_chunk();

    Params bad_mode = make_params(builtin_registry().require("MatchPhotos"));
    bad_mode.set("reference_preselection_mode", std::string("bogus"));
    EXPECT_THROW(run_task(builtin_registry(), document, "MatchPhotos", bad_mode), ValueError);
    EXPECT_TRUE(*document.active_chunk() == before);

    Params bad_neighbors = make_params(builtin_registry().require("MatchPhotos"));
    bad_neighbors.set("reference_preselection_neighbors", std::int64_t{0});
    EXPECT_THROW(run_task(builtin_registry(), document, "MatchPhotos", bad_neighbors), ValueError);
    EXPECT_TRUE(*document.active_chunk() == before);
}

TEST(MatchPhotos, ProjectRoundTripsAfterMatching) {
    Document document = make_scene_document();
    run_match_photos(document);
    const Chunk before = *document.active_chunk();

    TempDir dir("alphashape_match_photos_roundtrip");
    const std::string project = dir.file("proj.psx");
    document.set_path(project);
    ProjectIO::save(document, project);
    auto reloaded = ProjectIO::load(project);

    ASSERT_NE(reloaded->active_chunk(), nullptr);
    EXPECT_TRUE(*reloaded->active_chunk() == before);
}

TEST(MatchPhotos, RecordsAllEightParamsInMeta) {
    Document document = make_scene_document();
    run_match_photos(document);
    const Meta& meta = document.active_chunk()->meta();
    const std::vector<std::string> keys = {
        "keypoint_limit",          "keypoint_limit_per_mpx",         "tiepoint_limit",
        "generic_preselection",    "reference_preselection",         "reference_preselection_mode",
        "reference_preselection_neighbors", "reset_matches"};
    for (const std::string& key : keys) {
        EXPECT_NE(meta.find("MatchPhotos/" + key), nullptr) << "missing MatchPhotos/" << key;
    }
}

}  // namespace
}  // namespace alphashape


namespace alphashape {
namespace {

// Re-running detection replaces keypoint sets, which invalidates every stored
// keypoint index. DetectFeatures must drop the stale matches and tracks rather
// than leave them pointing at the wrong keypoints.
TEST(MatchPhotos, RedetectingFeaturesDropsStaleMatchesAndTracks) {
    Document document = make_scene_document();
    run_match_photos(document);
    ASSERT_FALSE(document.active_chunk()->tie_points().matches.empty());

    Params detect = make_params(builtin_registry().require("DetectFeatures"));
    detect.set("keypoint_limit_per_mpx", std::int64_t{5000});
    run_task(builtin_registry(), document, "DetectFeatures", detect);
    EXPECT_TRUE(document.active_chunk()->tie_points().matches.empty());
    EXPECT_TRUE(document.active_chunk()->tie_points().tracks.empty());

    run_match_photos(document);
    EXPECT_EQ(document.active_chunk()->tie_points().matches.size(), 3U);
}

}  // namespace
}  // namespace alphashape
