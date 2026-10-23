#include "common/algorithm/match/match_photos_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"
#include "common/algorithm/match/hamming_matcher.hpp"
#include "common/algorithm/match/preselection.hpp"
#include "common/algorithm/match/tracks.hpp"
#include "common/algorithm/sfm/essential.hpp"
#include "common/core/frame.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

// ---------------------------------------------------------------------------
// Constants that are OUR engineering choices. They are gathered here so the tuning surface is one block.
// ---------------------------------------------------------------------------

/// Minimum number of putative matches a pair must have before RANSAC is even
/// attempted, and the minimum number of RANSAC inliers to keep the pair.
///
/// The first use also bounds RANSAC's cost: a pair that does not clear this
/// floor is rejected before estimate_essential_ransac runs, so the expensive
/// geometric verification only ever sees pairs that already look promising.
constexpr std::size_t kMinPairInliers = 15;

/// Epipolar inlier threshold in pixels. It is converted, per pair, to a
/// symmetric angular threshold on the unit sphere by
///     threshold_rad = kEpipolarThresholdPx / sqrt(f_a * f_b)
/// which is the small-angle form of theta ~= r / f for both views.
constexpr double kEpipolarThresholdPx = 2.0;

/// Deterministic per-pair RANSAC seed: a splitmix64 mix of the two camera ids,
/// so the sampled hypothesis set does not depend on thread scheduling. The
/// result of a pair is therefore reproducible run to run.
std::uint64_t pair_seed(std::int32_t camera_a, std::int32_t camera_b) {
    std::uint64_t x = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(camera_a)) << 32) |
                      static_cast<std::uint64_t>(static_cast<std::uint32_t>(camera_b));
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// Sort PairMatches by (camera_a, camera_b), the order the chunk and the
/// serialiser require.
void sort_pairs(std::vector<PairMatches>& pairs) {
    std::sort(pairs.begin(), pairs.end(),
              [](const PairMatches& x, const PairMatches& y) {
                  if (x.camera_a != y.camera_a) return x.camera_a < y.camera_a;
                  return x.camera_b < y.camera_b;
              });
}

/// Sort one pair's matches by index_a (the stored invariant).
void sort_pairs_matches(std::vector<KeypointMatch>& matches) {
    std::sort(matches.begin(), matches.end(),
              [](const KeypointMatch& x, const KeypointMatch& y) {
                  return x.index_a < y.index_a;
              });
}

}  // namespace

void MatchPhotosTask::declare_params(Params& params) const {
    // Names mirror Metashape's public matchPhotos() Python API.
    params.declare({"keypoint_limit", ParamType::Int, std::int64_t{40000},
                    "Maximum keypoints per image when detecting missing keypoints. 0 = no limit."});
    params.declare({"keypoint_limit_per_mpx", ParamType::Int, std::int64_t{1000},
                    "Maximum keypoints per megapixel when detecting. 0 = no limit."});
    params.declare({"tiepoint_limit", ParamType::Int, std::int64_t{4000},
                    "Maximum tie points per camera, longest tracks first. 0 = unlimited."});
    params.declare({"generic_preselection", ParamType::Bool, true,
                    "Prematch pairs and keep each image's strongest partners."});
    params.declare({"reference_preselection", ParamType::Bool, true,
                    "Choose pairs from camera references (GPS) or capture order."});
    params.declare({"reference_preselection_mode", ParamType::String, std::string{"source"},
                    "'source' = GPS neighbours, 'sequential' = capture order."});
    params.declare({"reference_preselection_neighbors", ParamType::Int, std::int64_t{20},
                    "Neighbours per camera for reference preselection (ours: 20). Must be >= 1."});
    params.declare({"reset_matches", ParamType::Bool, false,
                    "Discard existing matches and recompute everything from scratch."});
}

std::vector<Stage> MatchPhotosTask::stages() const {
    return {Stage{"detect", 1}, Stage{"preselect", 0}, Stage{"match", 1}, Stage{"tracks", 0}};
}

namespace {

/// One thread's worth of work items, so a batch keeps all cores busy while a
/// cancellation is honoured at batch granularity (see DetectFeatures).
std::ptrdiff_t batch_size() {
#if defined(ALPHASHAPE_HAVE_OPENMP)
    return std::max<std::ptrdiff_t>(1, static_cast<std::ptrdiff_t>(omp_get_max_threads()));
#else
    return 1;
#endif
}

/// Pack a normalised (a < b) pair into a 64-bit key for set membership.
std::uint64_t pair_key(std::int32_t a, std::int32_t b) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
}

}  // namespace

void MatchPhotosTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- validate every parameter before touching the chunk ------------------
    const std::int64_t keypoint_limit = params.get_int("keypoint_limit");
    const std::int64_t keypoint_limit_per_mpx = params.get_int("keypoint_limit_per_mpx");
    const std::int64_t tiepoint_limit = params.get_int("tiepoint_limit");
    const bool generic_preselection = params.get_bool("generic_preselection");
    const bool reference_preselection = params.get_bool("reference_preselection");
    const std::string mode = params.get_string("reference_preselection_mode");
    const std::int64_t neighbors = params.get_int("reference_preselection_neighbors");
    const bool reset_matches = params.get_bool("reset_matches");

    if (keypoint_limit < 0 || keypoint_limit_per_mpx < 0) {
        throw ValueError("keypoint limits must be non-negative");
    }
    if (tiepoint_limit < 0) {
        throw ValueError("tiepoint_limit must be non-negative");
    }
    if (mode != "source" && mode != "sequential") {
        throw ValueError("reference_preselection_mode '" + mode +
                         "' is not supported (only 'source' or 'sequential')");
    }
    if (neighbors < 1) {
        throw ValueError("reference_preselection_neighbors must be >= 1");
    }

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("MatchPhotos needs an active chunk");
    }

    // -- cameras: every camera that has a frame linked to it -----------------
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk->frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());  // first frame wins
        }
    }
    std::vector<std::int32_t> camera_ids;
    std::unordered_map<std::int32_t, const Sensor*> sensor_of;
    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk->cameras()) {
        if (frame_of.find(camera.id) == frame_of.end()) {
            continue;  // no frame: not part of this match run
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("camera " + std::to_string(camera.id) + " has no sensor");
        }
        const Sensor& sensor = chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)];
        // A non-positive focal length makes the epipolar threshold undefined.
        if (!(sensor.calibration.f > 0.0)) {
            throw ValueError("camera " + std::to_string(camera.id) +
                             " has a non-positive focal length");
        }
        camera_ids.push_back(camera.id);
        sensor_of[camera.id] = &sensor;
        camera_by_id[camera.id] = &camera;
    }
    std::sort(camera_ids.begin(), camera_ids.end());  // camera-id order

    // -- detect: reuse the chunk's keypoint sets; detect only what is missing -
    // keypoint_sets starts as a copy of the chunk's sets, so a failed or
    // cancelled run leaves the chunk untouched and a commit is a single move.
    std::vector<KeypointSet> keypoint_sets = chunk->tie_points().keypoint_sets;
    std::unordered_set<std::int32_t> have_set;
    for (const KeypointSet& set : keypoint_sets) {
        have_set.insert(set.camera_id);
    }
    std::vector<std::int32_t> detect_ids;
    for (const std::int32_t id : camera_ids) {
        if (have_set.find(id) == have_set.end()) detect_ids.push_back(id);
    }

    progress.update("detect", 0.0);
    if (!detect_ids.empty()) {
        const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(detect_ids.size());
        std::vector<std::optional<KeypointSet>> results(detect_ids.size());
        std::vector<std::string> errors(detect_ids.size());
        const std::ptrdiff_t batch = batch_size();
        std::ptrdiff_t done = 0;
        // Workers touch no callback: progress/cancelled are user callbacks that
        // may throw, and nothing may unwind out of an OpenMP region.
        while (done < total) {
            const std::ptrdiff_t end = std::min(total, done + batch);
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (std::ptrdiff_t i = done; i < end; ++i) {
                const std::int32_t id = detect_ids[static_cast<std::size_t>(i)];
                try {
                    results[static_cast<std::size_t>(i)] = detect_keypoints_for_frame(
                        document, *frame_of.at(id), keypoint_limit, keypoint_limit_per_mpx);
                } catch (const std::exception& e) {
                    errors[static_cast<std::size_t>(i)] =
                        "camera " + std::to_string(id) + ": " + e.what();
                }
            }
            done = end;
            progress.update("detect", static_cast<double>(done) / static_cast<double>(total));
            if (progress.is_cancelled()) {
                throw CancelledError("MatchPhotos was cancelled");
            }
        }
        for (const std::string& error : errors) {
            if (!error.empty()) throw Error(error);
        }
        for (auto& result : results) keypoint_sets.push_back(std::move(*result));
    }
    progress.update("detect", 1.0);

    std::unordered_map<std::int32_t, const KeypointSet*> set_of;
    for (const KeypointSet& set : keypoint_sets) {
        set_of[set.camera_id] = &set;
    }

    // -- preselect candidate pairs ------------------------------------------
    progress.update("preselect", 0.0);
    std::vector<match::CameraPair> candidates;
    if (reference_preselection) {
        if (mode == "sequential") {
            candidates = match::sequential_pairs(camera_ids, static_cast<int>(neighbors));
        } else {  // "source": GPS neighbours of the referenced cameras
            std::vector<std::pair<std::int32_t, Eigen::Vector3d>> locations;
            std::vector<std::int32_t> unreferenced;
            for (const std::int32_t id : camera_ids) {
                const Camera* camera = camera_by_id.at(id);
                if (camera->reference.location.has_value()) {
                    locations.emplace_back(id, *camera->reference.location);
                } else {
                    unreferenced.push_back(id);
                }
            }
            // With no referenced camera this reduces to exhaustive (every
            // unreferenced camera is paired with every other).
            candidates = match::reference_pairs(locations, unreferenced,
                                                static_cast<int>(neighbors));
        }
    } else {
        candidates = match::exhaustive_pairs(camera_ids);
    }

    if (generic_preselection) {
        candidates = match::generic_pairs(keypoint_sets, candidates, match::GenericOptions{});
    }

    // Unless resetting, drop pairs that already have matches: their existing
    // inliers stay valid because MatchPhotos never re-detects a keypoint set
    // that is already present.
    if (!reset_matches) {
        std::unordered_set<std::uint64_t> existing;
        for (const PairMatches& pair : chunk->tie_points().matches) {
            existing.insert(pair_key(pair.camera_a, pair.camera_b));
        }
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&existing](const match::CameraPair& pair) {
                                            return existing.count(pair_key(pair.a, pair.b)) != 0;
                                        }),
                         candidates.end());
    }
    progress.update("preselect", 1.0);

    // -- match each candidate pair, geometrically verified with RANSAC --------
    progress.update("match", 0.0);
    const std::size_t pair_count = candidates.size();
    std::vector<std::optional<PairMatches>> matched(pair_count);
    std::vector<std::string> match_errors(pair_count);

    if (pair_count > 0) {
        const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(pair_count);
        const std::ptrdiff_t batch = batch_size();
        std::ptrdiff_t done = 0;
        while (done < total) {
            const std::ptrdiff_t end = std::min(total, done + batch);
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (std::ptrdiff_t i = done; i < end; ++i) {
                const std::size_t index = static_cast<std::size_t>(i);
                const match::CameraPair& pair = candidates[index];
                try {
                    const KeypointSet& set_a = *set_of.at(pair.a);
                    const KeypointSet& set_b = *set_of.at(pair.b);

                    // RANSAC cost is bounded here: a pair with too few putative
                    // matches is rejected before estimate_essential_ransac runs.
                    const std::vector<KeypointMatch> putative = match::match_descriptors(
                        set_a.descriptors, set_b.descriptors, match::MatchOptions{});
                    if (putative.size() < kMinPairInliers) {
                        continue;  // matched[index] stays nullopt
                    }

                    const Calibration& cal_a = sensor_of.at(pair.a)->calibration;
                    const Calibration& cal_b = sensor_of.at(pair.b)->calibration;
                    const double f_a = cal_a.f;
                    const double f_b = cal_b.f;

                    std::vector<Eigen::Vector3d> b_a;
                    std::vector<Eigen::Vector3d> b_b;
                    b_a.reserve(putative.size());
                    b_b.reserve(putative.size());
                    for (const KeypointMatch& m : putative) {
                        b_a.push_back(cal_a.unproject(keypoint_to_pixel(set_a.keypoints[m.index_a])));
                        b_b.push_back(cal_b.unproject(keypoint_to_pixel(set_b.keypoints[m.index_b])));
                    }

                    sfm::RansacOptions options;
                    options.threshold_rad = kEpipolarThresholdPx / std::sqrt(f_a * f_b);
                    options.seed = pair_seed(pair.a, pair.b);
                    const sfm::EssentialResult essential =
                        sfm::estimate_essential_ransac(b_a, b_b, options);
                    if (!essential.success || essential.inlier_count < kMinPairInliers) {
                        continue;
                    }

                    PairMatches result;
                    result.camera_a = pair.a;
                    result.camera_b = pair.b;
                    for (std::size_t k = 0; k < putative.size(); ++k) {
                        if (essential.inliers[k] != 0) {
                            result.matches.push_back(putative[k]);
                        }
                    }
                    sort_pairs_matches(result.matches);  // sorted by index_a
                    matched[index] = std::move(result);
                } catch (const std::exception& e) {
                    match_errors[index] = "camera pair (" + std::to_string(pair.a) + ", " +
                                          std::to_string(pair.b) + "): " + e.what();
                }
            }
            done = end;
            progress.update("match", static_cast<double>(done) / static_cast<double>(total));
            if (progress.is_cancelled()) {
                throw CancelledError("MatchPhotos was cancelled");
            }
        }
        for (const std::string& error : match_errors) {
            if (!error.empty()) throw Error(error);
        }
    }
    progress.update("match", 1.0);

    // -- merge with existing pairs (or replace) ------------------------------
    std::vector<PairMatches> all_matches;
    if (!reset_matches) {
        all_matches = chunk->tie_points().matches;
    }
    for (auto& result : matched) {
        if (result.has_value()) {
            all_matches.push_back(std::move(*result));
        }
    }
    sort_pairs(all_matches);

    // -- tracks --------------------------------------------------------------
    progress.update("tracks", 0.0);
    std::vector<Track> tracks = match::build_tracks(all_matches);
    tracks = match::limit_tracks(tracks, static_cast<std::size_t>(tiepoint_limit));
    progress.update("tracks", 1.0);

    // -- commit atomically: every stage succeeded, so the chunk may change ----
    TiePoints& tie_points = chunk->tie_points();
    tie_points.keypoint_sets = std::move(keypoint_sets);
    tie_points.matches = std::move(all_matches);
    tie_points.tracks = std::move(tracks);
}

}  // namespace alphashape
