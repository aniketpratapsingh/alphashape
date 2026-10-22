// structure_builder.cpp -- incremental structure-from-motion.
//
// Technique (clean-room, from the literature only):
//   * N. Snavely, S. Seitz, R. Szeliski, "Photo Tourism: Exploring Photo
//     Collections in 3D" (SIGGRAPH 2006): pick an initial image pair, seed the
//     reconstruction with its relative pose and the triangulated shared points,
//     then repeatedly resect the next-best camera, triangulate the tracks it
//     extends, and refine with bundle adjustment.
//   * J. L. Schonberger & J.-M. Frahm, "Structure-from-Motion Revisited"
//     (CVPR 2016): the same loop stated modernly -- rank initial pairs and
//     require real parallax, register by number of visible triangulated points,
//     and filter observations by reprojection error and parallax between rounds.
//
// Pipeline (steps 0-4 of the milestone spec):
//   0. Precompute one bearing per keypoint with Calibration::unproject.
//   1. If >= 2 cameras carry an initial_pose, continue: adopt those poses and
//      the tracks' initial points, then go straight to registration.
//   2. Otherwise choose the initial pair: rank verified pairs by match count,
//      keep the top `init_pair_candidates`, and for each run a two-view
//      essential-matrix RANSAC, triangulate its inliers and measure the median
//      parallax. Prefer the parallax-qualifying pair with the most inliers,
//      else the pair with the largest median parallax. Camera a becomes the
//      world origin (identity); camera b gets the relative pose.
//   3. Register cameras: resect the camera that sees the most triangulated
//      points, triangulate every newly-eligible track, run a global BA on the
//      documented schedule, then clear points with a bad observation.
//   4. Final global BA, final filter, report the RMS.
//
// Determinism. The same input yields the same output: Ceres runs with
// num_threads = 1 (TODO: revisit once throughput matters more than bit-stable
// results) and every iteration over a std container walks it in index order.
//
// Engineering constants, all OURS (none copied from any reference code): the
// thresholds and schedules live in BuilderOptions (structure_builder.hpp) with
// the defaults documented there. Local to this file:
//   * kBehindCameraErrorPx: the reprojection error charged when a Frame camera
//     sees a point at z <= 0 (the model cannot project it); large enough to be
//     filtered, matching the convention bundle_adjustment.cpp uses.
//   * kDegToRad / kRadToDeg: plain pi/180 conversions.

#include "common/algorithm/sfm/structure_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "common/algorithm/sfm/essential.hpp"
#include "common/algorithm/sfm/triangulation.hpp"

namespace alphashape::sfm {
namespace {

constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);
constexpr double kPi = 3.14159265358979323846;
constexpr double kBehindCameraErrorPx = 1.0e9;

double deg2rad(double d) { return d * kPi / 180.0; }

/// The working state of one build. Inputs are borrowed; everything else is
/// owned. Camera and track indices are positions in the input vectors.
struct Scene {
    const BuilderInput* in = nullptr;
    const BuilderOptions* opt = nullptr;

    std::size_t ncams = 0;
    std::size_t ntracks = 0;

    std::vector<std::int32_t> camera_id;                        ///< cam -> id
    std::vector<std::vector<Eigen::Vector3d>> bearings;         ///< cam -> kp bearings
    std::vector<std::vector<std::size_t>> cam_tracks;           ///< cam -> track indices
    std::vector<std::vector<std::pair<std::size_t, std::uint32_t>>> track_obs;  ///< track -> (cam, kp)

    std::vector<std::uint8_t> registered;
    std::vector<std::uint8_t> failed;  ///< failed this round
    std::vector<AbsolutePose> pose;
    std::vector<std::optional<Eigen::Vector3d>> point;

    std::size_t index_a = kNoIndex, index_b = kNoIndex;  ///< gauge cameras
};

// --- helpers (definitions below) ---
std::map<std::int32_t, std::size_t> validate_input(const BuilderInput& in);
void init_scene(Scene& s, const std::map<std::int32_t, std::size_t>& id_to_index);
std::size_t registered_count(const Scene& s);
double camera_f(const Scene& s, std::size_t cam);
double reprojection_error_px(const Scene& s, std::size_t cam, std::uint32_t kp,
                             const Eigen::Vector3d& x);
void triangulate_track(Scene& s, std::size_t ti);
void triangulate_new_tracks(Scene& s);
void filter_points(Scene& s);
std::uint64_t mix_seed(std::uint64_t base, std::int32_t id_a, std::int32_t id_b);

struct BAIndex {
    BAProblem problem;
    std::vector<std::size_t> cam_ba;    ///< cam index -> ba index (kNoIndex if unregistered)
    std::vector<std::size_t> track_ba;  ///< track index -> ba index (kNoIndex if no point)
    std::vector<std::size_t> ba_to_cam;
    std::vector<std::size_t> ba_to_track;
};
BAIndex build_ba(Scene& s);
void run_global_ba(Scene& s);
bool choose_initial_pair(Scene& s);
std::size_t resection_point_count(const Scene& s, std::size_t cam);
bool resect_camera(Scene& s, std::size_t cam);

// --- input validation and indexing ---

/// Map each camera id to its index in BuilderInput::cameras. Throws ValueError
/// on a duplicate or invalid id.
std::map<std::int32_t, std::size_t> validate_input(const BuilderInput& in) {
    std::map<std::int32_t, std::size_t> id_to_index;
    for (std::size_t i = 0; i < in.cameras.size(); ++i) {
        const std::int32_t id = in.cameras[i].id;
        if (id < 0) {
            throw ValueError("structure_builder: camera has a negative id");
        }
        if (!id_to_index.emplace(id, i).second) {
            throw ValueError("structure_builder: duplicate camera id '" + std::to_string(id) + "'");
        }
    }
    auto require_camera = [&](std::int32_t id) -> std::size_t {
        const auto it = id_to_index.find(id);
        if (it == id_to_index.end()) {
            throw ValueError("structure_builder: unknown camera id '" + std::to_string(id) + "'");
        }
        return it->second;
    };
    auto require_keypoint = [&](std::size_t cam_index, std::uint32_t kp) {
        if (static_cast<std::size_t>(kp) >= in.cameras[cam_index].pixels.size()) {
            throw ValueError("structure_builder: keypoint index " + std::to_string(kp) +
                             " out of range in camera '" +
                             std::to_string(in.cameras[cam_index].id) + "'");
        }
    };

    for (const PairMatches& pair : in.matches) {
        const std::size_t ia = require_camera(pair.camera_a);
        const std::size_t ib = require_camera(pair.camera_b);
        if (pair.camera_a >= pair.camera_b) {
            throw ValueError("structure_builder: pair matches require camera_a < camera_b");
        }
        for (const KeypointMatch& m : pair.matches) {
            require_keypoint(ia, m.index_a);
            require_keypoint(ib, m.index_b);
        }
    }
    for (const Track& track : in.tracks) {
        if (track.observations.size() < 2) {
            throw ValueError("structure_builder: a track has fewer than two observations");
        }
        for (const Observation& obs : track.observations) {
            const std::size_t cam_index = require_camera(obs.camera_id);
            require_keypoint(cam_index, obs.keypoint_index);
        }
    }
    return id_to_index;
}

/// Fill the Scene's index structures and bearings from validated input.
void init_scene(Scene& s, const std::map<std::int32_t, std::size_t>& id_to_index) {
    const BuilderInput& in = *s.in;
    s.ncams = in.cameras.size();
    s.ntracks = in.tracks.size();

    s.camera_id.resize(s.ncams);
    s.bearings.resize(s.ncams);
    s.cam_tracks.resize(s.ncams);
    for (std::size_t c = 0; c < s.ncams; ++c) {
        const BuilderCamera& cam = in.cameras[c];
        s.camera_id[c] = cam.id;
        s.bearings[c].reserve(cam.pixels.size());
        for (const Eigen::Vector2d& px : cam.pixels) {
            s.bearings[c].push_back(cam.calibration.unproject(px));
        }
    }

    s.track_obs.resize(s.ntracks);
    for (std::size_t t = 0; t < s.ntracks; ++t) {
        for (const Observation& obs : in.tracks[t].observations) {
            const std::size_t c = id_to_index.at(obs.camera_id);
            s.track_obs[t].emplace_back(c, obs.keypoint_index);
            s.cam_tracks[c].push_back(t);
        }
    }
    // Deterministic iteration: sort each camera's track list by track index.
    for (std::vector<std::size_t>& list : s.cam_tracks) {
        std::sort(list.begin(), list.end());
    }

    s.registered.assign(s.ncams, 0);
    s.failed.assign(s.ncams, 0);
    s.pose.assign(s.ncams, AbsolutePose{});
    s.point.assign(s.ntracks, std::nullopt);
    for (std::size_t t = 0; t < s.ntracks; ++t) {
        s.point[t] = in.tracks[t].point;  // initial point if continuing
    }
}

std::size_t registered_count(const Scene& s) {
    std::size_t n = 0;
    for (std::uint8_t r : s.registered) {
        n += r != 0 ? 1 : 0;
    }
    return n;
}

double camera_f(const Scene& s, std::size_t cam) {
    const double f = s.in->cameras[cam].calibration.f;
    return f > 0.0 ? f : 1.0;
}

/// Reprojection error in pixels of point x as seen by camera cam at keypoint kp.
double reprojection_error_px(const Scene& s, std::size_t cam, std::uint32_t kp,
                             const Eigen::Vector3d& x) {
    const Calibration& cal = s.in->cameras[cam].calibration;
    const Eigen::Vector3d p_cam = s.pose[cam].R * x + s.pose[cam].t;
    if (cal.type == CameraType::Frame && !(p_cam.z() > 0.0)) {
        return kBehindCameraErrorPx;  // behind the image plane: unprojectable
    }
    const Eigen::Vector2d px = cal.project(p_cam);
    return (px - s.in->cameras[cam].pixels[kp]).norm();
}

std::uint64_t mix_seed(std::uint64_t base, std::int32_t id_a, std::int32_t id_b) {
    std::uint64_t h = base;
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(id_a)) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(id_b)) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    return h;
}

// --- triangulation ---

/// Triangulate track ti from its registered observations (>= 2). Writes
/// s.point[ti]; leaves it nullopt on failure. Uses min parallax from the
/// options and the max reprojection angle = max_reprojection_px / f (f = the
/// first registered observing camera's focal length).
void triangulate_track(Scene& s, std::size_t ti) {
    std::vector<AbsolutePose> poses;
    std::vector<Eigen::Vector3d> obs_bearings;
    double f = 0.0;
    for (const auto& [cam, kp] : s.track_obs[ti]) {
        if (s.registered[cam] == 0) {
            continue;
        }
        if (f == 0.0) {
            f = camera_f(s, cam);
        }
        poses.push_back(s.pose[cam]);
        obs_bearings.push_back(s.bearings[cam][kp]);
    }
    if (poses.size() < 2) {
        return;
    }
    TriangulationOptions options;
    options.min_parallax_rad = deg2rad(s.opt->triangulation_min_parallax_deg);
    options.max_angle_error_rad = s.opt->max_reprojection_px / f;
    s.point[ti] = triangulate_multiview(poses, obs_bearings, options);
}

/// Triangulate every track with >= 2 registered observations and no point yet.
void triangulate_new_tracks(Scene& s) {
    for (std::size_t ti = 0; ti < s.ntracks; ++ti) {
        if (s.point[ti].has_value()) {
            continue;
        }
        std::size_t registered_obs = 0;
        for (const auto& [cam, kp] : s.track_obs[ti]) {
            (void)kp;
            // Both arms are size_t: g++ rejects the int 1/0 mix with
            // -Wsign-conversion where clang accepts the usual arithmetic.
            registered_obs += s.registered[cam] != 0 ? std::size_t{1} : std::size_t{0};
        }
        if (registered_obs >= 2) {
            triangulate_track(s, ti);
        }
    }
}

/// Drop any point that has a registered observation with reprojection error >
/// max_reprojection_px, or whose max parallax among registered cameras is below
/// the triangulation minimum. Clearing makes it eligible for retriangulation.
void filter_points(Scene& s) {
    const double min_parallax = deg2rad(s.opt->triangulation_min_parallax_deg);
    for (std::size_t ti = 0; ti < s.ntracks; ++ti) {
        if (!s.point[ti].has_value()) {
            continue;
        }
        const Eigen::Vector3d x = *s.point[ti];
        bool drop = false;
        std::vector<AbsolutePose> poses;
        for (const auto& [cam, kp] : s.track_obs[ti]) {
            if (s.registered[cam] == 0) {
                continue;
            }
            poses.push_back(s.pose[cam]);
            if (reprojection_error_px(s, cam, kp, x) > s.opt->max_reprojection_px) {
                drop = true;
            }
        }
        if (!drop && max_parallax(poses, x) < min_parallax) {
            drop = true;
        }
        if (drop) {
            s.point[ti] = std::nullopt;
        }
    }
}

// --- bundle adjustment ---

/// Populate a BAProblem from the registered cameras and triangulated tracks.
/// Index maps let the caller write results back.
BAIndex build_ba(Scene& s) {
    BAIndex idx;
    idx.cam_ba.assign(s.ncams, kNoIndex);
    idx.track_ba.assign(s.ntracks, kNoIndex);

    for (std::size_t c = 0; c < s.ncams; ++c) {
        if (s.registered[c] == 0) {
            continue;
        }
        idx.cam_ba[c] = idx.problem.poses.size();
        idx.ba_to_cam.push_back(c);
        idx.problem.calibrations.push_back(s.in->cameras[c].calibration);
        BAPose pose;
        pose.R = s.pose[c].R;
        pose.t = s.pose[c].t;
        idx.problem.poses.push_back(pose);
    }
    for (std::size_t t = 0; t < s.ntracks; ++t) {
        if (!s.point[t].has_value()) {
            continue;
        }
        idx.track_ba[t] = idx.problem.points.size();
        idx.ba_to_track.push_back(t);
        idx.problem.points.push_back(*s.point[t]);
    }
    for (std::size_t t = 0; t < s.ntracks; ++t) {
        const std::size_t p = idx.track_ba[t];
        if (p == kNoIndex) {
            continue;
        }
        for (const auto& [cam, kp] : s.track_obs[t]) {
            const std::size_t c = idx.cam_ba[cam];
            if (c == kNoIndex) {
                continue;
            }
            BAObservation obs;
            obs.camera = c;
            obs.point = p;
            obs.pixel = s.in->cameras[cam].pixels[kp];
            idx.problem.observations.push_back(obs);
        }
    }

    // Gauge: the first initial camera (index_a) fixed (6 DoF), with a second
    // registered camera pinning the scale on choose_scale_anchor_axis's axis.
    idx.problem.fixed_pose.assign(idx.problem.poses.size(), 0);
    if (s.index_a != kNoIndex && idx.cam_ba[s.index_a] != kNoIndex) {
        idx.problem.fixed_pose[idx.cam_ba[s.index_a]] = 1;
        // The scale anchor is the second gauge camera (index_b: the second
        // camera of the initial pair, or the second seeded camera when
        // continuing) when it is registered. It is the natural partner of the
        // fixed camera -- the pair the gauge was actually built from -- whereas
        // the first registered camera by index may have a near-zero baseline to
        // index_a, which makes the anchored axis ill-conditioned. Fall back to
        // scanning for the first registered camera when index_b is unavailable.
        std::size_t anchor_cam = kNoIndex;
        if (s.index_b != kNoIndex && s.index_b != s.index_a && s.registered[s.index_b] != 0 &&
            idx.cam_ba[s.index_b] != kNoIndex) {
            anchor_cam = s.index_b;
        } else {
            for (std::size_t c = 0; c < s.ncams; ++c) {
                if (c != s.index_a && s.registered[c] != 0 && idx.cam_ba[c] != kNoIndex) {
                    anchor_cam = c;
                    break;
                }
            }
        }
        if (anchor_cam != kNoIndex) {
            const BAPose pose0 = idx.problem.poses[idx.cam_ba[s.index_a]];
            const BAPose pose1 = idx.problem.poses[idx.cam_ba[anchor_cam]];
            idx.problem.scale_anchor_camera = static_cast<int>(idx.cam_ba[anchor_cam]);
            idx.problem.scale_anchor_axis = choose_scale_anchor_axis(pose0, pose1);
        }
    }
    return idx;
}

/// Run a global bundle adjustment over all registered cameras and points, then
/// write the refined poses / points back into the Scene.
void run_global_ba(Scene& s) {
    if (s.index_a == kNoIndex) {
        return;
    }
    BAIndex idx = build_ba(s);
    if (idx.problem.poses.empty() || idx.problem.points.empty()) {
        return;
    }
    BAOptions options;
    options.loss_scale_px = s.opt->ba_loss_scale_px;
    options.num_threads = 1;  // TODO: determinism over speed; revisit later.
    bundle_adjust(idx.problem, options);
    for (std::size_t i = 0; i < idx.ba_to_cam.size(); ++i) {
        s.pose[idx.ba_to_cam[i]].R = idx.problem.poses[i].R;
        s.pose[idx.ba_to_cam[i]].t = idx.problem.poses[i].t;
    }
    for (std::size_t i = 0; i < idx.ba_to_track.size(); ++i) {
        s.point[idx.ba_to_track[i]] = idx.problem.points[i];
    }
}

// --- initial pair and resection ---

/// Rank the verified pairs by match count, evaluate the top candidates, and
/// seed the reconstruction from the best one. Pairs are inspected in
/// (match count desc, camera id asc) order for determinism.
bool choose_initial_pair(Scene& s) {
    struct Candidate {
        std::size_t match_count;
        std::int32_t id_a, id_b;
        const PairMatches* pair;
    };
    std::vector<Candidate> candidates;
    for (const PairMatches& pair : s.in->matches) {
        if (pair.matches.size() < s.opt->min_pair_inliers) {
            continue;
        }
        candidates.push_back({pair.matches.size(), pair.camera_a, pair.camera_b, &pair});
    }
    if (candidates.empty()) {
        return false;
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& l, const Candidate& r) {
        if (l.match_count != r.match_count) {
            return l.match_count > r.match_count;
        }
        if (l.id_a != r.id_a) {
            return l.id_a < r.id_a;
        }
        return l.id_b < r.id_b;
    });
    if (candidates.size() > s.opt->init_pair_candidates) {
        candidates.resize(s.opt->init_pair_candidates);
    }

    const auto index_of = [&](std::int32_t id) -> std::size_t {
        for (std::size_t c = 0; c < s.ncams; ++c) {
            if (s.camera_id[c] == id) {
                return c;
            }
        }
        return kNoIndex;
    };

    const double min_parallax = deg2rad(s.opt->min_init_parallax_deg);
    bool have_parallax_best = false, have_any_best = false;
    std::size_t par_a = kNoIndex, par_b = kNoIndex, any_a = kNoIndex, any_b = kNoIndex;
    std::size_t par_inliers = 0;
    double any_parallax = -1.0;
    RelativePose par_pose, any_pose;

    for (const Candidate& cand : candidates) {
        const std::size_t ca = index_of(cand.id_a);
        const std::size_t cb = index_of(cand.id_b);
        if (ca == kNoIndex || cb == kNoIndex) {
            continue;
        }
        std::vector<Eigen::Vector3d> b1, b2;
        b1.reserve(cand.match_count);
        b2.reserve(cand.match_count);
        for (const KeypointMatch& m : cand.pair->matches) {
            b1.push_back(s.bearings[ca][m.index_a]);
            b2.push_back(s.bearings[cb][m.index_b]);
        }
        RansacOptions options;
        options.threshold_rad =
            s.opt->resection_threshold_px / std::sqrt(camera_f(s, ca) * camera_f(s, cb));
        options.seed = mix_seed(s.opt->seed, cand.id_a, cand.id_b);
        const EssentialResult essential = estimate_essential_ransac(b1, b2, options);
        if (!essential.success || essential.inlier_count == 0) {
            continue;
        }

        // Median parallax over the inliers: the angle between the two viewing
        // rays, the max parallax of each triangulated inlier point.
        const AbsolutePose pose_a;  // identity: camera a is the world origin
        AbsolutePose pose_b;
        pose_b.R = essential.pose.R;
        pose_b.t = essential.pose.t;
        std::vector<double> parallaxes;
        for (std::size_t i = 0; i < b1.size(); ++i) {
            if (essential.inliers[i] == 0) {
                continue;
            }
            const std::optional<Eigen::Vector3d> x = triangulate(essential.pose, b1[i], b2[i]);
            if (!x.has_value()) {
                continue;
            }
            parallaxes.push_back(max_parallax({pose_a, pose_b}, *x));
        }
        if (parallaxes.empty()) {
            continue;
        }
        std::sort(parallaxes.begin(), parallaxes.end());
        const double median = parallaxes[parallaxes.size() / 2];

        if (!have_any_best || median > any_parallax) {
            have_any_best = true;
            any_parallax = median;
            any_a = ca;
            any_b = cb;
            any_pose = essential.pose;
        }
        if (median >= min_parallax && (!have_parallax_best || essential.inlier_count > par_inliers)) {
            have_parallax_best = true;
            par_inliers = essential.inlier_count;
            par_a = ca;
            par_b = cb;
            par_pose = essential.pose;
        }
    }

    if (have_parallax_best) {
        s.index_a = par_a;
        s.index_b = par_b;
        s.pose[s.index_b].R = par_pose.R;
        s.pose[s.index_b].t = par_pose.t;
    } else if (have_any_best) {
        s.index_a = any_a;
        s.index_b = any_b;
        s.pose[s.index_b].R = any_pose.R;
        s.pose[s.index_b].t = any_pose.t;
    } else {
        return false;
    }
    s.registered[s.index_a] = 1;
    s.registered[s.index_b] = 1;
    s.pose[s.index_a] = AbsolutePose{};  // identity: world origin
    return true;
}

/// Number of triangulated tracks camera `cam` observes (its resection support).
std::size_t resection_point_count(const Scene& s, std::size_t cam) {
    std::size_t n = 0;
    for (std::size_t ti : s.cam_tracks[cam]) {
        if (s.point[ti].has_value()) {
            ++n;
        }
    }
    return n;
}

/// Resect camera `cam` against the triangulated tracks it observes. Returns true
/// on success and writes its pose.
bool resect_camera(Scene& s, std::size_t cam) {
    std::vector<Eigen::Vector3d> world, obs_bearings;
    for (std::size_t ti : s.cam_tracks[cam]) {
        if (!s.point[ti].has_value()) {
            continue;
        }
        // The bearing of this camera's observation in the track.
        for (const auto& [obs_cam, kp] : s.track_obs[ti]) {
            if (obs_cam != cam) {
                continue;
            }
            world.push_back(*s.point[ti]);
            obs_bearings.push_back(s.bearings[cam][kp]);
        }
    }
    if (world.size() < s.opt->min_resection_points) {
        return false;
    }
    RansacOptions options;
    options.threshold_rad = s.opt->resection_threshold_px / camera_f(s, cam);
    options.seed = mix_seed(s.opt->seed, s.camera_id[cam], 0);
    const AbsolutePoseResult result =
        estimate_absolute_pose_ransac(world, obs_bearings, options);
    if (!result.success) {
        return false;
    }
    s.pose[cam] = result.pose;
    s.registered[cam] = 1;
    return true;
}

}  // namespace

BuilderResult build_incremental(const BuilderInput& input, const BuilderOptions& options,
                                const std::function<bool()>& cancelled) {
    const std::map<std::int32_t, std::size_t> id_to_index = validate_input(input);

    Scene s;
    s.in = &input;
    s.opt = &options;
    init_scene(s, id_to_index);

    BuilderResult result;
    result.poses.assign(s.ncams, std::nullopt);
    result.points.assign(s.ntracks, std::nullopt);

    // Step 1: continuing an alignment from supplied poses.
    std::vector<std::size_t> seeded;
    for (std::size_t c = 0; c < s.ncams; ++c) {
        if (input.cameras[c].initial_pose.has_value()) {
            seeded.push_back(c);
        }
    }
    const bool continuing = seeded.size() >= 2;
    if (continuing) {
        for (std::size_t c : seeded) {
            s.registered[c] = 1;
            s.pose[c] = *input.cameras[c].initial_pose;
        }
        s.index_a = seeded[0];
        s.index_b = seeded[1];
        triangulate_new_tracks(s);
    } else {
        // Step 2: choose and seed the initial pair.
        if (!choose_initial_pair(s)) {
            return result;  // registered = 0
        }
        triangulate_new_tracks(s);
        run_global_ba(s);  // two-camera BA, camera a fixed, scale anchored on b
    }
    result.initial_camera_a = s.camera_id[s.index_a];
    result.initial_camera_b = s.camera_id[s.index_b];

    // Step 3: register cameras.
    std::size_t since_ba = 0;
    while (true) {
        if (cancelled && cancelled()) {
            throw Error("cancelled");
        }
        // Best candidate: most triangulated-point views, ties by lower index.
        std::size_t best = kNoIndex, best_count = 0;
        for (std::size_t c = 0; c < s.ncams; ++c) {
            if (s.registered[c] != 0 || s.failed[c] != 0) {
                continue;
            }
            const std::size_t count = resection_point_count(s, c);
            if (count < options.min_resection_points) {
                continue;
            }
            if (count > best_count) {
                best_count = count;
                best = c;
            }
        }
        if (best == kNoIndex) {
            break;  // nothing left that can be registered
        }
        if (!resect_camera(s, best)) {
            s.failed[best] = 1;  // failed for this round; try the next camera
            continue;
        }
        // A successful registration starts a new round: cameras that could not
        // be resected earlier may now see more triangulated points.
        std::fill(s.failed.begin(), s.failed.end(), 0);

        triangulate_new_tracks(s);

        const std::size_t registered_now = registered_count(s);
        const std::size_t interval = registered_now < 50
                                         ? std::max<std::size_t>(1, options.global_ba_every)
                                         : std::max<std::size_t>(1, registered_now / 10);
        ++since_ba;
        if (since_ba >= interval) {
            run_global_ba(s);
            since_ba = 0;
        }
        filter_points(s);
    }

    // Step 4: final BA, final filter, RMS.
    run_global_ba(s);
    filter_points(s);

    BAIndex final_index = build_ba(s);
    result.final_rms_px = final_index.problem.observations.empty()
                              ? 0.0
                              : reprojection_rms_px(final_index.problem);

    for (std::size_t c = 0; c < s.ncams; ++c) {
        if (s.registered[c] != 0) {
            result.poses[c] = s.pose[c];
        }
    }
    for (std::size_t t = 0; t < s.ntracks; ++t) {
        result.points[t] = s.point[t];
    }
    result.registered = registered_count(s);
    return result;
}

}  // namespace alphashape::sfm
