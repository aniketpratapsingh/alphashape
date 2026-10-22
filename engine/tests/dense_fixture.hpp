#pragma once

// dense_fixture.hpp -- the shared harness for the dense-reconstruction tests
// (test_build_dense.cpp and test_build_model.cpp, milestone 6/7).
//
// The scene is analytic (data/make_scene_fixtures.py): a ground plane z = 0 for
// y <= 0 and a wall plane y = 0 for 0 <= z <= 6, |x| <= 10, seen from eight
// cameras on a 40 degree arc of radius 16 around the corner. Most helpers below
// come from the milestone-6 dense test; they are factored here so the milestone-7
// meshing test reuses exactly the same truth-pose setup.
//
// The truth file (data/seq_truth.json) carries, per image, the world->camera R
// and t (X_cam = R X + t) and the focal length.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/sfm/triangulation.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape {
namespace dense_fixture {

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
inline const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
inline const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

inline std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

inline std::vector<std::string> seq_files(int count = 8) {
    std::vector<std::string> files;
    for (int i = 0; i < count; ++i) {
        files.push_back(fixture("seq_" + std::to_string(i) + ".jpg"));
    }
    return files;
}

// -- truth parsing (balanced-bracket, as in test_align_cameras) --------------

inline std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

inline std::vector<double> numbers_in(const std::string& text) {
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

inline std::string balanced_array(const std::string& text, std::size_t open) {
    int depth = 0;
    for (std::size_t i = open; i < text.size(); ++i) {
        if (text[i] == '[') {
            ++depth;
        } else if (text[i] == ']') {
            if (--depth == 0) {
                return text.substr(open, i - open + 1);
            }
        }
    }
    return {};
}

struct TruthImage {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d centre() const { return -R.transpose() * t; }
};

inline std::vector<TruthImage> parse_truth_images(const std::string& name) {
    const std::string json = read_text_file(fixture(name));
    std::vector<TruthImage> images;
    std::size_t pos = 0;
    while ((pos = json.find("\"file\"", pos)) != std::string::npos) {
        const std::size_t r_key = json.find("\"R\"", pos);
        const std::size_t t_key = json.find("\"t\"", pos);
        if (r_key == std::string::npos || t_key == std::string::npos) {
            break;
        }
        const std::vector<double> r = numbers_in(balanced_array(json, json.find('[', r_key)));
        const std::vector<double> t = numbers_in(balanced_array(json, json.find('[', t_key)));
        if (r.size() != 9 || t.size() != 3) {
            throw std::runtime_error("unexpected truth array length in " + name);
        }
        TruthImage image;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                image.R(row, col) = r[static_cast<std::size_t>(row * 3 + col)];
            }
        }
        image.t = Eigen::Vector3d(t[0], t[1], t[2]);
        images.push_back(image);
        pos = t_key;
    }
    return images;
}

// -- document construction ---------------------------------------------------

inline Document make_document(const std::vector<std::string>& files) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{files});
    run_task(builtin_registry(), document, "AddPhotos", params);
    return document;
}

inline void run_match_photos(Document& document, std::int64_t keypoint_limit_per_mpx = 10000) {
    Params params = make_params(builtin_registry().require("MatchPhotos"));
    params.set("keypoint_limit_per_mpx", ParamValue{keypoint_limit_per_mpx});
    run_task(builtin_registry(), document, "MatchPhotos", params);
}

inline void run_align_cameras(Document& document) {
    Params params = make_params(builtin_registry().require("AlignCameras"));
    run_task(builtin_registry(), document, "AlignCameras", params);
}

/// AddPhotos + MatchPhotos + AlignCameras on the given files.
inline Document aligned_document(const std::vector<std::string>& files) {
    Document document = make_document(files);
    run_match_photos(document);
    run_align_cameras(document);
    return document;
}

// -- task runners ------------------------------------------------------------

inline void run_build_depth_maps(Document& document, int downscale = 4,
                                 const std::string& filter_mode = "mild") {
    Params params = make_params(builtin_registry().require("BuildDepthMaps"));
    params.set("downscale", ParamValue{static_cast<std::int64_t>(downscale)});
    params.set("filter_mode", ParamValue{filter_mode});
    run_task(builtin_registry(), document, "BuildDepthMaps", params);
}

inline void run_build_point_cloud(Document& document, bool point_colors = true,
                                  bool point_confidence = true) {
    Params params = make_params(builtin_registry().require("BuildPointCloud"));
    params.set("point_colors", ParamValue{point_colors});
    params.set("point_confidence", ParamValue{point_confidence});
    run_task(builtin_registry(), document, "BuildPointCloud", params);
}

// -- truth-frame reconstruction ----------------------------------------------

inline Eigen::Vector3d camera_centre(const Camera& camera) {
    const Eigen::Matrix3d r = camera.pose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = camera.pose.block<3, 1>(0, 3);
    return -r.transpose() * t;
}

inline constexpr std::size_t kNoTruth = static_cast<std::size_t>(-1);

/// Map each chunk camera to its index in the truth array by the seq_N label.
inline std::vector<std::size_t> truth_index_by_camera(const Chunk& chunk,
                                                      const std::vector<TruthImage>& truth) {
    std::vector<std::size_t> index(chunk.cameras().size(), kNoTruth);
    for (std::size_t c = 0; c < chunk.cameras().size(); ++c) {
        const std::string& label = chunk.cameras()[c].label;
        if (label.rfind("seq_", 0) != 0) {
            continue;
        }
        const int n = std::atoi(label.c_str() + 4);
        if (n < 0 || static_cast<std::size_t>(n) >= truth.size()) {
            throw std::runtime_error("unexpected camera label " + label);
        }
        index[c] = static_cast<std::size_t>(n);
    }
    return index;
}

/// Overwrite every camera pose with the truth pose (X_cam = R X_world + t).
inline void set_truth_poses(Document& document, const std::vector<TruthImage>& truth) {
    Chunk& chunk = *document.active_chunk();
    const std::vector<std::size_t> index = truth_index_by_camera(chunk, truth);
    for (std::size_t c = 0; c < chunk.cameras().size(); ++c) {
        if (index[c] == kNoTruth) {
            continue;
        }
        Camera& camera = chunk.cameras()[c];
        camera.pose = Transform::Identity();
        camera.pose.block<3, 3>(0, 0) = truth[index[c]].R;
        camera.pose.block<3, 1>(0, 3) = truth[index[c]].t;
        camera.enabled = true;
    }
}

/// Replace every sensor with the true pinhole calibration (no distortion).
inline void set_truth_calibration(Document& document, int width, int height, double f) {
    for (Sensor& sensor : document.active_chunk()->sensors()) {
        sensor.type = CameraType::Frame;
        sensor.width = width;
        sensor.height = height;
        sensor.calibration = Calibration::initial(CameraType::Frame, width, height, f);
    }
}

/// Re-triangulate every track with the (now truth) camera poses and the true
/// calibration, so the depth range the stereo stage derives from the tie points
/// is consistent with the truth poses.
inline std::size_t retriangulate_tracks(Chunk& chunk) {
    std::unordered_map<std::int32_t, const KeypointSet*> set_by_id;
    for (const KeypointSet& set : chunk.tie_points().keypoint_sets) {
        set_by_id[set.camera_id] = &set;
    }
    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk.cameras()) {
        camera_by_id[camera.id] = &camera;
    }

    const sfm::TriangulationOptions options;  // default 1.5 deg min parallax
    std::size_t kept = 0;
    for (Track& track : chunk.tie_points().tracks) {
        std::vector<sfm::AbsolutePose> poses;
        std::vector<Eigen::Vector3d> bearings;
        for (const Observation& observation : track.observations) {
            const auto camera_it = camera_by_id.find(observation.camera_id);
            const auto set_it = set_by_id.find(observation.camera_id);
            if (camera_it == camera_by_id.end() || set_it == set_by_id.end()) {
                continue;
            }
            const Camera& camera = *camera_it->second;
            const KeypointSet& set = *set_it->second;
            if (observation.keypoint_index >= set.keypoints.size() || camera.sensor_id < 0 ||
                camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
                continue;
            }
            const Calibration& calibration =
                chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;

            sfm::AbsolutePose pose;
            pose.R = camera.pose.block<3, 3>(0, 0);
            pose.t = camera.pose.block<3, 1>(0, 3);
            poses.push_back(pose);
            bearings.push_back(calibration.unproject(
                keypoint_to_pixel(set.keypoints[observation.keypoint_index])));
        }
        track.point = sfm::triangulate_multiview(poses, bearings, options);
        if (track.point.has_value()) {
            ++kept;
        }
    }
    return kept;
}

// -- statistics --------------------------------------------------------------

/// Nearest-rank percentile of a copy of `values` sorted ascending.
inline double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double idx = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double f = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - f) + sorted[hi] * f;
}

// -- scratch directory -------------------------------------------------------

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

}  // namespace dense_fixture
}  // namespace alphashape
