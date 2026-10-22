#pragma once

// aerial_fixture.hpp -- the shared harness for the milestone-9 aerial tests
// (DEMs + orthomosaics; test_aerial_fixture.cpp).
//
// The scene is analytic (data/make_aerial_fixture.py): a gentle slope plus a
// Gaussian hill, a ramp and an axis-aligned building box, imaged by a 4x3 nadir
// camera grid at 60 m. The world is a local ENU frame in metres (x east, y
// north, z up). The truth file (data/aerial_truth.json) carries, per image, the
// world->camera R and t (X_cam = R X_world + t), the centre, the focal length
// and the EXIF GPS; the DEM is the top surface on a 0.25 m cell grid.
//
// The helper layout mirrors tests/dense_fixture.hpp (balanced-bracket JSON
// parsing, AddPhotos + set truth poses), which the milestone-6/7 tests use.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace aerial_fixture {

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
inline const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
inline const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

inline std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

/// The 12 aerial images, named aerial_r{row}c{col}.jpg for row 0..2, col 0..3.
inline std::vector<std::string> image_files() {
    std::vector<std::string> files;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            files.push_back(
                fixture("aerial_r" + std::to_string(row) + "c" + std::to_string(col) + ".jpg"));
        }
    }
    return files;
}

// -- scene constants (must match data/make_aerial_fixture.py) ----------------

inline constexpr double kAreaX0 = -60.0;
inline constexpr double kAreaX1 = 60.0;
inline constexpr double kAreaY0 = -45.0;
inline constexpr double kAreaY1 = 45.0;

inline constexpr double kBuildX0 = -20.0;
inline constexpr double kBuildX1 = -4.0;
inline constexpr double kBuildY0 = 5.0;
inline constexpr double kBuildY1 = 17.0;
inline constexpr double kBuildBase = 2.0 + 0.04 * -12.0;
inline constexpr double kBuildTop = kBuildBase + 8.0;

inline constexpr double kRampX0 = 0.0;
inline constexpr double kRampX1 = 20.0;
inline constexpr double kRampY0 = 15.0;
inline constexpr double kRampY1 = 30.0;
inline constexpr double kRampSlope = 0.25;

inline constexpr int kImageWidth = 480;
inline constexpr int kImageHeight = 360;
inline constexpr double kFocal = 500.0;

/// Terrain height h(x, y), metres -- the ground *without* the building.
///
/// KEEP IN SYNC with truth_height in data/make_aerial_fixture.py: the DEM test
/// compares the two to 1e-4 m.
inline double truth_height(double x, double y) {
    double h = 2.0 + 0.04 * x;
    h += 3.0 * std::exp(-(((x - 25.0) * (x - 25.0) + (y + 15.0) * (y + 15.0)) / (2.0 * 12.0 * 12.0)));
    if (x >= kRampX0 && x <= kRampX1 && y >= kRampY0 && y <= kRampY1) {
        h += kRampSlope * (x - kRampX0);
    }
    return h;
}

// -- JSON helpers (balanced-bracket, as in dense_fixture) --------------------

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
        if (c == '-' || c == '+' || (c >= '0' && c <= '9')) {
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

/// The single scalar that follows `"key"` at or after `from`.
inline double json_scalar(const std::string& text, const std::string& key, std::size_t from = 0) {
    const std::size_t pos = text.find("\"" + key + "\"", from);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing JSON key \"" + key + "\"");
    }
    const std::size_t colon = text.find(':', pos);
    const std::vector<double> values = numbers_in(text.substr(colon + 1, 64));
    if (values.empty()) {
        throw std::runtime_error("no number after JSON key \"" + key + "\"");
    }
    return values.front();
}

// -- truth parsing -----------------------------------------------------------

struct TruthPose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // X_cam = R X_world + t
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();  // camera centre, ENU metres
    double focal = kFocal;
    std::string name;

    /// The camera centre implied by R and t (-R^T t), for cross-checking.
    Eigen::Vector3d centre_from_pose() const { return -R.transpose() * t; }
};

/// Parse aerial_truth.json into 12 poses in file order (row 0 first).
inline std::vector<TruthPose> load_truth() {
    const std::string json = read_text_file(fixture("aerial_truth.json"));
    std::vector<TruthPose> poses;
    std::size_t pos = 0;
    while ((pos = json.find("\"file\"", pos)) != std::string::npos) {
        const std::size_t r_key = json.find("\"R\"", pos);
        const std::size_t t_key = json.find("\"t\"", pos);
        const std::size_t c_key = json.find("\"centre\"", pos);
        if (r_key == std::string::npos || t_key == std::string::npos ||
            c_key == std::string::npos) {
            break;
        }
        const std::vector<double> r = numbers_in(balanced_array(json, json.find('[', r_key)));
        const std::vector<double> t = numbers_in(balanced_array(json, json.find('[', t_key)));
        const std::vector<double> centre = numbers_in(balanced_array(json, json.find('[', c_key)));
        if (r.size() != 9 || t.size() != 3 || centre.size() != 3) {
            throw std::runtime_error("unexpected truth array length in aerial_truth.json");
        }
        TruthPose pose;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                pose.R(row, col) = r[static_cast<std::size_t>(row * 3 + col)];
            }
        }
        pose.t = Eigen::Vector3d(t[0], t[1], t[2]);
        pose.centre = Eigen::Vector3d(centre[0], centre[1], centre[2]);
        pose.focal = json_scalar(json, "f", pos);
        const std::size_t name_begin = json.find('"', json.find(':', json.find("\"file\"", pos))) + 1;
        pose.name = json.substr(name_begin, json.find('"', name_begin) - name_begin);
        poses.push_back(pose);
        pos = c_key;
    }
    if (poses.size() != 12) {
        throw std::runtime_error("expected 12 truth poses, got " + std::to_string(poses.size()));
    }
    return poses;
}

// -- DEM grid ----------------------------------------------------------------

/// The DEM/ortho grid definition and its heights (metres, top surface).
struct TruthGrid {
    double origin_x = kAreaX0;   ///< west edge x (column 0's outer edge)
    double origin_y = kAreaY1;   ///< north edge y (row 0's outer edge)
    double cell = 0.25;
    int width = 480;
    int height = 360;
    std::vector<float> dem;  ///< row-major, row 0 = north, col 0 = west

    /// World (x, y) of the centre of cell (col, row).
    Eigen::Vector2d cell_centre(int col, int row) const {
        return Eigen::Vector2d(origin_x + (col + 0.5) * cell, origin_y - (row + 0.5) * cell);
    }

    float at(int col, int row) const {
        return dem[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(col)];
    }
};

/// Read aerial_truth_dem.f32: <int32 width><int32 height> then little-endian f32.
inline TruthGrid load_truth_dem() {
    const std::string path = fixture("aerial_truth_dem.f32");
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::int32_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    TruthGrid grid;
    grid.width = header[0];
    grid.height = header[1];
    const std::streamoff expected =
        2 * static_cast<std::streamoff>(sizeof(std::int32_t)) +
        static_cast<std::streamoff>(grid.width) * grid.height * 4;
    if (size != expected || grid.width <= 0 || grid.height <= 0) {
        throw std::runtime_error("aerial_truth_dem.f32 has an unexpected size or header");
    }
    grid.dem.resize(static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height));
    in.read(reinterpret_cast<char*>(grid.dem.data()),
            static_cast<std::streamsize>(grid.dem.size() * sizeof(float)));
    return grid;
}

/// Decode aerial_truth_ortho.png (the top-surface texture on the DEM grid).
inline Image load_truth_ortho() { return read_image(fixture("aerial_truth_ortho.png")); }

// -- chunk construction ------------------------------------------------------

/// AddPhotos on the 12 aerial images, overwrite every pose with the truth pose,
/// enable the cameras, place the chunk's local ENU frame on the truth origin
/// (transform = enu_to_ecef(origin), CRS 4326). This gives the DEM/ortho code a
/// fully georeferenced chunk in the same frame the truth products use.
inline Document make_truth_chunk_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());

    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{image_files()});
    run_task(builtin_registry(), document, "AddPhotos", params);

    const std::vector<TruthPose> poses = load_truth();
    for (std::size_t c = 0; c < chunk.cameras().size(); ++c) {
        Camera& camera = chunk.cameras()[c];
        camera.pose = Transform::Identity();
        camera.pose.block<3, 3>(0, 0) = poses[c].R;
        camera.pose.block<3, 1>(0, 3) = poses[c].t;
        camera.enabled = true;
    }
    for (Sensor& sensor : chunk.sensors()) {
        sensor.type = CameraType::Frame;
        sensor.width = kImageWidth;
        sensor.height = kImageHeight;
        sensor.calibration = Calibration::initial(CameraType::Frame, kImageWidth, kImageHeight, kFocal);
    }

    // The chunk's local frame is ENU; its placement maps it onto ECEF at the
    // truth origin (data/make_aerial_fixture.py writes the same origin).
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    chunk.transform() = geo::enu_to_ecef(origin);
    chunk.crs().epsg = 4326;
    return document;
}

}  // namespace aerial_fixture
}  // namespace alphashape
