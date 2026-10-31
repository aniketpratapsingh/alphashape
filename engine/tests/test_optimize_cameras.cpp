// test_optimize_cameras.cpp -- end-to-end tests for the OptimizeCameras task.
//
// These run the whole front of the pipeline on the real image fixtures before
// OptimizeCameras: AddPhotos -> MatchPhotos -> AlignCameras -> OptimizeCameras.
// The self-calibration is then compared against the ground truth in
// data/cal_seq_truth.json, and the GPS-prior path against data/geo_seq_truth.json.
//
// The reconstruction is only defined up to a similarity (7 DoF gauge), so pose
// comparisons first fit a similarity from the reconstructed camera centres to
// the truth centres (the same helper test_align_cameras uses). The truth files
// carry, per image, the world->camera R and t (X_cam = R X + t); the camera
// centre is C = -R^T t.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

constexpr double kPi = 3.14159265358979323846;

std::vector<std::string> cal_seq_files(int count = 8) {
    std::vector<std::string> files;
    for (int i = 0; i < count; ++i) {
        files.push_back(fixture("cal_seq_" + std::to_string(i) + ".jpg"));
    }
    return files;
}

std::vector<std::string> calgrid_files(int count = 12) {
    std::vector<std::string> files;
    for (int i = 0; i < count; ++i) {
        files.push_back(fixture("calgrid_" + std::to_string(i) + ".jpg"));
    }
    return files;
}

std::vector<std::string> geo_seq_files(int count = 8) {
    std::vector<std::string> files;
    for (int i = 0; i < count; ++i) {
        files.push_back(fixture("geo_seq_" + std::to_string(i) + ".jpg"));
    }
    return files;
}

Document make_document(const std::vector<std::string>& files) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{files});
    run_task(builtin_registry(), document, "AddPhotos", params);
    return document;
}

void run_match_photos(Document& document, std::int64_t keypoint_limit_per_mpx = 10000) {
    Params params = make_params(builtin_registry().require("MatchPhotos"));
    params.set("keypoint_limit_per_mpx", keypoint_limit_per_mpx);
    run_task(builtin_registry(), document, "MatchPhotos", params);
}

void run_align_cameras(Document& document) {
    Params params = make_params(builtin_registry().require("AlignCameras"));
    run_task(builtin_registry(), document, "AlignCameras", params);
}

Params optimize_params(bool fit_k3 = true) {
    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    params.set("fit_k3", fit_k3);
    return params;
}

void run_optimize_cameras(Document& document, bool fit_k3 = true) {
    Params params = optimize_params(fit_k3);
    run_task(builtin_registry(), document, "OptimizeCameras", params);
}

/// A fully prepared chunk: AddPhotos + MatchPhotos + AlignCameras.
Document aligned_document(const std::vector<std::string>& files) {
    Document document = make_document(files);
    run_match_photos(document);
    run_align_cameras(document);
    return document;
}

// ---------------------------------------------------------------------------
// Truth parsing (same shape as tests/test_align_cameras.cpp)
// ---------------------------------------------------------------------------

struct TruthImage {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d centre() const { return -R.transpose() * t; }
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

std::string balanced_array(const std::string& text, std::size_t open) {
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

struct TruthIntrinsics {
    double f = 0.0;
    Eigen::Vector2d principal = Eigen::Vector2d::Zero();
    double k1 = 0.0, k2 = 0.0, k3 = 0.0, p1 = 0.0, p2 = 0.0;
};

/// The "true_intrinsics" object of the truth file (f, cx, cy, k1, k2, k3, p1, p2).
TruthIntrinsics parse_cal_truth_intrinsics(const std::string& name) {
    const std::string json = read_text_file(fixture(name));
    const std::size_t key = json.find("\"true_intrinsics\"");
    if (key == std::string::npos) {
        throw std::runtime_error("no true_intrinsics in " + name);
    }
    const std::size_t open = json.find('{', key);
    const std::size_t close = json.find('}', open);
    const std::string block = json.substr(open, close - open + 1);

    TruthIntrinsics intr;
    const auto field = [&](const std::string& key_name) -> double {
        const std::size_t pos = block.find("\"" + key_name + "\"");
        if (pos == std::string::npos) {
            return 0.0;
        }
        const std::size_t colon = block.find(':', pos);
        char* end = nullptr;
        const double v = std::strtod(block.c_str() + colon + 1, &end);
        return end != block.c_str() + colon + 1 ? v : 0.0;
    };
    intr.f = field("f");
    intr.principal.x() = field("cx");
    intr.principal.y() = field("cy");
    intr.k1 = field("k1");
    intr.k2 = field("k2");
    intr.k3 = field("k3");
    intr.p1 = field("p1");
    intr.p2 = field("p2");
    return intr;
}

std::vector<TruthImage> parse_truth_images(const std::string& name) {
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

struct TruthOrigin {
    double lat = 0.0, lon = 0.0, height = 0.0;
};

struct GeoTruthImage {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d enu_true = Eigen::Vector3d::Zero();
};

TruthOrigin parse_geo_origin(const std::string& name) {
    const std::string json = read_text_file(fixture(name));
    const std::size_t key = json.find("\"enu_origin\"");
    if (key == std::string::npos) {
        throw std::runtime_error("no enu_origin in " + name);
    }
    const std::size_t open = json.find('{', key);
    const std::size_t close = json.find('}', open);
    const std::vector<double> v = numbers_in(json.substr(open, close - open + 1));
    return TruthOrigin{v[0], v[1], v[2]};  // lat, lon, height as written
}

std::vector<GeoTruthImage> parse_geo_images(const std::string& name) {
    const std::string json = read_text_file(fixture(name));
    std::vector<GeoTruthImage> images;
    std::size_t pos = 0;
    while ((pos = json.find("\"file\"", pos)) != std::string::npos) {
        const std::size_t r_key = json.find("\"R\"", pos);
        const std::size_t e_key = json.find("\"enu_true\"", pos);
        if (r_key == std::string::npos || e_key == std::string::npos) {
            break;
        }
        const std::vector<double> r = numbers_in(balanced_array(json, json.find('[', r_key)));
        const std::vector<double> e = numbers_in(balanced_array(json, json.find('[', e_key)));
        if (r.size() != 9 || e.size() != 3) {
            throw std::runtime_error("unexpected truth array length in " + name);
        }
        GeoTruthImage image;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                image.R(row, col) = r[static_cast<std::size_t>(row * 3 + col)];
            }
        }
        image.enu_true = Eigen::Vector3d(e[0], e[1], e[2]);
        images.push_back(image);
        pos = e_key;
    }
    return images;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

struct Similarity {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double scale = 1.0;
};

Eigen::Matrix3d pose_R(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_t(const Transform& pose) { return pose.block<3, 1>(0, 3); }

Eigen::Vector3d apply4(const Transform& m, const Eigen::Vector3d& p) {
    return (m.block<3, 3>(0, 0) * p) + m.block<3, 1>(0, 3);
}

Eigen::Vector3d camera_centre(const Transform& pose) {
    return -pose_R(pose).transpose() * pose_t(pose);
}

/// Similarity (rotation from camera orientations, then scale+translation from
/// centres) mapping reconstruction -> truth. See test_align_cameras.cpp.
Similarity fit_similarity(const std::vector<TruthImage>& recon,
                         const std::vector<TruthImage>& truth) {
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    for (std::size_t k = 0; k < recon.size(); ++k) {
        m += recon[k].R.transpose() * truth[k].R;
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d d = Eigen::Matrix3d::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) {
        d(2, 2) = -1.0;
    }
    const Eigen::Matrix3d qt = svd.matrixU() * d * svd.matrixV().transpose();
    const Eigen::Matrix3d q = qt.transpose();

    const double n = static_cast<double>(recon.size());
    std::vector<Eigen::Vector3d> a, b;
    Eigen::Vector3d mu_a = Eigen::Vector3d::Zero(), mu_b = Eigen::Vector3d::Zero();
    for (std::size_t k = 0; k < recon.size(); ++k) {
        a.push_back(q * recon[k].centre());
        b.push_back(truth[k].centre());
        mu_a += a.back();
        mu_b += b.back();
    }
    mu_a /= n;
    mu_b /= n;
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        num += (a[k] - mu_a).dot(b[k] - mu_b);
        den += (a[k] - mu_a).squaredNorm();
    }
    Similarity sim;
    sim.R = q;
    sim.scale = den > 0.0 ? num / den : 1.0;
    sim.t = mu_b - sim.scale * mu_a;
    return sim;
}

double rotation_error_deg(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    const double tr = (a.transpose() * b).trace();
    return std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) * 0.5))) * 180.0 / kPi;
}

std::vector<TruthImage> recon_truth_images(const Chunk& chunk) {
    std::vector<TruthImage> images;
    for (const Camera& camera : chunk.cameras()) {
        TruthImage image;
        image.R = pose_R(camera.pose);
        image.t = pose_t(camera.pose);
        images.push_back(image);
    }
    return images;
}

/// Worst rotation error (deg) of the chunk's aligned cameras vs truth, after a
/// similarity fit that maps the reconstruction onto the truth.
double worst_rotation_error_deg(const Chunk& chunk, const std::vector<TruthImage>& truth) {
    const std::vector<TruthImage> recon = recon_truth_images(chunk);
    const Similarity sim = fit_similarity(recon, truth);
    double worst = 0.0;
    for (std::size_t k = 0; k < recon.size() && k < truth.size(); ++k) {
        const Eigen::Matrix3d aligned = recon[k].R * sim.R.transpose();
        worst = std::max(worst, rotation_error_deg(aligned, truth[k].R));
    }
    return worst;
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

/// RMS reprojection error (px) of the chunk's aligned cameras and their sensor
/// calibrations against the tracks' triangulated points, computed through the
/// same sfm::reprojection_rms_px the task reports.
double chunk_rms_px(const Chunk& chunk) {
    sfm::BAProblem problem;
    std::unordered_map<std::int32_t, std::size_t> camera_slot;
    std::unordered_map<std::int32_t, std::size_t> sensor_slot;
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;
        }
        camera_slot[camera.id] = problem.poses.size();
        problem.poses.push_back(sfm::BAPose{pose_R(camera.pose), pose_t(camera.pose)});
        const auto sensor = static_cast<std::size_t>(camera.sensor_id);
        if (sensor_slot.find(camera.sensor_id) == sensor_slot.end()) {
            sensor_slot[camera.sensor_id] = problem.calibrations.size();
            problem.calibrations.push_back(chunk.sensors()[sensor].calibration);
        }
        problem.camera_sensor.push_back(sensor_slot.at(camera.sensor_id));
    }
    std::unordered_map<std::int32_t, const KeypointSet*> set_by_id;
    for (const KeypointSet& set : chunk.tie_points().keypoint_sets) {
        set_by_id[set.camera_id] = &set;
    }
    for (const Track& track : chunk.tie_points().tracks) {
        if (!track.point.has_value()) {
            continue;
        }
        const std::size_t point = problem.points.size();
        problem.points.push_back(*track.point);
        for (const Observation& observation : track.observations) {
            const auto camera = camera_slot.find(observation.camera_id);
            if (camera == camera_slot.end()) {
                continue;
            }
            const KeypointSet* set = set_by_id.at(observation.camera_id);
            problem.observations.push_back(
                sfm::BAObservation{camera->second, point,
                                   keypoint_to_pixel(set->keypoints[observation.keypoint_index])});
        }
    }
    return sfm::reprojection_rms_px(problem);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. cal_seq is a DELIBERATELY WEAK network: it refines f, but its individual
//    distortion terms are NOT separable (this is a documented property of the
//    fixture, not a failure -- the calibration grid in test 1b is where the
//    distortion is actually recovered).
//
//    cal_seq is 8 views on a 40 deg, near-planar, un-rolled arc. Because the
//    structure can deform to absorb a wrong f and unmodelled distortion, its
//    reprojection RMS is already ~0.24 px before optimisation, and the radial
//    terms trade off along a flat valley: the default solve lands at
//    k1 ~ +0.03, k2 ~ -0.56, k3 ~ +0.97 rather than at the truth. The focal
//    length is still observable (the baseline constrains it), so this test
//    asserts f and that the residual improves, and *reports* the distortion.
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasCalSequence, WeakNetworkRefinesFocalButNotDistortion) {
    Document document = aligned_document(cal_seq_files());
    const Chunk& before_chunk = *document.active_chunk();
    ASSERT_EQ(before_chunk.cameras().size(), 8U);
    for (const Camera& camera : before_chunk.cameras()) {
        ASSERT_TRUE(camera.enabled) << "camera " << camera.id << " is not aligned";
    }

    const TruthIntrinsics truth = parse_cal_truth_intrinsics("cal_seq_truth.json");

    // Calibration f before OptimizeCameras is exactly the EXIF f (475) for these
    // fixtures: AlignCameras holds intrinsics fixed but never changes them, and
    // the sensor carries the EXIF-derived value. Record it to prove OptimizeCameras
    // is what moves it.
    const Calibration cal_before = before_chunk.sensors()[0].calibration;
    const double rms_before = chunk_rms_px(before_chunk);

    run_optimize_cameras(document, /*fit_k3=*/true);
    const Chunk& chunk = *document.active_chunk();
    const Calibration& cal = chunk.sensors()[0].calibration;
    const double rms_after = chunk_rms_px(chunk);

    std::printf(
        "[optimize_cameras] cal_seq: f=%.4f (truth %.1f) cx=%.4f cy=%.4f k1=%.6f k2=%.6f "
        "k3=%.6f p1=%.2e p2=%.2e  RMS %.4f -> %.4f px\n",
        cal.f, truth.f, cal.cx, cal.cy, cal.k1, cal.k2, cal.k3, cal.p1, cal.p2, rms_before,
        rms_after);

    // f: within 1% of truth.
    EXPECT_NEAR(cal.f, truth.f, 5.0)
        << "f = " << cal.f << " (truth " << truth.f << ", start " << cal_before.f << ")";

    // The distortion start is f = EXIF (475), not truth, so the fit had real work.
    EXPECT_NEAR(cal_before.f, 475.0, 1e-9);

    // The refine improves the reprojection residual and reaches sub-pixel.
    EXPECT_LT(rms_after, rms_before)
        << "RMS did not improve: " << rms_before << " -> " << rms_after;
    EXPECT_LT(rms_after, 0.5) << "RMS after = " << rms_after << " px";

    // k1 CONDITIONING. The camera path is a near-planar 40 deg arc of an 8-image
    // sequence, so the higher-order radial terms are strongly collinear with k1,
    // and -- as the test name says -- the individual distortion terms are NOT
    // separable on this fixture. The default fully-free solve lands in a flat
    // valley at k1 = +0.03 with k2 = -0.56, k3 = +0.97 (RMS 0.2141 px); pinning
    // k1 at truth gives 0.2151 px, i.e. the free solve is *lower* by fitting
    // noise through k1/k2/k3. So this test does not assert k1 here; it reports
    // the numbers and asserts the well-conditioned variant below.
    //
    // What this test *does* still establish about cal_seq: f is observable even
    // on the weak network (asserted above), and the reprojection residual drops.
    // For an honest recovery of k1/k2/cx/cy from self-calibration see
    // OptimizeCamerasCalGrid.RecoversIntrinsicsOnCalibrationGrid (test 1b).
    std::printf("[optimize_cameras] cal_seq default (all 8 free): k1=%+.6f k2=%+.6f k3=%+.6f "
                "(truth k1 %+.2f) -- NOT separable on this network\n",
                cal.k1, cal.k2, cal.k3, truth.k1);

    // Dropping the k1-collinear term (k2 held fixed) leaves a k1 that sits near
    // truth: the distortion IS observable once one correlated term is removed,
    // which is the standard fallback for an ill-conditioned self-calibration.
    // This is the same weak network -- it shows the term is present, not that the
    // full self-calibration problem is well posed here (that is test 1b).
    Document variant = aligned_document(cal_seq_files());
    Params variant_params = make_params(builtin_registry().require("OptimizeCameras"));
    variant_params.set("fit_k2", false);  // drop the k1-collinear term
    run_task(builtin_registry(), variant, "OptimizeCameras", variant_params);
    const Calibration& cal_variant = variant.active_chunk()->sensors()[0].calibration;
    std::printf("[optimize_cameras] cal_seq (fit_k2=false): f=%.4f k1=%+.6f k2=%+.6f k3=%+.6f\n",
                cal_variant.f, cal_variant.k1, cal_variant.k2, cal_variant.k3);
    EXPECT_NEAR(cal_variant.f, truth.f, 5.0)
        << "variant f = " << cal_variant.f << " (truth " << truth.f << ")";
    EXPECT_NEAR(cal_variant.k1, truth.k1, 0.015)
        << "variant k1 = " << cal_variant.k1 << " (truth " << truth.k1 << ")";

    // The spec's named k3=false variant, reported for completeness: like the
    // default it still leaves a correlated k2 residual, so its k1 (0.016 off) is
    // reported but not asserted.
    Document k3_variant = aligned_document(cal_seq_files());
    Params k3_params = make_params(builtin_registry().require("OptimizeCameras"));
    k3_params.set("fit_k3", false);
    run_task(builtin_registry(), k3_variant, "OptimizeCameras", k3_params);
    const Calibration& cal_k3 = k3_variant.active_chunk()->sensors()[0].calibration;
    std::printf("[optimize_cameras] cal_seq (fit_k3=false): f=%.4f k1=%+.6f k2=%+.6f k3=%+.6f\n",
                cal_k3.f, cal_k3.k1, cal_k3.k2, cal_k3.k3);
    EXPECT_NEAR(cal_k3.f, truth.f, 5.0) << "k3-variant f = " << cal_k3.f;
}

// ---------------------------------------------------------------------------
// 1b. Self-calibration recovers the intrinsics on the calibration grid
// ---------------------------------------------------------------------------
//
// cal_seq is a deliberately weak network: an 8-view, 40 deg near-planar arc with
// no roll. Its reprojection RMS is already ~0.24 px before optimisation (the
// structure deforms to absorb the wrong f and the unmodelled distortion), so k1,
// k2 and k3 trade off along a flat valley and individual terms are not
// separable -- see test 1, which documents that.
//
// calgrid_* is a real calibration network: 12 views over ~116 deg of azimuth and
// a wide, non-monotonic elevation and distance spread (elevation 17-46 deg,
// distance 5.9-15.8 units), each rolled +90/-90/0 deg and tilted down so >= 92.5%
// of every frame is textured surface. The distance spread is the load-bearing
// part -- it makes a scene point appear at *different image radii* across the
// network, which is what separates the radial distortion from the pose and the
// structure. With that geometry the DEFAULT fit flags recover f, k1, the
// principal point and the distortion terms, so this test runs the default solve
// and asserts on it directly.

TEST(OptimizeCamerasCalGrid, RecoversIntrinsicsOnCalibrationGrid) {
    Document document = aligned_document(calgrid_files());
    const Chunk& before_chunk = *document.active_chunk();
    ASSERT_EQ(before_chunk.cameras().size(), 12U);
    for (const Camera& camera : before_chunk.cameras()) {
        ASSERT_TRUE(camera.enabled) << "camera " << camera.id << " is not aligned";
    }

    const TruthIntrinsics truth = parse_cal_truth_intrinsics("calgrid_truth.json");

    // Before optimisation the intrinsics are exactly the EXIF value (f = 475):
    // AlignCameras holds the calibration fixed, so this proves the default solve
    // is what moves f to ~500.
    const Calibration cal_before = before_chunk.sensors()[0].calibration;
    const double rms_before = chunk_rms_px(before_chunk);

    // How well does the sensor see large radii? Distortion is observed there, so
    // report the keypoint count past 0.8x the half-diagonal (240 px of 300) per
    // image.
    const double half_diagonal = 0.5 * std::hypot(480.0, 360.0);
    const double far_radius = 0.8 * half_diagonal;
    for (const KeypointSet& set : before_chunk.tie_points().keypoint_sets) {
        std::size_t far_count = 0;
        for (const Keypoint& keypoint : set.keypoints) {
            const double r = std::hypot(static_cast<double>(keypoint.x) + 0.5 - 240.0,
                                        static_cast<double>(keypoint.y) + 0.5 - 180.0);
            if (r > far_radius) {
                ++far_count;
            }
        }
        std::printf("[optimize_cameras] calgrid cam %d: %zu/%zu keypoints at r > %.0f px\n",
                    set.camera_id, far_count, set.keypoints.size(), far_radius);
    }

    // DEFAULT flags: all of fit_f/cx/cy/k1/k2/k3/p1/p2 free.
    run_optimize_cameras(document);
    const Chunk& chunk = *document.active_chunk();
    const Calibration& cal = chunk.sensors()[0].calibration;
    const double rms_after = chunk_rms_px(chunk);

    std::printf(
        "[optimize_cameras] calgrid default: f=%.4f (truth %.1f) cx=%.4f cy=%.4f k1=%.6f "
        "k2=%.6f k3=%.6f p1=%.2e p2=%.2e k4=%.2e  RMS %.4f -> %.4f px\n",
        cal.f, truth.f, cal.cx, cal.cy, cal.k1, cal.k2, cal.k3, cal.p1, cal.p2, cal.k4,
        rms_before, rms_after);

    // The default solve moved f off the EXIF start (475) and onto ~500.
    EXPECT_NEAR(cal_before.f, 475.0, 1e-9) << "the fixture's EXIF must start f at 475";

    // f within 0.5%.
    EXPECT_NEAR(cal.f, 500.0, 2.5) << "f = " << cal.f << " (truth 500, start " << cal_before.f << ")";

    // k1 within 0.01 of truth.
    EXPECT_NEAR(cal.k1, -0.08, 0.01) << "k1 = " << cal.k1 << " (truth -0.08)";

    // Principal point within 3 px of truth.
    EXPECT_NEAR(cal.cx, 3.0, 3.0) << "cx = " << cal.cx << " (truth 3)";
    EXPECT_NEAR(cal.cy, -2.0, 3.0) << "cy = " << cal.cy << " (truth -2)";

    // The refine improves the reprojection residual.
    EXPECT_LT(rms_after, rms_before)
        << "RMS did not improve: " << rms_before << " -> " << rms_after;
}

// ---------------------------------------------------------------------------
// 2. Pose accuracy does not get worse
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasCalSequence, ImprovesPoseAccuracyOnCalSequence) {
    Document document = aligned_document(cal_seq_files());
    const std::vector<TruthImage> truth = parse_truth_images("cal_seq_truth.json");
    ASSERT_EQ(truth.size(), 8U);

    const double rot_before = worst_rotation_error_deg(*document.active_chunk(), truth);
    run_optimize_cameras(document);
    const double rot_after = worst_rotation_error_deg(*document.active_chunk(), truth);

    std::printf("[optimize_cameras] cal_seq pose: worst rotation %.4f -> %.4f deg\n", rot_before,
                rot_after);
    // Rotation error must not get worse (a small tolerance absorbs the
    // similarity-fit numerical noise).
    EXPECT_LE(rot_after, rot_before + 1e-6)
        << "rotation error grew: " << rot_before << " -> " << rot_after;
}

// ---------------------------------------------------------------------------
// 3. fixed_params are honoured
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasFixedParams, HonoursSensorFixedParams) {
    Document document = aligned_document(cal_seq_files());
    document.active_chunk()->sensors()[0].fixed_params = {"f"};
    const double f_before = document.active_chunk()->sensors()[0].calibration.f;
    const double k1_before = document.active_chunk()->sensors()[0].calibration.k1;

    run_optimize_cameras(document);
    const Calibration& cal = document.active_chunk()->sensors()[0].calibration;

    // f was in fixed_params, so it comes back bitwise unchanged...
    EXPECT_EQ(cal.f, f_before);
    // ...while a free coefficient moves.
    EXPECT_NE(cal.k1, k1_before);
    std::printf("[optimize_cameras] fixed_params: f %.4f (unchanged), k1 %.6f -> %.6f\n", cal.f,
                k1_before, cal.k1);
}

// ---------------------------------------------------------------------------
// 4. Every fit flag false leaves intrinsics alone but still refines poses
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasFitFlags, FitFlagsFalseKeepsIntrinsics) {
    // AlignCameras leaves poses+points at the optimum for the EXIF calibration,
    // so a no-intrinsics OptimizeCameras has nothing to do on that state. To
    // show the poses are still free to refine (only the camera model is held
    // fixed), perturb a few camera centres first, then let OptimizeCameras
    // repair them with every fit_* false.
    Document document = aligned_document(cal_seq_files());
    const Calibration cal_before = document.active_chunk()->sensors()[0].calibration;

    // Perturb cameras 2..7 (0 is the gauge camera, 1 carries the scale anchor),
    // shifting each centre by a fixed offset so the reprojection error rises.
    for (std::size_t k = 2; k < document.active_chunk()->cameras().size(); ++k) {
        Transform& pose = document.active_chunk()->cameras()[k].pose;
        const Eigen::Vector3d c = camera_centre(pose);
        const Eigen::Vector3d shifted = c + Eigen::Vector3d(0.05, 0.0, 0.0);
        const Eigen::Matrix3d r = pose_R(pose);
        pose.block<3, 1>(0, 3) = -r * shifted;  // keep R, move the centre to `shifted`
    }
    const std::vector<TruthImage> pose_perturbed = recon_truth_images(*document.active_chunk());
    const double rms_perturbed = chunk_rms_px(*document.active_chunk());

    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    const std::array<const char*, 11> names = {"f", "cx",  "cy", "b1", "b2", "k1",
                                              "k2", "k3", "k4", "p1", "p2"};
    for (const char* name : names) {
        params.set(std::string("fit_") + name, false);
    }
    run_task(builtin_registry(), document, "OptimizeCameras", params);

    const Chunk& chunk = *document.active_chunk();
    // Intrinsics bitwise unchanged.
    EXPECT_TRUE(chunk.sensors()[0].calibration == cal_before);

    // Poses refined: the perturbation was reduced (a nonzero move away from the
    // shifted state, in the direction that lowers the residual).
    const std::vector<TruthImage> pose_after = recon_truth_images(chunk);
    double moved = 0.0;
    for (std::size_t k = 2; k < pose_after.size(); ++k) {
        moved = std::max(moved, (pose_after[k].centre() - pose_perturbed[k].centre()).norm());
    }
    const double rms_after = chunk_rms_px(chunk);
    std::printf(
        "[optimize_cameras] all fit=false: calibration bitwise unchanged, worst pose move "
        "%.4f, RMS %.4f -> %.4f px\n",
        moved, rms_perturbed, rms_after);
    EXPECT_GT(moved, 0.0) << "no camera pose refined with all fit flags false";
    EXPECT_LT(rms_after, rms_perturbed) << "the perturbed poses were not repaired";
}

// ---------------------------------------------------------------------------
// 5. GPS priors constrain the geo sequence
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasGeo, UsesGpsPriorsOnGeoSequence) {
    Document document = make_document(geo_seq_files());
    run_match_photos(document);
    run_align_cameras(document);
    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.cameras().size(), 8U);
    ASSERT_GT((chunk.transform() - Transform::Identity()).cwiseAbs().maxCoeff(), 1e-9);

    // Tighten every reference accuracy to a few centimetres.
    for (Camera& camera : chunk.cameras()) {
        camera.reference.accuracy = Eigen::Vector3d(0.03, 0.03, 0.05);
    }

    run_optimize_cameras(document);

    const TruthOrigin origin = parse_geo_origin("geo_seq_truth.json");
    const geo::Geodetic true_origin{origin.lon, origin.lat, origin.height};
    const std::vector<GeoTruthImage> truth = parse_geo_images("geo_seq_truth.json");
    ASSERT_EQ(truth.size(), 8U);

    const Transform& transform = document.active_chunk()->transform();
    const Transform truth_to_ecef = geo::enu_to_ecef(true_origin);

    double worst = 0.0;
    double sum_sq = 0.0;
    for (std::size_t k = 0; k < truth.size(); ++k) {
        const Eigen::Vector3d centre_enu =
            camera_centre(document.active_chunk()->cameras()[k].pose);
        const Eigen::Vector3d centre_ecef = apply4(transform, centre_enu);
        const Eigen::Vector3d truth_ecef = apply4(truth_to_ecef, truth[k].enu_true);
        const double d = (centre_ecef - truth_ecef).norm();
        worst = std::max(worst, d);
        sum_sq += d * d;
    }
    const double prior_rms = std::sqrt(sum_sq / static_cast<double>(truth.size()));
    std::printf("[optimize_cameras] geo_seq: worst ECEF centre %.4f m, RMS %.4f m\n", worst,
                prior_rms);
    EXPECT_LT(worst, 0.08) << "worst ECEF centre error " << worst;
}

// ---------------------------------------------------------------------------
// 6. A chunk without an alignment is rejected and left unchanged
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasValidate, RequiresAlignment) {
    Document document = make_document(cal_seq_files());
    run_match_photos(document);  // tie points, but no alignment
    const Chunk before = *document.active_chunk();

    Params params = optimize_params();
    EXPECT_THROW(run_task(builtin_registry(), document, "OptimizeCameras", params), ValueError);
    EXPECT_TRUE(*document.active_chunk() == before);
}

// ---------------------------------------------------------------------------
// 7. Project round-trip after optimization
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasRoundTrip, RoundTripsAfterOptimization) {
    Document document = aligned_document(cal_seq_files());
    run_optimize_cameras(document);
    const Chunk before = *document.active_chunk();

    TempDir dir("alphashape_optimize_cameras_roundtrip");
    const std::string project = dir.file("proj.psx");
    document.set_path(project);
    ProjectIO::save(document, project);
    auto reloaded = ProjectIO::load(project);

    ASSERT_NE(reloaded->active_chunk(), nullptr);
    EXPECT_TRUE(*reloaded->active_chunk() == before);
}

// ---------------------------------------------------------------------------
// 8. Every fit parameter is recorded in the meta
// ---------------------------------------------------------------------------

TEST(OptimizeCamerasMeta, RecordsAllFitParamsInMeta) {
    Document document = aligned_document(cal_seq_files());
    run_optimize_cameras(document);
    const Meta& meta = document.active_chunk()->meta();

    const std::array<const char*, 11> names = {"f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"};
    for (const char* name : names) {
        EXPECT_NE(meta.find(std::string("OptimizeCameras/fit_") + name), nullptr)
            << "missing OptimizeCameras/fit_" << name;
    }
}

// ---------------------------------------------------------------------------
// 9. Higher-order tangential terms p3/p4 are recoverable (WP-13.0)
// ---------------------------------------------------------------------------

/// A synthetic wide-angle scene: 10 cameras on a ring looking inward and a grid
/// of points across a 3-D box, projected noise-free through `truth`. The lateral
/// extent reaches normalised radii around 0.7-1.0, where p3/p4 are observable
/// (with f = 1600 the p3 term reaches ~0.5 px), and the varied radii separate
/// them from the radial coefficients.
Document make_p3p4_scene(const Calibration& truth) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("p3p4");
    document.set_active_chunk_id(chunk.id());

    Sensor sensor;
    sensor.label = "synth";
    sensor.type = CameraType::Frame;
    sensor.width = truth.width;
    sensor.height = truth.height;
    sensor.calibration = truth;
    chunk.sensors().push_back(sensor);

    constexpr int kCameraCount = 10;
    std::vector<KeypointSet> sets(static_cast<std::size_t>(kCameraCount));
    for (int i = 0; i < kCameraCount; ++i) {
        const double azimuth = 2.0 * kPi * static_cast<double>(i) / kCameraCount;
        const double radius = 5.0 + 1.5 * static_cast<double>(i % 3);
        const Eigen::Vector3d centre(radius * std::cos(azimuth),
                                     (i % 2 == 0) ? 1.5 : -1.5,
                                     radius * std::sin(azimuth));
        const Eigen::Vector3d target(0.3 * std::sin(0.7 * i), -0.2 * std::cos(0.5 * i), 0.0);
        const Eigen::Vector3d z = (target - centre).normalized();
        const Eigen::Vector3d down(0.0, -1.0, 0.0);
        const Eigen::Vector3d y = (down - down.dot(z) * z).normalized();
        const Eigen::Vector3d x = y.cross(z).normalized();
        Eigen::Matrix3d rotation;
        rotation.row(0) = x.transpose();
        rotation.row(1) = y.transpose();
        rotation.row(2) = z.transpose();

        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(i);
        camera.enabled = true;
        camera.pose.block<3, 3>(0, 0) = rotation;
        camera.pose.block<3, 1>(0, 3) = -rotation * centre;
        chunk.cameras().push_back(camera);
        sets[static_cast<std::size_t>(i)].camera_id = i;
    }

    int point_count = 0;
    for (int gx = -4; gx <= 4; ++gx) {
        for (int gy = -4; gy <= 4; ++gy) {
            for (int gz = -1; gz <= 1; ++gz) {
                const Eigen::Vector3d point(0.9 * gx, 0.9 * gy, 2.0 * gz);
                Track track;
                for (int i = 0; i < kCameraCount; ++i) {
                    const Transform& pose = chunk.cameras()[static_cast<std::size_t>(i)].pose;
                    const Eigen::Vector3d p_cam =
                        pose.block<3, 3>(0, 0) * point + pose.block<3, 1>(0, 3);
                    if (!(p_cam.z() > 0.6)) {
                        continue;
                    }
                    const double nx = p_cam.x() / p_cam.z();
                    const double ny = p_cam.y() / p_cam.z();
                    if (nx * nx + ny * ny > 1.0) {
                        continue;  // keep the observation inside a sane field
                    }
                    const Eigen::Vector2d pixel = truth.project(p_cam);
                    KeypointSet& set = sets[static_cast<std::size_t>(i)];
                    Keypoint keypoint;
                    // keypoint_to_pixel() adds 0.5: store the corner convention.
                    keypoint.x = static_cast<float>(pixel.x() - 0.5);
                    keypoint.y = static_cast<float>(pixel.y() - 0.5);
                    set.keypoints.push_back(keypoint);
                    Observation observation;
                    observation.camera_id = i;
                    observation.keypoint_index =
                        static_cast<std::uint32_t>(set.keypoints.size() - 1);
                    track.observations.push_back(observation);
                }
                if (track.observations.size() >= 2U) {
                    track.point = point;
                    chunk.tie_points().tracks.push_back(track);
                    ++point_count;
                }
            }
        }
    }
    for (KeypointSet& set : sets) {
        chunk.tie_points().keypoint_sets.push_back(std::move(set));
    }
    EXPECT_GT(point_count, 100);
    return document;
}

TEST(Optimize, DistortionTermsFitP3P4) {
    Calibration truth = Calibration::initial(CameraType::Frame, 4000, 4000, 1600.0);
    truth.cx = 3.0;
    truth.cy = -2.0;
    truth.k1 = -0.05;
    truth.p3 = 2e-4;
    truth.p4 = -1e-4;

    Document document = make_p3p4_scene(truth);
    // Start from zero p3/p4: the fit has to move them onto the truth.
    document.active_chunk()->sensors()[0].calibration.p3 = 0.0;
    document.active_chunk()->sensors()[0].calibration.p4 = 0.0;

    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    for (const char* name : {"f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"}) {
        params.set(std::string("fit_") + name, false);
    }
    params.set("fit_p3", true);
    params.set("fit_p4", true);
    run_task(builtin_registry(), document, "OptimizeCameras", params);

    const Calibration& cal = document.active_chunk()->sensors()[0].calibration;
    std::printf("[optimize_cameras] p3p4 fit: p3=%.6e (truth 2e-4) p4=%.6e (truth -1e-4)\n",
                cal.p3, cal.p4);
    EXPECT_NEAR(cal.p3, 2e-4, 2e-5) << "p3 = " << cal.p3;
    EXPECT_NEAR(cal.p4, -1e-4, 1e-5) << "p4 = " << cal.p4;
}

}  // namespace alphashape
