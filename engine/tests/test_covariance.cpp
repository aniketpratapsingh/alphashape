// Posterior covariance tests (WP-13.13a). Suite Covariance.
//
// The covariance bundles compute is sigma0^2 * (J^T J)^-1 for selected blocks.
// The dense-inverse test rebuilds the SAME Jacobian independently by central
// differences over the free parameters and compares every camera and point
// block, so a wrong block accessor or a wrong sigma0 factor cannot hide.

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "common/core/document.hpp"
#include "common/core/types.hpp"
#include "common/tasks/registry.hpp"
#include "model/camera/calibration.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using test_rng::draw2;
using test_rng::draw3;

Eigen::Matrix3d rotation_from_angle_axis(const Eigen::Vector3d& vector) {
    const double angle = vector.norm();
    if (!(angle > 1e-12)) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, vector / angle).toRotationMatrix();
}

Calibration frame_calibration(bool distorted) {
    Calibration calibration;
    calibration.type = CameraType::Frame;
    calibration.width = 640;
    calibration.height = 480;
    calibration.f = 800.0;
    if (distorted) {
        calibration.k1 = -0.05;
        calibration.k2 = 0.01;
        calibration.p1 = 1e-4;
    }
    return calibration;
}

struct Scene {
    BAProblem problem;
    std::vector<Eigen::Vector3d> true_points;
    std::vector<BAPose> true_poses;
};

/// A small, well-conditioned ring: `cameras` stations 5 units out looking at a
/// cube of `points` scene points, every camera seeing every point, plus
/// independent Gaussian pixel noise. `distorted` gives the sensors non-zero
/// radial/tangential truth so the self-calibration terms are identifiable.
Scene make_scene(std::size_t cameras, std::size_t points, double sigma_px,
                 std::mt19937& rng, bool distorted = false) {
    Scene scene;
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    const Calibration calibration = frame_calibration(distorted);
    scene.problem.calibrations.assign(cameras, calibration);

    const auto coordinate = [](std::mt19937& generator) {
        return rng::uniform_real(generator, -1.2, 1.2);
    };
    for (std::size_t p = 0; p < points; ++p) {
        const Eigen::Vector3d point = draw3(rng, coordinate, coordinate, coordinate);
        scene.true_points.push_back(point);
        scene.problem.points.push_back(point);
    }

    const auto noise = [sigma_px](std::mt19937& generator) {
        return rng::normal(generator, 0.0, sigma_px);
    };
    for (std::size_t c = 0; c < cameras; ++c) {
        const double angle = kTwoPi * static_cast<double>(c) / static_cast<double>(cameras);
        const Eigen::Vector3d centre(5.0 * std::cos(angle), 5.0 * std::sin(angle), 1.5);
        const Eigen::Vector3d forward = (-centre).normalized();
        const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();
        const Eigen::Vector3d up = right.cross(forward);
        Eigen::Matrix3d robot;
        robot.col(0) = right;
        robot.col(1) = -up;
        robot.col(2) = forward;
        BAPose pose;
        pose.R = robot.transpose();
        pose.t = -pose.R * centre;
        scene.true_poses.push_back(pose);
        scene.problem.poses.push_back(pose);
    }

    for (std::size_t c = 0; c < cameras; ++c) {
        for (std::size_t p = 0; p < points; ++p) {
            const Eigen::Vector3d point_camera =
                scene.problem.poses[c].R * scene.problem.points[p] +
                scene.problem.poses[c].t;
            BAObservation observation;
            observation.camera = c;
            observation.point = p;
            observation.pixel =
                calibration.project(point_camera) + draw2(rng, noise, noise);
            scene.problem.observations.push_back(observation);
        }
    }
    scene.problem.fixed_pose.assign(scene.problem.poses.size(), 0);
    return scene;
}

/// The free parameters of the same gauge-fixed problem: camera 0 held, the
/// scale-anchor translation component of `anchor_camera` held, all calibrations
/// constant. `x` is the baseline free vector at the solution.
struct DenseModel {
    std::vector<BAPose> base_poses;
    std::vector<Eigen::Vector3d> base_points;
    std::vector<std::array<int, 6>> camera_free;  // ambient component -> index
    std::size_t point_start = 0;
    std::vector<double> x;
};

DenseModel make_dense_model(const BAProblem& solved, std::size_t anchor_camera,
                            int anchor_axis) {
    DenseModel model;
    model.base_poses = solved.poses;
    model.base_points = solved.points;
    model.camera_free.assign(solved.poses.size(), {-1, -1, -1, -1, -1, -1});
    int next = 0;
    for (std::size_t c = 1; c < solved.poses.size(); ++c) {
        const Eigen::AngleAxisd angle_axis(solved.poses[c].R);
        const Eigen::Vector3d omega = angle_axis.angle() * angle_axis.axis();
        for (int k = 0; k < 6; ++k) {
            if (k < 3) {
                model.camera_free[c][static_cast<std::size_t>(k)] = next++;
                model.x.push_back(omega[k]);
            } else {
                const int axis = k - 3;
                if (c == anchor_camera && axis == anchor_axis) {
                    continue;  // SubsetManifold holds this component
                }
                model.camera_free[c][static_cast<std::size_t>(k)] = next++;
                model.x.push_back(solved.poses[c].t[axis]);
            }
        }
    }
    model.point_start = model.x.size();
    for (const Eigen::Vector3d& point : solved.points) {
        for (int k = 0; k < 3; ++k) {
            model.x.push_back(point[k]);
        }
    }
    return model;
}

std::vector<double> dense_residuals(const DenseModel& model,
                                    const std::vector<double>& x,
                                    const BAProblem& problem) {
    std::vector<BAPose> poses = model.base_poses;
    std::vector<Eigen::Vector3d> points = model.base_points;
    for (std::size_t c = 1; c < poses.size(); ++c) {
        const Eigen::AngleAxisd base_angle_axis(model.base_poses[c].R);
        const Eigen::Vector3d base_omega =
            base_angle_axis.angle() * base_angle_axis.axis();
        Eigen::Vector3d omega = base_omega;
        for (int k = 0; k < 3; ++k) {
            const int slot = model.camera_free[c][static_cast<std::size_t>(k)];
            if (slot >= 0) {
                omega[k] = x[static_cast<std::size_t>(slot)];
            }
        }
        poses[c].R = rotation_from_angle_axis(omega);
        for (int k = 3; k < 6; ++k) {
            const int slot = model.camera_free[c][static_cast<std::size_t>(k)];
            if (slot >= 0) {
                poses[c].t[k - 3] = x[static_cast<std::size_t>(slot)];
            }
        }
    }
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (int k = 0; k < 3; ++k) {
            points[p][k] = x[model.point_start + 3 * p + static_cast<std::size_t>(k)];
        }
    }
    std::vector<double> residuals;
    residuals.reserve(2 * problem.observations.size());
    for (const BAObservation& observation : problem.observations) {
        const Calibration& calibration = problem.calibrations[observation.camera];
        const Eigen::Vector3d point_camera =
            poses[observation.camera].R * points[observation.point] +
            poses[observation.camera].t;
        const Eigen::Vector2d residual =
            calibration.project(point_camera) - observation.pixel;
        residuals.push_back(residual.x());
        residuals.push_back(residual.y());
    }
    return residuals;
}

/// Central-difference Jacobian of the residual vector over the free parameters.
Eigen::MatrixXd dense_jacobian(const DenseModel& model, const BAProblem& problem) {
    constexpr double kStep = 1e-6;
    const std::vector<double> base = dense_residuals(model, model.x, problem);
    Eigen::MatrixXd jacobian(
        static_cast<Eigen::Index>(base.size()),
        static_cast<Eigen::Index>(model.x.size()));
    std::vector<double> shifted = model.x;
    for (std::size_t j = 0; j < model.x.size(); ++j) {
        shifted[j] = model.x[j] + kStep;
        const std::vector<double> plus = dense_residuals(model, shifted, problem);
        shifted[j] = model.x[j] - kStep;
        const std::vector<double> minus = dense_residuals(model, shifted, problem);
        shifted[j] = model.x[j];
        for (std::size_t i = 0; i < base.size(); ++i) {
            jacobian(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                (plus[i] - minus[i]) / (2.0 * kStep);
        }
    }
    return jacobian;
}

TEST(Covariance, CameraBlocksMatchDenseInverse) {
    std::mt19937 rng(20261104);
    Scene scene = make_scene(4, 40, 0.1, rng);
    scene.problem.fixed_pose[0] = 1;
    const int anchor_axis = choose_scale_anchor_axis(scene.problem.poses[0],
                                                     scene.problem.poses[1]);
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis = anchor_axis;

    BAOptions options;
    options.loss_scale_px = 0.0;  // plain least squares: no robust reweighting
    options.compute_camera_covariance = true;
    options.compute_point_covariance = true;
    const BAReport report = bundle_adjust(scene.problem, options);
    ASSERT_TRUE(report.usable);
    ASSERT_TRUE(report.covariance_valid) << report.covariance_reason;
    ASSERT_EQ(report.camera_covariance.size(), scene.problem.poses.size());
    ASSERT_EQ(report.point_covariance.size(), scene.problem.points.size());

    const DenseModel model = make_dense_model(scene.problem, 1, anchor_axis);
    const Eigen::MatrixXd jacobian = dense_jacobian(model, scene.problem);
    const Eigen::MatrixXd normal =
        jacobian.transpose() * jacobian;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(normal);
    ASSERT_TRUE(lu.isInvertible());
    const Eigen::MatrixXd dense =
        lu.inverse() * report.sigma0_squared;

    for (std::size_t c = 0; c < scene.problem.poses.size(); ++c) {
        Eigen::Matrix<double, 6, 6> expected = Eigen::Matrix<double, 6, 6>::Zero();
        for (int a = 0; a < 6; ++a) {
            for (int b = 0; b < 6; ++b) {
                const int ia = model.camera_free[c][static_cast<std::size_t>(a)];
                const int ib = model.camera_free[c][static_cast<std::size_t>(b)];
                if (ia >= 0 && ib >= 0) {
                    expected(a, b) = dense(ia, ib);
                }
            }
        }
        const double scale = std::max(report.camera_covariance[c].norm(), 1e-300);
        const double relative =
            (expected - report.camera_covariance[c]).norm() / scale;
        EXPECT_LE(relative, 1e-6) << "camera " << c;
    }

    for (std::size_t p = 0; p < scene.problem.points.size(); ++p) {
        Eigen::Matrix3d expected = Eigen::Matrix3d::Zero();
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const std::size_t ia =
                    model.point_start + 3 * p + static_cast<std::size_t>(a);
                const std::size_t ib =
                    model.point_start + 3 * p + static_cast<std::size_t>(b);
                expected(a, b) = dense(
                    static_cast<Eigen::Index>(ia),
                    static_cast<Eigen::Index>(ib));
            }
        }
        const double scale = std::max(report.point_covariance[p].norm(), 1e-300);
        const double relative =
            (expected - report.point_covariance[p]).norm() / scale;
        EXPECT_LE(relative, 1e-6) << "point " << p;
    }
}

TEST(Covariance, PointCovarianceSymmetricPositiveDefinite) {
    std::mt19937 rng(77001);
    Scene scene = make_scene(4, 30, 0.2, rng);
    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.problem.poses[0], scene.problem.poses[1]);

    BAOptions options;
    options.loss_scale_px = 0.0;
    options.compute_point_covariance = true;
    const BAReport report = bundle_adjust(scene.problem, options);
    ASSERT_TRUE(report.covariance_valid) << report.covariance_reason;
    ASSERT_EQ(report.point_covariance.size(), scene.problem.points.size());

    for (std::size_t p = 0; p < report.point_covariance.size(); ++p) {
        const Eigen::Matrix3d& covariance = report.point_covariance[p];
        const double asymmetry = (covariance - covariance.transpose()).norm();
        EXPECT_LE(asymmetry, 1e-12 * std::max(covariance.norm(), 1e-300))
            << "point " << p;
        const Eigen::LLT<Eigen::Matrix3d> llt(covariance);
        EXPECT_EQ(llt.info(), Eigen::Success) << "point " << p;
        EXPECT_GT(covariance.trace(), 0.0) << "point " << p;
    }
}

TEST(Covariance, FixedCameraHasZeroCovariance) {
    std::mt19937 rng(77002);
    Scene scene = make_scene(4, 25, 0.15, rng);
    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.problem.poses[0], scene.problem.poses[1]);

    BAOptions options;
    options.loss_scale_px = 0.0;
    options.compute_camera_covariance = true;
    const BAReport report = bundle_adjust(scene.problem, options);
    ASSERT_TRUE(report.covariance_valid) << report.covariance_reason;
    ASSERT_EQ(report.camera_covariance.size(), scene.problem.poses.size());

    EXPECT_TRUE(report.camera_covariance[0].isZero(0.0));
    for (std::size_t c = 1; c < report.camera_covariance.size(); ++c) {
        EXPECT_GT(report.camera_covariance[c].trace(), 0.0) << "camera " << c;
    }
}

TEST(Covariance, OffByDefaultIsBitIdentical) {
    std::mt19937 rng(77003);
    Scene scene = make_scene(4, 20, 0.1, rng);
    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.problem.poses[0], scene.problem.poses[1]);

    BAProblem first = scene.problem;
    BAProblem second = scene.problem;
    BAProblem with_covariance = scene.problem;
    BAOptions off;
    off.loss_scale_px = 0.0;
    off.num_threads = 1;  // deterministic reduction order
    BAOptions on = off;
    on.compute_camera_covariance = true;
    on.compute_point_covariance = true;
    on.compute_calibration_covariance = true;

    const BAReport plain = bundle_adjust(first, off);
    const BAReport repeated = bundle_adjust(second, off);
    const BAReport enhanced = bundle_adjust(with_covariance, on);
    ASSERT_TRUE(plain.usable);
    ASSERT_TRUE(enhanced.usable);
    EXPECT_FALSE(plain.covariance_valid);
    EXPECT_TRUE(plain.camera_covariance.empty());
    EXPECT_TRUE(plain.point_covariance.empty());
    EXPECT_TRUE(plain.calibration_covariance.empty());

    for (std::size_t c = 0; c < scene.problem.poses.size(); ++c) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_EQ(first.poses[c].R(i, j), second.poses[c].R(i, j));
                EXPECT_EQ(first.poses[c].R(i, j), with_covariance.poses[c].R(i, j));
            }
            EXPECT_EQ(first.poses[c].t[i], second.poses[c].t[i]);
            EXPECT_EQ(first.poses[c].t[i], with_covariance.poses[c].t[i]);
        }
    }
    for (std::size_t p = 0; p < scene.problem.points.size(); ++p) {
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(first.points[p][k], second.points[p][k]);
            EXPECT_EQ(first.points[p][k], with_covariance.points[p][k]);
        }
    }
    EXPECT_EQ(plain.iterations, repeated.iterations);
    EXPECT_EQ(plain.robust_cost, enhanced.robust_cost);
}

TEST(Covariance, CalibrationCorrelationIsValid) {
    std::mt19937 rng(77004);
    Scene scene = make_scene(4, 60, 0.2, rng, /*distorted=*/true);
    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.problem.poses[0], scene.problem.poses[1]);

    std::array<bool, kCalibrationParamCount> mask = {};
    for (const char* name : {"f", "cx", "cy", "k1", "k2"}) {
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            if (std::string(name) == kCalibrationParamNames[static_cast<std::size_t>(k)]) {
                mask[static_cast<std::size_t>(k)] = true;
            }
        }
    }
    scene.problem.fit_intrinsics.assign(scene.problem.calibrations.size(), mask);

    BAOptions options;
    options.loss_scale_px = 0.0;
    options.compute_calibration_covariance = true;
    const BAReport report = bundle_adjust(scene.problem, options);
    ASSERT_TRUE(report.covariance_valid) << report.covariance_reason;
    ASSERT_EQ(report.calibration_covariance.size(),
              scene.problem.calibrations.size());

    for (const Eigen::MatrixXd& covariance : report.calibration_covariance) {
        ASSERT_EQ(covariance.rows(), static_cast<Eigen::Index>(kCalibrationParamCount));
        ASSERT_EQ(covariance.cols(), static_cast<Eigen::Index>(kCalibrationParamCount));
        for (int i = 0; i < kCalibrationParamCount; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            if (!mask[ii]) {
                EXPECT_TRUE(covariance.row(i).isZero(1e-12)) << "held row " << i;
                EXPECT_TRUE(covariance.col(i).isZero(1e-12)) << "held col " << i;
                continue;
            }
            ASSERT_GT(covariance(i, i), 0.0) << "free term " << i;
            for (int j = 0; j < kCalibrationParamCount; ++j) {
                const std::size_t jj = static_cast<std::size_t>(j);
                if (!mask[jj]) {
                    continue;
                }
                const double rho =
                    covariance(i, j) / std::sqrt(covariance(i, i) * covariance(j, j));
                EXPECT_LE(std::abs(rho), 1.0 + 1e-12) << i << "," << j;
                if (i == j) {
                    EXPECT_NEAR(rho, 1.0, 1e-12);
                }
            }
        }
    }
}

TEST(Covariance, ScalesWithNoise) {
    double trace[2] = {0.0, 0.0};
    for (int sample = 0; sample < 2; ++sample) {
        std::mt19937 rng(77005);
        const double sigma_px = sample == 0 ? 0.1 : 0.2;
        Scene scene = make_scene(4, 30, sigma_px, rng);
        scene.problem.fixed_pose[0] = 1;
        scene.problem.scale_anchor_camera = 1;
        scene.problem.scale_anchor_axis =
            choose_scale_anchor_axis(scene.problem.poses[0], scene.problem.poses[1]);

        BAOptions options;
        options.loss_scale_px = 0.0;
        options.compute_camera_covariance = true;
        const BAReport report = bundle_adjust(scene.problem, options);
        ASSERT_TRUE(report.covariance_valid) << report.covariance_reason;
        for (const Eigen::Matrix<double, 6, 6>& covariance :
             report.camera_covariance) {
            trace[sample] += covariance.trace();
        }
    }
    ASSERT_GT(trace[0], 0.0);
    const double ratio = trace[1] / trace[0];
    EXPECT_GE(ratio, 3.0);
    EXPECT_LE(ratio, 5.0);
}

// ---------------------------------------------------------------------------
// The task adapter: OptimizeCameras must store the covariance on the chunk.
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

std::vector<std::string> cal_seq_files() {
    std::vector<std::string> files;
    for (int i = 0; i < 8; ++i) {
        files.push_back(fixture("cal_seq_" + std::to_string(i) + ".jpg"));
    }
    return files;
}

void run_named(Document& document, const std::string& name, Params& params) {
    run_task(builtin_registry(), document, name, params);
}

Document pipeline_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    Params add = make_params(builtin_registry().require("AddPhotos"));
    add.set("filenames", ParamValue{cal_seq_files()});
    run_named(document, "AddPhotos", add);
    Params match = make_params(builtin_registry().require("MatchPhotos"));
    run_named(document, "MatchPhotos", match);
    Params align = make_params(builtin_registry().require("AlignCameras"));
    run_named(document, "AlignCameras", align);
    return document;
}

TEST(Covariance, OptimizeCamerasStoresCovariance) {
    Document document = pipeline_document();
    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    params.set("tiepoint_covariance", true);
    run_named(document, "OptimizeCameras", params);

    Chunk& chunk = *document.active_chunk();
    const std::string* reason = chunk.meta().find("optimize_cameras.covariance");
    ASSERT_NE(reason, nullptr);
    EXPECT_EQ(*reason, "ok");
    const std::string* sigma0 = chunk.meta().find("optimize_cameras.sigma0");
    ASSERT_NE(sigma0, nullptr);
    EXPECT_TRUE(std::isfinite(std::stod(*sigma0)));

    std::size_t enabled = 0;
    std::size_t positive = 0;
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;
        }
        ++enabled;
        const Eigen::Matrix3d& covariance = camera.location_covariance;
        const Eigen::Matrix3d& rotation = camera.rotation_covariance;
        const double asymmetry = (covariance - covariance.transpose()).norm();
        EXPECT_LE(asymmetry, 1e-12 * std::max(covariance.norm(), 1.0));
        EXPECT_GE(covariance.trace(), 0.0);
        if (covariance.trace() > 0.0) {
            ++positive;
            EXPECT_EQ(Eigen::LLT<Eigen::Matrix3d>(covariance).info(), Eigen::Success);
            EXPECT_GT(rotation.trace(), 0.0);
        }
    }
    EXPECT_GE(enabled, 2U);
    EXPECT_GE(positive, enabled - 1U);

    std::size_t calibrated = 0;
    for (const Sensor& sensor : chunk.sensors()) {
        if (sensor.calibration.covariance.size() == 0) {
            continue;
        }
        ++calibrated;
        EXPECT_EQ(sensor.calibration.covariance.rows(),
                  static_cast<Eigen::Index>(kCalibrationParamCount));
        EXPECT_EQ(sensor.calibration.covariance_params.size(),
                  static_cast<std::size_t>(kCalibrationParamCount));
    }
    EXPECT_GE(calibrated, 1U);
}

}  // namespace
}  // namespace alphashape::sfm

