// Camera model: forward projection, inverse projection, and the closed forms.
//
// Suites are prefixed CameraModel so `ctest -R CameraModel` selects them.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

/// A realistic Frame calibration: mild barrel distortion, tangential terms,
/// a non-zero principal-point offset and a non-orthogonal/skewed sensor.
Calibration realistic_frame() {
    Calibration c = Calibration::initial(CameraType::Frame, 4000, 3000, 3000.0);
    c.cx = 12.0;
    c.cy = -7.0;
    c.b1 = 2.0;
    c.b2 = 0.5;
    c.k1 = -0.1;
    c.k2 = 0.05;
    c.p1 = 1e-4;
    return c;
}

/// A wide fisheye with a mild distortion so the inverse stays well-posed.
Calibration realistic_fisheye() {
    Calibration c = Calibration::initial(CameraType::Fisheye, 4000, 3000, 1000.0);
    c.cx = 8.0;
    c.cy = -5.0;
    c.b1 = 1.5;
    c.b2 = 0.25;
    c.k1 = -0.02;
    c.k2 = 0.005;
    c.p1 = 5e-5;
    c.p2 = -3e-5;
    return c;
}

std::vector<Eigen::Vector2d> pixel_grid(int w, int h, int steps) {
    std::vector<Eigen::Vector2d> pixels;
    for (int iy = 0; iy <= steps; ++iy) {
        for (int ix = 0; ix <= steps; ++ix) {
            const double x = 0.5 + (static_cast<double>(w) - 1.0) *
                                       static_cast<double>(ix) / static_cast<double>(steps);
            const double y = 0.5 + (static_cast<double>(h) - 1.0) *
                                       static_cast<double>(iy) / static_cast<double>(steps);
            pixels.emplace_back(x, y);
        }
    }
    return pixels;
}

// -- initial() ---------------------------------------------------------------

TEST(CameraModelInitial, ZeroesEveryDistortionTerm) {
    const Calibration c = Calibration::initial(CameraType::Frame, 640, 480, 700.0);
    EXPECT_EQ(c.type, CameraType::Frame);
    EXPECT_EQ(c.width, 640);
    EXPECT_EQ(c.height, 480);
    EXPECT_DOUBLE_EQ(c.f, 700.0);
    EXPECT_DOUBLE_EQ(c.cx, 0.0);
    EXPECT_DOUBLE_EQ(c.cy, 0.0);
    EXPECT_DOUBLE_EQ(c.b1, 0.0);
    EXPECT_DOUBLE_EQ(c.b2, 0.0);
    EXPECT_DOUBLE_EQ(c.k1, 0.0);
    EXPECT_DOUBLE_EQ(c.k2, 0.0);
    EXPECT_DOUBLE_EQ(c.k3, 0.0);
    EXPECT_DOUBLE_EQ(c.k4, 0.0);
    EXPECT_DOUBLE_EQ(c.p1, 0.0);
    EXPECT_DOUBLE_EQ(c.p2, 0.0);
}

TEST(CameraModelInitial, EqualsItselfAndDiffersWhenChanged) {
    const Calibration a = Calibration::initial(CameraType::Fisheye, 100, 50, 60.0);
    const Calibration b = Calibration::initial(CameraType::Fisheye, 100, 50, 60.0);
    EXPECT_TRUE(a == b);
    Calibration c = a;
    c.k1 = 1e-9;
    EXPECT_FALSE(a == c);
    Calibration d = a;
    d.type = CameraType::Frame;
    EXPECT_FALSE(a == d);
}

// -- closed forms ------------------------------------------------------------

TEST(CameraModelPinhole, ZeroDistortionMatchesClosedForm) {
    const Calibration c = Calibration::initial(CameraType::Frame, 4000, 3000, 2500.0);
    const Eigen::Vector3d p(0.3, -0.2, 2.0);
    const double x = p.x() / p.z();
    const double y = p.y() / p.z();
    const Eigen::Vector2d got = c.project(p);
    // u = W/2 + cx + x f,  v = H/2 + cy + y f  with every distortion term zero.
    EXPECT_DOUBLE_EQ(got.x(), 2000.0 + 0.0 + x * 2500.0);
    EXPECT_DOUBLE_EQ(got.y(), 1500.0 + 0.0 + y * 2500.0);
}

TEST(CameraModelPinhole, PrincipalPointLandsAtCentrePlusOffset) {
    Calibration c = Calibration::initial(CameraType::Frame, 4000, 3000, 2500.0);
    c.cx = 12.0;
    c.cy = -7.0;
    const Eigen::Vector2d got = c.project(Eigen::Vector3d(0.0, 0.0, 1.0));
    EXPECT_DOUBLE_EQ(got.x(), 2000.0 + 12.0);
    EXPECT_DOUBLE_EQ(got.y(), 1500.0 - 7.0);
}

TEST(CameraModelPinhole, UnprojectReturnsUnitRay) {
    const Calibration c = Calibration::initial(CameraType::Frame, 4000, 3000, 2500.0);
    const Eigen::Vector3d ray = c.unproject(Eigen::Vector2d(1234.5, 987.25));
    EXPECT_NEAR(ray.norm(), 1.0, 1e-15);
    EXPECT_GT(ray.z(), 0.0);
}

// -- Frame round-trip --------------------------------------------------------

TEST(CameraModelFrame, ProjectUnprojectRoundTripOverGrid) {
    const Calibration c = realistic_frame();
    for (const Eigen::Vector2d& pixel : pixel_grid(c.width, c.height, 12)) {
        const Eigen::Vector3d ray = c.unproject(pixel);
        EXPECT_NEAR(ray.norm(), 1.0, 1e-12);
        // A Frame point must be in front of the camera; scale the ray up.
        const Eigen::Vector3d p = ray * 5.0;
        const Eigen::Vector2d back = c.project(p);
        EXPECT_LT((back - pixel).norm(), 1e-6) << "pixel (" << pixel.x() << ", " << pixel.y() << ")";
    }
}

TEST(CameraModelFrame, ProjectRejectsPointsAtOrBehindImagePlane) {
    const Calibration c = realistic_frame();
    EXPECT_THROW(c.project(Eigen::Vector3d(0.1, 0.1, 0.0)), ValueError);
    EXPECT_THROW(c.project(Eigen::Vector3d(0.1, 0.1, -1.0)), ValueError);
}

/// Pack a calibration in the order try_project_point / project_point expect.
void pack(const Calibration& c, double* params) {
    params[0] = c.f;
    params[1] = c.cx;
    params[2] = c.cy;
    params[3] = c.b1;
    params[4] = c.b2;
    params[5] = c.k1;
    params[6] = c.k2;
    params[7] = c.k3;
    params[8] = c.k4;
    params[9] = c.p1;
    params[10] = c.p2;
    params[11] = c.p3;
    params[12] = c.p4;
}

TEST(CameraModelFrame, ProjectPointTemplateMatchesProject) {
    const Calibration c = realistic_frame();
    double params[kCalibrationParamCount];
    pack(c, params);
    const std::vector<Eigen::Vector3d> points = {
        Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d(0.4, -0.3, 1.7),
        Eigen::Vector3d(-0.9, 0.6, 3.2)};
    for (const Eigen::Vector3d& p : points) {
        double out[2] = {0.0, 0.0};
        project_point<double>(c.type, c.width, c.height, params, p.data(), out);
        const Eigen::Vector2d expected = c.project(p);
        EXPECT_DOUBLE_EQ(out[0], expected.x());
        EXPECT_DOUBLE_EQ(out[1], expected.y());
    }
}

// -- try_project_point: the throw-free forward model -------------------------

/// A deterministic pseudo-random camera-frame point. The generator is a plain
/// splitmix64 so the test is reproducible without a library dependency.
Eigen::Vector3d random_point(std::uint64_t& state) {
    const auto next = [&state]() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t x = state;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x ^= x >> 31;
        // Map to [-2, 2) via the top 53 bits.
        return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) * 4.0 - 2.0;
    };
    return Eigen::Vector3d(next(), next(), next());
}

/// try_project_point must agree with project_point bit for bit wherever the
/// model is defined, and return false exactly where project_point throws.
/// Checked on random points for a distorted Frame and a distorted Fisheye, plus
/// deliberately-problematic points (on the plane, behind it, at the origin).
void check_try_matches_project(const Calibration& c) {
    double params[kCalibrationParamCount];
    pack(c, params);

    std::vector<Eigen::Vector3d> points = {
        Eigen::Vector3d(0.0, 0.0, 1.0),     // on axis
        Eigen::Vector3d(0.0, 0.0, 0.0),     // the origin (Fisheye: undefined)
        Eigen::Vector3d(0.3, -0.2, 0.0),    // exactly on the image plane
        Eigen::Vector3d(0.3, -0.2, -1.5),   // behind the image plane
    };
    std::uint64_t state = 0xC0FFEEULL;
    for (int i = 0; i < 4000; ++i) {
        points.push_back(random_point(state));
    }

    int defined = 0, undefined = 0;
    for (const Eigen::Vector3d& p : points) {
        bool threw = false;
        Eigen::Vector2d expected;
        try {
            expected = c.project(p);
        } catch (const ValueError&) {
            threw = true;
        }

        double out[2] = {-999.0, -999.0};
        const bool ok = try_project_point<double>(c.type, c.width, c.height, params, p.data(), out);
        ASSERT_EQ(ok, !threw) << "return/throw disagree for (" << p.transpose() << ")";

        if (ok) {
            ++defined;
            // Bit-exact: the same expression flow, so EXPECT_DOUBLE_EQ is right.
            EXPECT_DOUBLE_EQ(out[0], expected.x()) << "u differs for (" << p.transpose() << ")";
            EXPECT_DOUBLE_EQ(out[1], expected.y()) << "v differs for (" << p.transpose() << ")";
        } else {
            ++undefined;
            // On a false return the caller's buffer must be left untouched.
            EXPECT_DOUBLE_EQ(out[0], -999.0);
            EXPECT_DOUBLE_EQ(out[1], -999.0);
        }
    }
    // Both outcomes must actually occur on this sample set, or the test is vacuous.
    EXPECT_GT(defined, 0);
    if (c.type == CameraType::Fisheye) {
        EXPECT_GT(undefined, 0) << "no origin/undefined point in the sample set";
    } else {
        EXPECT_GT(undefined, 0) << "no behind-plane point in the sample set";
    }
}

TEST(CameraModelTryProject, MatchesProjectBitExactlyFrameWithDistortion) {
    check_try_matches_project(realistic_frame());
}

TEST(CameraModelTryProject, MatchesProjectBitExactlyFisheye) {
    check_try_matches_project(realistic_fisheye());
}

TEST(CameraModelTryProject, ReturnsFalseExactlyWhereProjectThrowsFrame) {
    const Calibration c = realistic_frame();
    double params[kCalibrationParamCount];
    pack(c, params);
    double out[2] = {0.0, 0.0};
    EXPECT_FALSE(try_project_point<double>(c.type, c.width, c.height, params,
                                           Eigen::Vector3d(0.1, 0.1, 0.0).data(), out));
    EXPECT_FALSE(try_project_point<double>(c.type, c.width, c.height, params,
                                           Eigen::Vector3d(0.1, 0.1, -2.0).data(), out));
    EXPECT_TRUE(try_project_point<double>(c.type, c.width, c.height, params,
                                          Eigen::Vector3d(0.1, 0.1, 2.0).data(), out));
}

TEST(CameraModelTryProject, ReturnsFalseOnlyAtTheOriginForFisheye) {
    const Calibration c = realistic_fisheye();
    double params[kCalibrationParamCount];
    pack(c, params);
    double out[2] = {0.0, 0.0};
    EXPECT_FALSE(try_project_point<double>(c.type, c.width, c.height, params,
                                           Eigen::Vector3d(0.0, 0.0, 0.0).data(), out));
    // Behind the image plane is valid for a fisheye (up to the antipode).
    EXPECT_TRUE(try_project_point<double>(c.type, c.width, c.height, params,
                                          Eigen::Vector3d(1.0, 0.0, -1.0).data(), out));
}

// -- Fisheye -----------------------------------------------------------------

TEST(CameraModelFisheye, ProjectUnprojectRoundTripOverGrid) {
    const Calibration c = realistic_fisheye();
    for (const Eigen::Vector2d& pixel : pixel_grid(c.width, c.height, 12)) {
        const Eigen::Vector3d ray = c.unproject(pixel);
        EXPECT_NEAR(ray.norm(), 1.0, 1e-12);
        const Eigen::Vector2d back = c.project(ray);
        EXPECT_LT((back - pixel).norm(), 1e-6) << "pixel (" << pixel.x() << ", " << pixel.y() << ")";
    }
}

TEST(CameraModelFisheye, RoundTripBeyondNinetyDegreesOffAxis) {
    const Calibration c = realistic_fisheye();
    for (const double theta : {1.8, 2.0, 2.4, 2.8}) {  // 103 deg .. 160 deg
        for (const double phi : {-2.0, -0.5, 0.0, 0.7, 2.5}) {
            const Eigen::Vector3d ray(std::sin(theta) * std::cos(phi),
                                      std::sin(theta) * std::sin(phi), std::cos(theta));
            const Eigen::Vector2d pixel = c.project(ray);
            const Eigen::Vector3d back = c.unproject(pixel);
            // Compare directions: a dot product of 1 means the same ray.
            EXPECT_NEAR(ray.normalized().dot(back), 1.0, 1e-9)
                << "theta=" << theta << " phi=" << phi;
        }
    }
}

TEST(CameraModelFisheye, ProjectAcceptsPointsBehindTheImagePlane) {
    const Calibration c = realistic_fisheye();
    // 150 degrees off axis: z is negative but theta < pi, so this is valid.
    const double theta = 2.6;
    const Eigen::Vector3d ray(std::sin(theta), 0.0, std::cos(theta));
    // Initialised before the throwing call: g++'s inliner reports the
    // assignment inside EXPECT_NO_THROW as maybe-uninitialized otherwise.
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    EXPECT_NO_THROW(pixel = c.project(ray));
    EXPECT_TRUE(pixel.allFinite());
}

TEST(CameraModelFisheye, ProjectKeepsForwardPointOnAxis) {
    const Calibration c = realistic_fisheye();
    // A point exactly on the optical axis projects to the principal point.
    const Eigen::Vector2d got = c.project(Eigen::Vector3d(0.0, 0.0, 1.0));
    EXPECT_DOUBLE_EQ(got.x(), 2000.0 + 8.0);
    EXPECT_DOUBLE_EQ(got.y(), 1500.0 - 5.0);
}

// -- scaled_calibration ------------------------------------------------------

/// A distorted frame calibration whose width and height are NOT divisible by 4,
/// so the principal-point rounding and the integer dimensions are both exercised.
Calibration odd_frame() {
    Calibration c = Calibration::initial(CameraType::Frame, 481, 359, 400.0);
    c.cx = 12.0;
    c.cy = -7.0;
    c.b1 = 2.0;
    c.b2 = 0.5;
    c.k1 = -0.1;
    c.k2 = 0.05;
    c.p1 = 1e-4;
    c.p2 = -3e-5;
    return c;
}

TEST(CameraModelScaled, ProjectionScalesExactly) {
    const Calibration full = odd_frame();
    const int ds = 4;
    const Calibration scaled = scaled_calibration(full, ds);

    EXPECT_EQ(scaled.width, full.width / ds);
    EXPECT_EQ(scaled.height, full.height / ds);
    EXPECT_TRUE(scaled.type == full.type);
    // Distortion is resolution independent.
    EXPECT_DOUBLE_EQ(scaled.k1, full.k1);
    EXPECT_DOUBLE_EQ(scaled.k2, full.k2);
    EXPECT_DOUBLE_EQ(scaled.p1, full.p1);
    EXPECT_DOUBLE_EQ(scaled.p2, full.p2);

    // Points in front of both cameras: u under the reduced model must equal u/ds
    // (the top-left pixel centre convention makes this exact).
    double max_err = 0.0;
    for (int iy = 1; iy <= 8; ++iy) {
        for (int ix = 1; ix <= 8; ++ix) {
            const Eigen::Vector3d p(0.15 * ix - 0.7, 0.15 * iy - 0.7, 1.0);
            const Eigen::Vector2d u_full = full.project(p);
            const Eigen::Vector2d u_scaled = scaled.project(p);
            max_err = std::max(max_err, (u_scaled - u_full / static_cast<double>(ds)).norm());
        }
    }
    EXPECT_LT(max_err, 1e-9);
}

TEST(CameraModelScaled, DownscaleOneIsIdentity) {
    const Calibration full = odd_frame();
    EXPECT_TRUE(scaled_calibration(full, 1) == full);
}

TEST(CameraModelScaled, PrincipalPointLandsExactly) {
    // The absolute principal point (W/2 + cx) must map to (W/2 + cx)/ds.
    const Calibration full = odd_frame();
    const int ds = 4;
    const Calibration scaled = scaled_calibration(full, ds);
    const double abs_x = static_cast<double>(full.width) * 0.5 + full.cx;
    const double abs_y = static_cast<double>(full.height) * 0.5 + full.cy;
    const double scaled_abs_x =
        static_cast<double>(scaled.width) * 0.5 + scaled.cx;
    const double scaled_abs_y =
        static_cast<double>(scaled.height) * 0.5 + scaled.cy;
    EXPECT_DOUBLE_EQ(scaled_abs_x, abs_x / static_cast<double>(ds));
    EXPECT_DOUBLE_EQ(scaled_abs_y, abs_y / static_cast<double>(ds));
}

TEST(CameraModelScaled, NonPowerOfTwoThrows) {
    const Calibration full = odd_frame();
    EXPECT_THROW(scaled_calibration(full, 3), ValueError);
    EXPECT_THROW(scaled_calibration(full, 0), ValueError);
    EXPECT_THROW(scaled_calibration(full, -2), ValueError);
}

// -- p3 / p4 (WP-13.0) -------------------------------------------------------

/// A tiny forward-mode autodiff scalar over the 13 calibration parameters, so
/// the templated forward model can be differentiated without pulling Ceres into
/// this translation unit.
struct Dual13 {
    double v = 0.0;
    std::array<double, kCalibrationParamCount> d{};

    Dual13() = default;
    Dual13(double value) : v(value) {}  // NOLINT(google-explicit-constructor)
};

Dual13 operator+(const Dual13& a, const Dual13& b) {
    Dual13 r(a.v + b.v);
    for (std::size_t i = 0; i < r.d.size(); ++i) r.d[i] = a.d[i] + b.d[i];
    return r;
}
Dual13 operator*(const Dual13& a, const Dual13& b) {
    Dual13 r(a.v * b.v);
    for (std::size_t i = 0; i < r.d.size(); ++i) {
        r.d[i] = a.d[i] * b.v + a.v * b.d[i];
    }
    return r;
}
Dual13 operator/(const Dual13& a, const Dual13& b) {
    Dual13 r(a.v / b.v);
    const double inv = 1.0 / (b.v * b.v);
    for (std::size_t i = 0; i < r.d.size(); ++i) {
        r.d[i] = (a.d[i] * b.v - a.v * b.d[i]) * inv;
    }
    return r;
}
bool operator==(const Dual13& a, const Dual13& b) { return a.v == b.v; }
bool operator>(const Dual13& a, const Dual13& b) { return a.v > b.v; }
Dual13 sqrt(const Dual13& a) {
    const double s = std::sqrt(a.v);
    Dual13 r(s);
    for (std::size_t i = 0; i < r.d.size(); ++i) {
        r.d[i] = a.d[i] / (2.0 * s);
    }
    return r;
}
Dual13 atan2(const Dual13& y, const Dual13& x) {
    const double denom = x.v * x.v + y.v * y.v;
    Dual13 r(std::atan2(y.v, x.v));
    for (std::size_t i = 0; i < r.d.size(); ++i) {
        r.d[i] = (x.v * y.d[i] - y.v * x.d[i]) / denom;
    }
    return r;
}

TEST(CameraModel, P3P4DefaultZeroLeavesProjectionUnchanged) {
    // With p3 = p4 = 0 the folded q1/q2 form must reproduce the historical
    // Brown-Conrady expression bit for bit over a large random sample.
    Calibration c = realistic_frame();
    c.k3 = 0.011;
    c.k4 = -0.002;
    c.p3 = 0.0;
    c.p4 = 0.0;
    std::uint64_t state = 0x5EED1234ULL;
    int checked = 0;
    for (int i = 0; i < 10000; ++i) {
        Eigen::Vector3d p = random_point(state);
        p.z() = 2.0 + std::abs(p.z());  // depth in [2, 4): a sane field of view
        const double x = p.x() / p.z();
        const double y = p.y() / p.z();
        const double r2 = x * x + y * y;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double radial = 1.0 + c.k1 * r2 + c.k2 * r4 + c.k3 * r6 + c.k4 * r8;
        const double xd = x * radial + c.p1 * (r2 + 2.0 * x * x) + 2.0 * c.p2 * x * y;
        const double yd = y * radial + c.p2 * (r2 + 2.0 * y * y) + 2.0 * c.p1 * x * y;
        const double u = c.width * 0.5 + c.cx + xd * (c.f + c.b1) + yd * c.b2;
        const double v = c.height * 0.5 + c.cy + yd * c.f;
        const Eigen::Vector2d got = c.project(p);
        EXPECT_EQ(got.x(), u) << "u differs at (" << p.transpose() << ")";
        EXPECT_EQ(got.y(), v) << "v differs at (" << p.transpose() << ")";
        ++checked;
    }
    EXPECT_EQ(checked, 10000);
}

/// main's invert_distortion, kept as an independent reference: Newton on the
/// historical p1/p2 Jacobian, then the fixed-point fallback, retaining the
/// estimate with the smallest residual. p3/p4 did not exist there, so the
/// p3 = p4 = 0 inverse must reproduce this bit for bit.
Eigen::Vector2d reference_invert(const Calibration& c, double xd, double yd) {
    constexpr int kMaxIter = 20;
    constexpr double kTol = 1e-12;
    double params[kCalibrationParamCount];
    pack(c, params);
    params[0] = 1.0;   // f
    params[1] = 0.0;   // cx
    params[2] = 0.0;   // cy
    params[3] = 0.0;   // b1
    params[4] = 0.0;   // b2
    params[11] = 0.0;  // p3
    params[12] = 0.0;  // p4
    const auto eval = [&params](double x, double y, double& ox, double& oy) {
        const double p_in[3] = {x, y, 1.0};
        double out[2];
        project_point<double>(CameraType::Frame, 0, 0, params, p_in, out);
        ox = out[0];
        oy = out[1];
    };
    const double k1 = params[5], k2 = params[6], k3 = params[7], k4 = params[8];
    const double p1 = params[9], p2 = params[10];

    double nx = xd;
    double ny = yd;
    double best_x = nx;
    double best_y = ny;
    double best_residual = std::numeric_limits<double>::infinity();
    for (int i = 0; i < kMaxIter; ++i) {
        const double r2 = nx * nx + ny * ny;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double s = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        const double q = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
        double fx_model = 0.0;
        double fy_model = 0.0;
        eval(nx, ny, fx_model, fy_model);
        const double fx = fx_model - xd;
        const double fy = fy_model - yd;
        const double residual = std::hypot(fx, fy);
        if (residual < best_residual) {
            best_residual = residual;
            best_x = nx;
            best_y = ny;
        }
        if (residual < kTol) {
            return Eigen::Vector2d(nx, ny);
        }
        const double j00 = s + 2.0 * nx * nx * q + 6.0 * p1 * nx + 2.0 * p2 * ny;
        const double j01 = 2.0 * nx * ny * q + 2.0 * p1 * ny + 2.0 * p2 * nx;
        const double j10 = 2.0 * nx * ny * q + 2.0 * p2 * nx + 2.0 * p1 * ny;
        const double j11 = s + 2.0 * ny * ny * q + 6.0 * p2 * ny + 2.0 * p1 * nx;
        const double det = j00 * j11 - j01 * j10;
        if (std::abs(det) < 1e-300) {
            break;
        }
        nx -= (j11 * fx - j01 * fy) / det;
        ny -= (j00 * fy - j10 * fx) / det;
    }

    double px = xd;
    double py = yd;
    for (int i = 0; i < kMaxIter; ++i) {
        const double r2 = px * px + py * py;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double s = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        if (std::abs(s) < 1e-300) {
            break;
        }
        const double tx = p1 * (r2 + 2.0 * px * px) + 2.0 * p2 * px * py;
        const double ty = p2 * (r2 + 2.0 * py * py) + 2.0 * p1 * px * py;
        const double ox = (xd - tx) / s;
        const double oy = (yd - ty) / s;
        double exd = 0.0;
        double eyd = 0.0;
        eval(ox, oy, exd, eyd);
        const double residual = std::hypot(exd - xd, eyd - yd);
        if (residual < best_residual) {
            best_residual = residual;
            best_x = ox;
            best_y = oy;
        }
        const bool settled = std::abs(ox - px) < kTol && std::abs(oy - py) < kTol;
        px = ox;
        py = oy;
        if (settled || residual < kTol) {
            break;
        }
    }
    return Eigen::Vector2d(best_x, best_y);
}

TEST(CameraModel, P3P4ZeroUndistortBitIdentical) {
    // With p3 = p4 = 0 the inverse must reproduce main's Jacobian-based
    // iteration bit for bit: the new terms are skipped entirely, not added as
    // zero, so no rounding (or signed-zero) difference can creep in.
    Calibration c = realistic_frame();
    c.k3 = 0.011;
    c.k4 = -0.002;
    c.p3 = 0.0;
    c.p4 = 0.0;
    const double half_w = c.width * 0.5;
    const double half_h = c.height * 0.5;
    std::uint64_t state = 0x1234ABCDULL;
    const auto next = [&state]() {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t x = state;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    };
    int checked = 0;
    for (int i = 0; i < 10000; ++i) {
        const double u = next() * c.width;
        const double v = next() * c.height;
        const double yd = (v - half_h - c.cy) / c.f;
        const double xd = (u - half_w - c.cx - yd * c.b2) / (c.f + c.b1);
        const Eigen::Vector2d ref = reference_invert(c, xd, yd);
        const Eigen::Vector3d want = Eigen::Vector3d(ref.x(), ref.y(), 1.0).normalized();
        const Eigen::Vector3d got = c.unproject(Eigen::Vector2d(u, v));
        EXPECT_EQ(got.x(), want.x()) << "pixel (" << u << ", " << v << ")";
        EXPECT_EQ(got.y(), want.y()) << "pixel (" << u << ", " << v << ")";
        EXPECT_EQ(got.z(), want.z()) << "pixel (" << u << ", " << v << ")";
        ++checked;
    }
    EXPECT_EQ(checked, 10000);
}

TEST(CameraModel, P3P4FiniteDifferenceGradientMatchesAutodiff) {
    Calibration c = realistic_frame();
    c.k3 = 0.011;
    c.p3 = 2e-4;
    c.p4 = -1e-4;
    double base[kCalibrationParamCount];
    pack(c, base);

    std::array<Dual13, kCalibrationParamCount> params{};
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        params[static_cast<std::size_t>(i)] = Dual13(base[i]);
    }
    // Seed the two coefficients under test; the function is exactly linear in
    // each, so the forward-mode derivative is the analytic one.
    constexpr std::size_t p3_index = 11;
    constexpr std::size_t p4_index = 12;
    params[p3_index].d[p3_index] = 1.0;
    params[p4_index].d[p4_index] = 1.0;

    const std::vector<Eigen::Vector3d> points = {
        Eigen::Vector3d(1.0, 0.7, 1.0),  Eigen::Vector3d(-0.9, 0.8, 1.2),
        Eigen::Vector3d(0.6, -1.1, 1.4), Eigen::Vector3d(-1.2, -0.5, 1.6)};
    const double h = 1e-5;  // linear in p3/p4: no truncation error
    for (const Eigen::Vector3d& p : points) {
        const std::array<Dual13, 3> p_cam{Dual13(p.x()), Dual13(p.y()), Dual13(p.z())};
        std::array<Dual13, 2> dual_pixel{};
        ASSERT_TRUE(try_project_point<Dual13>(c.type, c.width, c.height, params.data(),
                                              p_cam.data(), dual_pixel.data()));

        for (int index = 11; index <= 12; ++index) {
            double plus[kCalibrationParamCount];
            double minus[kCalibrationParamCount];
            for (int i = 0; i < kCalibrationParamCount; ++i) {
                plus[i] = base[i];
                minus[i] = base[i];
            }
            plus[index] += h;
            minus[index] -= h;
            double up[2] = {0.0, 0.0};
            double um[2] = {0.0, 0.0};
            ASSERT_TRUE(try_project_point<double>(c.type, c.width, c.height, plus, p.data(), up));
            ASSERT_TRUE(try_project_point<double>(c.type, c.width, c.height, minus, p.data(), um));
            for (int out = 0; out < 2; ++out) {
                const double fd = (up[out] - um[out]) / (2.0 * h);
                const double ad =
                    dual_pixel[static_cast<std::size_t>(out)].d[static_cast<std::size_t>(index)];
                const double rel = std::abs(fd - ad) / std::max(std::abs(ad), 1e-12);
                EXPECT_LE(rel, 1e-6) << "param " << index << " output " << out
                                     << " fd " << fd << " autodiff " << ad;
            }
        }
    }
}

}  // namespace
}  // namespace alphashape
