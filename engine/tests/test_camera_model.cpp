// Camera model: forward projection, inverse projection, and the closed forms.
//
// Suites are prefixed CameraModel so `ctest -R CameraModel` selects them.

#include <algorithm>
#include <cmath>
#include <cstdint>
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

TEST(CameraModelFrame, ProjectPointTemplateMatchesProject) {
    const Calibration c = realistic_frame();
    const double params[kCalibrationParamCount] = {c.f,  c.cx, c.cy, c.b1, c.b2, c.k1,
                                                   c.k2, c.k3, c.k4, c.p1, c.p2};
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

}  // namespace
}  // namespace alphashape
