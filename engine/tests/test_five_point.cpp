// Tests for the five-point relative orientation solver (Stewenius/Nister).
#include "common/algorithm/sfm/five_point.hpp"

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using test_rng::draw3;

struct Scene {
    std::array<Eigen::Vector3d, 5> b1;
    std::array<Eigen::Vector3d, 5> b2;
    Eigen::Matrix3d E_true;
};

// Portable draws, not std::distributions: those map engine bits to values
// differently under libc++ and libstdc++ (TESTING.md, "Numeric consistency").
const auto unit_draw = [](std::mt19937_64& g) { return rng::uniform_real(g, -1.0, 1.0); };

Eigen::Matrix3d random_rotation(std::mt19937_64& rng) {
    Eigen::Vector3d axis = draw3(rng, unit_draw, unit_draw, unit_draw);
    if (axis.norm() < 1e-6) {
        axis = Eigen::Vector3d::UnitX();
    }
    axis.normalize();
    const double ang = rng::uniform_real(rng, 0.0, 30.0 * M_PI / 180.0);
    return Eigen::AngleAxisd(ang, axis).toRotationMatrix();
}

Eigen::Matrix3d skew(const Eigen::Vector3d& t) {
    Eigen::Matrix3d s;
    s << 0.0, -t.z(), t.y(), t.z(), 0.0, -t.x(), -t.y(), t.x(), 0.0;
    return s;
}

Scene make_scene(std::mt19937_64& rng, bool wide_angle) {
    const Eigen::Matrix3d R = random_rotation(rng);
    Eigen::Vector3d t = draw3(rng, unit_draw, unit_draw, unit_draw);
    if (t.norm() < 1e-6) {
        t = Eigen::Vector3d::UnitX();
    }
    t.normalize();

    Scene s;
    for (int i = 0; i < 5; ++i) {
        Eigen::Vector3d X;
        bool ok = false;
        for (int attempt = 0; attempt < 200 && !ok; ++attempt) {
            if (wide_angle) {
                // Points may lie behind the principal plane (z < 0) in camera 1;
                // require depth 2..10 along the optical axis up to sign.
                const auto d = [](std::mt19937_64& g) { return rng::uniform_real(g, -10.0, -2.0); };
                const auto xy = [](std::mt19937_64& g) { return rng::uniform_real(g, -10.0, 10.0); };
                // g++ drew the third argument first, then y, then x; for i >= 2
                // that third argument consumed an xy draw of its own.
                double z_arg = 0.0;
                if (i < 2) {
                    z_arg = d(rng);
                } else {
                    z_arg = std::abs(xy(rng)) + 2.0;
                }
                const Eigen::Vector2d xyw = test_rng::draw2(rng, xy, xy);
                X = Eigen::Vector3d(xyw.x(), xyw.y(), z_arg);
            } else {
                const auto d = [](std::mt19937_64& g) { return rng::uniform_real(g, 2.0, 10.0); };
                const auto xy = [](std::mt19937_64& g) { return rng::uniform_real(g, -6.0, 6.0); };
                X = draw3(rng, xy, xy, d);
            }
            const Eigen::Vector3d X2 = R * X + t;
            ok = X2.z() != 0.0 && (wide_angle ? std::abs(X2.z()) > 1e-3 : X2.z() > 0.5);
        }
        s.b1[static_cast<std::size_t>(i)] = X.normalized();
        s.b2[static_cast<std::size_t>(i)] = (R * X + t).normalized();
    }
    s.E_true = (skew(t) * R).normalized();
    return s;
}

bool matches_true(const std::vector<Eigen::Matrix3d>& sols, const Eigen::Matrix3d& E_true,
                  double tol) {
    for (const Eigen::Matrix3d& E : sols) {
        if ((E - E_true).norm() < tol || (E + E_true).norm() < tol) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(FivePoint, RecoversTrueEssentialOver200Scenes) {
    std::mt19937_64 rng(20261017ULL);
    int failures = 0;
    for (int scene = 0; scene < 200; ++scene) {
        const Scene s = make_scene(rng, /*wide_angle=*/false);
        const std::vector<Eigen::Matrix3d> sols = five_point(s.b1, s.b2);
        if (!matches_true(sols, s.E_true, 1e-6)) {
            ++failures;
        }
    }
    std::printf("[FivePoint] 200 scenes, failures = %d\n", failures);
    EXPECT_LE(failures, 1);
}

TEST(FivePoint, EverySolutionSatisfiesConstraints) {
    std::mt19937_64 rng(424242ULL);
    for (int scene = 0; scene < 50; ++scene) {
        const Scene s = make_scene(rng, /*wide_angle=*/false);
        const std::vector<Eigen::Matrix3d> sols = five_point(s.b1, s.b2);
        ASSERT_LE(sols.size(), static_cast<std::size_t>(10));
        for (const Eigen::Matrix3d& E : sols) {
            for (int i = 0; i < 5; ++i) {
                const double res =
                    s.b2[static_cast<std::size_t>(i)].dot(E * s.b1[static_cast<std::size_t>(i)]);
                EXPECT_LT(std::abs(res), 1e-9);
            }
            EXPECT_LT(std::abs(E.determinant()), 1e-9);
            const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
            const Eigen::Vector3d sv = svd.singularValues();
            EXPECT_LT(std::abs(sv(0) - sv(1)) / sv(0), 1e-6);
            EXPECT_LT(sv(2), 1e-9);
        }
    }
}

TEST(FivePoint, WideAngleBearings) {
    std::mt19937_64 rng(777ULL);
    int recovered = 0;
    for (int scene = 0; scene < 50; ++scene) {
        const Scene s = make_scene(rng, /*wide_angle=*/true);
        const std::vector<Eigen::Matrix3d> sols = five_point(s.b1, s.b2);
        if (matches_true(sols, s.E_true, 1e-6)) {
            ++recovered;
        }
    }
    std::printf("[FivePoint] wide-angle scenes recovered = %d / 50\n", recovered);
    EXPECT_EQ(recovered, 50);
}

TEST(FivePoint, ReturnsAtMostTenSolutions) {
    std::mt19937_64 rng(9001ULL);
    for (int scene = 0; scene < 50; ++scene) {
        const Scene s = make_scene(rng, /*wide_angle=*/false);
        const std::vector<Eigen::Matrix3d> sols = five_point(s.b1, s.b2);
        EXPECT_LE(sols.size(), static_cast<std::size_t>(10));
    }
}

}  // namespace alphashape::sfm
