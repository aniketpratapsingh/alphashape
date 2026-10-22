// Tests for 3D similarity estimation (Umeyama 1991, weighted, and RANSAC).
#include "common/algorithm/sfm/similarity.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using test_rng::draw3;
using test_rng::draw4;

// Portable draws, not std::distributions: those map engine bits to values
// differently under libc++ and libstdc++, so a seed selected a different scene
// on macOS and Linux (TESTING.md, "Numeric consistency").
const auto gauss = [](std::mt19937_64& g) { return rng::normal(g, 0.0, 1.0); };

// A random rotation via a unit quaternion drawn from the RNG.
Mat3 random_rotation(std::mt19937_64& rng) {
    // draw4 sequences the four draws in g++'s right-to-left order, which is the
    // order these scenes were calibrated with (see test_rng_helpers.hpp).
    const Eigen::Vector4d xyzw = draw4(rng, gauss, gauss, gauss, gauss);
    Eigen::Quaterniond q(xyzw.w(), xyzw.x(), xyzw.y(), xyzw.z());
    q.normalize();
    return q.toRotationMatrix();
}

// A non-degenerate random point cloud (box, so not collinear).
std::vector<Vec3> random_cloud(std::mt19937_64& rng, std::size_t n, double extent) {
    const auto u = [extent](std::mt19937_64& g) { return rng::uniform_real(g, -extent, extent); };
    std::vector<Vec3> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = draw3(rng, u, u, u);
    }
    return pts;
}

TEST(Similarity, ExactRecovery) {
    std::mt19937_64 rng(12345);
    // 10^x in [0.01, 100]
    const auto scale_dist = [](std::mt19937_64& g) { return rng::uniform_real(g, -2.0, 2.0); };
    const auto t_dist = [](std::mt19937_64& g) { return rng::uniform_real(g, -100.0, 100.0); };
    const std::vector<Vec3> src = random_cloud(rng, 50, 10.0);

    for (int trial = 0; trial < 20; ++trial) {
        Similarity truth;
        truth.scale = std::pow(10.0, scale_dist(rng));
        truth.R = random_rotation(rng);
        truth.t = draw3(rng, t_dist, t_dist, t_dist);

        std::vector<Vec3> dst(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            dst[i] = truth.apply(src[i]);
        }

        const std::optional<Similarity> est = estimate_similarity(src, dst);
        ASSERT_TRUE(est.has_value());
        EXPECT_LT(std::abs(est->scale - truth.scale) / truth.scale, 1e-12);
        EXPECT_LT((est->R - truth.R).norm(), 1e-12);
        double scene = 0.0;
        for (const Vec3& p : src) {
            scene = std::max(scene, p.norm());
        }
        EXPECT_LT((est->t - truth.t).norm(), 1e-9 * scene);
    }
}

TEST(Similarity, NoisyRecovery) {
    std::mt19937_64 rng(999);
    Similarity truth;
    truth.scale = 2.5;
    truth.R = random_rotation(rng);
    truth.t = Vec3(3.0, -4.0, 5.0);

    const std::vector<Vec3> src = random_cloud(rng, 200, 5.0);
    const auto noise = [](std::mt19937_64& g) { return rng::normal(g, 0.0, 0.01); };
    std::vector<Vec3> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = truth.apply(src[i]) + draw3(rng, noise, noise, noise);
    }

    const std::optional<Similarity> est = estimate_similarity(src, dst);
    ASSERT_TRUE(est.has_value());
    EXPECT_LT(std::abs(est->scale - truth.scale), 0.05);
    EXPECT_LT((est->R - truth.R).norm(), 0.05);
}

TEST(Similarity, ReflectionNeverReturned) {
    std::mt19937_64 rng(77);
    const std::vector<Vec3> src = random_cloud(rng, 60, 8.0);
    // Mirror the source through a plane: the best-fit similarity through this
    // correspondence would be a reflection if we did not force det = +1.
    std::vector<Vec3> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = Vec3(-src[i].x(), src[i].y(), src[i].z());
    }
    const std::optional<Similarity> est = estimate_similarity(src, dst);
    ASSERT_TRUE(est.has_value());
    EXPECT_NEAR(est->R.determinant(), 1.0, 1e-9);
}

TEST(Similarity, CollinearIsNullopt) {
    std::vector<Vec3> src;
    std::vector<Vec3> dst;
    for (int i = 0; i < 10; ++i) {
        const Vec3 p(1.0 * i, 2.0 * i, 3.0 * i);
        src.push_back(p);
        dst.push_back(2.0 * p + Vec3(1.0, 1.0, 1.0));
    }
    EXPECT_FALSE(estimate_similarity(src, dst).has_value());
}

TEST(Similarity, TooFewPoints) {
    EXPECT_FALSE(estimate_similarity({Vec3::Zero(), Vec3::Ones()}, {Vec3::Zero(), Vec3::Ones()})
                     .has_value());
}

TEST(Similarity, WeightedZeroOutlierIgnored) {
    std::mt19937_64 rng(4242);
    Similarity truth;
    truth.scale = 1.7;
    truth.R = random_rotation(rng);
    truth.t = Vec3(-2.0, 6.0, 1.0);

    const std::vector<Vec3> src = random_cloud(rng, 40, 6.0);
    std::vector<Vec3> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = truth.apply(src[i]);
    }
    // Corrupt one point grossly, then weight it to zero.
    dst[7] = Vec3(1000.0, -1000.0, 1000.0);
    std::vector<double> w(src.size(), 1.0);
    w[7] = 0.0;

    const std::optional<Similarity> est = estimate_similarity_weighted(src, dst, w);
    ASSERT_TRUE(est.has_value());
    EXPECT_LT(std::abs(est->scale - truth.scale) / truth.scale, 1e-12);
    EXPECT_LT((est->R - truth.R).norm(), 1e-12);

    // The outlier is exactly ignored: the estimate equals the unweighted one
    // computed on the clean points only.
    std::vector<Vec3> cs;
    std::vector<Vec3> cd;
    for (std::size_t i = 0; i < src.size(); ++i) {
        if (i != 7) {
            cs.push_back(src[i]);
            cd.push_back(dst[i]);
        }
    }
    const std::optional<Similarity> clean = estimate_similarity(cs, cd);
    ASSERT_TRUE(clean.has_value());
    EXPECT_LT(std::abs(est->scale - clean->scale), 1e-12);
    EXPECT_LT((est->t - clean->t).norm(), 1e-12);
}

TEST(Similarity, RansacThirtyPercentOutliers) {
    std::mt19937_64 rng(2024);
    Similarity truth;
    truth.scale = 0.3;
    truth.R = random_rotation(rng);
    truth.t = Vec3(5.0, -3.0, 2.0);

    const std::size_t n = 100;
    const std::vector<Vec3> src = random_cloud(rng, n, 5.0);
    std::vector<Vec3> dst(src.size());
    std::vector<bool> is_outlier(n, false);
    const auto big = [](std::mt19937_64& g) { return rng::uniform_real(g, -500.0, 500.0); };
    // Deterministically designate 30% as gross outliers.
    for (std::size_t i = 0; i < n; ++i) {
        if (i % 10 < 3) {
            is_outlier[i] = true;
            dst[i] = truth.apply(src[i]) + draw3(rng, big, big, big);
        } else {
            dst[i] = truth.apply(src[i]);
        }
    }

    const double threshold = 0.01;
    const SimilarityRansacResult r = estimate_similarity_ransac(src, dst, threshold, 0x5eed);
    ASSERT_TRUE(r.similarity.has_value());
    EXPECT_LT(std::abs(r.similarity->scale - truth.scale) / truth.scale, 1e-9);
    EXPECT_LT((r.similarity->R - truth.R).norm(), 1e-9);
    EXPECT_LT((r.similarity->t - truth.t).norm(), 1e-9);

    // Recall 100%: every true inlier is marked. Precision 100%: no outlier is.
    std::size_t tp = 0;
    std::size_t fp = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool marked = r.inliers[i] != 0;
        if (is_outlier[i]) {
            if (marked) {
                ++fp;
            }
        } else if (marked) {
            ++tp;
        }
    }
    const std::size_t n_in = n - 30;
    EXPECT_EQ(tp, n_in);
    EXPECT_EQ(fp, 0u);
    EXPECT_EQ(r.inlier_count, n_in);
}

TEST(Similarity, RansacDeterministic) {
    std::mt19937_64 rng(31337);
    Similarity truth;
    truth.scale = 1.3;
    truth.R = random_rotation(rng);
    truth.t = Vec3(1.0, 2.0, 3.0);
    const std::size_t n = 80;
    const std::vector<Vec3> src = random_cloud(rng, n, 5.0);
    std::vector<Vec3> dst(src.size());
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = truth.apply(src[i]);
    }
    dst[0] = Vec3(999.0, 999.0, 999.0);
    dst[1] = Vec3(-999.0, 5.0, 5.0);

    const SimilarityRansacResult a = estimate_similarity_ransac(src, dst, 0.01, 0x5eed);
    const SimilarityRansacResult b = estimate_similarity_ransac(src, dst, 0.01, 0x5eed);
    ASSERT_TRUE(a.similarity.has_value());
    ASSERT_TRUE(b.similarity.has_value());
    EXPECT_EQ(a.similarity->scale, b.similarity->scale);
    EXPECT_EQ((a.similarity->R - b.similarity->R).norm(), 0.0);
    EXPECT_EQ(a.inliers, b.inliers);
    EXPECT_EQ(a.inlier_count, b.inlier_count);
}

TEST(Similarity, InverseAndMatrixConsistent) {
    std::mt19937_64 rng(5150);
    Similarity s;
    s.scale = 4.0;
    s.R = random_rotation(rng);
    s.t = Vec3(7.0, -8.0, 9.0);

    const Vec3 p(1.5, -2.5, 3.5);
    const Vec3 q = s.apply(p);

    // matrix() agrees with apply().
    const Eigen::Vector4d hq = s.matrix() * p.homogeneous();
    EXPECT_LT((hq.head<3>() - q).norm(), 1e-12);

    // inverse() undoes apply().
    const Vec3 back = s.inverse().apply(q);
    EXPECT_LT((back - p).norm(), 1e-9);

    // inverse().matrix() is the matrix inverse.
    const Eigen::Matrix4d prod = s.matrix() * s.inverse().matrix();
    EXPECT_LT((prod - Eigen::Matrix4d::Identity()).norm(), 1e-9);
}

}  // namespace
}  // namespace alphashape::sfm
