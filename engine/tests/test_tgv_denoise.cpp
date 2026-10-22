#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/elevation/tgv_denoise.hpp"

// Test-only hooks implemented in tgv_denoise.cpp (not in the public header).
namespace alphashape::elevation {
void tgv_apply_K(int width, int height, const double* u, const double* w1,
                 const double* w2, double* p1, double* p2, double* q11, double* q22,
                 double* q12);
void tgv_apply_Kt(int width, int height, const double* p1, const double* p2,
                  const double* q11, const double* q22, const double* q12, double* u,
                  double* w1, double* w2);
void tv_apply_grad(int width, int height, const double* u, double* p1, double* p2);
void tv_apply_div(int width, int height, const double* p1, const double* p2, double* u);
}  // namespace alphashape::elevation

namespace {

using alphashape::elevation::TgvOptions;
using alphashape::elevation::TgvResult;

constexpr int kWidth = 128;
constexpr int kHeight = 128;

// Deterministic Gaussian: own Box-Muller so the byte stream does not depend on
// the standard library's normal_distribution implementation.
struct Gauss {
    std::mt19937 rng;
    double spare = 0.0;
    bool has_spare = false;
    explicit Gauss(std::uint32_t seed) : rng(seed) {}
    double next() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        // Portable draws: std::uniform_real_distribution is
        // implementation-defined across libc++ and libstdc++ (TESTING.md).
        const double u1 = alphashape::rng::uniform_real(rng, 1e-12, 1.0);
        const double u2 = alphashape::rng::uniform_real(rng, 1e-12, 1.0);
        const double r = std::sqrt(-2.0 * std::log(u1));
        spare = r * std::sin(2.0 * M_PI * u2);
        has_spare = true;
        return r * std::cos(2.0 * M_PI * u2);
    }
};

// Piecewise-affine truth: tilted plane (slope 0.3) on the left half, a +5
// plateau and a 2D ramp on the right half.
double truth_at(int x, int y) {
    const double fx = static_cast<double>(x);
    const double fy = static_cast<double>(y);
    if (x < kWidth / 2) {
        return 0.3 * fx;  // tilted plane
    }
    if (x < 3 * kWidth / 4 && y < kHeight / 2) {
        return 5.0;  // flat plateau
    }
    return 0.2 * fx + 0.1 * fy;  // 2D ramp
}

std::vector<double> make_truth() {
    std::vector<double> t(static_cast<std::size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            t[static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)] = truth_at(x, y);
        }
    }
    return t;
}

struct Scene {
    std::vector<float> f;           // noisy observation
    std::vector<float> confidence;  // 0 where unobserved
    std::vector<char> salt;         // 1 where a +10 outlier was injected
    std::vector<char> hole;         // 1 inside the 10x10 hole
    std::vector<double> truth;
};

// One shared scene for all data-driven tests.
const Scene& scene() {
    static const Scene s = [] {
        Scene sc;
        const std::size_t n = static_cast<std::size_t>(kWidth) * kHeight;
        sc.truth = make_truth();
        sc.f.resize(n);
        sc.confidence.assign(n, 1.0f);
        sc.salt.assign(n, 0);
        sc.hole.assign(n, 0);

        Gauss g(12345u);
        for (std::size_t i = 0; i < n; ++i) {
            sc.f[i] = static_cast<float>(sc.truth[i] + 0.3 * g.next());
        }

        // 10x10 hole on the tilted plane.
        for (int y = 20; y < 30; ++y) {
            for (int x = 30; x < 40; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x);
                sc.confidence[i] = 0.0f;
                sc.hole[i] = 1;
            }
        }
        // ~25% unobserved in random blobs.
        std::mt19937 blob(999u);
        const int n_blobs = 40;
        for (int b = 0; b < n_blobs; ++b) {
            const int cx = static_cast<int>(blob() % kWidth);
            const int cy = static_cast<int>(blob() % kHeight);
            const int half = 3 + static_cast<int>(blob() % 4);
            for (int y = cy - half; y <= cy + half; ++y) {
                for (int x = cx - half; x <= cx + half; ++x) {
                    if (x < 0 || y < 0 || x >= kWidth || y >= kHeight) continue;
                    sc.confidence[static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)] = 0.0f;
                }
            }
        }
        // 2% salt outliers (+10) on observed cells.
        std::mt19937 salt_rng(4242u);
        int injected = 0;
        const int target = static_cast<int>(n * 2 / 100);
        while (injected < target) {
            const std::size_t i = salt_rng() % n;
            if (sc.salt[i] || sc.confidence[i] == 0.0f) continue;
            sc.f[i] += 10.0f;
            sc.salt[i] = 1;
            ++injected;
        }
        return sc;
    }();
    return s;
}

double rmse(const std::vector<float>& a, const std::vector<float>& b, const std::vector<char>& mask) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!mask.empty() && !mask[i]) continue;
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
        ++count;
    }
    return count ? std::sqrt(sum / static_cast<double>(count)) : 0.0;
}

// Random double field, deterministic.
std::vector<double> random_field(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const auto uni = [](std::mt19937& g) { return alphashape::rng::uniform_real(g, -1.0, 1.0); };
    std::vector<double> v(n);
    for (auto& val : v) val = uni(rng);
    return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double as_float_rmse(const std::vector<float>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        sum += d * d;
    }
    return std::sqrt(sum / static_cast<double>(a.size()));
}

}  // namespace

// ---------------------------------------------------------------------------
// Adjointness of the assembled operator K: <K x, y> == <x, K^T y>.
// ---------------------------------------------------------------------------
TEST(Tgv, AdjointK) {
    const int w = 17, h = 13;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    const std::vector<double> u = random_field(n, 1u);
    const std::vector<double> w1 = random_field(n, 2u);
    const std::vector<double> w2 = random_field(n, 3u);
    const std::vector<double> p1 = random_field(n, 4u);
    const std::vector<double> p2 = random_field(n, 5u);
    const std::vector<double> q11 = random_field(n, 6u);
    const std::vector<double> q22 = random_field(n, 7u);
    const std::vector<double> q12 = random_field(n, 8u);

    std::vector<double> kp1(n), kp2(n), kq11(n), kq22(n), kq12(n);
    alphashape::elevation::tgv_apply_K(w, h, u.data(), w1.data(), w2.data(), kp1.data(),
                                       kp2.data(), kq11.data(), kq22.data(), kq12.data());
    // The q slot pairs under the matrix Frobenius product, whose weight on the
    // off-diagonal entry is 2 (|Q|_F = sqrt(q11^2+q22^2+2*q12^2)).
    const double lhs = dot(kp1, p1) + dot(kp2, p2) + dot(kq11, q11) + dot(kq22, q22) +
                       2.0 * dot(kq12, q12);

    std::vector<double> ku(n), kw1(n), kw2(n);
    alphashape::elevation::tgv_apply_Kt(w, h, p1.data(), p2.data(), q11.data(), q22.data(),
                                        q12.data(), ku.data(), kw1.data(), kw2.data());
    const double rhs = dot(ku, u) + dot(kw1, w1) + dot(kw2, w2);

    EXPECT_NEAR(lhs, rhs, 1e-9);
}

TEST(Tgv, AdjointGrad) {
    const int w = 21, h = 9;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    const std::vector<double> u = random_field(n, 11u);
    const std::vector<double> p1 = random_field(n, 12u);
    const std::vector<double> p2 = random_field(n, 13u);
    std::vector<double> gx(n), gy(n);
    alphashape::elevation::tv_apply_grad(w, h, u.data(), gx.data(), gy.data());
    const double lhs = dot(gx, p1) + dot(gy, p2);
    std::vector<double> ku(n);
    alphashape::elevation::tv_apply_div(w, h, p1.data(), p2.data(), ku.data());
    EXPECT_NEAR(lhs, dot(ku, u), 1e-9);
}

// ---------------------------------------------------------------------------
// Fixed point: an exact plane with full confidence is left untouched.
// ---------------------------------------------------------------------------
TEST(Tgv, PlaneFixedPoint) {
    const int w = 32, h = 24;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    std::vector<float> f(n), c(n, 1.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            f[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)] =
                static_cast<float>(0.3 * x + 0.7 * y + 1.0);
        }
    }
    std::vector<float> u = f;  // start exactly on the solution
    TgvOptions opt;
    opt.max_iterations = 300;
    const TgvResult r = alphashape::elevation::tgv2_denoise(w, h, f, c, u, opt);
    EXPECT_LT(as_float_rmse(u, [&] {
                  std::vector<double> t(n);
                  for (int y = 0; y < h; ++y) {
                      for (int x = 0; x < w; ++x) {
                          t[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)] = 0.3 * x + 0.7 * y + 1.0;
                      }
                  }
                  return t;
              }()),
              1e-4);
    EXPECT_GT(r.iterations, 0);
}

// ---------------------------------------------------------------------------
// Main quality tests on the 128x128 scene.
// ---------------------------------------------------------------------------
TEST(Tgv, L1BeatsNoisyInput) {
    const Scene& sc = scene();
    const std::size_t n = sc.f.size();
    std::vector<float> truth_f(n);
    for (std::size_t i = 0; i < n; ++i) truth_f[i] = static_cast<float>(sc.truth[i]);
    // The comparison is made on the *observed* cells only: the noisy RMSE the
    // solver has to beat is defined over c > 0 (unobserved cells carry no
    // measurement to denoise), so the TGV RMSE must use the same mask. Scoring
    // over all cells would fold in inpainting error at the scene's cliffs,
    // which is a different (and much larger) quantity.
    std::vector<char> observed(n, 0);
    for (std::size_t i = 0; i < n; ++i) observed[i] = sc.confidence[i] > 0.0f ? 1 : 0;
    const double noisy_rmse = [&] {
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (sc.confidence[i] <= 0.0f) continue;
            const double d = static_cast<double>(sc.f[i]) - sc.truth[i];
            sum += d * d;
            ++count;
        }
        return std::sqrt(sum / static_cast<double>(count));
    }();

    TgvOptions opt;
    opt.max_iterations = 500;
    std::vector<float> u;
    const auto t0 = std::chrono::steady_clock::now();
    const TgvResult r =
        alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u, opt);
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const double tgv_rmse = rmse(u, truth_f, observed);
    std::printf("[L1] noisy RMSE=%.4f  TGV RMSE=%.4f  iters=%d converged=%d  %.3fs\n",
                noisy_rmse, tgv_rmse, r.iterations, static_cast<int>(r.converged), secs);
    EXPECT_LT(tgv_rmse, 0.35 * noisy_rmse);
    // Wall-clock budget, not a correctness bound. 0.4 s alone, but a polluted
    // 128^2 solve is pure CPU work and starves when a sibling test holds the
    // cores -- measured 17-20 s on the 4-core x86 build machine while a
    // llvmpipe (CPU Vulkan) test ran beside it. 60 s still catches an
    // order-of-magnitude algorithmic regression (1 s was tuned on the M1).
    EXPECT_LT(secs, 60.0) << "128x128 solve regressed past its budget";
}

TEST(Tgv, SlopesBetterThanTv) {
    const Scene& sc = scene();
    const std::size_t n = sc.f.size();
    std::vector<float> truth_f(n);
    for (std::size_t i = 0; i < n; ++i) truth_f[i] = static_cast<float>(sc.truth[i]);

    TgvOptions opt;
    std::vector<float> u_tgv, u_tv;
    alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u_tgv, opt);
    alphashape::elevation::tv_denoise(kWidth, kHeight, sc.f, sc.confidence, u_tv, opt);

    // Mask the tilted plane (x < 64) and the 2D ramp (both observed), excluding
    // 3 cells around each discontinuity we model.
    std::vector<char> mask(n, 0);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x);
            if (sc.confidence[i] <= 0.0f) continue;
            bool in_slope = false;
            // tilted plane, staying clear of x = 64 and x = 96 boundaries
            if (x < 61) in_slope = true;
            // 2D ramp region
            if (x >= 99 && y >= 64) in_slope = true;
            if (in_slope) mask[i] = 1;
        }
    }
    const double tv_rmse = rmse(u_tv, truth_f, mask);
    const double tgv_rmse = rmse(u_tgv, truth_f, mask);
    std::printf("[slopes] TV RMSE=%.4f  TGV RMSE=%.4f\n", tv_rmse, tgv_rmse);
    EXPECT_LT(tgv_rmse, tv_rmse);
}

TEST(Tgv, OutliersRejectedL1) {
    const Scene& sc = scene();
    TgvOptions opt;
    std::vector<float> u;
    alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u, opt);

    // The +10 spikes are unidentifiable where they sit inside a genuine jump of
    // the truth (the scene's 14/18-unit mesa cliffs) or next to a cell with no
    // observation: there the regulariser legitimately follows the discontinuity
    // rather than the corrupt sample. Restrict to salt cells whose 3x3
    // neighbourhood is locally affine and fully observed, exactly the
    // discontinuity-adjacency exclusion SlopesBetterThanTv already uses.
    double worst = 0.0, worst_all = 0.0;
    int counted = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        if (!sc.salt[i]) continue;
        const double err = std::abs(static_cast<double>(u[i]) - sc.truth[i]);
        worst_all = std::max(worst_all, err);
        const int x = static_cast<int>(i % kWidth), y = static_cast<int>(i / kWidth);
        bool near_feature = false;
        double lo = sc.truth[i], hi = sc.truth[i];
        for (int dy = -1; dy <= 1 && !near_feature; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= kWidth || ny >= kHeight) continue;
                const std::size_t j =
                    static_cast<std::size_t>(ny) * kWidth + static_cast<std::size_t>(nx);
                if (sc.confidence[j] <= 0.0f) { near_feature = true; break; }
                lo = std::min(lo, sc.truth[j]);
                hi = std::max(hi, sc.truth[j]);
            }
        }
        if (near_feature || hi - lo > 2.0) continue;
        ++counted;
        worst = std::max(worst, err);
    }
    std::printf("[outliers] max |u-truth| at salt cells = %.4f (locally-affine set, n=%d)\n", worst,
                counted);
    std::printf("[outliers] unfiltered max over all salt cells = %.4f\n", worst_all);
    EXPECT_GT(counted, 200);
    EXPECT_LT(worst, 1.0);
}

TEST(Tgv, HoleInpainted) {
    const Scene& sc = scene();
    const std::size_t n = sc.f.size();
    std::vector<float> truth_f(n);
    for (std::size_t i = 0; i < n; ++i) truth_f[i] = static_cast<float>(sc.truth[i]);
    TgvOptions opt;
    std::vector<float> u;
    alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u, opt);

    // Interior of the 10x10 hole (skip a 2-cell rim, which the surrounding
    // discontinuity-free plane should extend across).
    std::vector<char> hole_mask(n, 0);
    for (int y = 22; y < 28; ++y) {
        for (int x = 32; x < 38; ++x) {
            hole_mask[static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)] = 1;
        }
    }
    const double hole_rmse = rmse(u, truth_f, hole_mask);
    std::printf("[hole] plane-continuation RMSE = %.4f\n", hole_rmse);
    EXPECT_LT(hole_rmse, 0.3);
}

TEST(Tgv, L2ReducesRmse) {
    const Scene& sc = scene();
    const std::size_t n = sc.f.size();
    std::vector<float> truth_f(n);
    for (std::size_t i = 0; i < n; ++i) truth_f[i] = static_cast<float>(sc.truth[i]);
    const std::vector<char> all(n, 1);
    double noisy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(sc.f[i]) - sc.truth[i];
        noisy += d * d;
    }
    noisy = std::sqrt(noisy / static_cast<double>(n));
    TgvOptions opt;
    opt.data_term = alphashape::elevation::TgvDataTerm::L2;
    std::vector<float> u;
    const TgvResult r =
        alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u, opt);
    const double l2_rmse = rmse(u, truth_f, all);
    std::printf("[L2] noisy RMSE=%.4f  TGV-L2 RMSE=%.4f  iters=%d\n", noisy, l2_rmse,
                r.iterations);
    EXPECT_LT(l2_rmse, noisy);
}

TEST(Tgv, InvalidInputs) {
    const int w = 8, h = 8;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    std::vector<float> f(n, 1.0f), c(n, 1.0f), u;
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(1, h, f, c, u), alphashape::ValueError);
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, 1, f, c, u), alphashape::ValueError);
    std::vector<float> bad_f(n - 1, 1.0f);
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, h, bad_f, c, u),
                 alphashape::ValueError);
    std::vector<float> neg_c(n, 1.0f);
    neg_c[0] = -1.0f;
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, h, f, neg_c, u),
                 alphashape::ValueError);
    std::vector<float> zero_c(n, 0.0f);
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, h, f, zero_c, u),
                 alphashape::ValueError);
    std::vector<float> nan_f = f;
    nan_f[3] = std::nanf("");
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, h, nan_f, c, u),
                 alphashape::ValueError);
    TgvOptions bad;
    bad.alpha1 = 0.0;
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, h, f, c, u, bad),
                 alphashape::ValueError);
    bad = TgvOptions();
    bad.max_iterations = 0;
    EXPECT_THROW(alphashape::elevation::tgv2_denoise(w, h, f, c, u, bad),
                 alphashape::ValueError);
    // A NaN in f where c == 0 must be accepted (unobserved).
    std::vector<float> nan_unobs = f;
    nan_unobs[0] = std::nanf("");
    std::vector<float> c0 = c;
    c0[0] = 0.0f;
    EXPECT_NO_THROW(alphashape::elevation::tgv2_denoise(w, h, nan_unobs, c0, u));
}

// ---------------------------------------------------------------------------
// Determinism across thread counts.
// ---------------------------------------------------------------------------
TEST(Tgv, DeterminismAcrossThreads) {
    const Scene& sc = scene();
    std::vector<float> u1, u4;
    omp_set_num_threads(1);
    alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u1);
    omp_set_num_threads(4);
    alphashape::elevation::tgv2_denoise(kWidth, kHeight, sc.f, sc.confidence, u4);
    omp_set_num_threads(1);
    ASSERT_EQ(u1.size(), u4.size());
    for (std::size_t i = 0; i < u1.size(); ++i) {
        EXPECT_EQ(u1[i], u4[i]) << "at " << i;
    }
}



