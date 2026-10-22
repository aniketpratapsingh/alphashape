#include "common/algorithm/elevation/tgv_denoise.hpp"

// Discretisation. The grid is W x H, row-major, index i = y*W + x.
//
//   grad u   = (Dx u, Dy u)        Dx forward difference, zero on the last
//                                  column (Neumann), Dy likewise on the last row
//   eps(w)   = (Dx w1, Dy w2, (Dy w1 + Dx w2)/2)      the symmetrised gradient
//
// The divergence used everywhere is minus the *exact* transpose of these
// forward operators, so <grad u, p> == <u, -div p> to machine precision; the
// transpose of a forward difference with a zeroed last row/column is the usual
// backward difference with its own boundary terms (see adj_forward_x below),
// which is exactly the consistency the primal-dual pairing needs. Both
// operators share the same difference scheme, so eps^T is the transpose of eps.
//
// Chambolle-Pock first-order primal-dual (2011, Algorithm 1) on
//   K = [[grad, -I], [0, eps]],  F*(p,q) = i_{|.|<=a1}(p) + i_{|.|<=a0}(q),
//   G(u,w) = data term D(u) only (w carries no data).
// with over-relaxation theta = 1. Norms are pointwise isotropic:
//   |p| = sqrt(p1^2 + p2^2),  |q| = sqrt(q11^2 + q22^2 + 2*q12^2).
// The dual prox is the pointwise projection onto a ball; the primal prox for u
// is a weighted soft-threshold toward f (L1) or a closed-form linear solve (L2).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::elevation {
namespace detail {

// out = A_x^T v, where A_x is the forward difference with a zeroed last column.
// This is the backward difference with the matching boundary terms:
//   out[0] = -v[0]; out[x] = v[x-1] - v[x]; out[W-1] = v[W-2].
void adj_forward_x(int width, int height, const double* v, double* out) {
    const std::size_t w = static_cast<std::size_t>(width);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(height); ++y) {
        const std::size_t r = static_cast<std::size_t>(y) * w;
        out[r] = -v[r];
        for (std::size_t x = 1; x + 1 < w; ++x) {
            out[r + x] = v[r + x - 1] - v[r + x];
        }
        out[r + w - 1] = v[r + w - 2];
    }
}

// out = A_y^T v, the same in the vertical direction (zeroed last row).
void adj_forward_y(int width, int height, const double* v, double* out) {
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(width); ++x) {
        const std::size_t cx = static_cast<std::size_t>(x);
        out[cx] = -v[cx];
        for (std::size_t y = 1; y + 1 < h; ++y) {
            const std::size_t i = y * w + cx;
            out[i] = v[i - w] - v[i];
        }
        const std::size_t last = (h - 1) * w + cx;
        out[last] = v[last - w];
    }
}

// Forward difference with Neumann boundary; the zeroed last column/row.
void grad_forward(int width, int height, const double* u, double* gx, double* gy) {
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(height); ++y) {
        const std::size_t cy = static_cast<std::size_t>(y);
        const std::size_t r = cy * w;
        for (std::size_t x = 0; x + 1 < w; ++x) {
            gx[r + x] = u[r + x + 1] - u[r + x];
        }
        gx[r + w - 1] = 0.0;
        if (cy + 1 < h) {
            for (std::size_t x = 0; x < w; ++x) {
                gy[r + x] = u[r + x + w] - u[r + x];
            }
        } else {
            for (std::size_t x = 0; x < w; ++x) {
                gy[r + x] = 0.0;
            }
        }
    }
}

// eps(w), the symmetrised gradient, as (Dx w1, Dy w2, (Dy w1 + Dx w2)/2).
void eps_sym(int width, int height, const double* w1, const double* w2, double* e11,
             double* e22, double* e12, std::vector<double>& scratch) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    scratch.resize(2 * n);
    double* t1x = scratch.data();
    double* t1y = t1x + n;
    grad_forward(width, height, w1, t1x, t1y);
    double* t2x = e12;  // reuse the caller's output as scratch for Dx w2
    double* t2y = e22;
    grad_forward(width, height, w2, t2x, t2y);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        e11[i] = t1x[i];
        // e22 already holds t2y == Dy w2; e12 still holds t2x == Dx w2.
        e12[i] = 0.5 * (t1y[i] + e12[i]);
    }
}

// eps^T Q, the exact transpose of eps under the matrix Frobenius pairing.
//
// eps(w) is stored as the three numbers (e11, e22, e12) with e12 the single
// off-diagonal entry (Dy w1 + Dx w2)/2. The primal penalty and the dual ball
// both use the Frobenius norm of the symmetric matrix
//   |E|_F = sqrt(e11^2 + e22^2 + 2*e12^2),   |Q|_F = sqrt(q11^2 + q22^2 + 2*q12^2),
// whose pairing is <E, Q> = e11 q11 + e22 q22 + 2 e12 q12. Substituting
// e12 = (Dy w1 + Dx w2)/2 gives <eps(w), Q> = w1.(Dx^T q11 + Dy^T q12) +
// w2.(Dy^T q22 + Dx^T q12): the transpose carries a full weight 1 on the q12
// component (the 0.5 from e12 cancels the factor 2 of the Frobenius pairing).
// Using 0.5 here instead would pair eps with q under the plain Euclidean product
// while the energy/ball stay Frobenius -- an inconsistent model in which the
// shear is over-penalised by 4x.
void eps_sym_transpose(int width, int height, const double* q11, const double* q22,
                       const double* q12, double* out1, double* out2) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> tmp(n, 0.0);
    adj_forward_x(width, height, q11, out1);
    adj_forward_y(width, height, q12, tmp.data());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        out1[i] += tmp[i];
    }
    adj_forward_y(width, height, q22, out2);
    adj_forward_x(width, height, q12, tmp.data());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        out2[i] += tmp[i];
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Test-only hooks. Forward-declared in the test; exposed here so the adjoint
// identity <K x, y> == <x, K^T y> can be checked on the assembled operator
// without widening the public header.
// ---------------------------------------------------------------------------
void tgv_apply_K(int width, int height, const double* u, const double* w1,
                 const double* w2, double* p1, double* p2, double* q11, double* q22,
                 double* q12);
void tgv_apply_Kt(int width, int height, const double* p1, const double* p2,
                  const double* q11, const double* q22, const double* q12, double* u,
                  double* w1, double* w2);
void tv_apply_grad(int width, int height, const double* u, double* p1, double* p2);
void tv_apply_div(int width, int height, const double* p1, const double* p2, double* u);

// K = [[grad, -I], [0, eps]] applied to (u, w).  p = grad u - w, q = eps(w).
void tgv_apply_K(int width, int height, const double* u, const double* w1,
                 const double* w2, double* p1, double* p2, double* q11, double* q22,
                 double* q12) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> gx(n), gy(n), scratch;
    detail::grad_forward(width, height, u, gx.data(), gy.data());
    detail::eps_sym(width, height, w1, w2, q11, q22, q12, scratch);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        p1[i] = gx[i] - w1[i];
        p2[i] = gy[i] - w2[i];
    }
}

// K^T = [[-div, 0], [-I, eps^T]] applied to (p, q).
void tgv_apply_Kt(int width, int height, const double* p1, const double* p2,
                  const double* q11, const double* q22, const double* q12, double* u,
                  double* w1, double* w2) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> dx(n, 0.0);
    detail::adj_forward_x(width, height, p1, u);
    detail::adj_forward_y(width, height, p2, dx.data());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        u[i] += dx[i];  // div p = -(grad^T p) = -(adj_x p1 + adj_y p2)
    }
    std::vector<double> s1(n, 0.0), s2(n, 0.0);
    detail::eps_sym_transpose(width, height, q11, q22, q12, s1.data(), s2.data());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        w1[i] = -p1[i] + s1[i];
        w2[i] = -p2[i] + s2[i];
    }
}

void tv_apply_grad(int width, int height, const double* u, double* p1, double* p2) {
    detail::grad_forward(width, height, u, p1, p2);
}

void tv_apply_div(int width, int height, const double* p1, const double* p2, double* u) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> dx(n, 0.0);
    detail::adj_forward_x(width, height, p1, u);
    detail::adj_forward_y(width, height, p2, dx.data());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        u[i] += dx[i];
    }
}

namespace {

void validate(int width, int height, const std::vector<float>& f,
              const std::vector<float>& confidence, const TgvOptions& options) {
    if (width < 2 || height < 2) {
        throw ValueError("tgv_denoise: width and height must be >= 2");
    }
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (f.size() != n) {
        throw ValueError("tgv_denoise: f size (" + std::to_string(f.size()) +
                         ") does not match width*height (" + std::to_string(n) + ")");
    }
    if (confidence.size() != n) {
        throw ValueError("tgv_denoise: confidence size (" +
                         std::to_string(confidence.size()) + ") does not match width*height (" +
                         std::to_string(n) + ")");
    }
    bool any_positive = false;
    for (std::size_t i = 0; i < n; ++i) {
        const float c = confidence[i];
        if (c < 0.0f) {
            throw ValueError("tgv_denoise: confidence must be >= 0");
        }
        if (c > 0.0f) {
            any_positive = true;
            if (!std::isfinite(f[i])) {
                throw ValueError("tgv_denoise: f must be finite where confidence > 0");
            }
        }
    }
    if (!any_positive) {
        throw ValueError("tgv_denoise: at least one confidence must be > 0");
    }
    if (!(options.alpha1 > 0.0) || !(options.alpha0 > 0.0) || !(options.lambda > 0.0)) {
        throw ValueError("tgv_denoise: alpha1, alpha0 and lambda must be positive");
    }
    if (options.max_iterations <= 0) {
        throw ValueError("tgv_denoise: max_iterations must be positive");
    }
}

// Initialise u from f where observed, with the mean of observed f elsewhere.
void init_from_data(int width, int height, const std::vector<float>& f,
                    const std::vector<float>& confidence, std::vector<double>& u) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (confidence[i] > 0.0f) {
            sum += static_cast<double>(f[i]);
            ++count;
        }
    }
    const double mean = (count > 0) ? sum / static_cast<double>(count) : 0.0;
    u.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        u[i] = (confidence[i] > 0.0f) ? static_cast<double>(f[i]) : mean;
    }
}

// Pointwise isotropic projection onto the dual norm balls.
void project_dual2(std::size_t n, double alpha, std::vector<double>& a,
                   std::vector<double>& b) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const double norm = std::sqrt(a[i] * a[i] + b[i] * b[i]);
        if (norm > alpha) {
            const double s = alpha / norm;
            a[i] *= s;
            b[i] *= s;
        }
    }
}

// Pointwise projection onto the dual ball of the second-order term, using the
// isotropic norm |q| = sqrt(q11^2 + q22^2 + 2*q12^2) (the Frobenius norm of the
// symmetric matrix [[q11, q12], [q12, q22]], i.e. the conjugate of the primal
// penalty alpha0 * |eps(w)| with the same isotropic norm on eps).
void project_dual3(std::size_t n, double alpha, std::vector<double>& a,
                   std::vector<double>& b, std::vector<double>& c) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const double norm = std::sqrt(a[i] * a[i] + b[i] * b[i] + 2.0 * c[i] * c[i]);
        if (norm > alpha) {
            const double s = alpha / norm;
            a[i] *= s;
            b[i] *= s;
            c[i] *= s;
        }
    }
}

// Primal prox of G (the data term) with step tau, applied to v -> u.
void data_prox(TgvDataTerm term, int width, int height, const std::vector<float>& f,
               const std::vector<float>& confidence, double lambda, double tau,
               const std::vector<double>& v, std::vector<double>& u) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (term == TgvDataTerm::L1) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const double thr = tau * lambda * static_cast<double>(confidence[i]);
            const double diff = v[i] - static_cast<double>(f[i]);
            u[i] = static_cast<double>(f[i]) +
                   std::copysign(std::max(std::abs(diff) - thr, 0.0), diff);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const double wgt = static_cast<double>(confidence[i]);
            const double denom = 1.0 + tau * lambda * wgt;
            const double fi = static_cast<double>(f[i]);
            u[i] = (v[i] + tau * lambda * wgt * fi) / denom;
        }
    }
}

// Primal objective a1*|grad u - w|_1 + a0*|eps(w)|_1 + D(u).
double primal_energy(const TgvOptions& options, int width, int height,
                     const std::vector<float>& f, const std::vector<float>& confidence,
                     bool first_order, const std::vector<double>& u,
                     const std::vector<double>& w1, const std::vector<double>& w2) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> gx(n), gy(n), e11(n), e22(n), e12(n), scratch;
    detail::grad_forward(width, height, u.data(), gx.data(), gy.data());
    double tv = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : tv)
    for (std::size_t i = 0; i < n; ++i) {
        double p1 = gx[i];
        double p2 = gy[i];
        if (!first_order) {
            p1 -= w1[i];
            p2 -= w2[i];
        }
        tv += std::sqrt(p1 * p1 + p2 * p2);
    }
    double second = 0.0;
    if (!first_order) {
        detail::eps_sym(width, height, w1.data(), w2.data(), e11.data(), e22.data(),
                        e12.data(), scratch);
#pragma omp parallel for schedule(static) reduction(+ : second)
        for (std::size_t i = 0; i < n; ++i) {
            second += std::sqrt(e11[i] * e11[i] + e22[i] * e22[i] + 2.0 * e12[i] * e12[i]);
        }
    }
    double data = 0.0;
    if (options.data_term == TgvDataTerm::L1) {
#pragma omp parallel for schedule(static) reduction(+ : data)
        for (std::size_t i = 0; i < n; ++i) {
            data += options.lambda * static_cast<double>(confidence[i]) *
                    std::abs(u[i] - static_cast<double>(f[i]));
        }
    } else {
#pragma omp parallel for schedule(static) reduction(+ : data)
        for (std::size_t i = 0; i < n; ++i) {
            const double d = u[i] - static_cast<double>(f[i]);
            data += 0.5 * options.lambda * static_cast<double>(confidence[i]) * d * d;
        }
    }
    return options.alpha1 * tv + options.alpha0 * second + data;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

// Shared Chambolle-Pock (2011, Alg. 1) loop. `second_order` selects the TGV^2
// model (w active, q/eps dual) versus plain TV (w == 0, q absent).
TgvResult run_primal_dual(int width, int height, const std::vector<float>& f,
                          const std::vector<float>& confidence, std::vector<float>& u,
                          const TgvOptions& options, bool second_order) {
    validate(width, height, f, confidence, options);

    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> x(n);  // u
    if (u.size() == n) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(u[i]);
        }
    } else {
        init_from_data(width, height, f, confidence, x);
    }
    // Over-relaxed primal extragradient holders.
    std::vector<double> x_old(n);
    std::vector<double> x_bar(n);

    std::vector<double> w1(n, 0.0), w2(n, 0.0), w1_old(n), w2_old(n);
    std::vector<double> w1_bar(n, 0.0), w2_bar(n, 0.0);

    // Duals: p (gradient of u), q (symmetric gradient of w).
    std::vector<double> p1(n, 0.0), p2(n, 0.0);
    std::vector<double> q11(n, 0.0), q22(n, 0.0), q12(n, 0.0);

    std::vector<double> gx(n), gy(n);       // grad x_bar
    std::vector<double> e11(n), e22(n), e12(n), scratch;  // eps w_bar
    std::vector<double> gtp(n), gtp_y(n);   // grad^T p = A_x^T p1 + A_y^T p2
    std::vector<double> s1(n), s2(n);       // eps^T q

    // sigma = tau = 1/||K|| at the guaranteed bound ||K||^2 <= 12 (the plain
    // diagonal preconditioning of Pock & Chambolle 2011 collapses to this
    // constant step here, since grad^T grad <= 4, eps^T eps <= 3 and the two
    // blocks are as separated as the bound allows). theta = 1.
    const double step = 1.0 / std::sqrt(12.0);
    const double tau = step;
    const double sigma = step;

    const int check_every = 10;
    TgvResult result;
    result.iterations = options.max_iterations;
    for (int it = 0; it < options.max_iterations; ++it) {
        // --- dual update: p <- proj_{|.|<=a1}(p + sigma*K x_bar) ---
        detail::grad_forward(width, height, x_bar.data(), gx.data(), gy.data());
        for (std::size_t i = 0; i < n; ++i) {
            gx[i] = p1[i] + sigma * (gx[i] - w1_bar[i]);
            gy[i] = p2[i] + sigma * (gy[i] - w2_bar[i]);
        }
        project_dual2(n, options.alpha1, gx, gy);
        p1.swap(gx);
        p2.swap(gy);
        if (second_order) {
            detail::eps_sym(width, height, w1_bar.data(), w2_bar.data(), e11.data(),
                            e22.data(), e12.data(), scratch);
            for (std::size_t i = 0; i < n; ++i) {
                e11[i] = q11[i] + sigma * e11[i];
                e22[i] = q22[i] + sigma * e22[i];
                e12[i] = q12[i] + sigma * e12[i];
            }
            project_dual3(n, options.alpha0, e11, e22, e12);
            q11.swap(e11);
            q22.swap(e22);
            q12.swap(e12);
        }

        // --- primal update: x <- prox_{tau G}(x - tau K^T (p, q)) ---
        x_old = x;
        // u block: grad^T p = A_x^T p1 + A_y^T p2; x_tmp = x - tau * grad^T p.
        detail::adj_forward_x(width, height, p1.data(), gtp.data());
        detail::adj_forward_y(width, height, p2.data(), gtp_y.data());
        std::vector<double> x_tmp(n);
        for (std::size_t i = 0; i < n; ++i) {
            x_tmp[i] = x[i] - tau * (gtp[i] + gtp_y[i]);
        }
        data_prox(options.data_term, width, height, f, confidence, options.lambda, tau,
                  x_tmp, x);

        // w block: w <- w + tau*(p - eps^T q).
        if (second_order) {
            detail::eps_sym_transpose(width, height, q11.data(), q22.data(), q12.data(),
                                      s1.data(), s2.data());
            for (std::size_t i = 0; i < n; ++i) {
                w1[i] = w1[i] + tau * (p1[i] - s1[i]);
                w2[i] = w2[i] + tau * (p2[i] - s2[i]);
            }
        }

        // --- relaxed extrapolation (theta = 1) ---
        for (std::size_t i = 0; i < n; ++i) {
            x_bar[i] = 2.0 * x[i] - x_old[i];
        }
        if (second_order) {
            for (std::size_t i = 0; i < n; ++i) {
                w1_bar[i] = 2.0 * w1[i] - w1_old[i];
                w2_bar[i] = 2.0 * w2[i] - w2_old[i];
            }
            w1_old = w1;
            w2_old = w2;
        }

        // --- convergence: RMS change of u per iteration against RMS|u|+1,
        // evaluated every `check_every` iterations, using a deterministic
        // per-row partial sum (never an OpenMP reduction).
        if ((it + 1) % check_every == 0 || it + 1 == options.max_iterations) {
            std::vector<double> row_delta(static_cast<std::size_t>(height), 0.0);
            std::vector<double> row_abs(static_cast<std::size_t>(height), 0.0);
            const std::size_t w = static_cast<std::size_t>(width);
#pragma omp parallel for schedule(static)
            for (std::size_t y = 0; y < row_delta.size(); ++y) {
                double d = 0.0;
                double a = 0.0;
                const std::size_t r = y * w;
                for (std::size_t xx = 0; xx < w; ++xx) {
                    const double diff = x[r + xx] - x_old[r + xx];
                    d += diff * diff;
                    a += x[r + xx] * x[r + xx];
                }
                row_delta[y] = d;
                row_abs[y] = a;
            }
            double delta = 0.0;
            double absu = 0.0;
            for (std::size_t y = 0; y < row_delta.size(); ++y) {
                delta += row_delta[y];
                absu += row_abs[y];
            }
            const double rms_change = std::sqrt(delta / static_cast<double>(n));
            const double scale = std::sqrt(absu / static_cast<double>(n)) + 1.0;
            if (rms_change / scale < options.tolerance) {
                result.iterations = it + 1;
                result.converged = true;
                break;
            }
        }
    }

    result.primal_energy = primal_energy(options, width, height, f, confidence,
                                         !second_order, x, w1, w2);
    u.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        u[i] = static_cast<float>(x[i]);
    }
    return result;
}

TgvResult tgv2_denoise(int width, int height, const std::vector<float>& f,
                       const std::vector<float>& confidence, std::vector<float>& u,
                       const TgvOptions& options) {
    return run_primal_dual(width, height, f, confidence, u, options, true);
}

TgvResult tv_denoise(int width, int height, const std::vector<float>& f,
                     const std::vector<float>& confidence, std::vector<float>& u,
                     const TgvOptions& options) {
    return run_primal_dual(width, height, f, confidence, u, options, false);
}

}  // namespace alphashape::elevation