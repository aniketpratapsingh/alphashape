// five_point.cpp -- five-point relative pose via the Stewenius / Engels /
// Nister 10x20 action-matrix method (ISPRS J. 2006, "Recent developments on
// direct relative orientation").  Clean-room implementation from the paper.
//
// Algorithm outline
// -----------------
// 1. Null space.  Each correspondence gives a row
//        q = vec(b2 * b1^T)  (row-major)
//    so the epipolar constraint is q . vec(E) = 0 with E stored row-major.
//    Stacking 5 rows into a 5x9 matrix and taking the 4 right singular vectors
//    for the smallest singular values gives X, Y, Z, W, with
//        E = x*X + y*Y + z*Z + W.
// 2. Polynomial helper.  A polynomial of total degree <= 3 in (x,y,z) is a
//    20-vector over this monomial order (first 10 = degree 3, eliminated):
//        0  x^3    1 x^2y   2 x^2z   3 xy^2   4 xyz
//        5  xz^2   6 y^3    7 y^2z   8 yz^2   9 z^3
//       10  x^2   11 xy   12 xz   13 y^2   14 yz   15 z^2   16 x  17 y
//       18  z     19 1
//    Multiplication maps exponents through a small lookup table.
// 3. The 10 cubic constraints, assembled with the helper: det(E) = 0 and the
//    9 entries of  2*E*E^T*E - trace(E*E^T)*E = 0  (E entries are affine in
//    x,y,z, so these are cubics).  Result: a 10x20 coefficient matrix M.
// 4. Action matrix for multiplication by z.  Splitting columns as
//    [degree-3 D | degree<=2 B] and solving the 10x10 left block gives
//        D = -C * B,   C = L^{-1} R   (L = left 10 cols, R = right 10 cols).
//    Multiplication by z maps B entirely into D union B, so the 10x10 action
//    matrix A (rows indexed by B) has row = -C.row(k) when z*m lands on the
//    degree-3 monomial k, and a unit row when z*m lands back in B.
// 5. Eigenvectors of A are proportional to [x^2,xy,xz,y^2,yz,z^2,x,y,z,1]
//    evaluated at a solution, so  x = v[6]/v[9], y = v[7]/v[9], z = v[8]/v[9].
//    Every real candidate is formed into E, normalised to unit Frobenius norm,
//    and verified against the 5 epipolar constraints and the cubic constraints.

#include "common/algorithm/sfm/five_point.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace alphashape::sfm {
namespace {

constexpr int kNumMonomials = 20;

// Coefficients of a polynomial of total degree <= 3 in (x, y, z).
using Poly = std::array<double, kNumMonomials>;

struct Exp {
    int a;  // exponent of x
    int b;  // exponent of y
    int c;  // exponent of z
};

constexpr std::array<Exp, kNumMonomials> kExps = {{
    {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1},
    {1, 0, 2}, {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3},
    {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1},
    {0, 0, 2}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0},
}};

// Indices of the 20 monomials useful for the algorithm steps above.
constexpr int kIdxX = 16;
constexpr int kIdxY = 17;
constexpr int kIdxZ = 18;
constexpr int kIdxOne = 19;

// Monomial index (0..19) for an exponent triple, or -1 when degree > 3.
struct MonTable {
    std::array<int, 343> lookup;  // (a*7 + b)*7 + c, a,b,c in [0,6]
    MonTable() {
        lookup.fill(-1);
        for (int i = 0; i < kNumMonomials; ++i) {
            const Exp& e = kExps[static_cast<std::size_t>(i)];
            lookup[static_cast<std::size_t>((e.a * 7 + e.b) * 7 + e.c)] = i;
        }
    }
    int get(int a, int b, int c) const {
        return lookup[static_cast<std::size_t>((a * 7 + b) * 7 + c)];
    }
};

const MonTable& mon_table() {
    static const MonTable table;
    return table;
}

Poly zero_poly() {
    Poly p{};
    p.fill(0.0);
    return p;
}

// p * q, valid when deg(p) + deg(q) <= 3.
Poly mul(const Poly& p, const Poly& q) {
    Poly r = zero_poly();
    const MonTable& table = mon_table();
    for (int i = 0; i < kNumMonomials; ++i) {
        const double pv = p[static_cast<std::size_t>(i)];
        if (pv == 0.0) {
            continue;
        }
        const Exp& ei = kExps[static_cast<std::size_t>(i)];
        for (int j = 0; j < kNumMonomials; ++j) {
            const double qv = q[static_cast<std::size_t>(j)];
            if (qv == 0.0) {
                continue;
            }
            const Exp& ej = kExps[static_cast<std::size_t>(j)];
            const int k = table.get(ei.a + ej.a, ei.b + ej.b, ei.c + ej.c);
            assert(k >= 0);  // degree > 3 must never occur
            r[static_cast<std::size_t>(k)] += pv * qv;
        }
    }
    return r;
}

Poly add(const Poly& p, const Poly& q) {
    Poly r = zero_poly();
    for (int i = 0; i < kNumMonomials; ++i) {
        const auto s = static_cast<std::size_t>(i);
        r[s] = p[s] + q[s];
    }
    return r;
}

Poly sub(const Poly& p, const Poly& q) {
    Poly r = zero_poly();
    for (int i = 0; i < kNumMonomials; ++i) {
        const auto s = static_cast<std::size_t>(i);
        r[s] = p[s] - q[s];
    }
    return r;
}

Poly scale(const Poly& p, double s) {
    Poly r = zero_poly();
    for (int i = 0; i < kNumMonomials; ++i) {
        r[static_cast<std::size_t>(i)] = p[static_cast<std::size_t>(i)] * s;
    }
    return r;
}

// A single linear monomial c*x + ... or the plain constant 1.
Poly const_poly(double c) {
    Poly p = zero_poly();
    p[static_cast<std::size_t>(kIdxOne)] = c;
    return p;
}

Poly var_poly(int monomial_index, double c) {
    Poly p = zero_poly();
    p[static_cast<std::size_t>(monomial_index)] = c;
    return p;
}

// Build a 10x20 coefficient matrix whose rows are the cubic constraints
// det(E)=0 and the 9 entries of 2*E*E^T*E - trace(E*E^T)*E = 0, expressed as
// polynomials in (x,y,z) with E = x*X + y*Y + z*Z + W.
Eigen::Matrix<double, 10, 20> constraint_matrix(const Eigen::Matrix3d& X,
                                                const Eigen::Matrix3d& Y,
                                                const Eigen::Matrix3d& Z,
                                                const Eigen::Matrix3d& W) {
    // 3x3 of polynomials for E.
    Poly E[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const auto r = static_cast<Eigen::Index>(i);
            const auto c = static_cast<Eigen::Index>(j);
            E[i][j] = const_poly(W(r, c));
            E[i][j] = add(E[i][j], scale(var_poly(kIdxX, 1.0), X(r, c)));
            E[i][j] = add(E[i][j], scale(var_poly(kIdxY, 1.0), Y(r, c)));
            E[i][j] = add(E[i][j], scale(var_poly(kIdxZ, 1.0), Z(r, c)));
        }
    }

    // EEt = E * E^T  (degree-2 polynomials).
    Poly EEt[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EEt[i][j] = zero_poly();
            for (int k = 0; k < 3; ++k) {
                EEt[i][j] = add(EEt[i][j], mul(E[i][k], E[j][k]));
            }
        }
    }
    // trace(E*E^T)
    Poly trace = add(add(EEt[0][0], EEt[1][1]), EEt[2][2]);

    Eigen::Matrix<double, 10, 20> M;
    M.setZero();

    // Row 0: det(E).
    Poly det = zero_poly();
    {
        Poly t0 = mul(E[0][0], sub(mul(E[1][1], E[2][2]), mul(E[1][2], E[2][1])));
        Poly t1 = mul(E[0][1], sub(mul(E[1][0], E[2][2]), mul(E[1][2], E[2][0])));
        Poly t2 = mul(E[0][2], sub(mul(E[1][0], E[2][1]), mul(E[1][1], E[2][0])));
        det = add(sub(t0, t1), t2);
    }
    for (int c = 0; c < kNumMonomials; ++c) {
        M(0, c) = det[static_cast<std::size_t>(c)];
    }

    // Rows 1..9: 2*E*EEt - trace*E.
    int row = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Poly p = zero_poly();
            for (int k = 0; k < 3; ++k) {
                p = add(p, mul(EEt[i][k], E[k][j]));
            }
            p = scale(p, 2.0);
            p = sub(p, mul(trace, E[i][j]));
            for (int c = 0; c < kNumMonomials; ++c) {
                M(row, c) = p[static_cast<std::size_t>(c)];
            }
            ++row;
        }
    }
    return M;
}

// Basis of the action matrix (degree <= 2 monomials), in this order:
// [x^2, xy, xz, y^2, yz, z^2, x, y, z, 1].
constexpr std::array<int, 10> kBasis = {
    10, 11, 12, 13, 14, 15, kIdxX, kIdxY, kIdxZ, kIdxOne};

// Row of the action matrix for multiplication by z applied to basis element i.
int z_row(int i) {
    switch (kBasis[static_cast<std::size_t>(i)]) {
        case kIdxX:  // z*x = xz  -> monomial 12
            return 12;
        case kIdxY:  // z*y = yz  -> monomial 14
            return 14;
        case kIdxZ:  // z*z = z^2 -> monomial 15
            return 15;
        case 10:  // z*x^2 = x^2 z -> monomial 2
            return 2;
        case 11:  // z*xy = xyz   -> monomial 4
            return 4;
        case 12:  // z*xz = xz^2  -> monomial 5
            return 5;
        case 13:  // z*y^2 = y^2 z -> monomial 7
            return 7;
        case 14:  // z*yz = yz^2  -> monomial 8
            return 8;
        case 15:  // z*z^2 = z^3 -> monomial 9
            return 9;
        default:  // z*1 = z -> monomial 18
            return 18;
    }
}

// Solve the ten cubic constraints for (x,y,z) and return E candidates.
std::vector<Eigen::Matrix3d> solve(const Eigen::Matrix3d& X, const Eigen::Matrix3d& Y,
                                   const Eigen::Matrix3d& Z, const Eigen::Matrix3d& W) {
    const Eigen::Matrix<double, 10, 20> M = constraint_matrix(X, Y, Z, W);
    const Eigen::Matrix<double, 10, 10> L = M.leftCols<10>();
    const Eigen::Matrix<double, 10, 10> R = M.rightCols<10>();
    const Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> lu(L);
    if (!lu.isInvertible()) {
        return {};
    }
    const Eigen::Matrix<double, 10, 10> C = lu.solve(R);  // D = -C * B

    // Build the action matrix for multiplication by z.
    Eigen::Matrix<double, 10, 10> A;
    A.setZero();
    for (int i = 0; i < 10; ++i) {
        const int m = z_row(i);
        if (m < 10) {  // z*m is an eliminated degree-3 monomial
            A.row(i) = -C.row(static_cast<Eigen::Index>(m));
        } else {  // z*m remains in the basis: unit row
            for (int j = 0; j < 10; ++j) {
                if (kBasis[static_cast<std::size_t>(j)] == m) {
                    A(i, j) = 1.0;
                    break;
                }
            }
        }
    }

    const Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> es(A, /*computeEigenvectors=*/true);
    const Eigen::Matrix<std::complex<double>, 10, 10> V = es.eigenvectors();

    const Eigen::Index ix = 6;  // position of x   in the basis
    const Eigen::Index iy = 7;  // position of y   in the basis
    const Eigen::Index iz = 8;  // position of z   in the basis
    const Eigen::Index i1 = 9;  // position of 1   in the basis

    std::vector<Eigen::Matrix3d> solutions;
    for (int k = 0; k < 10; ++k) {
        const auto kk = static_cast<Eigen::Index>(k);
        const std::complex<double> v1 = V(i1, kk);
        const std::complex<double> vx = V(ix, kk);
        const std::complex<double> vy = V(iy, kk);
        const std::complex<double> vz = V(iz, kk);
        const double mag = std::abs(v1);
        if (mag < 1e-12) {
            continue;
        }
        if (std::abs(vx.imag()) > 1e-8 * mag || std::abs(vy.imag()) > 1e-8 * mag ||
            std::abs(vz.imag()) > 1e-8 * mag || std::abs(v1.imag()) > 1e-8 * mag) {
            continue;
        }
        const double x = vx.real() / v1.real();
        const double y = vy.real() / v1.real();
        const double z = vz.real() / v1.real();
        Eigen::Matrix3d E = x * X + y * Y + z * Z + W;
        const double fro = E.norm();
        if (fro < 1e-12) {
            continue;
        }
        E /= fro;
        solutions.push_back(E);
    }
    return solutions;
}

}  // namespace

std::vector<Eigen::Matrix3d> five_point(const std::array<Eigen::Vector3d, 5>& b1,
                                        const std::array<Eigen::Vector3d, 5>& b2) {
    // Build the 5x9 epipolar constraint matrix, rows = vec(b2 * b1^T) row-major.
    Eigen::Matrix<double, 5, 9> Q;
    for (int i = 0; i < 5; ++i) {
        const auto r = static_cast<Eigen::Index>(i);
        const Eigen::Vector3d& a = b1[static_cast<std::size_t>(i)];
        const Eigen::Vector3d& b = b2[static_cast<std::size_t>(i)];
        for (int u = 0; u < 3; ++u) {
            for (int v = 0; v < 3; ++v) {
                Q(r, static_cast<Eigen::Index>(u * 3 + v)) =
                    b(static_cast<Eigen::Index>(u)) * a(static_cast<Eigen::Index>(v));
            }
        }
    }

    const Eigen::JacobiSVD<Eigen::Matrix<double, 5, 9>> svd(Q, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 9, 9> V = svd.matrixV();
    // Last four right singular vectors span the (4-dimensional) null space.
    const auto to_mat = [&](Eigen::Index col) -> Eigen::Matrix3d {
        Eigen::Matrix3d m;
        const Eigen::VectorXd vec = V.col(col);
        for (int u = 0; u < 3; ++u) {
            for (int v = 0; v < 3; ++v) {
                m(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) =
                    vec(static_cast<Eigen::Index>(u * 3 + v));
            }
        }
        return m;
    };
    const Eigen::Matrix3d X = to_mat(5);
    const Eigen::Matrix3d Y = to_mat(6);
    const Eigen::Matrix3d Z = to_mat(7);
    const Eigen::Matrix3d W = to_mat(8);

    std::vector<Eigen::Matrix3d> candidates = solve(X, Y, Z, W);

    // Verify every candidate against the epipolar constraints and reject the
    // ones that do not actually satisfy the essential-matrix conditions.
    std::vector<Eigen::Matrix3d> result;
    for (const Eigen::Matrix3d& E : candidates) {
        double max_epi = 0.0;
        for (int i = 0; i < 5; ++i) {
            const double res = b2[static_cast<std::size_t>(i)].dot(E * b1[static_cast<std::size_t>(i)]);
            max_epi = std::max(max_epi, std::abs(res));
        }
        const double det = std::abs(E.determinant());
        // Essential matrix: two equal singular values, one zero.
        const Eigen::JacobiSVD<Eigen::Matrix3d> esvd(E);
        const Eigen::Vector3d sv = esvd.singularValues();
        const double rel = std::abs(sv(0) - sv(1)) / std::max(sv(0), 1e-30);
        if (max_epi < 1e-8 && det < 1e-8 && rel < 1e-5 && sv(2) < 1e-6) {
            result.push_back(E);
        }
    }
    return result;
}

}  // namespace alphashape::sfm
