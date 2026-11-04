// Posterior covariance for bundle adjustment: sigma0^2 * (J^T J)^-1 for the
// requested blocks. The point columns are eliminated with the Schur complement
// (Triggs et al. 2000, sec. 4; Forstner & Wrobel 2016) so the cost is linear in
// the number of points. See covariance.hpp for the contract.

#include "common/algorithm/sfm/covariance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <ceres/crs_matrix.h>
#include <ceres/manifold.h>

#include "model/camera/calibration.hpp"

namespace alphashape::sfm {
namespace {

/// A point's normal block is treated as singular below this eigenvalue ratio
/// (one camera cannot fix all three coordinates of a point).
constexpr double kPointRankTolerance = 1e-12;

/// One free camera or calibration parameter block, in Jacobian column order.
/// `ambient` maps every tangent column to its ambient component: Ceres drops
/// SubsetManifold-held components from the tangent space.
struct ReducedBlock {
    enum class Kind { Rotation, Translation, Calibration };
    Kind kind = Kind::Rotation;
    std::size_t entity = 0;
    int start = 0;
    int size = 0;
    std::vector<int> ambient;
};

/// Column layout of the Jacobian: camera/intrinsic columns first, then three
/// columns per registered point.
struct ColumnLayout {
    int reduced = 0;
    int total = 0;
    std::vector<ReducedBlock> blocks;
    std::vector<int> column_block;  // reduced block index, or -1
    std::vector<int> column_point;  // point index, or -1
    std::vector<int> point_start;   // first column per point, or -1
};

/// Dense block accumulator keyed by an ordered pair of reduced blocks. Holds
/// Hcc and the -Hcp Hpp^-1 Hpc contribution to the Schur complement.
using BlockMap = std::unordered_map<std::uint64_t, Eigen::MatrixXd>;

std::uint64_t pair_key(std::size_t a, std::size_t b, std::size_t count) {
    const std::size_t lo = a < b ? a : b;
    const std::size_t hi = a < b ? b : a;
    return static_cast<std::uint64_t>(lo * count + hi);
}

/// Returns the dense block for (a, b) and reports whether the caller's order
/// was swapped relative to the stored (lower-key) orientation.
Eigen::MatrixXd& block_at(BlockMap* map, const std::vector<int>& sizes, std::size_t a,
                          std::size_t b, bool* transposed) {
    *transposed = a > b;
    const std::size_t lo = a > b ? b : a;
    const std::size_t hi = a > b ? a : b;
    const std::uint64_t key = pair_key(lo, hi, sizes.size());
    auto it = map->find(key);
    if (it == map->end()) {
        it = map->emplace(key, Eigen::MatrixXd::Zero(sizes[lo], sizes[hi])).first;
    }
    return it->second;
}

/// Adds sign * (ua * ub^T) to block (a, b).
void add_outer(BlockMap* map, const std::vector<int>& sizes, std::size_t a,
               const std::vector<double>& ua, std::size_t b,
               const std::vector<double>& ub, double sign) {
    bool transposed = false;
    Eigen::MatrixXd& block = block_at(map, sizes, a, b, &transposed);
    const int rows = static_cast<int>(ua.size());
    const int cols = static_cast<int>(ub.size());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const double value = sign * ua[static_cast<std::size_t>(i)] *
                                 ub[static_cast<std::size_t>(j)];
            if (transposed) {
                block(j, i) += value;
            } else {
                block(i, j) += value;
            }
        }
    }
}

/// Adds sign * (wa * wb^T) to block (a, b); wa and wb are (block size x k).
void add_gram(BlockMap* map, const std::vector<int>& sizes, std::size_t a,
              const Eigen::MatrixXd& wa, std::size_t b, const Eigen::MatrixXd& wb,
              double sign) {
    bool transposed = false;
    Eigen::MatrixXd& block = block_at(map, sizes, a, b, &transposed);
    const Eigen::MatrixXd product = wa * wb.transpose();
    if (transposed) {
        block += sign * product.transpose();
    } else {
        block += sign * product;
    }
}

/// Appends one free Ceres block to the layout and to the Evaluate request.
void add_block_layout(ceres::Problem& problem, double* data, int ambient_size,
                      ReducedBlock::Kind kind, std::size_t entity,
                      ColumnLayout* layout, std::vector<double*>* eval_blocks) {
    if (data == nullptr || !problem.HasParameterBlock(data) ||
        problem.IsParameterBlockConstant(data)) {
        return;
    }
    const int tangent = problem.ParameterBlockTangentSize(data);
    if (tangent <= 0) {
        return;
    }
    ReducedBlock block;
    block.kind = kind;
    block.entity = entity;
    block.start = layout->reduced;
    block.size = tangent;
    block.ambient.resize(static_cast<std::size_t>(tangent));
    for (int k = 0; k < tangent; ++k) {
        block.ambient[static_cast<std::size_t>(k)] = k;
    }
    // A SubsetManifold drops components from the tangent space. Recover which
    // ambient component each tangent column stands for by probing Plus().
    if (tangent != ambient_size) {
        const ceres::Manifold* manifold = problem.GetManifold(data);
        if (manifold != nullptr && manifold->AmbientSize() == ambient_size) {
            std::vector<double> delta(static_cast<std::size_t>(tangent), 0.0);
            std::vector<double> plus(static_cast<std::size_t>(ambient_size), 0.0);
            for (int k = 0; k < tangent; ++k) {
                std::fill(delta.begin(), delta.end(), 0.0);
                delta[static_cast<std::size_t>(k)] = 1.0;
                manifold->Plus(data, delta.data(), plus.data());
                int best = 0;
                double best_change = -1.0;
                for (int a = 0; a < ambient_size; ++a) {
                    const double change =
                        std::abs(plus[static_cast<std::size_t>(a)] - data[a]);
                    if (change > best_change) {
                        best_change = change;
                        best = a;
                    }
                }
                block.ambient[static_cast<std::size_t>(k)] = best;
            }
        }
    }
    layout->reduced += tangent;
    layout->blocks.push_back(std::move(block));
    eval_blocks->push_back(data);
}

ColumnLayout build_layout(ceres::Problem& problem, const CovarianceBlocks& blocks,
                          std::vector<double*>* eval_blocks) {
    ColumnLayout layout;
    for (std::size_t c = 0; c < blocks.cameras.size(); ++c) {
        add_block_layout(problem, blocks.cameras[c], 3, ReducedBlock::Kind::Rotation,
                         c, &layout, eval_blocks);
        add_block_layout(problem, blocks.cameras[c] + 3, 3,
                         ReducedBlock::Kind::Translation, c, &layout, eval_blocks);
    }
    for (std::size_t s = 0; s < blocks.calibrations.size(); ++s) {
        add_block_layout(problem, blocks.calibrations[s], kCalibrationParamCount,
                         ReducedBlock::Kind::Calibration, s, &layout, eval_blocks);
    }
    layout.point_start.assign(blocks.points.size(), -1);
    layout.total = layout.reduced;
    for (std::size_t p = 0; p < blocks.points.size(); ++p) {
        double* point = blocks.points[p];
        if (point == nullptr || !problem.HasParameterBlock(point) ||
            problem.IsParameterBlockConstant(point) ||
            problem.ParameterBlockTangentSize(point) != 3) {
            continue;
        }
        layout.point_start[p] = layout.total;
        layout.total += 3;
        eval_blocks->push_back(point);
    }
    layout.column_block.assign(static_cast<std::size_t>(layout.total), -1);
    layout.column_point.assign(static_cast<std::size_t>(layout.total), -1);
    for (std::size_t b = 0; b < layout.blocks.size(); ++b) {
        const ReducedBlock& block = layout.blocks[b];
        for (int k = 0; k < block.size; ++k) {
            layout.column_block[static_cast<std::size_t>(block.start + k)] =
                static_cast<int>(b);
        }
    }
    for (std::size_t p = 0; p < layout.point_start.size(); ++p) {
        if (layout.point_start[p] < 0) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            layout.column_point[static_cast<std::size_t>(layout.point_start[p] + k)] =
                static_cast<int>(p);
        }
    }
    return layout;
}

}  // namespace

CovarianceOutput compute_covariance(ceres::Problem& problem,
                                    const CovarianceBlocks& blocks,
                                    const CovarianceRequest& request,
                                    double sigma0_squared) {
    CovarianceOutput out;
    const std::size_t camera_count = blocks.cameras.size();
    const std::size_t point_count = blocks.points.size();
    const std::size_t calibration_count = blocks.calibrations.size();

    bool want_cameras = request.cameras;
    bool want_points = request.points;
    bool want_calibrations = request.calibrations;
    std::string reason;
    if (want_cameras && camera_count > request.max_cameras) {
        want_cameras = false;
        reason = "camera covariance skipped: " + std::to_string(camera_count) +
                 " cameras exceed the cap of " + std::to_string(request.max_cameras);
    }
    if (want_points && point_count > request.max_points) {
        want_points = false;
        if (!reason.empty()) {
            reason += "; ";
        }
        reason += "point covariance skipped: " + std::to_string(point_count) +
                  " points exceed the cap of " + std::to_string(request.max_points);
    }
    if (!want_cameras && !want_points && !want_calibrations) {
        out.reason = reason.empty() ? "covariance skipped: not requested" : reason;
        return out;
    }
    if (!(sigma0_squared >= 0.0) || !std::isfinite(sigma0_squared)) {
        out.reason = "covariance skipped: sigma0 squared is not finite";
        return out;
    }

    std::vector<double*> eval_blocks;
    const ColumnLayout layout = build_layout(problem, blocks, &eval_blocks);
    if (eval_blocks.empty() || layout.total == 0) {
        out.reason = "covariance skipped: no free parameters";
        return out;
    }
    if (layout.reduced > static_cast<int>(request.max_columns)) {
        out.reason = "covariance skipped: " + std::to_string(layout.reduced) +
                     " free camera/intrinsic columns exceed the cap of " +
                     std::to_string(request.max_columns);
        return out;
    }
    try {
        // Evaluate returns tangent-space columns in eval_blocks order; the
        // robust loss is applied so the covariance matches the solved objective.
        ceres::Problem::EvaluateOptions eval;
        eval.apply_loss_function = true;
        eval.parameter_blocks = eval_blocks;
        ceres::CRSMatrix jacobian;
        double cost = 0.0;
        if (!problem.Evaluate(eval, &cost, nullptr, nullptr, &jacobian)) {
            out.reason = "covariance computation failed: Jacobian evaluation failed";
            return out;
        }
        const int row_count = jacobian.num_rows;
        const std::vector<int>& row_begin = jacobian.rows;
        const std::vector<int>& entry_cols = jacobian.cols;
        const std::vector<double>& entry_values = jacobian.values;
        std::fprintf(stderr,
                     "[cov] cols=%d layout_total=%d reduced=%d blocks=%zu points=%zu "
                     "registered=%zu rows=%d\n",
                     jacobian.num_cols, layout.total, layout.reduced,
                     layout.blocks.size(), point_count, eval_blocks.size(), row_count);
        if (jacobian.num_cols != layout.total) {
            out.reason = "covariance computation failed: Jacobian has " +
                         std::to_string(jacobian.num_cols) +
                         " columns, expected " + std::to_string(layout.total);
            return out;
        }

        // Hpp for every point, from its own three Jacobian columns.
        std::vector<Eigen::Matrix3d> point_normal(point_count,
                                                  Eigen::Matrix3d::Zero());
        for (int r = 0; r < row_count; ++r) {
            const int begin = row_begin[static_cast<std::size_t>(r)];
            const int end = row_begin[static_cast<std::size_t>(r) + 1];
            std::size_t points[2] = {0, 0};
            Eigen::Vector3d values[2] = {Eigen::Vector3d::Zero(),
                                         Eigen::Vector3d::Zero()};
            int npoints = 0;
            for (int j = begin; j < end; ++j) {
                const int col = entry_cols[static_cast<std::size_t>(j)];
                const int p = layout.column_point[static_cast<std::size_t>(col)];
                if (p < 0) {
                    continue;
                }
                int slot = -1;
                for (int t = 0; t < npoints; ++t) {
                    if (points[t] == static_cast<std::size_t>(p)) {
                        slot = t;
                    }
                }
                if (slot < 0) {
                    slot = npoints++;
                    points[slot] = static_cast<std::size_t>(p);
                }
                values[slot][col - layout.point_start[static_cast<std::size_t>(p)]] =
                    entry_values[static_cast<std::size_t>(j)];
            }
            for (int t = 0; t < npoints; ++t) {
                point_normal[points[t]] += values[t] * values[t].transpose();
            }
        }

        // Drop rank-deficient point blocks (for example a point seen by one
        // camera) together with their contribution to the normal equations.
        std::vector<std::uint8_t> point_included(point_count, 0);
        std::size_t included_count = 0;
        for (std::size_t p = 0; p < point_count; ++p) {
            if (layout.point_start[p] < 0) {
                continue;
            }
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
                point_normal[p]);
            const double largest = eigen.eigenvalues().maxCoeff();
            const double smallest = eigen.eigenvalues().minCoeff();
            if (largest > 0.0 && smallest > kPointRankTolerance * largest) {
                point_included[p] = 1;
                ++included_count;
            } else {
                ++out.points_dropped;
            }
        }
        // A row contributes to Hcc unless every point it touches was dropped.
        std::vector<std::uint8_t> row_included(static_cast<std::size_t>(row_count), 1);
        std::vector<int> point_row_count(point_count, 0);
        for (int r = 0; r < row_count; ++r) {
            const int begin = row_begin[static_cast<std::size_t>(r)];
            const int end = row_begin[static_cast<std::size_t>(r) + 1];
            bool has_point = false;
            bool any_included = false;
            std::size_t seen[2] = {0, 0};
            int nseen = 0;
            for (int j = begin; j < end; ++j) {
                const int col = entry_cols[static_cast<std::size_t>(j)];
                const int p = layout.column_point[static_cast<std::size_t>(col)];
                if (p < 0) {
                    continue;
                }
                has_point = true;
                bool duplicate = false;
                for (int t = 0; t < nseen; ++t) {
                    if (seen[t] == static_cast<std::size_t>(p)) {
                        duplicate = true;
                    }
                }
                if (duplicate) {
                    continue;
                }
                if (nseen < 2) {
                    seen[nseen++] = static_cast<std::size_t>(p);
                }
                ++point_row_count[static_cast<std::size_t>(p)];
                if (point_included[static_cast<std::size_t>(p)] != 0) {
                    any_included = true;
                }
            }
            row_included[static_cast<std::size_t>(r)] =
                static_cast<std::uint8_t>(!has_point || any_included);
        }

        std::vector<int> point_row_offset(point_count + 1, 0);
        for (std::size_t p = 0; p < point_count; ++p) {
            point_row_offset[p + 1] = point_row_offset[p] + point_row_count[p];
        }
        std::vector<int> point_rows(
            static_cast<std::size_t>(point_row_offset[point_count]), 0);
        std::vector<int> cursor(point_row_offset.begin(), point_row_offset.end() - 1);
        for (int r = 0; r < row_count; ++r) {
            const int begin = row_begin[static_cast<std::size_t>(r)];
            const int end = row_begin[static_cast<std::size_t>(r) + 1];
            std::size_t seen[2] = {0, 0};
            int nseen = 0;
            for (int j = begin; j < end; ++j) {
                const int col = entry_cols[static_cast<std::size_t>(j)];
                const int p = layout.column_point[static_cast<std::size_t>(col)];
                if (p < 0) {
                    continue;
                }
                bool duplicate = false;
                for (int t = 0; t < nseen; ++t) {
                    if (seen[t] == static_cast<std::size_t>(p)) {
                        duplicate = true;
                    }
                }
                if (duplicate) {
                    continue;
                }
                if (nseen < 2) {
                    seen[nseen++] = static_cast<std::size_t>(p);
                }
                int& position = cursor[static_cast<std::size_t>(p)];
                point_rows[static_cast<std::size_t>(position)] = r;
                ++position;
            }
        }
        std::vector<int> block_sizes(layout.blocks.size());
        for (std::size_t b = 0; b < layout.blocks.size(); ++b) {
            block_sizes[b] = layout.blocks[b].size;
        }
        BlockMap accumulator;
        const auto accumulator_norm = [&accumulator]() {
            double sum = 0.0;
            for (const auto& entry : accumulator) {
                sum += entry.second.cwiseAbs().sum();
            }
            return sum;
        };
        for (int r = 0; r < row_count; ++r) {
            if (row_included[static_cast<std::size_t>(r)] == 0) {
                continue;
            }
            const int begin = row_begin[static_cast<std::size_t>(r)];
            const int end = row_begin[static_cast<std::size_t>(r) + 1];
            std::vector<std::pair<int, std::vector<double>>> groups;
            for (int j = begin; j < end; ++j) {
                const int col = entry_cols[static_cast<std::size_t>(j)];
                const int b = layout.column_block[static_cast<std::size_t>(col)];
                if (b < 0) {
                    continue;
                }
                if (groups.empty() || groups.back().first != b) {
                    groups.emplace_back(
                        b, std::vector<double>(
                               static_cast<std::size_t>(
                                   block_sizes[static_cast<std::size_t>(b)]),
                               0.0));
                }
                const ReducedBlock& block = layout.blocks[static_cast<std::size_t>(b)];
                groups.back().second[static_cast<std::size_t>(col - block.start)] =
                    entry_values[static_cast<std::size_t>(j)];
            }
            for (std::size_t i = 0; i < groups.size(); ++i) {
                for (std::size_t j = i; j < groups.size(); ++j) {
                    add_outer(&accumulator, block_sizes,
                              static_cast<std::size_t>(groups[i].first),
                              groups[i].second,
                              static_cast<std::size_t>(groups[j].first),
                              groups[j].second, 1.0);
                }
            }
        }
        const double hcc_norm = accumulator_norm();
        struct PointWork {
            std::size_t point = 0;
            Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
            std::vector<int> columns;
            Eigen::MatrixXd weight;  // Hcp * Hpp^-1, one row per column
        };
        std::vector<PointWork> point_work;
        double weight_norm = 0.0;
        for (std::size_t p = 0; p < point_count; ++p) {
            if (point_included[p] == 0) {
                continue;
            }
            const int start = layout.point_start[p];
            std::vector<int> columns;
            std::vector<Eigen::Vector3d> cross;
            for (int idx = point_row_offset[p]; idx < point_row_offset[p + 1]; ++idx) {
                const int r = point_rows[static_cast<std::size_t>(idx)];
                const int begin = row_begin[static_cast<std::size_t>(r)];
                const int end = row_begin[static_cast<std::size_t>(r) + 1];
                Eigen::Vector3d own = Eigen::Vector3d::Zero();
                for (int j = begin; j < end; ++j) {
                    const int col = entry_cols[static_cast<std::size_t>(j)];
                    if (layout.column_point[static_cast<std::size_t>(col)] ==
                        static_cast<int>(p)) {
                        own[col - start] = entry_values[static_cast<std::size_t>(j)];
                    }
                }
                for (int j = begin; j < end; ++j) {
                    const int col = entry_cols[static_cast<std::size_t>(j)];
                    if (layout.column_block[static_cast<std::size_t>(col)] < 0) {
                        continue;
                    }
                    std::size_t slot = columns.size();
                    for (std::size_t t = 0; t < columns.size(); ++t) {
                        if (columns[t] == col) {
                            slot = t;
                            break;
                        }
                    }
                    if (slot == columns.size()) {
                        columns.push_back(col);
                        cross.push_back(Eigen::Vector3d::Zero());
                    }
                    cross[slot] += entry_values[static_cast<std::size_t>(j)] * own;
                }
            }
            if (columns.empty()) {
                continue;
            }
            std::vector<std::size_t> order(columns.size());
            for (std::size_t t = 0; t < order.size(); ++t) {
                order[t] = t;
            }
            std::sort(order.begin(), order.end(), [&columns](std::size_t a, std::size_t b) {
                return columns[a] < columns[b];
            });
            PointWork work;
            work.point = p;
            work.columns.resize(columns.size());
            work.weight.resize(static_cast<Eigen::Index>(columns.size()), 3);
            const Eigen::LLT<Eigen::Matrix3d> llt(point_normal[p]);
            if (llt.info() != Eigen::Success) {
                continue;
            }
            work.inverse = llt.solve(Eigen::Matrix3d::Identity());
            for (std::size_t t = 0; t < order.size(); ++t) {
                work.columns[t] = columns[order[t]];
                work.weight.row(static_cast<Eigen::Index>(t)) =
                    cross[order[t]].transpose();
            }
            work.weight = work.weight * work.inverse;
            weight_norm += work.weight.norm();
            std::vector<std::pair<int, std::vector<int>>> by_block;
            for (std::size_t t = 0; t < work.columns.size(); ++t) {
                const int b =
                    layout.column_block[static_cast<std::size_t>(work.columns[t])];
                if (by_block.empty() || by_block.back().first != b) {
                    by_block.emplace_back(b, std::vector<int>());
                }
                by_block.back().second.push_back(static_cast<int>(t));
            }
            std::vector<Eigen::MatrixXd> local(by_block.size());
            for (std::size_t i = 0; i < by_block.size(); ++i) {
                const ReducedBlock& block =
                    layout.blocks[static_cast<std::size_t>(by_block[i].first)];
                local[i] = Eigen::MatrixXd::Zero(block.size, 3);
                for (const int t : by_block[i].second) {
                    const int row =
                        work.columns[static_cast<std::size_t>(t)] - block.start;
                    local[i].row(row) = work.weight.row(t);
                }
            }
            for (std::size_t i = 0; i < by_block.size(); ++i) {
                for (std::size_t j = i; j < by_block.size(); ++j) {
                    add_gram(&accumulator, block_sizes,
                             static_cast<std::size_t>(by_block[i].first), local[i],
                             static_cast<std::size_t>(by_block[j].first), local[j],
                             -1.0);
                }
            }
            if (want_points) {
                point_work.push_back(std::move(work));
            }
        }
        std::fprintf(stderr,
                     "[cov] included=%zu work=%zu accum_blocks=%zu dropped=%zu "
                     "hcc_norm=%g schur_norm=%g weight_norm=%g\n",
                     included_count, point_work.size(), accumulator.size(),
                     out.points_dropped, hcc_norm, accumulator_norm(), weight_norm);
        Eigen::SparseMatrix<double> normal;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        if (layout.reduced > 0) {
            const std::size_t block_count = layout.blocks.size();
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(accumulator.size() * 9);
            for (const auto& entry : accumulator) {
                const std::size_t lo =
                    static_cast<std::size_t>(entry.first / block_count);
                const std::size_t hi =
                    static_cast<std::size_t>(entry.first % block_count);
                const Eigen::MatrixXd& values = entry.second;
                for (int i = 0; i < static_cast<int>(values.rows()); ++i) {
                    for (int j = 0; j < static_cast<int>(values.cols()); ++j) {
                        const double value = values(i, j);
                        if (value == 0.0) {
                            continue;
                        }
                        const int row = layout.blocks[lo].start + i;
                        const int col = layout.blocks[hi].start + j;
                        triplets.emplace_back(row, col, value);
                        if (lo != hi) {
                            triplets.emplace_back(col, row, value);
                        }
                    }
                }
            }
            normal = Eigen::SparseMatrix<double>(layout.reduced, layout.reduced);
            normal.setFromTriplets(triplets.begin(), triplets.end());
            normal.makeCompressed();
            ldlt.compute(normal);
            if (ldlt.info() != Eigen::Success) {
                out.reason =
                    "covariance: normal matrix is singular (free network?)";
                return out;
            }
            const auto& diagonal = ldlt.vectorD();
            const double largest = diagonal.cwiseAbs().maxCoeff();
            std::fprintf(stderr, "[cov] pivot_ratio=%g\n",
                         largest > 0.0 ? diagonal.minCoeff() / largest : 0.0);
            if (!(largest > 0.0) ||
                diagonal.minCoeff() <= kPointRankTolerance * largest) {
                out.reason =
                    "covariance: normal matrix is singular (free network?)";
                return out;
            }
        }
        if (want_cameras) {
            out.camera_covariance.assign(
                camera_count, Eigen::Matrix<double, 6, 6>::Zero());
            for (std::size_t c = 0; c < camera_count; ++c) {
                std::vector<int> global_columns;
                std::vector<int> ambient;
                for (const ReducedBlock& block : layout.blocks) {
                    if (block.entity != c) {
                        continue;
                    }
                    if (block.kind != ReducedBlock::Kind::Rotation &&
                        block.kind != ReducedBlock::Kind::Translation) {
                        continue;
                    }
                    const int base =
                        block.kind == ReducedBlock::Kind::Rotation ? 0 : 3;
                    for (int k = 0; k < block.size; ++k) {
                        global_columns.push_back(block.start + k);
                        ambient.push_back(
                            base + block.ambient[static_cast<std::size_t>(k)]);
                    }
                }
                if (global_columns.empty() || layout.reduced == 0) {
                    continue;  // constant camera (or constant block): zero
                }
                Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(
                    layout.reduced, static_cast<Eigen::Index>(global_columns.size()));
                for (std::size_t k = 0; k < global_columns.size(); ++k) {
                    rhs(global_columns[k], static_cast<Eigen::Index>(k)) = 1.0;
                }
                const Eigen::MatrixXd solution = ldlt.solve(rhs);
                Eigen::Matrix<double, 6, 6> block =
                    Eigen::Matrix<double, 6, 6>::Zero();
                for (std::size_t a = 0; a < global_columns.size(); ++a) {
                    for (std::size_t b = 0; b < global_columns.size(); ++b) {
                        block(ambient[a], ambient[b]) =
                            solution(global_columns[a],
                                     static_cast<Eigen::Index>(b));
                    }
                }
                out.camera_covariance[c] = block * sigma0_squared;
            }
        }
        if (want_calibrations) {
            const Eigen::Index n = static_cast<Eigen::Index>(kCalibrationParamCount);
            out.calibration_covariance.assign(
                calibration_count, Eigen::MatrixXd::Zero(n, n));
            for (std::size_t s = 0; s < calibration_count; ++s) {
                const ReducedBlock* block = nullptr;
                for (const ReducedBlock& candidate : layout.blocks) {
                    if (candidate.entity == s &&
                        candidate.kind == ReducedBlock::Kind::Calibration) {
                        block = &candidate;
                    }
                }
                if (block == nullptr || layout.reduced == 0) {
                    continue;  // fully held calibration: zero
                }
                Eigen::MatrixXd rhs =
                    Eigen::MatrixXd::Zero(layout.reduced, block->size);
                for (int k = 0; k < block->size; ++k) {
                    rhs(block->start + k, k) = 1.0;
                }
                const Eigen::MatrixXd solution = ldlt.solve(rhs);
                Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(n, n);
                for (int a = 0; a < block->size; ++a) {
                    for (int b = 0; b < block->size; ++b) {
                        covariance(block->ambient[static_cast<std::size_t>(a)],
                                   block->ambient[static_cast<std::size_t>(b)]) =
                            solution(block->start + a, b);
                    }
                }
                out.calibration_covariance[s] = covariance * sigma0_squared;
            }
        }
        if (want_points) {
            out.point_covariance.assign(point_count, Eigen::Matrix3d::Zero());
            for (const PointWork& work : point_work) {
                Eigen::Matrix3d covariance = work.inverse;
                if (layout.reduced > 0 && !work.columns.empty()) {
                    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(layout.reduced, 3);
                    for (std::size_t k = 0; k < work.columns.size(); ++k) {
                        rhs.row(work.columns[k]) =
                            work.weight.row(static_cast<Eigen::Index>(k));
                    }
                    const Eigen::MatrixXd solution = ldlt.solve(rhs);
                    for (std::size_t k = 0; k < work.columns.size(); ++k) {
                        covariance +=
                            work.weight.row(static_cast<Eigen::Index>(k))
                                .transpose() *
                            solution.row(work.columns[k]);
                    }
                }
                out.point_covariance[work.point] = covariance * sigma0_squared;
            }
        }
        out.valid = reason.empty();
        out.reason = reason.empty() ? std::string("ok") : reason;
        return out;
    } catch (const std::exception& error) {
        out.reason = std::string("covariance computation failed: ") + error.what();
        return out;
    } catch (...) {
        out.reason = "covariance computation failed";
        return out;
    }
}
}  // namespace alphashape::sfm
