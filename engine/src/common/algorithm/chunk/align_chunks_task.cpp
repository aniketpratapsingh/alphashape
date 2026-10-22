// AlignChunks: marker-based alignment of the active chunk onto a reference
// chunk (M11 §4.8, WP-11.7).
//
// The estimation is the published one (Umeyama 1991) already used by
// Georeference and AlignCameras: `sfm::estimate_similarity_ransac` with the
// deterministic seed 0x5eed. The task adds the chunk-level parts:
//
//   * marker correspondence by exact label, using each marker's ESTIMATED
//     position in its own chunk (`Marker::position`), so both sides are in the
//     same kind of coordinate;
//   * the RANSAC threshold, `3 x median sigma` of the marker positions, from
//     the per-marker covariance (a marker with a zero covariance contributes
//     the plan's floor of 0.01 m rather than collapsing the threshold to 0);
//   * a post-fit RMS gate (`accuracy_m`, 0 = auto = the threshold) before
//     anything is written, so a bad fit never half-transforms a chunk;
//   * the similarity applied to the active chunk and recorded in BOTH chunks'
//     meta as "AlignChunks/<other_id>/matrix".
//
// Direction. The fitted similarity maps the ACTIVE chunk's marker positions to
// the REFERENCE chunk's: dst = s R src + t with src in the active frame. Applying
// it to the active chunk moves it into the reference frame, which is what
// "align the active chunk onto the reference chunk" means in §4.8 and what the
// WP-11.7 acceptance test measures (camera centres match the reference's).
//
// Rotation and scale act on CHUNK-LOCAL coordinates, so a camera pose moves as
//   pose' = pose . S^-1      with S = [sR | t],
// because a world point p becomes S p, and X_cam = pose (S p) must equal
// pose' p. The translation therefore enters the pose as pose . S^-1, not as a
// left-sided composition. Translation-only degenerate fits (1 or 2 distinct
// markers) cannot be estimated by Umeyama and are refused.

#include "common/algorithm/chunk/align_chunks_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/sfm/similarity.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk_geometry.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"

namespace alphashape {
namespace {

/// Floor for the auto threshold. Marker position covariances are frequently
/// zero (a marker triangulated from exactly two rays, or imported without
/// uncertainty), which would otherwise make a zero-width inlier band and
/// reject every point.
constexpr double kMinThresholdM = 0.01;

/// A fit needs three non-collinear points: Umeyama's own test rejects a
/// collinear triple, and a similarity from less than three cannot fix scale and
/// rotation.
constexpr std::size_t kMinMarkers = 3;

/// Formatted with 17 significant digits so a saved matrix round-trips exactly.
std::string format_matrix(const Eigen::Matrix4d& matrix) {
    std::ostringstream out;
    out.precision(17);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (row != 0 || col != 0) {
                out << ' ';
            }
            out << matrix(row, col);
        }
    }
    return out.str();
}

/// One matched marker pair, in active-chunk and reference-chunk coordinates.
struct MarkerPair {
    Eigen::Vector3d active;
    Eigen::Vector3d reference;
    double sigma = 0.0;  ///< joint position sigma used by the auto threshold
};

/// Per-marker 1-sigma position uncertainty. A marker's covariance contributes
/// its largest axis; a zero covariance falls back to the plan's floor.
double marker_sigma(const Marker& marker) {
    const Eigen::Matrix3d& covariance = marker.position_covariance;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return kMinThresholdM;
    }
    const double largest = solver.eigenvalues().maxCoeff();
    if (!(largest > 0.0) || !std::isfinite(largest)) {
        return kMinThresholdM;
    }
    return std::max(std::sqrt(largest), kMinThresholdM);
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    const double upper = values[mid];
    if (values.size() % 2 != 0) {
        return upper;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                     values.end());
    return 0.5 * (values[mid - 1] + upper);
}

/// Markers whose labels match exactly and whose estimated position is set.
/// Ids are not compared: a marker pair is a *label* pair (Metashape
/// semantics), and the two chunks are expected to hold different id spaces.
std::vector<MarkerPair> match_markers(const Chunk& active, const Chunk& reference) {
    std::unordered_map<std::string, const Marker*> by_label;
    by_label.reserve(reference.markers().size());
    for (const Marker& marker : reference.markers()) {
        if (!marker.position.has_value()) {
            continue;
        }
        // A duplicated label in the reference is ambiguous: skip every copy
        // rather than pick one arbitrarily.
        if (by_label.find(marker.label) != by_label.end()) {
            by_label[marker.label] = nullptr;
            continue;
        }
        by_label.emplace(marker.label, &marker);
    }

    std::vector<MarkerPair> pairs;
    for (const Marker& marker : active.markers()) {
        if (!marker.position.has_value()) {
            continue;
        }
        const auto it = by_label.find(marker.label);
        if (it == by_label.end() || it->second == nullptr) {
            continue;
        }
        MarkerPair pair;
        pair.active = *marker.position;
        pair.reference = *it->second->position;
        pair.sigma = 0.5 * (marker_sigma(marker) + marker_sigma(*it->second));
        pairs.push_back(pair);
    }
    return pairs;
}

/// True when three or more matched pairs span two dimensions -- Umeyama cannot
/// fix rotation while every point lies on one line, and §4.8 requires
/// non-collinear common markers.
bool non_collinear(const std::vector<MarkerPair>& pairs) {
    if (pairs.size() < kMinMarkers) {
        return false;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const MarkerPair& pair : pairs) {
        mean += pair.active;
    }
    mean /= static_cast<double>(pairs.size());
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (const MarkerPair& pair : pairs) {
        const Eigen::Vector3d centred = pair.active - mean;
        scatter += centred * centred.transpose();
    }
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(scatter);
    const Eigen::Vector3d values = svd.singularValues();
    // Same test Umeyama's solver documents: second singular value / first.
    return values(0) > 0.0 && values(1) / values(0) > 1e-6;
}

}  // namespace

void AlignChunksTask::declare_params(Params& params) const {
    params.declare({"reference_chunk_id", ParamType::Int, std::int64_t{-1},
                    "Id of the chunk the active chunk is aligned onto. -1 takes "
                    "the first other chunk in the workspace."});
    params.declare({"method", ParamType::String, std::string("markers"),
                    "Alignment method: \"markers\" (label-matched marker "
                    "positions) or \"reference\" (both chunks georeferenced in "
                    "the same CRS; only the CRS placement is recorded)."});
    params.declare({"min_common_markers", ParamType::Int, std::int64_t{3},
                    "Refuse an alignment that matches fewer than this many "
                    "labelled markers with an estimated position."});
    params.declare({"accuracy_m", ParamType::Float, 0.0,
                    "Maximum accepted RMS of the marker residuals, in metres. "
                    "0 selects the RANSAC threshold (3 x median marker sigma)."});
}

std::vector<Stage> AlignChunksTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"estimate", 0}, Stage{"transform", 0},
            Stage{"finalize", 0}};
}

namespace {

/// Resolve the reference chunk: an explicit id, or the first chunk that is not
/// the active one. Returns nullptr when there is none.
Chunk* resolve_reference(Document& document, const Chunk& active,
                         std::int64_t requested) {
    if (requested < 0) {
        for (const auto& chunk : document.workspace().chunks()) {
            if (chunk.get() != &active) {
                return chunk.get();
            }
        }
        return nullptr;
    }
    if (requested > std::int64_t{INT32_MAX}) {
        throw ValueError("AlignChunks: reference_chunk_id " +
                         std::to_string(requested) + " is out of range");
    }
    for (const auto& chunk : document.workspace().chunks()) {
        if (chunk->id() == static_cast<std::int32_t>(requested)) {
            return chunk.get();
        }
    }
    throw ValueError("AlignChunks: no chunk with id " + std::to_string(requested));
}

/// RANSAC threshold from the marker sigmas: 3 x the median joint sigma, never
/// below the floor. This is the §4.8 rule; a caller can override the residual
/// gate (accuracy_m) but not the inlier band.
double ransac_threshold(const std::vector<MarkerPair>& pairs) {
    std::vector<double> sigmas;
    sigmas.reserve(pairs.size());
    for (const MarkerPair& pair : pairs) {
        sigmas.push_back(pair.sigma);
    }
    return std::max(3.0 * median(std::move(sigmas)), kMinThresholdM);
}

/// Apply a similarity (dst = s R p + t) to one chunk-local geometry state.
/// Camera poses are world->camera, and their world side is what the similarity
/// rewrites, so each pose is post-multiplied by the INVERSE similarity:
/// pose' = pose . S^-1 with S = [sR | t].
void apply_similarity_impl(ChunkGeometry& geometry,
                           const sfm::Similarity& similarity) {
    const Eigen::Matrix4d similarity_matrix = similarity.matrix();
    Eigen::Matrix4d inverse = Eigen::Matrix4d::Zero();
    // Closed form rather than .inverse(): S^-1 = [R^T/s | -R^T t/s].
    inverse.block<3, 3>(0, 0) = similarity.R.transpose() / similarity.scale;
    inverse.block<3, 1>(0, 3) =
        -(similarity.R.transpose() * similarity.t) / similarity.scale;
    inverse(3, 3) = 1.0;

    for (Transform& pose : geometry.camera_poses) {
        pose = pose * inverse;
    }
    for (std::optional<Eigen::Vector3d>& point : geometry.track_points) {
        if (point.has_value()) {
            *point = similarity.apply(*point);
        }
    }
    for (std::optional<Eigen::Vector3d>& position : geometry.marker_positions) {
        if (position.has_value()) {
            *position = similarity.apply(*position);
        }
    }
    for (std::vector<Eigen::Vector3d>& vertices : geometry.shape_vertices) {
        for (Eigen::Vector3d& vertex : vertices) {
            vertex = similarity.apply(vertex);
        }
    }
    for (std::vector<Eigen::Vector3d>& positions : geometry.trajectory_positions) {
        for (Eigen::Vector3d& position : positions) {
            position = similarity.apply(position);
        }
    }
    for (std::shared_ptr<PointCloud>& cloud : geometry.point_clouds) {
        if (!cloud) {
            continue;
        }
        // Clone on write: the snapshot may share this payload with the chunk
        // it was captured from.
        cloud = std::make_shared<PointCloud>(*cloud);
        for (Eigen::Vector3d& point : cloud->points) {
            point = similarity.apply(point);
        }
        if (!cloud->normals.empty()) {
            for (Eigen::Vector3f& normal : cloud->normals) {
                const Eigen::Vector3d rotated =
                    similarity.R * normal.cast<double>();
                normal = rotated.normalized().cast<float>();
            }
        }
    }
    for (std::shared_ptr<Mesh>& model : geometry.models) {
        if (!model) {
            continue;
        }
        model = std::make_shared<Mesh>(*model);
        for (Eigen::Vector3d& vertex : model->vertices) {
            vertex = similarity.apply(vertex);
        }
    }

    // The region is an axis-aligned box in the old frame; its corners map to a
    // rotated box, so the region becomes the axis-aligned bound of the mapped
    // corners (the same rule merge_chunks uses).
    const Eigen::Vector3d half = 0.5 * geometry.region.size;
    Eigen::Vector3d low = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    Eigen::Vector3d high = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    for (int corner = 0; corner < 8; ++corner) {
        Eigen::Vector3d offset;
        offset.x() = (corner & 1) != 0 ? half.x() : -half.x();
        offset.y() = (corner & 2) != 0 ? half.y() : -half.y();
        offset.z() = (corner & 4) != 0 ? half.z() : -half.z();
        const Eigen::Vector3d mapped =
            similarity.apply(geometry.region.centre + offset);
        low = low.cwiseMin(mapped);
        high = high.cwiseMax(mapped);
    }
    geometry.region.centre = 0.5 * (low + high);
    geometry.region.size = high - low;

    // The chunk transform is the local->world placement; the similarity acts on
    // chunk-local coordinates, so it is composed on the RIGHT: T' = T . S.
    geometry.transform = geometry.transform * similarity_matrix;
}

void set_meta(std::vector<std::pair<std::string, std::string>>& items,
              const std::string& key, const std::string& value) {
    for (auto& item : items) {
        if (item.first == key) {
            item.second = value;
            return;
        }
    }
    items.emplace_back(key, value);
}

std::string matrix_key(std::int32_t other_chunk_id) {
    return "AlignChunks/" + std::to_string(other_chunk_id) + "/matrix";
}

}  // namespace


void apply_similarity_to_geometry(ChunkGeometry& geometry,
                                  const sfm::Similarity& similarity) {
    if (!(similarity.scale > 0.0) || !std::isfinite(similarity.scale) ||
        !similarity.R.allFinite() || !similarity.t.allFinite()) {
        throw ValueError("align_chunks: the similarity is degenerate (scale " +
                         std::to_string(similarity.scale) + ")");
    }
    apply_similarity_impl(geometry, similarity);
}

MarkerAlignmentEstimate estimate_marker_alignment(const Chunk& active,
                                                  const Chunk& reference,
                                                  std::size_t min_common_markers,
                                                  double accuracy_m) {
    MarkerAlignmentEstimate estimate;
    const std::vector<MarkerPair> pairs = match_markers(active, reference);
    estimate.markers_matched = pairs.size();
    const std::size_t required = std::max(kMinMarkers, min_common_markers);
    if (pairs.size() < required) {
        estimate.reason = "align_chunks: need at least " + std::to_string(required) +
                          " common markers (matched " +
                          std::to_string(pairs.size()) + ")";
        return estimate;
    }
    if (!non_collinear(pairs)) {
        estimate.reason = "align_chunks: the common markers are collinear";
        return estimate;
    }

    std::vector<Eigen::Vector3d> src;
    std::vector<Eigen::Vector3d> dst;
    src.reserve(pairs.size());
    dst.reserve(pairs.size());
    for (const MarkerPair& pair : pairs) {
        src.push_back(pair.active);
        dst.push_back(pair.reference);
    }
    estimate.threshold_m = ransac_threshold(pairs);
    const sfm::SimilarityRansacResult ransac =
        sfm::estimate_similarity_ransac(src, dst, estimate.threshold_m, 0x5eed);
    if (!ransac.similarity.has_value()) {
        estimate.reason = "align_chunks: the marker similarity fit failed";
        return estimate;
    }
    estimate.similarity = *ransac.similarity;
    estimate.markers_used = ransac.inlier_count;
    if (estimate.markers_used < kMinMarkers) {
        estimate.reason = "align_chunks: only " +
                          std::to_string(estimate.markers_used) +
                          " markers agree with the fitted similarity";
        return estimate;
    }

    // Residual gate on the inliers: the fit is accepted only if it explains
    // the agreeing markers to the requested accuracy (auto: the inlier band).
    double squared = 0.0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        if (ransac.inliers.size() == pairs.size() && ransac.inliers[i] == 0) {
            continue;
        }
        squared += (estimate.similarity.apply(src[i]) - dst[i]).squaredNorm();
    }
    estimate.rms_m =
        std::sqrt(squared / static_cast<double>(estimate.markers_used));
    const double gate = accuracy_m > 0.0 ? accuracy_m : estimate.threshold_m;
    if (!(estimate.rms_m <= gate)) {
        estimate.reason = "align_chunks: marker RMS " +
                          std::to_string(estimate.rms_m) +
                          " m exceeds the accuracy gate " + std::to_string(gate) + " m";
        return estimate;
    }
    estimate.ok = true;
    return estimate;
}

void AlignChunksTask::apply(Document& document, Params& params,
                            const Progress& progress) {
    progress.update("initialize", 0.0);

    Chunk* active = document.active_chunk();
    if (active == nullptr) {
        throw Error("AlignChunks needs an active chunk");
    }
    const std::int64_t requested = params.get_int("reference_chunk_id");
    Chunk* reference = resolve_reference(document, *active, requested);
    if (reference == nullptr) {
        throw ValueError("AlignChunks: the workspace holds only the active chunk; "
                         "reference_chunk_id is required");
    }
    if (reference == active) {
        throw ValueError("AlignChunks: the reference chunk must not be the active chunk");
    }
    const std::string method = params.get_string("method");
    if (method != "markers" && method != "reference") {
        throw ValueError("AlignChunks: unknown method '" + method +
                         "' (use \"markers\" or \"reference\")");
    }
    const std::int64_t min_common = params.get_int("min_common_markers");
    if (min_common < 0) {
        throw ValueError("AlignChunks: min_common_markers must not be negative");
    }
    const double accuracy = params.get_float("accuracy_m");
    if (!(accuracy >= 0.0) || !std::isfinite(accuracy)) {
        throw ValueError("AlignChunks: accuracy_m must be finite and >= 0");
    }
    progress.update("initialize", 1.0);

    const std::int32_t active_id = active->id();
    const std::int32_t reference_id = reference->id();

    // -- estimate ---------------------------------------------------------
    progress.update("estimate", 0.0);
    sfm::Similarity similarity;
    std::size_t markers_used = 0;
    double rms_m = 0.0;
    std::string reason;

    if (method == "reference") {
        // Both chunks are already placed by their CRS; nothing to transform,
        // the placement is the identity in the shared frame.
        if (!active->crs().is_set() || !reference->crs().is_set()) {
            throw Error("align_chunks: method=reference requires both chunks to be "
                        "georeferenced");
        }
        if (!(active->crs() == reference->crs())) {
            throw Error("align_chunks: method=reference requires both chunks to share "
                        "a CRS (active epsg " + std::to_string(active->crs().epsg) +
                        ", reference epsg " + std::to_string(reference->crs().epsg) + ")");
        }
        reason = "reference CRS";
    } else {
        const MarkerAlignmentEstimate estimate = estimate_marker_alignment(
            *active, *reference,
            static_cast<std::size_t>(std::max<std::int64_t>(min_common, 0)), accuracy);
        if (!estimate.ok) {
            throw Error(estimate.reason);
        }
        similarity = estimate.similarity;
        markers_used = estimate.markers_used;
        rms_m = estimate.rms_m;
        reason = "markers";
        if (Log::enabled(LogLevel::Debug)) {
            log(LogLevel::Debug) << "AlignChunks: rms " << rms_m << " m, threshold "
                                 << estimate.threshold_m << " m, scale "
                                 << similarity.scale << ", inliers " << markers_used
                                 << "/" << estimate.markers_matched;
        }
    }
    progress.update("estimate", 1.0);

    // -- transform --------------------------------------------------------
    // Build the post-state first; a structural mismatch throws before the
    // document is touched.
    progress.update("transform", 0.0);
    ChunkGeometry after = capture_geometry(*active);
    apply_similarity_to_geometry(after, similarity);
    const Eigen::Matrix4d applied = similarity.matrix();
    set_meta(after.meta, matrix_key(reference_id), format_matrix(applied));
    progress.update("transform", 1.0);

    // -- finalize ---------------------------------------------------------
    // Both edits are commands, executed as one atomic step, so a caller that
    // keeps the command (or drives the task through UndoStack) can undo the
    // alignment as a single entry.
    progress.update("finalize", 0.0);
    CommandTransaction transaction("Align Chunks");
    transaction.add(make_align_chunks_command(active_id, std::move(after)));
    transaction.add(make_set_chunk_meta_command(
        reference_id, matrix_key(active_id), format_matrix(applied.inverse())));
    transaction.apply(document);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "AlignChunks: applied=true markers_used="
                            << markers_used << " rms_m=" << rms_m
                            << " reason=" << reason;
    }
}

}  // namespace alphashape
