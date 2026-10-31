#include "common/algorithm/stereo/depth_filter.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

namespace {

/// The calibration parameters packed in the order try_project_point() expects.
/// The support test projects every candidate per pixel per neighbour, so it
/// packs once per view and calls the throw-free forward model directly rather
/// than going through Calibration::project() (which throws, and so cannot be
/// inlined into the loop).
struct Projector {
    CameraType type = CameraType::Frame;
    int width = 0;
    int height = 0;
    double params[kCalibrationParamCount] = {};

    explicit Projector(const Calibration& c)
        : type(c.type), width(c.width), height(c.height) {
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

    /// Project a camera-frame point to pixels; false where the model is
    /// undefined (Frame behind the image plane, Fisheye at the origin).
    bool operator()(const Eigen::Vector3d& p_cam, Eigen::Vector2d& pixel_out) const {
        const double p[3] = {p_cam.x(), p_cam.y(), p_cam.z()};
        double out[2] = {0.0, 0.0};
        if (!try_project_point<double>(type, width, height, params, p, out)) {
            return false;
        }
        pixel_out = Eigen::Vector2d(out[0], out[1]);
        return true;
    }
};


// Minimum number of supporting neighbours per mode (Schönberger 2016 geometric
// consistency / PMVS-style support, our own constants — to be tuned).
constexpr int kSupportMild = 1;
constexpr int kSupportModerate = 2;
constexpr int kSupportAggressive = 3;

// Speckle removal: a connected component is dropped when its size is below
// pixel_count / divisor, floored at kMinSpecklePixels. Ours, to be tuned.
constexpr std::size_t kSpeckleDivisorMild = 4000;
constexpr std::size_t kSpeckleDivisorModerate = 2000;
constexpr std::size_t kSpeckleDivisorAggressive = 1000;
constexpr std::size_t kMinSpecklePixels = 4;

// Two 4-connected pixels join the same component when their depths agree to
// within this relative difference. Ours, to be tuned.
constexpr double kSpeckleRelativeDepth = 0.03;

// Anything at or below this is treated as "behind the camera / no ray".
constexpr double kDepthEps = 1e-9;

std::string lower_ascii(const std::string& text) {
    std::string out = text;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

// Throws when a DepthView is unusable: no map, a map whose array sizes do not
// match its pixel count, or a calibration that disagrees with the map geometry.
void validate_view(const DepthView& view, const char* role) {
    const DepthMap* map = view.map;
    if (map == nullptr) {
        throw ValueError(std::string("filter_depth_map: ") + role + " depth map is null");
    }
    if (map->width < 0 || map->height < 0) {
        throw ValueError(std::string("filter_depth_map: ") + role + " map has negative size");
    }
    const std::size_t pixels = static_cast<std::size_t>(map->width) *
                               static_cast<std::size_t>(map->height);
    if (map->depths.size() != pixels) {
        throw ValueError(std::string("filter_depth_map: ") + role +
                         " depths array size does not match map size");
    }
    if (!map->normals.empty() && map->normals.size() != 3 * pixels) {
        throw ValueError(std::string("filter_depth_map: ") + role +
                         " normals array size does not match map size");
    }
    if (!map->confidence.empty() && map->confidence.size() != pixels) {
        throw ValueError(std::string("filter_depth_map: ") + role +
                         " confidence array size does not match map size");
    }
    if (!map->flags.empty() && map->flags.size() != pixels) {
        throw ValueError(std::string("filter_depth_map: ") + role +
                         " flags array size does not match map size");
    }
    if (view.calibration.width != map->width || view.calibration.height != map->height) {
        throw ValueError(std::string("filter_depth_map: ") + role +
                         " calibration size does not match its depth map");
    }
}

// Neighbours needed to keep a pixel; clamped to the number actually available
// (never below 1 when at least one neighbour exists; 0 when none do).
int required_support(FilterMode mode, int neighbour_count) {
    int req = 0;
    switch (mode) {
        case FilterMode::Disabled: req = 0; break;
        case FilterMode::Mild: req = kSupportMild; break;
        case FilterMode::Moderate: req = kSupportModerate; break;
        case FilterMode::Aggressive: req = kSupportAggressive; break;
    }
    if (neighbour_count <= 0) {
        return 0;
    }
    return std::min(req, neighbour_count);
}

std::size_t speckle_divisor(FilterMode mode) {
    switch (mode) {
        case FilterMode::Moderate: return kSpeckleDivisorModerate;
        case FilterMode::Aggressive: return kSpeckleDivisorAggressive;
        case FilterMode::Mild:
        case FilterMode::Disabled:
        default: return kSpeckleDivisorMild;
    }
}

}  // namespace

FilterMode parse_filter_mode(const std::string& text) {
    const std::string key = lower_ascii(text);
    if (key == "disabled") return FilterMode::Disabled;
    if (key == "mild") return FilterMode::Mild;
    if (key == "moderate") return FilterMode::Moderate;
    if (key == "aggressive") return FilterMode::Aggressive;
    throw ValueError("stereo::parse_filter_mode: unknown filter mode '" + text +
                     "' (valid: disabled, mild, moderate, aggressive)");
}

std::string to_string(FilterMode mode) {
    switch (mode) {
        case FilterMode::Disabled: return "disabled";
        case FilterMode::Mild: return "mild";
        case FilterMode::Moderate: return "moderate";
        case FilterMode::Aggressive: return "aggressive";
    }
    return "disabled";
}

DepthMap filter_depth_map(const DepthView& reference, const std::vector<DepthView>& neighbours,
                          const FilterOptions& options, FilterStats* stats) {
    validate_view(reference, "reference");
    for (const DepthView& view : neighbours) {
        validate_view(view, "neighbour");
    }

    const DepthMap& src = *reference.map;
    const int width = src.width;
    const int height = src.height;
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t ref_stride = static_cast<std::size_t>(width);

    DepthMap out;
    out.camera_id = src.camera_id;
    out.width = width;
    out.height = height;
    out.downscale = src.downscale;
    // Filtering changes the values, never the geometry they were estimated
    // against, so the fingerprint carries through unchanged and a filtered map
    // stays reusable by a later run. The output records WHICH mode produced it,
    // so a later run reuses it only when it asked for the same strength.
    out.alignment_fingerprint = src.alignment_fingerprint;
    out.filter_mode = static_cast<int>(options.mode);

    // Output arrays are always full-size: the caller asked for a filtered copy.
    // (`flags` is the exception: empty stays empty, so an unrepaired map is not
    // silently given a provenance array it never had.)
    out.depths.assign(pixels, -1.0f);
    out.normals.assign(src.normals.empty() ? 0 : 3 * pixels, 0.0f);
    out.confidence.assign(pixels, 0.0f);  // Confidence exists even for Disabled.
    // Provenance follows the pixel it describes: a repaired source pixel that
    // survives stays flagged (and a rejected one is cleared below with the rest
    // of its record), so re-filtering can never launder a synthetic depth into
    // a measured one. An empty source vector stays empty.
    out.flags = src.flags;

    const bool have_normals = !src.normals.empty();
    const bool disabled = options.mode == FilterMode::Disabled;
    const int neighbour_count = static_cast<int>(neighbours.size());
    const int need = required_support(options.mode, neighbour_count);
    const double rel_tol = options.relative_depth_tolerance;
    const double reproj_tol = options.reprojection_tolerance_px;

    // Reference pose (X_cam = R * X_world + t) and its transpose for world reconstruction.
    const Eigen::Matrix3d R_ref_T = reference.R.transpose();
    const Eigen::Matrix3d& R_ref = reference.R;
    const Eigen::Vector3d& t_ref = reference.t;

    // Packed calibrations for the two projections in the inner loop, so the
    // throw-free forward model is called directly (no exception, no try/catch).
    const Projector project_reference(reference.calibration);
    std::vector<Projector> project_neighbour;
    project_neighbour.reserve(neighbours.size());
    for (const DepthView& nb : neighbours) {
        project_neighbour.emplace_back(nb.calibration);
    }

    // Per-pixel raw support count, parallel-friendly and output-only.
    std::vector<float> support_count(pixels, 0.0f);
    std::vector<std::uint8_t> pass_support(pixels, 0);

    std::size_t input_valid = 0;
    std::size_t supported_total = 0;

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : input_valid, supported_total)
#endif
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * ref_stride +
                                    static_cast<std::size_t>(i);
            const float depth_f = src.depths[idx];
            if (!(depth_f > 0.0f)) {
                continue;  // includes the -1 sentinel and any non-positive value
            }
            ++input_valid;

            // Ray for this depth-map pixel, in the map's own convention.
            const double u = static_cast<double>(i) + 0.5;
            const double v = static_cast<double>(j) + 0.5;
            const Eigen::Vector3d bearing = reference.calibration.unproject({u, v});
            const double depth = static_cast<double>(depth_f);

            // World point this pixel asserts.
            const Eigen::Vector3d x_cam = bearing * depth;
            const Eigen::Vector3d x_world = R_ref_T * (x_cam - t_ref);

            int votes = 0;
            for (std::size_t nb_index = 0; nb_index < neighbours.size(); ++nb_index) {
                const DepthView& nb = neighbours[nb_index];
                const DepthMap& nmap = *nb.map;

                // World -> neighbour camera.
                const Eigen::Vector3d x_n = nb.R * x_world + nb.t;
                // Project through the throw-free model: false wherever it is
                // undefined (a Frame camera behind the image plane).
                Eigen::Vector2d uv_n;
                if (!project_neighbour[nb_index](x_n, uv_n)) {
                    continue;
                }
                const double col = std::floor(uv_n.x());
                const double row = std::floor(uv_n.y());
                if (col < 0.0 || row < 0.0 ||
                    col >= static_cast<double>(nmap.width) ||
                    row >= static_cast<double>(nmap.height)) {
                    continue;  // projects outside the neighbour map
                }
                const int ni = static_cast<int>(col);
                const int nj = static_cast<int>(row);
                const std::size_t nidx = static_cast<std::size_t>(nj) *
                                             static_cast<std::size_t>(nmap.width) +
                                         static_cast<std::size_t>(ni);
                const float n_depth_f = nmap.depths[nidx];
                if (!(n_depth_f > 0.0f)) {
                    continue;  // neighbour has no depth here: no vote
                }

                const double predicted = x_n.norm();
                if (predicted <= kDepthEps) {
                    continue;
                }
                const double n_depth = static_cast<double>(n_depth_f);
                if (std::abs(n_depth - predicted) / predicted >= rel_tol) {
                    continue;  // depths disagree
                }

                // Back-project the neighbour's own depth and project into the
                // reference to measure a reprojection error in pixels.
                const double nu = static_cast<double>(ni) + 0.5;
                const double nv = static_cast<double>(nj) + 0.5;
                const Eigen::Vector3d nb_bearing = nb.calibration.unproject({nu, nv});
                const Eigen::Vector3d nb_cam = nb_bearing * n_depth;
                const Eigen::Vector3d nb_world = nb.R.transpose() * (nb_cam - nb.t);
                const Eigen::Vector3d back_ref = R_ref * nb_world + t_ref;
                Eigen::Vector2d back_uv;
                if (!project_reference(back_ref, back_uv)) {
                    continue;  // undefined projection (Frame behind the plane)
                }
                const double du = back_uv.x() - u;
                const double dv = back_uv.y() - v;
                if (std::sqrt(du * du + dv * dv) >= reproj_tol) {
                    continue;  // reprojects too far from the reference pixel
                }

                ++votes;
            }

            support_count[idx] = static_cast<float>(votes);
            if (votes >= need) {
                pass_support[idx] = 1;
                ++supported_total;
            }
        }
    }

    // Seed the output from pixels that survived the support test.
    for (std::size_t idx = 0; idx < pixels; ++idx) {
        if (pass_support[idx] == 0) {
            continue;
        }
        out.depths[idx] = src.depths[idx];
        out.confidence[idx] = support_count[idx];
        if (have_normals) {
            out.normals[3 * idx + 0] = src.normals[3 * idx + 0];
            out.normals[3 * idx + 1] = src.normals[3 * idx + 1];
            out.normals[3 * idx + 2] = src.normals[3 * idx + 2];
        }
    }

    std::size_t rejected_speckle = 0;

    // Speckle removal: drop small 4-connected components whose neighbours agree
    // in depth. Skipped entirely for Disabled.
    if (!disabled && width > 0 && height > 0) {
        const std::size_t min_pixels =
            std::max(kMinSpecklePixels, pixels / speckle_divisor(options.mode));
        std::vector<std::uint8_t> visited(pixels, 0);
        std::vector<std::size_t> stack;
        std::vector<std::size_t> component;

        for (std::size_t start = 0; start < pixels; ++start) {
            if (pass_support[start] == 0 || visited[start] != 0) {
                continue;
            }
            component.clear();
            stack.clear();
            stack.push_back(start);
            visited[start] = 1;
            while (!stack.empty()) {
                const std::size_t cur = stack.back();
                stack.pop_back();
                component.push_back(cur);

                const int ci = static_cast<int>(cur % ref_stride);
                const int cj = static_cast<int>(cur / ref_stride);
                const float cdepth = out.depths[cur];

                const int di[4] = {1, -1, 0, 0};
                const int dj[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k) {
                    const int ni = ci + di[k];
                    const int nj = cj + dj[k];
                    if (ni < 0 || nj < 0 || ni >= width || nj >= height) {
                        continue;
                    }
                    const std::size_t nidx = static_cast<std::size_t>(nj) * ref_stride +
                                             static_cast<std::size_t>(ni);
                    if (pass_support[nidx] == 0 || visited[nidx] != 0) {
                        continue;
                    }
                    const float ndepth = out.depths[nidx];
                    const double denom = std::max(std::abs(static_cast<double>(cdepth)),
                                                  std::abs(static_cast<double>(ndepth)));
                    if (denom <= 0.0) {
                        continue;
                    }
                    if (std::abs(static_cast<double>(cdepth) - static_cast<double>(ndepth)) /
                            denom >= kSpeckleRelativeDepth) {
                        continue;  // a depth edge does not join components
                    }
                    visited[nidx] = 1;
                    stack.push_back(nidx);
                }
            }

            if (component.size() < min_pixels) {
                for (const std::size_t idx : component) {
                    out.depths[idx] = -1.0f;
                    out.confidence[idx] = 0.0f;
                    if (!out.flags.empty()) {
                        out.flags[idx] = 0U;  // the pixel is gone: so is its provenance
                    }
                    if (have_normals) {
                        out.normals[3 * idx + 0] = 0.0f;
                        out.normals[3 * idx + 1] = 0.0f;
                        out.normals[3 * idx + 2] = 0.0f;
                    }
                }
                rejected_speckle += component.size();
            }
        }
    }

    if (stats != nullptr) {
        stats->input = input_valid;
        stats->supported = supported_total - rejected_speckle;
        stats->rejected_support = input_valid - supported_total;
        stats->rejected_speckle = rejected_speckle;
    }
    return out;
}

}  // namespace alphashape::stereo

