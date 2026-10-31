#include "common/algorithm/stereo/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::stereo {
namespace {

/// One back-projected depth-map sample. `normal` is already in the chunk-local
/// (world) frame; `colour` is RGB in [0, 255]; `footprint` is the sample's
/// world-space pixel footprint (depth / f), or -1 when f is degenerate.
struct Sample {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    float colour[3] = {0.0F, 0.0F, 0.0F};
    float footprint = -1.0F;
    float confidence = 0.0F;
    bool has_colour = false;
    /// Synthesised by the post-filter repair pass rather than measured by the
    /// search. Such a sample is still merged (it is the only evidence some
    /// voxels have) but it carries a fraction of a measured sample's weight, so
    /// where both describe one voxel the measurement dominates the support.
    bool repaired = false;
};

/// Weight of one repaired sample relative to one measured sample in the merged
/// point cloud. A repaired depth is a plane's guess, so it must not outvote a
/// measurement; it is still merged, because for some voxels it is the only
/// evidence there is. Applied as a fraction of a MEASURED SAMPLE'S OWN weight
/// (support 1), not of the filtered confidence, so a repaired pixel can never
/// exceed a measured one whatever the filter's scale. 0.1 keeps a lone repaired
/// point above min_confidence 0 (so it still appears) while ten measured
/// samples outweigh a hundred repaired ones.
constexpr double kRepairedFusionWeight = 0.1;

/// Voxel grid key = floor(world / voxel) per axis.
struct VoxelKey {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    /// Lexicographic, so a sorted key list gives a deterministic output order.
    bool operator<(const VoxelKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
        const std::uint64_t parts[3] = {static_cast<std::uint64_t>(k.x),
                                        static_cast<std::uint64_t>(k.y),
                                        static_cast<std::uint64_t>(k.z)};
        for (std::uint64_t v : parts) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

struct VoxelAccumulator {
    Eigen::Vector3d position_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3f normal_sum = Eigen::Vector3f::Zero();
    double colour_sum[3] = {0.0, 0.0, 0.0};
    double confidence_sum = 0.0;
    std::int64_t count = 0;
    std::int64_t colour_count = 0;
};

void accumulate(VoxelAccumulator& acc, const Sample& s) {
    acc.position_sum += s.position;
    acc.normal_sum += s.normal;
    if (s.has_colour) {
        acc.colour_sum[0] += static_cast<double>(s.colour[0]);
        acc.colour_sum[1] += static_cast<double>(s.colour[1]);
        acc.colour_sum[2] += static_cast<double>(s.colour[2]);
        ++acc.colour_count;
    }
    acc.confidence_sum += static_cast<double>(s.confidence);
    ++acc.count;
}

std::int64_t floor_div(double v, double voxel) {
    return static_cast<std::int64_t>(std::floor(v / voxel));
}

/// Mean of the `ds`x`ds` full-resolution block covering depth-map pixel
/// (i, j). A 1-channel image is replicated to RGB; >=3 channels use the first
/// three.
void block_colour(const Image& image, int i, int j, int ds, float out[3]) {
    const int used = image.channels() >= 3 ? 3 : 1;
    const int x0 = i * ds;
    const int y0 = j * ds;
    double sum[3] = {0.0, 0.0, 0.0};
    for (int dy = 0; dy < ds; ++dy) {
        for (int dx = 0; dx < ds; ++dx) {
            for (int c = 0; c < used; ++c) {
                sum[c] += static_cast<double>(image.at(x0 + dx, y0 + dy, c));
            }
        }
    }
    const double n = static_cast<double>(ds) * static_cast<double>(ds);
    for (int c = 0; c < used; ++c) {
        out[c] = static_cast<float>(sum[c] / n);
    }
    if (used == 1) {
        out[1] = out[0];
        out[2] = out[0];
    }
}

void push_colour(PointCloud& cloud, double mean) {
    const long rounded = std::lround(mean);
    const long clamped = rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded);
    cloud.colours.push_back(static_cast<std::uint8_t>(clamped));
}

}  // namespace

/// Automatic voxel size: the median world-space pixel footprint across every
/// valid sample (depth / f per map). 0.0 when no sample yields a footprint.
double auto_voxel_size(const std::vector<std::vector<Sample>>& per_input) {
    std::vector<float> footprints;
    for (const std::vector<Sample>& samples : per_input) {
        for (const Sample& s : samples) {
            if (s.footprint > 0.0F) {
                footprints.push_back(s.footprint);
            }
        }
    }
    if (footprints.empty()) {
        return 0.0;
    }
    const std::size_t mid = footprints.size() / 2;
    std::nth_element(footprints.begin(), footprints.begin() + static_cast<std::ptrdiff_t>(mid),
                     footprints.end());
    const float hi = footprints[mid];
    if (footprints.size() % 2 == 1) {
        return static_cast<double>(hi);
    }
    const float lo = *std::max_element(footprints.begin(), footprints.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (static_cast<double>(lo) + static_cast<double>(hi));
}

/// Validate one input and return its map's sample count check flags. Throws
/// ValueError on any structural inconsistency.
const DepthMap& validate_input(const FusionInput& in, std::size_t index) {
    const std::string prefix = "stereo::fuse_depth_maps: input " + std::to_string(index) + ": ";
    if (in.map == nullptr) {
        throw ValueError(prefix + "null depth map");
    }
    const DepthMap& map = *in.map;
    if (map.width <= 0 || map.height <= 0) {
        throw ValueError(prefix + "depth map has non-positive size");
    }
    if (map.downscale < 1) {
        throw ValueError(prefix + "depth map has invalid downscale");
    }
    if (in.calibration.width != map.width || in.calibration.height != map.height) {
        throw ValueError(prefix + "calibration size does not match the depth map");
    }
    const std::size_t pixels = static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height);
    if (map.depths.size() != pixels) {
        throw ValueError(prefix + "depth buffer size does not match the map");
    }
    if (!map.normals.empty() && map.normals.size() != pixels * 3) {
        throw ValueError(prefix + "normal buffer size does not match the map");
    }
    if (!map.confidence.empty() && map.confidence.size() != pixels) {
        throw ValueError(prefix + "confidence buffer size does not match the map");
    }
    if (!map.flags.empty() && map.flags.size() != pixels) {
        throw ValueError(prefix + "flags buffer size does not match the map");
    }
    if (in.image != nullptr) {
        const std::size_t need_w = static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.downscale);
        const std::size_t need_h = static_cast<std::size_t>(map.height) * static_cast<std::size_t>(map.downscale);
        if (static_cast<std::size_t>(in.image->width()) < need_w ||
            static_cast<std::size_t>(in.image->height()) < need_h) {
            throw ValueError(prefix + "image is smaller than the depth map's full resolution");
        }
        if (in.image->channels() < 1) {
            throw ValueError(prefix + "image has no channels");
        }
    }
    return map;
}

PointCloud fuse_depth_maps(const std::vector<FusionInput>& inputs, const FusionOptions& options) {
    PointCloud cloud;

    // -- validate every input up front --------------------------------------
    bool any_normals = false;
    bool all_normals = !inputs.empty();
    bool all_confidence = !inputs.empty();
    bool all_image = !inputs.empty();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const DepthMap& map = validate_input(inputs[i], i);
        if (!map.normals.empty()) {
            any_normals = true;
        } else {
            all_normals = false;
        }
        if (map.confidence.empty()) {
            all_confidence = false;
        }
        if (inputs[i].image == nullptr) {
            all_image = false;
        }
    }
    // A normal is only emitted when *every* contributing map supplies one;
    // likewise a meaningful summed confidence needs the container everywhere.
    const bool use_normals = any_normals && all_normals;
    // A repaired pixel is only ever produced on a FILTERED map, which always
    // carries a confidence buffer, so the weighted sum below (taken only when
    // every input has one) is where a repaired sample's low weight applies.
    // Colours need both the option and an image on *every* input: a voxel can be
    // fed by several maps, and a map with no image contributes nothing to its
    // colour. Emitting colours while some inputs are image-less would leave
    // those voxels black, so require an image everywhere.
    const bool coloured = options.colours && all_image;

    // -- back-project every valid pixel into per-input sample buffers --------
    // Each input writes only its own buffer, so the loop is embarrassingly
    // parallel; the merge below is serial and in input order, keeping the
    // result independent of the thread count.
    std::vector<std::vector<Sample>> per_input(inputs.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::int64_t input_index = 0;
         input_index < static_cast<std::int64_t>(inputs.size()); ++input_index) {
        const FusionInput& in = inputs[static_cast<std::size_t>(input_index)];
        const DepthMap& map = *in.map;
        const int ds = map.downscale;
        const bool want_colour = coloured && in.image != nullptr;
        std::vector<Sample>& out = per_input[static_cast<std::size_t>(input_index)];
        out.reserve(static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height));

        const Eigen::Matrix3d Rt = in.R.transpose();
        for (int j = 0; j < map.height; ++j) {
            for (int ii = 0; ii < map.width; ++ii) {
                const std::size_t idx =
                    static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                    static_cast<std::size_t>(ii);
                const float depth = map.depths[idx];
                if (!(depth > 0.0F)) {  // -1 sentinel and 0 both mean "no depth"
                    continue;
                }
                // An UNFILTERED map carries no confidence buffer at all (a map
                // checkpointed by a cancelled BuildDepthMaps, or a map that was
                // never cross-view filtered). Treating that as confidence 0
                // would silently drop every point of it under any
                // min_confidence > 0, which is not what "unfiltered" means: the
                // absence of a measurement is not a failing measurement. Such a
                // map passes the filter untouched; BuildPointCloud logs a
                // warning so the caller knows the filter could not judge it.
                const bool unfiltered = map.confidence.empty();
                const float conf = unfiltered ? 0.0F : map.confidence[idx];
                if (!unfiltered && conf < options.min_confidence) {
                    continue;
                }
                // A repaired pixel IS a filtered one at low confidence; it passes
                // the filter's gate only when min_confidence admits 0.25.

                Sample s;
                const Eigen::Vector2d centre(static_cast<double>(ii) + 0.5,
                                             static_cast<double>(j) + 0.5);
                const Eigen::Vector3d ray = in.calibration.unproject(centre);
                const Eigen::Vector3d p_cam = static_cast<double>(depth) * ray;
                s.position = Rt * (p_cam - in.t);

                if (use_normals) {
                    const Eigen::Vector3f n_cam(map.normals[idx * 3 + 0],
                                                map.normals[idx * 3 + 1],
                                                map.normals[idx * 3 + 2]);
                    s.normal = (Rt * n_cam.cast<double>()).cast<float>();
                }
                s.repaired = !map.flags.empty() && (map.flags[idx] & kDepthRepairedFlag) != 0;
                s.confidence = s.repaired ? static_cast<float>(kRepairedFusionWeight) : conf;
                if (std::abs(in.calibration.f) > 1e-12) {
                    s.footprint = static_cast<float>(static_cast<double>(depth) / in.calibration.f);
                }
                if (want_colour) {
                    float c[3] = {0.0F, 0.0F, 0.0F};
                    block_colour(*in.image, ii, j, ds, c);
                    s.colour[0] = c[0];
                    s.colour[1] = c[1];
                    s.colour[2] = c[2];
                    s.has_colour = true;
                }
                out.push_back(s);
            }
        }
    }

    // -- choose the voxel size ----------------------------------------------
    double voxel = options.voxel_size;
    if (!(voxel > 0.0)) {
        voxel = auto_voxel_size(per_input);
    }
    if (!(voxel > 0.0)) {
        // No valid samples (or no usable footprint): nothing to merge.
        return cloud;
    }

    // -- merge samples, deterministically, in input order --------------------
    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> grid;
    for (const std::vector<Sample>& samples : per_input) {
        for (const Sample& s : samples) {
            const VoxelKey key{floor_div(s.position.x(), voxel),
                               floor_div(s.position.y(), voxel),
                               floor_div(s.position.z(), voxel)};
            accumulate(grid[key], s);
        }
    }

    // -- deterministic output order: lexicographic by voxel key -------------
    std::vector<VoxelKey> keys;
    keys.reserve(grid.size());
    for (const auto& entry : grid) {
        keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());

    cloud.points.reserve(keys.size());
    if (use_normals) {
        cloud.normals.reserve(keys.size());
    }
    if (coloured) {
        cloud.colours.reserve(keys.size() * 3);
    }
    cloud.confidence.reserve(keys.size());

    for (const VoxelKey& key : keys) {
        const VoxelAccumulator& acc = grid.at(key);
        const double inv = 1.0 / static_cast<double>(acc.count);
        cloud.points.push_back(acc.position_sum * inv);
        if (use_normals) {
            Eigen::Vector3f n = acc.normal_sum;
            const float norm = n.norm();
            if (norm > 1e-12F) {
                n /= norm;
            } else {
                n = Eigen::Vector3f::Zero();
            }
            cloud.normals.push_back(n);
        }
        if (coloured) {
            const double denom =
                acc.colour_count > 0 ? static_cast<double>(acc.colour_count) : 1.0;
            push_colour(cloud, acc.colour_sum[0] / denom);
            push_colour(cloud, acc.colour_sum[1] / denom);
            push_colour(cloud, acc.colour_sum[2] / denom);
        }
        // With no confidence anywhere every summed value is 0, so fall back to
        // the sample count as the support weight instead.
        const double conf = all_confidence ? acc.confidence_sum : static_cast<double>(acc.count);
        cloud.confidence.push_back(static_cast<float>(conf));
    }

    return cloud;
}

}  // namespace alphashape::stereo
