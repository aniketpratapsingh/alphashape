#include "common/algorithm/markers/circular_decoder.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include "common/algorithm/markers/target_patterns.hpp"

namespace alphashape {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;
constexpr int kBorderRays = 64;
constexpr int kMinBorderPoints = 38;   ///< 60% of the rays, as in the detector.
constexpr double kEdgeMinMagnitude = 0.01;    ///< Gradient-magnitude floor;
                                              ///< images are in [0, 1].
constexpr double kEdgePeakFraction = 0.3;     ///< The border edge is the last
                                              ///< strong edge peak on the ray.
constexpr double kMinContrast = 0.08;         ///< Light gap minus dark border.
constexpr double kCentreDotContrast = 0.15;   ///< Dark ring at 4r versus gap.
constexpr double kMinInBoundsFraction = 0.6;
constexpr int kAngularBins = 8;         ///< Uniformity check bins per annulus.
constexpr double kSectorSeparation = 0.2;  ///< Fraction of light-dark contrast a
                                           ///< sector mean must clear the
                                           ///< threshold by.

/// Bilinear read at image coordinates where integer values are pixel centres.
bool sample_pixel(const FloatImage& image, double x, double y, double& value) {
    if (!(x >= 0.0) || !(y >= 0.0) || x > static_cast<double>(image.width() - 1) ||
        y > static_cast<double>(image.height() - 1)) {
        return false;
    }
    value = static_cast<double>(image.sample(static_cast<float>(x), static_cast<float>(y)));
    return true;
}

/// Sample intensities along one ray over [lower, upper]. Returns false when any
/// part of the window leaves the image, so clipped rays never invent an edge.
bool sample_ray(const FloatImage& image, const Eigen::Vector2d& centre, double dx, double dy,
                double lower, double upper, int steps, std::vector<double>& values) {
    values.clear();
    if (steps < 2) return false;
    values.reserve(static_cast<std::size_t>(steps + 1));
    const double step = (upper - lower) / static_cast<double>(steps);
    for (int i = 0; i <= steps; ++i) {
        const double t = lower + static_cast<double>(i) * step;
        double value = 0.0;
        if (!sample_pixel(image, centre.x() + dx * t, centre.y() + dy * t, value)) {
            return false;
        }
        values.push_back(value);
    }
    return true;
}

/// Position of the outermost strong edge peak in one ray's gradient-magnitude
/// samples, with parabolic subpixel correction. Gradient magnitude (rather
/// than a signed intensity derivative) keeps the outer border's edge distinct
/// from the nearby inner border edge. Returns a negative value when the ray
/// shows no such edge.
double outer_edge_position(const std::vector<double>& values, double lower, double upper) {
    const int count = static_cast<int>(values.size());
    if (count < 4) return -1.0;
    const double step = (upper - lower) / static_cast<double>(count - 1);
    double peak = 0.0;
    for (int i = 1; i + 1 < count; ++i) {
        const double value = values[static_cast<std::size_t>(i)];
        if (value >= values[static_cast<std::size_t>(i - 1)] &&
            value > values[static_cast<std::size_t>(i + 1)] && value > peak) {
            peak = value;
        }
    }
    if (!(peak > kEdgeMinMagnitude)) return -1.0;
    const double threshold = std::max(kEdgeMinMagnitude, kEdgePeakFraction * peak);
    int best = -1;
    for (int i = 1; i + 1 < count; ++i) {
        const double value = values[static_cast<std::size_t>(i)];
        if (value >= values[static_cast<std::size_t>(i - 1)] &&
            value > values[static_cast<std::size_t>(i + 1)] && value >= threshold) {
            best = i;
        }
    }
    if (best <= 0) return -1.0;
    const double left = values[static_cast<std::size_t>(best - 1)];
    const double mid = values[static_cast<std::size_t>(best)];
    const double right = values[static_cast<std::size_t>(best + 1)];
    const double denominator = left - 2.0 * mid + right;
    double shift = 0.0;
    if (denominator < -1e-12) {
        shift = 0.5 * (left - right) / denominator;
    }
    shift = std::clamp(shift, -0.5, 0.5);
    return lower + (static_cast<double>(best) + shift) * step;
}

double ellipse_residual(const Ellipse& ellipse, const Eigen::Vector2d& point) {
    const double c = std::cos(ellipse.angle_rad);
    const double s = std::sin(ellipse.angle_rad);
    const double dx = point.x() - ellipse.center.x();
    const double dy = point.y() - ellipse.center.y();
    const double u = dx * c + dy * s;
    const double v = -dx * s + dy * c;
    const double radius = std::sqrt(u * u + v * v);
    const double rho = std::sqrt((u * u) / (ellipse.a * ellipse.a) +
                                 (v * v) / (ellipse.b * ellipse.b));
    return std::abs(rho - 1.0) * radius;
}

/// Fit, drop points far off the first ellipse and refit once. A ray that landed
/// on a neighbouring edge is removed instead of dragging the border ellipse.
std::optional<Ellipse> robust_border_fit(const std::vector<Eigen::Vector2d>& points) {
    const std::optional<Ellipse> first = fit_ellipse(points);
    if (!first) return std::nullopt;
    const double tolerance = std::max(1.5, 0.08 * first->a);
    std::vector<Eigen::Vector2d> kept;
    kept.reserve(points.size());
    for (const Eigen::Vector2d& point : points) {
        if (ellipse_residual(*first, point) <= tolerance) kept.push_back(point);
    }
    if (kept.size() < 5 || kept.size() == points.size()) return first;
    const std::optional<Ellipse> second = fit_ellipse(kept);
    return second ? second : first;
}

/// Locate the outer border (the edge at radius R) from a target centre using
/// the gradient-magnitude image. `expected` is the semi-major outer radius
/// estimate; the search spans 0.35x to 1.6x of it so an unknown tilt (axis
/// ratio down to 0.5) stays inside.
std::optional<Ellipse> locate_border(const FloatImage& magnitude,
                                     const Eigen::Vector2d& centre, double expected) {
    if (!std::isfinite(expected) || !(expected >= 2.0) || !centre.allFinite()) {
        return std::nullopt;
    }
    const double lower = 0.35 * expected;
    const double upper = 1.60 * expected;
    const double span = upper - lower;
    if (!(span >= 2.0) || !std::isfinite(span)) return std::nullopt;
    // Cap before the cast: a caller-supplied radius close to the int range
    // would otherwise overflow the conversion (UB).
    const int steps = std::max(16, static_cast<int>(std::min(std::ceil(span / 0.5), 4096.0)));
    std::vector<Eigen::Vector2d> points;
    points.reserve(static_cast<std::size_t>(kBorderRays));
    std::vector<double> values;
    for (int ray = 0; ray < kBorderRays; ++ray) {
        const double angle = kTwoPi * (static_cast<double>(ray) + 0.5) /
                             static_cast<double>(kBorderRays);
        const double dx = std::cos(angle);
        const double dy = std::sin(angle);
        if (!sample_ray(magnitude, centre, dx, dy, lower, upper, steps, values)) continue;
        const double t = outer_edge_position(values, lower, upper);
        if (!(t > 0.0)) continue;
        points.emplace_back(centre.x() + dx * t, centre.y() + dy * t);
    }
    if (static_cast<int>(points.size()) < kMinBorderPoints) return std::nullopt;
    const std::optional<Ellipse> fitted = robust_border_fit(points);
    if (!fitted || !fitted->center.allFinite() || !(fitted->a >= 2.0) ||
        !(fitted->b > 0.0) || fitted->b > fitted->a * 1.01) {
        return std::nullopt;
    }
    if (fitted->b / fitted->a < 0.2) return std::nullopt;
    return fitted;
}

/// Affine map from target-normalised coordinates (unit circle at radius 1) to
/// the image. The basis has positive determinant, so the pattern is never
/// mirrored; rotation about the target axis becomes a cyclic shift of the ring
/// word, which the decoder resolves.
struct EllipseMap {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d major = Eigen::Vector2d(1.0, 0.0);
    Eigen::Vector2d minor = Eigen::Vector2d(0.0, 1.0);
    double a = 1.0;
    double b = 1.0;

    Eigen::Vector2d to_image(double u, double v) const {
        return center + major * (a * u) + minor * (b * v);
    }
};

EllipseMap make_map(const Ellipse& border) {
    EllipseMap map;
    map.center = border.center;
    map.a = border.a;
    map.b = border.b;
    const double c = std::cos(border.angle_rad);
    const double s = std::sin(border.angle_rad);
    map.major = Eigen::Vector2d(c, s);
    map.minor = Eigen::Vector2d(-s, c);
    return map;
}

/// Mean intensity over an annulus in normalised target coordinates. Returns
/// nullopt when too few samples fall inside the image.
std::optional<double> polar_mean(const FloatImage& image, const EllipseMap& map, double inner,
                                 double outer, int radii, int angles) {
    if (radii < 1 || angles < 3) return std::nullopt;
    double sum = 0.0;
    int count = 0;
    const int total = radii * angles;
    for (int ai = 0; ai < angles; ++ai) {
        const double phi = kTwoPi * (static_cast<double>(ai) + 0.5) /
                           static_cast<double>(angles);
        for (int ri = 0; ri < radii; ++ri) {
            const double rho =
                radii == 1 ? inner : inner + (outer - inner) * static_cast<double>(ri) /
                                               static_cast<double>(radii - 1);
            const Eigen::Vector2d p = map.to_image(rho * std::cos(phi), rho * std::sin(phi));
            double value = 0.0;
            if (!sample_pixel(image, p.x(), p.y(), value)) continue;
            sum += value;
            ++count;
        }
    }
    if (static_cast<double>(count) < kMinInBoundsFraction * static_cast<double>(total)) {
        return std::nullopt;
    }
    return sum / static_cast<double>(count);
}

/// Mean intensity of one angular bin of an annulus. `bins` equal bins span the
/// full turn; this samples bin `index` only. Used to prove that the light gap
/// and dark border are uniform around the whole target.
std::optional<double> angular_bin_mean(const FloatImage& image, const EllipseMap& map,
                                       double inner, double outer, int radii, int bins,
                                       int index) {
    if (radii < 1 || bins < 1 || index < 0 || index >= bins) return std::nullopt;
    constexpr int kAnglesPerBin = 6;
    const double width = kTwoPi / static_cast<double>(bins);
    const double begin = width * static_cast<double>(index);
    double sum = 0.0;
    int count = 0;
    for (int ai = 0; ai < kAnglesPerBin; ++ai) {
        const double phi = begin + width * (static_cast<double>(ai) + 0.5) /
                                       static_cast<double>(kAnglesPerBin);
        for (int ri = 0; ri < radii; ++ri) {
            const double rho = radii == 1
                                   ? inner
                                   : inner + (outer - inner) * static_cast<double>(ri) /
                                                 static_cast<double>(radii - 1);
            const Eigen::Vector2d p = map.to_image(rho * std::cos(phi), rho * std::sin(phi));
            double value = 0.0;
            if (!sample_pixel(image, p.x(), p.y(), value)) continue;
            sum += value;
            ++count;
        }
    }
    const int total = radii * kAnglesPerBin;
    if (static_cast<double>(count) < kMinInBoundsFraction * static_cast<double>(total)) {
        return std::nullopt;
    }
    return sum / static_cast<double>(count);
}

/// Find the angular phase of the N-sector grid. A circular border fixes the
/// ellipse but not the rotation of the printed ring, so the sector boundaries
/// are not known a priori. The ring is sampled densely (16 samples per sector)
/// and the partition offset with the smallest within-sector variance is used.
/// `phase_centre` is the frame angle of sector 0's centre; subsequent sectors
/// advance by one sector width. Returns nullopt if the ring is not visible.
std::optional<double> estimate_sector_phase(const FloatImage& image, const EllipseMap& map,
                                            int bits) {
    constexpr int kFine = 16;
    constexpr int kRadii = 5;
    const int samples = bits * kFine;
    const double step = kTwoPi / static_cast<double>(samples);
    std::vector<double> profile(static_cast<std::size_t>(samples), 0.0);
    for (int k = 0; k < samples; ++k) {
        const double angle = (static_cast<double>(k) + 0.5) * step;
        double sum = 0.0;
        int count = 0;
        for (int ri = 0; ri < kRadii; ++ri) {
            const double rho = 0.58 + 0.20 * static_cast<double>(ri) /
                                          static_cast<double>(kRadii - 1);
            const Eigen::Vector2d p = map.to_image(rho * std::cos(angle), rho * std::sin(angle));
            double value = 0.0;
            if (!sample_pixel(image, p.x(), p.y(), value)) continue;
            sum += value;
            ++count;
        }
        if (count < kRadii / 2) return std::nullopt;
        profile[static_cast<std::size_t>(k)] = sum / static_cast<double>(count);
    }
    double best_score = std::numeric_limits<double>::infinity();
    int best_offset = 0;
    for (int offset = 0; offset < kFine; ++offset) {
        double score = 0.0;
        for (int sector = 0; sector < bits; ++sector) {
            double sum = 0.0;
            for (int k = 0; k < kFine; ++k) {
                const int index = (offset + sector * kFine + k) % samples;
                sum += profile[static_cast<std::size_t>(index)];
            }
            const double mean = sum / static_cast<double>(kFine);
            for (int k = 0; k < kFine; ++k) {
                const int index = (offset + sector * kFine + k) % samples;
                const double difference = profile[static_cast<std::size_t>(index)] - mean;
                score += difference * difference;
            }
        }
        if (score < best_score) {
            best_score = score;
            best_offset = offset;
        }
    }
    return (static_cast<double>(best_offset) + 0.5 * static_cast<double>(kFine)) * step;
}

/// Mean over sector `index` of the coding ring, in the sector-aligned band
/// 0.55..0.80 R with a +/-0.35 sector angular guard.
std::optional<double> sector_mean(const FloatImage& image, const EllipseMap& map, int index,
                                  int bits, double phase_centre) {
    const double delta = kTwoPi / static_cast<double>(bits);
    const double centre_angle = phase_centre + static_cast<double>(index) * delta;
    const double half = 0.35 * delta;
    constexpr int kRadii = 7;
    constexpr int kAngles = 7;
    double sum = 0.0;
    int count = 0;
    for (int ai = 0; ai < kAngles; ++ai) {
        const double offset =
            kAngles == 1 ? 0.0 : -half + 2.0 * half * static_cast<double>(ai) /
                                          static_cast<double>(kAngles - 1);
        const double phi = centre_angle + offset;
        for (int ri = 0; ri < kRadii; ++ri) {
            const double rho = 0.55 + (0.80 - 0.55) * static_cast<double>(ri) /
                                         static_cast<double>(kRadii - 1);
            const Eigen::Vector2d p = map.to_image(rho * std::cos(phi), rho * std::sin(phi));
            double value = 0.0;
            if (!sample_pixel(image, p.x(), p.y(), value)) continue;
            sum += value;
            ++count;
        }
    }
    if (count < (kRadii * kAngles) / 2) return std::nullopt;
    return sum / static_cast<double>(count);
}

/// Mean over an isotropic annulus (used before the tilt is known).
std::optional<double> isotropic_mean(const FloatImage& image, const Eigen::Vector2d& centre,
                                     double inner, double outer, int radii, int angles) {
    double sum = 0.0;
    int count = 0;
    const int total = radii * angles;
    for (int ai = 0; ai < angles; ++ai) {
        const double phi =
            kTwoPi * (static_cast<double>(ai) + 0.5) / static_cast<double>(angles);
        for (int ri = 0; ri < radii; ++ri) {
            const double rho = radii == 1
                                   ? inner
                                   : inner + (outer - inner) * static_cast<double>(ri) /
                                                 static_cast<double>(radii - 1);
            double value = 0.0;
            if (!sample_pixel(image, centre.x() + rho * std::cos(phi),
                              centre.y() + rho * std::sin(phi), value)) {
                continue;
            }
            sum += value;
            ++count;
        }
    }
    if (static_cast<double>(count) < kMinInBoundsFraction * static_cast<double>(total)) {
        return std::nullopt;
    }
    return sum / static_cast<double>(count);
}

/// The detector may report the centre dot (at 0.25 R) instead of the border.
/// The dark outer ring sits at about 4x the dot radius and is separated from it
/// by the white ring gap, so a dark annulus at 4r is the signature.
bool looks_like_centre_dot(const FloatImage& image, const Eigen::Vector2d& centre,
                           double radius) {
    if (!(radius > 1.0)) return false;
    const std::optional<double> gap =
        isotropic_mean(image, centre, 1.3 * radius, 1.6 * radius, 3, 24);
    const std::optional<double> ring =
        isotropic_mean(image, centre, 3.6 * radius, 4.0 * radius, 3, 24);
    if (!gap || !ring) return false;
    return *ring < *gap - kCentreDotContrast;
}

/// Otsu's threshold on the N sector means (all splits of the sorted values).
/// Falls back to the midpoint between the dark border and the light gaps when
/// the sectors do not form two distinct levels.
double otsu_threshold(std::vector<double> values, double dark_reference,
                      double light_reference) {
    std::sort(values.begin(), values.end());
    const double fallback = 0.5 * (dark_reference + light_reference);
    if (values.size() < 2) return fallback;
    const double range = values.back() - values.front();
    if (!(range > 1e-6)) return fallback;
    double total = 0.0;
    for (double value : values) total += value;
    double best = 0.0;
    std::size_t best_split = 0;
    double low_sum = 0.0;
    for (std::size_t split = 1; split < values.size(); ++split) {
        low_sum += values[split - 1];
        const double low_mean = low_sum / static_cast<double>(split);
        const double high_mean =
            (total - low_sum) / static_cast<double>(values.size() - split);
        const double between = static_cast<double>(split) *
                               static_cast<double>(values.size() - split) *
                               (low_mean - high_mean) * (low_mean - high_mean);
        if (between > best) {
            best = between;
            best_split = split;
        }
    }
    if (!(best > 1e-12) || best_split == 0) return fallback;
    return 0.5 * (values[best_split - 1] + values[best_split]);
}

std::uint32_t rotate_right(std::uint32_t word, int bits) {
    const std::uint32_t mask = (1u << static_cast<unsigned>(bits)) - 1u;
    return ((word >> 1) | ((word & 1u) << static_cast<unsigned>(bits - 1))) & mask;
}

/// Distance from `word` to the closest rotation of `table_word`.
int rotation_distance(std::uint32_t word, std::uint32_t table_word, int bits) {
    int best = bits;
    std::uint32_t rotated = table_word;
    for (int shift = 0; shift < bits; ++shift) {
        best = std::min(best, static_cast<int>(std::popcount(word ^ rotated)));
        rotated = rotate_right(rotated, bits);
    }
    return best;
}

/// Decode using a known outer-border ellipse. Returns nullopt if the ring
/// validation fails, the table lookup exceeds the allowed distance, or the
/// target is too small.
std::optional<CodedTarget> decode_with_border(const FloatImage& image,
                                              const GradientField& gradient,
                                              const Ellipse& border, int bits,
                                              int allowed_hamming) {
    const double radius = border.a;
    if (bits >= 16 ? !(radius >= 6.0) : !(radius >= 4.0)) return std::nullopt;
    if (!border.center.allFinite() || !(border.b > 0.0)) return std::nullopt;
    const EllipseMap map = make_map(border);

    const std::optional<double> inner_gap = polar_mean(image, map, 0.30, 0.45, 3, 32);
    const std::optional<double> dark_border = polar_mean(image, map, 0.92, 0.98, 3, 32);
    if (!inner_gap || !dark_border) return std::nullopt;
    // The wide inner gap (0.25 R across) is the reliable light reference. The
    // design's 0.05 R outer gap at 0.86..0.89 R is narrower than the Gaussian
    // blur used on real photographs (and in the blur tests), which fills a gap
    // that thin; using it as a light reference would reject correctly decoded
    // targets. The dark outer border plus the wide inner gap carry the
    // contrast gate instead.
    if (!(*dark_border + kMinContrast <= *inner_gap)) return std::nullopt;

    const std::optional<double> phase = estimate_sector_phase(image, map, bits);
    if (!phase) return std::nullopt;
    std::vector<double> means;
    means.reserve(static_cast<std::size_t>(bits));
    for (int sector = 0; sector < bits; ++sector) {
        const std::optional<double> mean = sector_mean(image, map, sector, bits, *phase);
        if (!mean) return std::nullopt;
        means.push_back(*mean);
    }
    const double threshold = otsu_threshold(means, *dark_border, *inner_gap);
    // The wide inner gap must sit on the light side of the same threshold used
    // for the sectors. The border is not required to be darker than the
    // threshold: a blur wider than its 0.1 R width lifts its mean towards the
    // threshold, while the contrast check above already proves it is dark.
    if (!(*inner_gap > threshold)) return std::nullopt;

    // A sector whose mean is close to the threshold is a coin flip; rejecting
    // it stops a partially covered (foreign) candidate from decoding. The
    // contrast gate above guarantees a positive light-minus-dark span.
    const double separation = kSectorSeparation * (*inner_gap - *dark_border);
    int dark_sectors = 0;
    int light_sectors = 0;
    for (double mean : means) {
        if (std::abs(mean - threshold) < separation) return std::nullopt;
        if (mean < threshold) {
            ++dark_sectors;
        } else {
            ++light_sectors;
        }
    }
    // The tables exclude the all-zero and all-one words (checked by
    // CircularCodesMinDistance), so a genuine ring always has both classes.
    if (dark_sectors == 0 || light_sectors == 0) return std::nullopt;

    // Half-light/half-dark annuli (a foreign shape straddling the hypothesis)
    // pass a global mean test but fail an angular one. Every inner-gap bin must
    // be light. A border bin must stay below the midpoint of the light and dark
    // references. The light reference for the border is the brighter of the
    // inner gap and the quiet zone just outside the border: at a 2 px blur the
    // centre dot depresses the inner-gap mean, and a tilted border is thin
    // along the minor axis, so the inner gap alone gives a limit the genuine
    // border cannot stay below. A half-light annulus still has bins near the
    // quiet-zone level and is rejected.
    const std::optional<double> quiet = polar_mean(image, map, 1.10, 1.30, 4, 32);
    const double border_light = quiet ? std::max(*inner_gap, *quiet) : *inner_gap;
    const double mid_level = 0.5 * (border_light + *dark_border);
    for (int bin = 0; bin < kAngularBins; ++bin) {
        const std::optional<double> gap_bin =
            angular_bin_mean(image, map, 0.30, 0.45, 3, kAngularBins, bin);
        if (!gap_bin || !(*gap_bin > threshold)) return std::nullopt;
        const std::optional<double> border_bin =
            angular_bin_mean(image, map, 0.92, 0.98, 3, kAngularBins, bin);
        if (!border_bin || !(*border_bin < mid_level)) return std::nullopt;
    }

    const std::optional<double> centre = polar_mean(image, map, 0.02, 0.20, 4, 24);
    if (!centre) return std::nullopt;
    const int parity = (*centre < threshold) ? 1 : 0;

    std::uint32_t word = 0;
    for (int sector = 0; sector < bits; ++sector) {
        if (means[static_cast<std::size_t>(sector)] < threshold) {
            word |= 1u << static_cast<unsigned>(bits - 1 - sector);
        }
    }

    // Distance 3 between all rotations makes the nearest code unique, so the
    // accept test is a plain bound on the best distance. There is no runner-up
    // margin: that margin would reject nearly every genuine one-sector error.
    const std::optional<WordMatch> match = match_circular_word(word, bits, allowed_hamming);
    if (!match) return std::nullopt;
    const int best_distance = match->distance;
    const int best_index = match->index;

    // Centre refinement: `refine_ellipse_center` fits the first radial edge
    // from the centre, which is the centre dot for parity 1 and a mixture of
    // ring-sector edges otherwise.
    MarkerCandidate seed;
    seed.center = border.center + Eigen::Vector2d(0.5, 0.5);
    seed.radius_px = radius;
    seed.axis_ratio = border.b / border.a;
    seed.angle_rad = border.angle_rad;
    Eigen::Vector2d centre_calibration = seed.center;
    if (parity == 1) {
        // With a centre dot the first radial edge is the dot boundary on every
        // ray, so its fit is the target's fiducial and is preferred even when
        // the blurred border ellipse centre is a few tenths of a pixel away.
        // Without a dot the first edges mix dark sectors at 0.5 R with light
        // sectors at 0.85 R, so the border ellipse centre is kept: the parity-0
        // centre test measured 0.3774 px RMS with the refinement versus
        // 0.1766 px without it.
        const std::optional<Eigen::Vector2d> refined =
            refine_ellipse_center(gradient, seed);
        if (refined && (*refined - seed.center).norm() <= 0.35 * radius) {
            centre_calibration = *refined;
        }
    }

    CodedTarget target;
    target.code = 2 * best_index + parity;
    target.center = centre_calibration;
    target.bits = bits;
    target.hamming = best_distance;
    target.radius_px = radius;
    return target;
}

}  // namespace

std::optional<WordMatch> match_circular_word(std::uint32_t word, int bits,
                                             int max_hamming) {
    if (!is_supported_circular_bits(bits) || max_hamming < 0) return std::nullopt;
    const int allowed = std::min(max_hamming, 1);
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    if (table.empty()) return std::nullopt;
    int best_distance = bits + 1;
    int best_index = -1;
    for (std::size_t index = 0; index < table.size(); ++index) {
        const int distance = rotation_distance(word, table[index], bits);
        if (distance < best_distance) {
            best_distance = distance;
            best_index = static_cast<int>(index);
        }
    }
    if (best_index < 0 || best_distance > allowed) return std::nullopt;
    WordMatch match;
    match.index = best_index;
    match.distance = best_distance;
    return match;
}

namespace {

std::optional<CodedTarget> decode_circular_impl(const FloatImage& gray,
                                                const GradientField& gradient,
                                                const MarkerCandidate& candidate, int bits,
                                                int max_hamming) {
    if (!is_supported_circular_bits(bits) || gray.empty() ||
        gradient.magnitude.empty()) {
        return std::nullopt;
    }
    if (!candidate.center.allFinite() || max_hamming < 0) return std::nullopt;
    const int allowed_hamming = std::min(max_hamming, 1);
    const double hint = candidate.radius_px;
    // A non-finite or absurd hint would make the border search and its step
    // count meaningless (and the conversion to int undefined).
    if (!std::isfinite(hint) || !(hint > 0.5)) return std::nullopt;
    if (hint > 0.5 * static_cast<double>(std::max(gray.width(), gray.height()))) {
        return std::nullopt;
    }
    const Eigen::Vector2d centre = candidate.center - Eigen::Vector2d(0.5, 0.5);
    if (!(centre.x() >= 0.0) || !(centre.y() >= 0.0) ||
        centre.x() > static_cast<double>(gray.width() - 1) ||
        centre.y() > static_cast<double>(gray.height() - 1)) {
        return std::nullopt;
    }

    const double centre_dot_radius = 4.0 * hint;
    const double border_radius = hint;
    double first = border_radius;
    double second = centre_dot_radius;
    if (looks_like_centre_dot(gray, centre, hint)) {
        first = centre_dot_radius;
        second = border_radius;
    }

    const auto attempt = [&](double expected) -> std::optional<CodedTarget> {
        const std::optional<Ellipse> border =
            locate_border(gradient.magnitude, centre, expected);
        if (!border) return std::nullopt;
        // The located border must be consistent with the hypothesis. Without
        // this, a hypothesis near the centre dot can lock onto an unrelated
        // edge and decode a foreign pattern as a valid target.
        if (!(border->a >= expected / 1.5) || !(border->a <= expected * 1.5)) {
            return std::nullopt;
        }
        return decode_with_border(gray, gradient, *border, bits, allowed_hamming);
    };

    std::optional<CodedTarget> decoded = attempt(first);
    if (decoded) return decoded;
    if (std::abs(first - second) > 0.5) {
        decoded = attempt(second);
        if (decoded) return decoded;
    }
    return std::nullopt;
}

}  // namespace

std::optional<CodedTarget> decode_circular(const FloatImage& gray,
                                           const MarkerCandidate& candidate, int bits,
                                           int max_hamming) {
    if (gray.empty()) return std::nullopt;
    return decode_circular(gray, compute_gradient_field(gray), candidate, bits,
                           max_hamming);
}

std::optional<CodedTarget> decode_circular(const FloatImage& gray,
                                           const GradientField& gradient,
                                           const MarkerCandidate& candidate, int bits,
                                           int max_hamming) {
    return decode_circular_impl(gray, gradient, candidate, bits, max_hamming);
}

}  // namespace alphashape

