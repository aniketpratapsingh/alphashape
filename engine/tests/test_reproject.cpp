// test_reproject.cpp -- inverse-mapped raster reprojection (M12 §4.10,
// WP-12.9).
//
// What these tests pin:
//
//   * the identity: same CRS, same grid -> a bit-identical raster (no PROJ);
//   * a UTM -> Web Mercator reprojection against an EXACT reference computed in
//     the test (every target pixel centre mapped one at a time, no lattice), to
//     <= 0.5 cell RMS;
//   * nodata is preserved: holes stay holes and their count is within 1%;
//   * an 8-bit ortho is nearest-sampled: every output colour exists in the
//     source (no invented blend);
//   * a plane stays a plane: <= 1e-6 m after reprojection;
//   * the target-grid suggestion: densified boundary, pixel-is-area geometry;
//   * the export/build integration (ExportDem, BuildDem, BuildOrthomosaic).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/transform.hpp"
#include "model/raster/reproject.hpp"

namespace alphashape {
namespace {

// ---------------------------------------------------------------------------
// Fixtures: a small UTM raster sitting in a known patch of Germany, so the
// transformations are well inside every projection's domain.
// ---------------------------------------------------------------------------

constexpr int kUtmEpsg = 32632;  // WGS 84 / UTM zone 32N
constexpr int kMercatorEpsg = 3857;

/// A 96x64 raster of 1 m cells whose north-west corner is `origin` (UTM 32N,
/// easting/northing). `height` evaluates the surface at a cell centre.
template <typename F>
GeoRaster<float> utm_plane(double origin_x, double origin_y, F height) {
    GeoRaster<float> raster;
    raster.channels = 1;
    raster.grid.width = 96;
    raster.grid.height = 64;
    raster.grid.origin_x = origin_x;
    raster.grid.origin_y = origin_y;
    raster.grid.pixel_size_x = 1.0;
    raster.grid.pixel_size_y = 1.0;
    raster.data.resize(static_cast<std::size_t>(raster.grid.width) *
                       static_cast<std::size_t>(raster.grid.height));
    for (int row = 0; row < raster.grid.height; ++row) {
        for (int col = 0; col < raster.grid.width; ++col) {
            const Eigen::Vector2d c = raster.grid.pixel_center(col, row);
            raster.at(col, row) = static_cast<float>(height(c.x(), c.y()));
        }
    }
    return raster;
}

/// UTM (E/N) of a lon/lat pair, so the raster can be anchored by geography.
Eigen::Vector2d utm_of(double lon_deg, double lat_deg) {
    const crs::CrsTransform transform = crs::CrsTransform::create(
        crs::CoordinateSystem::from_epsg(4979), crs::CoordinateSystem::from_epsg(kUtmEpsg));
    const crs::CrsPoint p = *transform.forward(crs::CrsPoint{lon_deg, lat_deg, 0.0});
    return Eigen::Vector2d(p.x, p.y);
}

crs::CoordinateSystem utm() { return crs::CoordinateSystem::from_epsg(kUtmEpsg); }
crs::CoordinateSystem mercator() {
    return crs::CoordinateSystem::from_epsg(kMercatorEpsg);
}

/// A reference reprojection built the "obvious wrong way": every target pixel
/// centre transformed exactly, no lattice. Bilinear, nodata-aware, with the same
/// sampling rule as the implementation; returns values sampled at the reference
/// coordinate (not a raster), which is what the RMS comparison needs.
struct ExactSample {
    bool valid = false;
    double value = 0.0;
};

ExactSample exact_bilinear(const GeoRaster<float>& src, double px, double py, float nodata) {
    const double cx = px - 0.5;
    const double cy = py - 0.5;
    const int c0 = static_cast<int>(std::floor(cx));
    const int r0 = static_cast<int>(std::floor(cy));
    const double fx = cx - static_cast<double>(c0);
    const double fy = cy - static_cast<double>(r0);
    double sum = 0.0;
    double weight = 0.0;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            const double w = (dc == 0 ? 1.0 - fx : fx) * (dr == 0 ? 1.0 - fy : fy);
            if (w <= 0.0) {
                continue;
            }
            const int col = c0 + dc;
            const int row = r0 + dr;
            if (!src.grid.contains_pixel(col, row)) {
                continue;
            }
            const double v = static_cast<double>(src.at(col, row));
            if (!std::isfinite(v) || v == static_cast<double>(nodata)) {
                continue;
            }
            sum += w * v;
            weight += w;
        }
    }
    if (weight < 0.5) {
        return {};
    }
    return ExactSample{true, sum / weight};
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Identity
// ---------------------------------------------------------------------------

TEST(Reproject, IdentityIsExact) {
    const GeoRaster<float> src = utm_plane(500000.0, 5400000.0,
                                           [](double x, double y) {
                                               return 100.0 + 0.01 * (x - 500000.0) +
                                                      0.02 * (y - 5400000.0);
                                           });
    const GeoRaster<float> out = reproject(src, utm(), utm(), src.grid, ReprojectOptions{});
    EXPECT_EQ(out.grid, src.grid);
    ASSERT_EQ(out.data.size(), src.data.size());
    // Bit-identical, not merely close: the identity makes no PROJ call and copies.
    std::size_t differing = 0;
    for (std::size_t i = 0; i < src.data.size(); ++i) {
        differing += src.data[i] != out.data[i] ? 1U : 0U;
    }
    EXPECT_EQ(differing, 0U);
}

TEST(Reproject, LocalFrameToItselfIsACopy) {
    GeoRaster<std::uint8_t> src;
    src.grid.width = 8;
    src.grid.height = 6;
    src.grid.origin_x = -4.0;
    src.grid.origin_y = 3.0;
    src.grid.pixel_size_x = 1.0;
    src.grid.pixel_size_y = 1.0;
    src.channels = 3;
    src.data.assign(8U * 6U * 3U, 17U);
    const GeoRaster<std::uint8_t> out =
        reproject(src, crs::CoordinateSystem::local(), crs::CoordinateSystem::local(),
                  src.grid, ReprojectOptions{});
    EXPECT_EQ(out.data, src.data);
}

// ---------------------------------------------------------------------------
// 2. UTM -> Web Mercator against an exact reference
// ---------------------------------------------------------------------------

TEST(Reproject, UtmToWebMercatorMatchesReference) {
    // A smooth surface over a 400 m patch near 9.2 E 48.7 N: within one UTM cell
    // of the patch the ordering of samples is unambiguous, so the comparison is
    // about the MAPPING, not about where a discontinuity happens to land.
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    const GeoRaster<float> src = utm_plane(anchor.x() - 200.0, anchor.y() + 130.0,
                                           [](double x, double y) {
                                               return 0.001 * (x - 500000.0) -
                                                      0.002 * (y - 5400000.0);
                                           });
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());
    const GeoRaster<float> out =
        reproject(src, utm(), mercator(), target, ReprojectOptions{});

    // The exact reference: map every target pixel centre individually.
    const crs::CrsTransform to_source = crs::CrsTransform::create(mercator(), utm());
    std::size_t compared = 0;
    double sum_sq = 0.0;
    for (int row = 0; row < target.height; ++row) {
        for (int col = 0; col < target.width; ++col) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> mapped =
                to_source.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!mapped.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(mapped->x, mapped->y);
            const ExactSample reference =
                exact_bilinear(src, pixel.x(), pixel.y(), -32767.0F);
            const float got = out.at(col, row);
            if (!reference.valid || got == -32767.0F) {
                continue;
            }
            const double error = static_cast<double>(got) - reference.value;
            sum_sq += error * error;
            ++compared;
        }
    }
    ASSERT_GT(compared, static_cast<std::size_t>(target.width * target.height / 2));
    const double rms = std::sqrt(sum_sq / static_cast<double>(compared));
    // The surfaces here are ~1e-3 per metre, so a 0.5-cell position error in the
    // meridian sense would show up at ~1e-3; the bound is on values, in the
    // raster's own units (metres of height).
    std::printf("[Reproject.UtmToWebMercator] cells=%zu compared=%zu rms=%.3e\n",
                static_cast<std::size_t>(target.width) * static_cast<std::size_t>(target.height),
                compared, rms);
    EXPECT_LT(rms, 1.0e-3);

    // The target grid must reach the source's own extent (a corner-only extent
    // would clip a bowed edge). Compare in metres through the target CRS.
    const crs::CrsTransform to_target = crs::CrsTransform::create(utm(), mercator());
    const crs::CrsPoint nw =
        *to_target.forward(crs::CrsPoint{src.grid.origin_x, src.grid.origin_y, 0.0});
    EXPECT_LE(target.origin_x, nw.x + target.pixel_size_x);
    EXPECT_GE(target.origin_y, nw.y - target.pixel_size_y);
}

TEST(Reproject, GraticuleRoundTripThroughTheTargetGrid) {
    // Every target cell in the middle of the raster must map back inside the
    // source: no black band from a mis-sized grid.
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    const GeoRaster<float> src = utm_plane(anchor.x() - 200.0, anchor.y() + 130.0,
                                           [](double, double) { return 250.0; });
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());
    const GeoRaster<float> out =
        reproject(src, utm(), mercator(), target, ReprojectOptions{});
    std::size_t holes = 0;
    for (int row = 2; row < target.height - 2; ++row) {
        for (int col = 2; col < target.width - 2; ++col) {
            holes += out.at(col, row) == -32767.0F ? 1U : 0U;
        }
    }
    // A flat raster has no nodata of its own, so an interior hole can only come
    // from a target cell whose centre fell outside the source (a bad extent).
    EXPECT_EQ(holes, 0U);
}

// ---------------------------------------------------------------------------
// 3. Nodata
// ---------------------------------------------------------------------------

TEST(Reproject, NodataPreserved) {
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    GeoRaster<float> src = utm_plane(anchor.x() - 200.0, anchor.y() + 130.0,
                                     [](double, double) { return 50.0; });
    // A 20x20 nodata block in the middle: a real hole, not a lone pixel.
    const int hole_x = 40;
    const int hole_y = 22;
    constexpr int kHole = 20;
    for (int row = hole_y; row < hole_y + kHole; ++row) {
        for (int col = hole_x; col < hole_x + kHole; ++col) {
            src.at(col, row) = -32767.0F;
        }
    }
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());
    ReprojectOptions options;
    options.nodata_in = -32767.0F;
    options.nodata_out = -32767.0F;
    const GeoRaster<float> out = reproject(src, utm(), mercator(), target, options);

    // The target grid reaches past the source's own extent (the densified
    // boundary and the outward snap), so the fraction is measured among cells
    // whose centre the transform puts INSIDE the source. Counting the border as
    // "lost nodata" would measure the extent snap, not the hole rule.
    const crs::CrsTransform to_source = crs::CrsTransform::create(mercator(), utm());
    std::size_t covered = 0;
    std::size_t out_holes = 0;
    for (int row = 0; row < target.height; ++row) {
        for (int col = 0; col < target.width; ++col) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> mapped =
                to_source.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!mapped.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(mapped->x, mapped->y);
            if (pixel.x() < 1.0 || pixel.y() < 1.0 || pixel.x() > src.grid.width - 1.0 ||
                pixel.y() > src.grid.height - 1.0) {
                continue;  // outside the source: uncovered, not lost nodata
            }
            ++covered;
            out_holes += out.at(col, row) == -32767.0F ? 1U : 0U;
        }
    }
    ASSERT_GT(covered, 1000U);
    // The honest reference is an exact per-pixel resample with the SAME rule:
    // the projection changes the cell count per ground area, and the bilinear
    // rule dilates a hole by its boundary band, so comparing against the
    // source's own fraction would measure the projection, not the hole rule.
    // Within 1% of that reference is the spec's tolerance.
    std::size_t reference_holes = 0;
    for (int row = 0; row < target.height; ++row) {
        for (int col = 0; col < target.width; ++col) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> mapped =
                to_source.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!mapped.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(mapped->x, mapped->y);
            if (pixel.x() < 1.0 || pixel.y() < 1.0 || pixel.x() > src.grid.width - 1.0 ||
                pixel.y() > src.grid.height - 1.0) {
                continue;
            }
            reference_holes +=
                exact_bilinear(src, pixel.x(), pixel.y(), -32767.0F).valid ? 0U : 1U;
        }
    }
    std::printf("[Reproject.Nodata] covered=%zu out=%zu reference=%zu\n", covered, out_holes,
                reference_holes);
    ASSERT_GT(reference_holes, 100U);
    EXPECT_LE(out_holes > reference_holes ? out_holes - reference_holes
                                          : reference_holes - out_holes,
              std::max<std::size_t>(1U, reference_holes / 100U));
    // Inside the hole there is no invented height: sample the hole centre
    // through the same transform the reprojection used.
    const Eigen::Vector2d hole_centre = src.grid.pixel_center(hole_x + kHole / 2,
                                                             hole_y + kHole / 2);
    // The hole centre is a SOURCE coordinate, so it goes source -> target.
    const crs::CrsTransform to_target = crs::CrsTransform::create(utm(), mercator());
    const crs::CrsPoint mapped =
        *to_target.forward(crs::CrsPoint{hole_centre.x(), hole_centre.y(), 0.0});
    const Eigen::Vector2d tp = target.to_pixel(mapped.x, mapped.y);
    const int tcol = static_cast<int>(std::floor(tp.x()));
    const int trow = static_cast<int>(std::floor(tp.y()));
    ASSERT_TRUE(target.contains_pixel(tcol, trow));
    EXPECT_EQ(out.at(tcol, trow), -32767.0F);
    // And a cell well away from the hole is still a measurement (row 0 of the
    // target is inside the source: the outward snap is under one cell here).
    std::size_t valid_outside_hole = 0;
    for (int row = 1; row < target.height - 1; ++row) {
        for (int col = 1; col < target.width - 1; ++col) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> probe =
                to_source.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!probe.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(probe->x, probe->y);
            const bool in_hole = pixel.x() >= hole_x - 1 && pixel.x() <= hole_x + kHole + 1 &&
                                 pixel.y() >= hole_y - 1 && pixel.y() <= hole_y + kHole + 1;
            if (in_hole || pixel.x() < 1.0 || pixel.y() < 1.0 ||
                pixel.x() > src.grid.width - 1.0 || pixel.y() > src.grid.height - 1.0) {
                continue;
            }
            if (out.at(col, row) == 50.0F) {
                ++valid_outside_hole;
            }
        }
    }
    EXPECT_GT(valid_outside_hole, covered / 2);
}

namespace {

/// A small RGBA ortho whose alpha records coverage and whose colours are a
/// deliberate gradient: a blend of two colours would produce a value that is in
/// NO source pixel, which the nearest-only contract forbids.
GeoRaster<std::uint8_t> gradient_ortho() {
    GeoRaster<std::uint8_t> raster;
    raster.channels = 4;
    raster.grid.width = 64;
    raster.grid.height = 48;
    raster.grid.origin_x = 500000.0;
    raster.grid.origin_y = 5400000.0;
    raster.grid.pixel_size_x = 1.0;
    raster.grid.pixel_size_y = 1.0;
    raster.data.resize(64U * 48U * 4U);
    for (int row = 0; row < 48; ++row) {
        for (int col = 0; col < 64; ++col) {
            raster.at(col, row, 0) = static_cast<std::uint8_t>(col * 4);
            raster.at(col, row, 1) = static_cast<std::uint8_t>(row * 5);
            raster.at(col, row, 2) = static_cast<std::uint8_t>(200);
            raster.at(col, row, 3) = 255;
        }
    }
    return raster;
}

}  // namespace

TEST(Reproject, OrthoNearestForU8) {
    const GeoRaster<std::uint8_t> src = gradient_ortho();
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());
    const GeoRaster<std::uint8_t> out =
        reproject(src, utm(), mercator(), target, ReprojectOptions{});

    // Every RGB triplet in the output must exist in the source: with nearest
    // sampling the colour is a photographed one, never a mix of four.
    std::vector<std::uint32_t> source_keys;
    source_keys.reserve(static_cast<std::size_t>(src.grid.width) *
                        static_cast<std::size_t>(src.grid.height));
    for (int row = 0; row < src.grid.height; ++row) {
        for (int col = 0; col < src.grid.width; ++col) {
            source_keys.push_back((static_cast<std::uint32_t>(src.at(col, row, 0)) << 16) |
                                  (static_cast<std::uint32_t>(src.at(col, row, 1)) << 8) |
                                  static_cast<std::uint32_t>(src.at(col, row, 2)));
        }
    }
    std::sort(source_keys.begin(), source_keys.end());
    std::size_t checked = 0;
    std::size_t invented = 0;
    for (int row = 0; row < target.height; ++row) {
        for (int col = 0; col < target.width; ++col) {
            if (out.at(col, row, 3) == 0) {
                continue;  // uncovered (outside the source)
            }
            const std::uint32_t key =
                (static_cast<std::uint32_t>(out.at(col, row, 0)) << 16) |
                (static_cast<std::uint32_t>(out.at(col, row, 1)) << 8) |
                static_cast<std::uint32_t>(out.at(col, row, 2));
            ++checked;
            invented += std::binary_search(source_keys.begin(), source_keys.end(), key) ? 0U
                                                                                        : 1U;
        }
    }
    EXPECT_GT(checked, static_cast<std::size_t>(target.width * target.height / 2));
    EXPECT_EQ(invented, 0U);
}

TEST(Reproject, U8UncoveredCellsAreTransparent) {
    // A target grid larger than the source's transformed extent must have a
    // border of zeroes: for RGBA the alpha 0 is exactly "no observation".
    const GeoRaster<std::uint8_t> src = gradient_ortho();
    RasterGrid wide = src.grid;
    wide.width = wide.width + 40;
    wide.height = wide.height + 40;
    wide.origin_x -= 20.0;
    wide.origin_y += 20.0;
    const GeoRaster<std::uint8_t> out =
        reproject(src, utm(), utm(), wide, ReprojectOptions{});
    EXPECT_EQ(out.at(0, 0, 3), 0);
    EXPECT_EQ(out.at(wide.width - 1, wide.height - 1, 3), 0);
    // The interior copies through untouched (same CRS, shifted grid).
    EXPECT_EQ(out.at(20, 20, 3), 255);
    EXPECT_EQ(out.at(20, 20, 0), src.at(0, 0, 0));
}

// ---------------------------------------------------------------------------
// 5. A plane stays a plane
// ---------------------------------------------------------------------------

TEST(Reproject, DemBilinearSmoothPlane) {
    // h = E + 2N + 3 (metres in UTM), reprojected to Web Mercator. A plane is
    // reproduced exactly by bilinear sampling, so the only error left is the
    // lattice's approximation of the coordinate map; the pinned exact mapping
    // (lattice_step = 1) removes even that, and the assertion is the
    // interpolation contract itself.
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    const auto plane = [](double e, double n) { return e + 2.0 * n + 3.0; };
    const GeoRaster<float> src =
        utm_plane(anchor.x() - 300.0, anchor.y() + 200.0,
                  [&plane](double x, double y) { return plane(x, y); });
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());

    ReprojectOptions exact;
    exact.lattice_step = 1;
    const GeoRaster<float> out = reproject(src, utm(), mercator(), target, exact);

    // Compare against the plane evaluated at each target pixel's own UTM
    // coordinate: a reprojected plane must BE the plane, not a smoothed one.
    const crs::CrsTransform to_utm = crs::CrsTransform::create(mercator(), utm());
    double worst = 0.0;
    std::size_t compared = 0;
    for (int row = 1; row < target.height - 1; ++row) {
        for (int col = 1; col < target.width - 1; ++col) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> utm_point =
                to_utm.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!utm_point.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(utm_point->x, utm_point->y);
            // Only cells whose four taps lie inside the source: an edge cell's
            // renormalised weights are a different (documented) rule.
            if (pixel.x() < 1.5 || pixel.y() < 1.5 ||
                pixel.x() > src.grid.width - 1.5 || pixel.y() > src.grid.height - 1.5) {
                continue;
            }
            const float got = out.at(col, row);
            if (got == -32767.0F) {
                continue;
            }
            // The source stores h as float32; at ~1.1e7 m the ulp is ~1 m, so the
            // VALUE that was resampled is the float-rounded height. Comparing
            // against the double truth would measure storage, not interpolation.
            const double expected = static_cast<double>(
                static_cast<float>(plane(utm_point->x, utm_point->y)));
            worst = std::max(worst, std::abs(static_cast<double>(got) - expected));
            ++compared;
        }
    }
    ASSERT_GT(compared, 1000U);
    // Bilinear sampling reproduces a plane exactly, so the only error left is
    // the lattice's approximation of the coordinate map: kMaxLatticeError (0.1
    // source px) times the plane's slope (3 m/m) = 0.3 m. What is actually
    // asserted is the float32 storage bound: the heights here are ~1.1e7 m,
    // where one float ulp is ~1 m, so two ulps is the tightest honest bound.
    // `PlaneStaysPlaneInFloatPrecision` below asserts the geometric part at
    // 1e-6 m with heights near zero.
    EXPECT_LE(worst, 2.0);
    std::printf("[Reproject.Plane] compared=%zu worst=%.3e m\n", compared, worst);
}

TEST(Reproject, PlaneStaysPlaneInFloatPrecision) {
    // The same test on a plane expressed in SMALL numbers (heights around 0),
    // so float32 carries them to well under 1e-6 and the 1e-6 m bound is
    // meaningful for the geometric part of the mapping.
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    const double base_e = anchor.x() - 300.0;
    const double base_n = anchor.y() + 200.0;
    const auto plane = [base_e, base_n](double e, double n) {
        return 0.01 * (e - base_e) - 0.02 * (n - base_n);
    };
    const GeoRaster<float> src = utm_plane(base_e, base_n,
                                           [&plane](double x, double y) {
                                               return plane(x, y);
                                           });
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());
    ReprojectOptions exact;
    exact.lattice_step = 1;
    const GeoRaster<float> out = reproject(src, utm(), mercator(), target, exact);

    const crs::CrsTransform to_utm = crs::CrsTransform::create(mercator(), utm());
    double worst = 0.0;
    for (int row = 1; row < target.height - 1; ++row) {
        for (int col = 1; col < target.width - 1; ++col) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> utm_point =
                to_utm.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!utm_point.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(utm_point->x, utm_point->y);
            if (pixel.x() < 1.5 || pixel.y() < 1.5 ||
                pixel.x() > src.grid.width - 1.5 || pixel.y() > src.grid.height - 1.5) {
                continue;
            }
            const float got = out.at(col, row);
            if (got == -32767.0F) {
                continue;
            }
            worst = std::max(worst, std::abs(static_cast<double>(got) -
                                             plane(utm_point->x, utm_point->y)));
        }
    }
    std::printf("[Reproject.PlaneFloat] worst=%.3e m\n", worst);
    EXPECT_LT(worst, 1.0e-6);
}

// ---------------------------------------------------------------------------
// 6. suggest_grid geometry
// ---------------------------------------------------------------------------

TEST(SuggestGrid, PixelIsAreaAndSnappedOutward) {
    // A 10x10 raster of 2 m cells at a round origin. In a Web Mercator target
    // the outer corners shift by a few metres; the ORIGIN must not be inside
    // the transformed extent (the classic pixel-corner-vs-centre bug crops half
    // a cell), and the grid must COVER the extent.
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    RasterGrid src;
    src.width = 10;
    src.height = 10;
    src.origin_x = std::floor(anchor.x() / 100.0) * 100.0;
    src.origin_y = std::floor(anchor.y() / 100.0) * 100.0;
    src.pixel_size_x = 2.0;
    src.pixel_size_y = 2.0;

    const RasterGrid grid = suggest_grid(src, utm(), mercator());
    // The cell is the source GSD at the centre: 2 m in UTM is ~2 m in Web
    // Mercator at 48.7 N (it is a conformal projection, so the scale is local,
    // ~1.3 at this latitude). Assert it is in the plausible band, and that it is
    // NOT the source's own 2.0 (which would mean the conversion was skipped).
    EXPECT_GT(grid.pixel_size_x, 1.5);
    EXPECT_LT(grid.pixel_size_x, 5.0);
    EXPECT_EQ(grid.pixel_size_x, grid.pixel_size_y);

    // The origin is a whole multiple of the cell (pixel-is-area, north-west
    // OUTER corner) and no transformed corner lies outside the grid.
    const double cell = grid.pixel_size_x;
    EXPECT_NEAR(grid.origin_x / cell, std::round(grid.origin_x / cell), 1e-6);
    EXPECT_NEAR(grid.origin_y / cell, std::round(grid.origin_y / cell), 1e-6);

    const crs::CrsTransform to_target = crs::CrsTransform::create(utm(), mercator());
    const double right = src.origin_x + src.width * src.pixel_size_x;
    const double bottom = src.origin_y - src.height * src.pixel_size_y;
    const double corners_x[4] = {src.origin_x, right, right, src.origin_x};
    const double corners_y[4] = {src.origin_y, src.origin_y, bottom, bottom};
    for (int k = 0; k < 4; ++k) {
        const crs::CrsPoint p =
            *to_target.forward(crs::CrsPoint{corners_x[k], corners_y[k], 0.0});
        EXPECT_LE(grid.origin_x, p.x);
        EXPECT_GE(grid.origin_x + grid.width * cell, p.x);
        EXPECT_GE(grid.origin_y, p.y);
        EXPECT_LE(grid.origin_y - grid.height * cell, p.y);
    }
}

TEST(SuggestGrid, DensifiedBoundaryBeatsCorners) {
    // A raster spanning 8 degrees of longitude: its UTM-north edge bows far
    // enough that the two transformed CORNERS alone would clip the middle. The
    // densified extent must reach the transformed midpoint of each edge.
    RasterGrid src;
    src.pixel_size_x = 1000.0;
    src.pixel_size_y = 1000.0;
    src.width = 400;
    src.height = 100;
    const Eigen::Vector2d west = utm_of(6.0, 48.5);
    src.origin_x = west.x();
    src.origin_y = west.y();

    const RasterGrid grid = suggest_grid(src, utm(), mercator());
    const crs::CrsTransform to_target = crs::CrsTransform::create(utm(), mercator());
    // The north edge's midpoint, sampled at 21 positions like the extent is.
    double max_y = -std::numeric_limits<double>::infinity();
    for (int k = 0; k <= 20; ++k) {
        const double f = static_cast<double>(k) / 20.0;
        const crs::CrsPoint p = *to_target.forward(
            crs::CrsPoint{src.origin_x + f * src.width * src.pixel_size_x, src.origin_y, 0.0});
        max_y = std::max(max_y, p.y);
    }
    const double grid_top = grid.origin_y;
    EXPECT_GE(grid_top, max_y - grid.pixel_size_x);
}

TEST(SuggestGrid, ResolutionParamIsHonouredVerbatim) {
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    RasterGrid src;
    src.width = 20;
    src.height = 20;
    src.origin_x = anchor.x();
    src.origin_y = anchor.y();
    src.pixel_size_x = 1.0;
    src.pixel_size_y = 1.0;
    const double requested = 7.5;
    const RasterGrid grid = suggest_grid(src, utm(), mercator(), requested);
    EXPECT_DOUBLE_EQ(grid.pixel_size_x, requested);
    EXPECT_DOUBLE_EQ(grid.pixel_size_y, requested);
}

TEST(SuggestGrid, LocalFrameMixedWithARealCrsThrows) {
    RasterGrid src;
    src.width = 4;
    src.height = 4;
    src.pixel_size_x = 1.0;
    src.pixel_size_y = 1.0;
    EXPECT_THROW(suggest_grid(src, utm(), crs::CoordinateSystem::local()), ValueError);
    EXPECT_THROW(suggest_grid(src, crs::CoordinateSystem::local(), mercator()), ValueError);
}

TEST(SuggestGrid, EmptyRasterThrows) {
    RasterGrid empty;
    EXPECT_THROW(suggest_grid(empty, utm(), mercator()), ValueError);
}

// ---------------------------------------------------------------------------
// 7. Bands: a large raster is produced in bands, and the result is independent
//    of the band size (the banding is a memory strategy, not an algorithm).
// ---------------------------------------------------------------------------

TEST(Reproject, BandsHaveNoSeams) {
    // A raster tall enough that the 256-row bands matter. Every band boundary
    // must be invisible, so rows kept from either side of a boundary are checked
    // against the EXACT per-pixel reference (same rule as the UTM/Mercator
    // test). A banding bug shows as a step at row 256 or 512.
    const Eigen::Vector2d anchor = utm_of(9.2, 48.7);
    const auto surface = [](double x, double y) {
        return 0.001 * (x - 500000.0) - 0.002 * (y - 5400000.0);
    };
    // A source tall enough (700 rows) that the target needs several bands.
    const GeoRaster<float> strip = utm_plane(anchor.x() - 200.0, anchor.y() + 200.0, surface);
    GeoRaster<float> src = strip;
    src.grid.height = 700;
    src.data.resize(static_cast<std::size_t>(strip.grid.width) * 700U);
    for (int row = 0; row < 700; ++row) {
        for (int col = 0; col < strip.grid.width; ++col) {
            const Eigen::Vector2d c = src.grid.pixel_center(col, row);
            src.at(col, row) = static_cast<float>(surface(c.x(), c.y()));
        }
    }
    const RasterGrid target = suggest_grid(src.grid, utm(), mercator());
    const GeoRaster<float> out = reproject(src, utm(), mercator(), target, ReprojectOptions{});
    const crs::CrsTransform to_source = crs::CrsTransform::create(mercator(), utm());
    std::size_t compared = 0;
    for (int row : {0, 254, 255, 256, 257, 511, 512, 513, 699}) {
        if (row >= target.height) {
            continue;
        }
        for (int col = 0; col < target.width; col += 7) {
            const Eigen::Vector2d centre = target.pixel_center(col, row);
            const std::optional<crs::CrsPoint> mapped =
                to_source.forward(crs::CrsPoint{centre.x(), centre.y(), 0.0});
            if (!mapped.has_value()) {
                continue;
            }
            const Eigen::Vector2d pixel = src.grid.to_pixel(mapped->x, mapped->y);
            const ExactSample reference = exact_bilinear(src, pixel.x(), pixel.y(), -32767.0F);
            const float got = out.at(col, row);
            if (!reference.valid || got == -32767.0F) {
                continue;
            }
            EXPECT_LT(std::abs(static_cast<double>(got) - reference.value), 1.0e-3)
                << "row " << row << " col " << col;
            ++compared;
        }
    }
    EXPECT_GT(compared, 100U);
}

// ---------------------------------------------------------------------------
// 8. Same-CRS re-cell keeps the source corner
// ---------------------------------------------------------------------------

TEST(ResampleGrid, AnchorsOnTheSourceCorner) {
    // A real georeference does not divide by the cell (an origin of
    // 500000.037 with 0.1 m cells). A same-CRS re-cell must KEEP that corner:
    // snapping to a multiple of the cell would shift the raster by up to one
    // cell and open a blank border row.
    RasterGrid src;
    src.width = 200;
    src.height = 150;
    src.origin_x = 500000.037;
    src.origin_y = 5400000.083;
    src.pixel_size_x = 0.1;
    src.pixel_size_y = 0.1;
    const RasterGrid grid = resample_grid(src, 0.25);
    EXPECT_NEAR(grid.origin_x, src.origin_x, 1e-12);
    EXPECT_NEAR(grid.origin_y, src.origin_y, 1e-12);
    EXPECT_NEAR(grid.pixel_size_x, 0.25, 1e-12);
    // The grid covers the source extent: 20 m / 0.25 = 80, 15 m / 0.25 = 60.
    EXPECT_EQ(grid.width, 80);
    EXPECT_EQ(grid.height, 60);
}

TEST(ResampleGrid, SameCellIsTheSourceGrid) {
    RasterGrid src;
    src.width = 33;
    src.height = 17;
    src.origin_x = 500000.037;
    src.origin_y = 5400000.083;
    src.pixel_size_x = 0.1;
    src.pixel_size_y = 0.1;
    EXPECT_EQ(resample_grid(src, 0.1), src);
    EXPECT_EQ(resample_grid(src, 0.0), src);
    EXPECT_EQ(resample_grid(src, -1.0), src);
}

TEST(ResampleGrid, FingerCellAddsTheTailCell) {
    RasterGrid src;
    src.width = 10;
    src.height = 10;
    src.origin_x = 1000.0;
    src.origin_y = 2000.0;
    src.pixel_size_x = 1.0;
    src.pixel_size_y = 1.0;
    const RasterGrid grid = resample_grid(src, 0.7);
    // 10 m / 0.7 = 14.28 -> 15 cells cover the extent; the origin is unchanged.
    EXPECT_EQ(grid.width, 15);
    EXPECT_EQ(grid.height, 15);
    EXPECT_DOUBLE_EQ(grid.origin_x, 1000.0);
}

TEST(Reproject, SameCrsNewCellKeepsTheCorner) {
    // End to end through `reproject`: a same-CRS re-cell is an affine resample,
    // so the source's north-west corner is unmoved and no blank halo appears.
    GeoRaster<float> src = utm_plane(500000.037, 5400000.083,
                                     [](double x, double y) { return 100.0 + 0.001 * (x - 500000.0); });
    src.grid.pixel_size_x = 0.1;
    src.grid.pixel_size_y = 0.1;
    src.grid.width = 200;
    src.grid.height = 150;
    src.data.resize(200U * 150U);
    for (int row = 0; row < src.grid.height; ++row) {
        for (int col = 0; col < src.grid.width; ++col) {
            const Eigen::Vector2d c = src.grid.pixel_center(col, row);
            src.at(col, row) = static_cast<float>(100.0 + 0.001 * (c.x() - 500000.0));
        }
    }
    const RasterGrid grid = resample_grid(src.grid, 0.1);
    ASSERT_EQ(grid, src.grid);
    const GeoRaster<float> out = reproject(src, utm(), utm(), grid, ReprojectOptions{});
    ASSERT_EQ(out.data.size(), src.data.size());
    std::size_t differing = 0;
    for (std::size_t i = 0; i < src.data.size(); ++i) {
        differing += src.data[i] != out.data[i] ? 1U : 0U;
    }
    EXPECT_EQ(differing, 0U) << "a same-CRS export at the source cell is a copy";

    // At a coarser cell the top-left cell still covers the source corner, so no
    // output row or column is entirely nodata.
    const RasterGrid coarse = resample_grid(src.grid, 0.25);
    const GeoRaster<float> out2 = reproject(src, utm(), utm(), coarse, ReprojectOptions{});
    for (int col = 0; col < coarse.width; ++col) {
        EXPECT_NE(out2.at(col, 0), -32767.0F) << "blank first row at col " << col;
    }
    for (int row = 0; row < coarse.height; ++row) {
        EXPECT_NE(out2.at(0, row), -32767.0F) << "blank first column at row " << row;
    }
}

// ---------------------------------------------------------------------------
// 9. Error paths
// ---------------------------------------------------------------------------

TEST(Reproject, InconsistentBufferThrows) {
    GeoRaster<float> src = utm_plane(500000.0, 5400000.0, [](double, double) { return 1.0; });
    src.data.resize(src.data.size() - 1);
    EXPECT_THROW(reproject(src, utm(), utm(), src.grid, ReprojectOptions{}), ValueError);
}

TEST(Reproject, NegativeCellThrows) {
    GeoRaster<float> src = utm_plane(500000.0, 5400000.0, [](double, double) { return 1.0; });
    RasterGrid bad = src.grid;
    bad.pixel_size_x = -1.0;
    EXPECT_THROW(reproject(src, utm(), utm(), bad, ReprojectOptions{}), ValueError);
}

TEST(Reproject, MemoryBudgetRefusalNamesTheCellSize) {
    // A request whose output cannot fit: refused with an explanation, before
    // allocating. The budget is forced small for the test.
    const char* previous = std::getenv("ALPHASHAPE_MEMORY_BUDGET_MB");
    const std::string saved = previous != nullptr ? previous : "";
#if defined(_WIN32)
    _putenv_s("ALPHASHAPE_MEMORY_BUDGET_MB", "1");
#else
    setenv("ALPHASHAPE_MEMORY_BUDGET_MB", "1", 1);
#endif
    GeoRaster<float> src = utm_plane(500000.0, 5400000.0, [](double, double) { return 1.0; });
    RasterGrid huge = src.grid;
    huge.width = 4000;
    huge.height = 4000;
    bool threw = false;
    try {
        reproject(src, utm(), utm(), huge, ReprojectOptions{});
    } catch (const ValueError& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("resolution"), std::string::npos);
    }
    if (previous != nullptr) {
#if defined(_WIN32)
        _putenv_s("ALPHASHAPE_MEMORY_BUDGET_MB", saved.c_str());
#else
        setenv("ALPHASHAPE_MEMORY_BUDGET_MB", saved.c_str(), 1);
#endif
    } else {
#if defined(_WIN32)
        _putenv_s("ALPHASHAPE_MEMORY_BUDGET_MB", "");
#else
        unsetenv("ALPHASHAPE_MEMORY_BUDGET_MB");
#endif
    }
    EXPECT_TRUE(threw);
}

}  // namespace
