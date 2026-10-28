// WP-12.4: datum transformations and grid shifts.
//
// Two layers are tested.
//
// The *selection* layer (`available_transformations`, `with_transformation`) is
// checked against its contract: increasing accuracy with unknown accuracy last;
// pinning an operation must reproduce it; and a grid that is not installed must
// be named rather than silently degrading to a ballpark.
//
// The *automatic* layer is checked against main: an unpinned transform must
// transform exactly as the bare PROJ pipeline did — per point, so an
// area-limited national operation applies inside its area and not outside it.
// Those numbers are pinned below in `AutomaticMatchesBareProjPipeline`.
//
// The *numeric* layer checks the Helmert shifts against IOGP Publication
// 373-7-2 (Geomatics Guidance Note 7, part 2, September 2019). Rather than
// hard-coding PROJ's answers, the test carries a small independent
// implementation of the two formula sections the shifts are defined by —
// §4.1.1 geographic <-> geocentric and §4.2.3 Position Vector 7-parameter — and
// first checks that implementation against the document's own printed worked
// examples. Only then is PROJ compared with it. That way the assertion is
// "PROJ agrees with the published formulas", not "PROJ agrees with a number
// somebody once copied out of PROJ".
//
// Grid cases (NTv2/NADCON/OSTN15) need a grid on disk. They GTEST_SKIP with a
// message naming the grid when it is absent, so the suite runs everywhere; no
// grid file is committed.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <proj.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/datum.hpp"
#include "model/crs/proj_context.hpp"
#include "model/crs/transform.hpp"

namespace alphashape::crs {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Degrees from degrees/minutes/seconds, the form the Guidance Note publishes.
/// Negative `degrees` makes the whole value negative; a longitude written as
/// 1°43'W is dms(-1, 43, 4.5177).
double dms(double degrees, double minutes, double seconds) {
    const double sign = degrees < 0.0 ? -1.0 : 1.0;
    return sign * (std::fabs(degrees) + minutes / 60.0 + seconds / 3600.0);
}

// ---------------------------------------------------------------------------
// Independent reference: GN 7-2 §4.1.1 and §4.2.3
// ---------------------------------------------------------------------------

/// An ellipsoid by semi-major axis and inverse flattening, as the document
/// tabulates them.
struct Ellipsoid {
    double a;
    double inverse_flattening;
};

constexpr Ellipsoid kAiry1830{6377563.396, 299.3249646};
constexpr Ellipsoid kInternational1924{6378388.0, 297.0};
constexpr Ellipsoid kWgs84{6378137.0, 298.257223563};

struct Xyz {
    double x;
    double y;
    double z;
};

/// GN 7-2 §4.1.1 forward: geographic (degrees, metres) to geocentric.
Xyz geodetic_to_geocentric(const CrsPoint& point, const Ellipsoid& ellipsoid) {
    const double f = 1.0 / ellipsoid.inverse_flattening;
    const double e2 = 2.0 * f - f * f;
    const double lat = point.y * kPi / 180.0;
    const double lon = point.x * kPi / 180.0;
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double nu = ellipsoid.a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    return Xyz{(nu + point.z) * cos_lat * std::cos(lon), (nu + point.z) * cos_lat * std::sin(lon),
               ((1.0 - e2) * nu + point.z) * sin_lat};
}

/// GN 7-2 §4.1.1 reverse: geocentric to geographic, iterating on latitude as
/// the document prescribes.
CrsPoint geocentric_to_geodetic(const Xyz& xyz, const Ellipsoid& ellipsoid) {
    const double f = 1.0 / ellipsoid.inverse_flattening;
    const double e2 = 2.0 * f - f * f;
    const double lon = std::atan2(xyz.y, xyz.x);
    const double p = std::hypot(xyz.x, xyz.y);
    double lat = std::atan2(xyz.z, p * (1.0 - e2));
    for (int i = 0; i < 100; ++i) {
        const double sin_lat = std::sin(lat);
        const double nu = ellipsoid.a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
        const double next = std::atan2(xyz.z + e2 * nu * sin_lat, p);
        const bool converged = std::fabs(next - lat) < 1e-15;
        lat = next;
        if (converged) {
            break;
        }
    }
    const double sin_lat = std::sin(lat);
    const double nu = ellipsoid.a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    const double height = p / std::cos(lat) - nu;
    return CrsPoint{lon * 180.0 / kPi, lat * 180.0 / kPi, height};
}

/// GN 7-2 §4.2.3 Position Vector 7-parameter transformation (EPSG method
/// 1033), geocentric domain. Rotations in arc-seconds, scale in ppm.
Xyz position_vector(const Xyz& source, double tx, double ty, double tz, double rx_arcsec,
                    double ry_arcsec, double rz_arcsec, double scale_ppm) {
    const double arcsec = kPi / (180.0 * 3600.0);
    const double rx = rx_arcsec * arcsec;
    const double ry = ry_arcsec * arcsec;
    const double rz = rz_arcsec * arcsec;
    const double m = 1.0 + scale_ppm * 1e-6;
    return Xyz{tx + m * (source.x - rz * source.y + ry * source.z),
               ty + m * (rz * source.x + source.y - rx * source.z),
               tz + m * (-ry * source.x + rx * source.y + source.z)};
}

/// The Helmert shift PROJ applies for EPSG:1314 "OSGB36 to WGS 84 (6)", the
/// OSGB36 datum shift as a Position Vector transformation on Airy 1830. The
/// rotations and scale are proj.db's: 0.15"/0.247"/0.842", -20.489 ppm. The
/// EPSG registry carries one more digit on each (0.1502"/0.2470"/0.8421",
/// -20.4894 ppm), and that difference alone moves a British point by about
/// 3 mm, so the reference must use the parameters PROJ actually applies; the
/// parity with the registry's four-decimal figures is recorded in
/// engine/LIMITATIONS.md rather than asserted here.
Xyz osgb36_helmert_to_wgs84_geocentric(const Xyz& source) {
    return position_vector(source, 446.448, -125.157, 542.060, 0.15, 0.247, 0.842, -20.489);
}

/// Horizontal gap in metres at `latitude_deg`, from two lon/lat points. The
/// comparison is millimetre-scale between points that are arcseconds apart, so
/// a local flat-earth scale (the meridian arc and the parallel arc) is exact to
/// far better than a micrometre over such an offset.
double horizontal_gap_m(const CrsPoint& a, const CrsPoint& b, double latitude_deg) {
    constexpr double kMetresPerDegree = 111319.49079327358;  // at the equator
    const double scale = std::cos(latitude_deg * kPi / 180.0);
    const double dx = (a.x - b.x) * kMetresPerDegree * scale;
    const double dy = (a.y - b.y) * kMetresPerDegree;
    return std::sqrt(dx * dx + dy * dy);
}

bool grid_installed(const std::string& name) { return !find_data_file(name).empty(); }

const DatumTransformation* find_code(const std::vector<DatumTransformation>& list,
                                     const std::string& code) {
    for (const DatumTransformation& entry : list) {
        if (entry.code == code) {
            return &entry;
        }
    }
    return nullptr;
}

const CoordinateSystem& osgb36() {
    static const CoordinateSystem cs = CoordinateSystem::from_epsg(4277);
    return cs;
}

const CoordinateSystem& wgs84() {
    static const CoordinateSystem cs = CoordinateSystem::from_epsg(4326);
    return cs;
}

// ---------------------------------------------------------------------------
// The reference implementation agrees with the document's printed examples
// ---------------------------------------------------------------------------

TEST(Crs, ReferenceFormulasMatchPublishedExamples) {
    // Before the reference implementation is used as an oracle, it must
    // reproduce what the document prints. Each block quotes the worked example
    // it checks. The document prints geocentric XYZ to a centimetre, so a
    // centimetre is the tightest honest tolerance there (the printed figures
    // themselves are 2 dp); the millimetre agreement is asserted against the
    // formulas in DatumShiftMatchesEpsgExamples.

    // §4.2.3: WGS 72 -> WGS 84 (EPSG 1238), tZ=+4.5 m, rZ=+0.554", dS=+0.219 ppm;
    //   3657660.66 255768.55 5201382.11 -> 3657660.78 255778.43 5201387.75
    {
        const Xyz got = position_vector(Xyz{3657660.66, 255768.55, 5201382.11}, 0.0, 0.0, 4.5, 0.0,
                                        0.0, 0.554, 0.219);
        EXPECT_NEAR(got.x, 3657660.78, 0.01);
        EXPECT_NEAR(got.y, 255778.43, 0.01);
        EXPECT_NEAR(got.z, 5201387.75, 0.01);
    }

    // §4.2.4: WGS 84 -> ED50 3-parameter, t=(+84.87, +96.49, +116.95) m;
    //   3771793.97 140253.34 5124304.35 -> 3771878.84 140349.83 5124421.30
    {
        const Xyz got = position_vector(Xyz{3771793.97, 140253.34, 5124304.35}, 84.87, 96.49,
                                        116.95, 0.0, 0.0, 0.0, 0.0);
        EXPECT_NEAR(got.x, 3771878.84, 0.01);
        EXPECT_NEAR(got.y, 140349.83, 0.01);
        EXPECT_NEAR(got.z, 5124421.30, 0.01);
    }

    // §4.3.2: the same WGS 84 -> ED50 shift applied to a geographic point,
    //   WGS 84 53°48'33.82"N 2°07'46.38"E, h=73.0 m
    //   -> ED50 53°48'36.565"N 2°07'51.477"E (concatenated method; the
    //      abridged variant prints ...36.56"N / ...51.48"E)
    // The document prints to a thousandth of a second of arc, so the honest
    // assertion is that the formula rounds to the printed figures; the
    // millimetre agreement is assertable against the formula themselves.
    {
        const CrsPoint source{dms(2, 7, 46.38), dms(53, 48, 33.82), 73.0};
        const Xyz geocentric = geodetic_to_geocentric(source, kWgs84);
        const Xyz shifted =
            position_vector(geocentric, 84.87, 96.49, 116.95, 0.0, 0.0, 0.0, 0.0);
        const CrsPoint got = geocentric_to_geodetic(shifted, kInternational1924);
        // Rounded to 0.001" as printed.
        EXPECT_NEAR(got.x, dms(2, 7, 51.477), 0.0005 / 3600.0);
        EXPECT_NEAR(got.y, dms(53, 48, 36.565), 0.0005 / 3600.0);
        // The document prints the ellipsoidal height to a centimetre.
        EXPECT_NEAR(got.z, 28.02, 0.01);
    }

    // §4.1.1 round trip: the two geographic/geocentric formulas are inverses.
    {
        const CrsPoint point{dms(2, 7, 46.38), dms(53, 48, 33.82), 73.0};
        const CrsPoint back = geocentric_to_geodetic(geodetic_to_geocentric(point, kWgs84), kWgs84);
        EXPECT_LE(horizontal_gap_m(back, point, point.y), 1e-6);
        EXPECT_NEAR(back.z, point.z, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// The engine agrees with those formulas to a millimetre
// ---------------------------------------------------------------------------

TEST(Crs, DatumShiftMatchesEpsgExamples) {
    // §4.2.3 Position Vector 7-parameter example, WGS 72 -> WGS 84 (EPSG 1238):
    //   tZ=+4.5 m, rZ=+0.554", dS=+0.219 ppm
    //   Xs=3657660.66 Ys=255768.55 Zs=5201382.11
    //   -> Xt=3657660.78 Yt=255778.43 Zt=5201387.75
    // EPSG:4984 and EPSG:4978 are the WGS 72 and WGS 84 geocentric CRSs, so the
    // pair is the document's own domain: the comparison is directly against the
    // printed XYZ. PROJ lists the +0.219 ppm variant first, which is EPSG:1238.
    {
        const CrsTransform shift = CrsTransform::create(CoordinateSystem::from_epsg(4984),
                                                        CoordinateSystem::from_epsg(4978));
        const std::optional<CrsPoint> got =
            shift.forward(CrsPoint{3657660.66, 255768.55, 5201382.11});
        ASSERT_TRUE(got.has_value());
        // The document prints XYZ to two decimals, so a centimetre is the
        // tightest honest bound there; the formula agreement is asserted at a
        // millimetre just below.
        EXPECT_NEAR(got->x, 3657660.78, 0.01) << "GN 7-2 §4.2.3 X";
        EXPECT_NEAR(got->y, 255778.43, 0.01) << "GN 7-2 §4.2.3 Y";
        EXPECT_NEAR(got->z, 5201387.75, 0.01) << "GN 7-2 §4.2.3 Z";

        const Xyz want = position_vector(Xyz{3657660.66, 255768.55, 5201382.11}, 0.0, 0.0, 4.5,
                                         0.0, 0.0, 0.554, 0.219);
        EXPECT_LE(std::hypot(std::hypot(got->x - want.x, got->y - want.y), got->z - want.z), 1e-3)
            << "GN 7-2 §4.2.3 vs the §4.1.1/§4.2.3 formulas";
    }

    // OSGB36 -> WGS 84 by the 7-parameter Helmert shift (EPSG:1314): the
    // §4.3.1 concatenation on Airy 1830 with tZ=542.060 m and proj.db's
    // 0.15"/0.247"/0.842", -20.489 ppm. There is no printed OSGB36
    // example, so the reference is the document's formulas fed those
    // parameters, over points spread across Great Britain. The pair is 2D, so
    // PROJ leaves the ellipsoidal height alone; only lon/lat are compared.
    {
        const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
        const DatumTransformation* helmert = find_code(list, "1314");
        ASSERT_NE(helmert, nullptr) << "EPSG:1314 (OSGB36 to WGS 84 (6)) is not offered";
        EXPECT_FALSE(helmert->needs_missing_grid()) << "the Helmert case must not need a grid";
        EXPECT_NEAR(helmert->accuracy_m, 2.0, 1e-9) << "EPSG:1314 is published at 2 m";

        const CrsTransform shift =
            CrsTransform::create(osgb36().with_transformation(*helmert), wgs84());

        // Greenwich, and points across Great Britain at a range of heights: a
        // 2D Helmert's horizontal result depends on the source height, so the
        // reference takes the same height.
        const std::vector<CrsPoint> points = {
            CrsPoint{dms(1, 43, 4.5177), dms(52, 39, 27.2531), 0.0},
            CrsPoint{dms(-0.1276, 0, 0), dms(51, 30, 26.0), 35.0},
            CrsPoint{dms(-2.2426, 0, 0), dms(53, 28, 51.0), 38.0},
            CrsPoint{dms(-3.1883, 0, 0), dms(55, 57, 12.0), 47.0},
            CrsPoint{dms(-4.2518, 0, 0), dms(55, 51, 51.0), 20.0},
            CrsPoint{dms(-1.5535, 0, 0), dms(53, 47, 48.0), 63.0},
        };

        for (const CrsPoint& point : points) {
            const std::optional<CrsPoint> got = shift.forward(point);
            ASSERT_TRUE(got.has_value());
            const Xyz geocentric = geodetic_to_geocentric(point, kAiry1830);
            const CrsPoint want =
                geocentric_to_geodetic(osgb36_helmert_to_wgs84_geocentric(geocentric), kWgs84);
            EXPECT_LE(horizontal_gap_m(*got, want, point.y), 1e-3)
                << "OSGB36 -> WGS 84 Helmert at lon " << point.x << " lat " << point.y;
            // A 2D Helmert is a horizontal shift: the height passes through.
            EXPECT_NEAR(got->z, point.z, 1e-9)
                << "a 2D Helmert must not change the ellipsoidal height";
        }
    }

    // §4.3.2 through the engine: WGS 84 -> a CRS on the International 1924
    // ellipsoid carrying the ED50 3-parameter shift, at the document's point.
    // The engine is held to the formula at a millimetre and to the printed
    // figures at their own rounding (0.001" of arc).
    {
        const CoordinateSystem ed50 = CoordinateSystem::from_proj(
            "+proj=longlat +ellps=intl +towgs84=-84.87,-96.49,-116.95,0,0,0,0");
        const CrsTransform to_ed50 = CrsTransform::create(wgs84(), ed50);
        const CrsPoint source{dms(2, 7, 46.38), dms(53, 48, 33.82), 73.0};
        const std::optional<CrsPoint> got = to_ed50.forward(source);
        ASSERT_TRUE(got.has_value());

        const CrsPoint reference = geocentric_to_geodetic(
            position_vector(geodetic_to_geocentric(source, kWgs84), 84.87, 96.49, 116.95, 0.0, 0.0,
                            0.0, 0.0),
            kInternational1924);
        EXPECT_LE(horizontal_gap_m(*got, reference, 53.8), 1e-3) << "GN 7-2 §4.3.2 formulas";
        EXPECT_NEAR(got->x, dms(2, 7, 51.477), 0.0005 / 3600.0)
            << "GN 7-2 §4.3.2 printed longitude";
        EXPECT_NEAR(got->y, dms(53, 48, 36.565), 0.0005 / 3600.0)
            << "GN 7-2 §4.3.2 printed latitude";
        EXPECT_NEAR(got->z, 73.0, 1e-9) << "a 2D shift must not change the height";
    }

    // NAD27 -> NAD83 across the CONUS is the grid case; its own test
    // (Crs.GridOperationRunsWhenInstalled) covers it, because a GTEST_SKIP here
    // would discard the Helmert assertions above. The list is still checked
    // without needing the grid: the NADCON operation must be offered and name
    // its grid.
    {
        const CoordinateSystem nad27 = CoordinateSystem::from_epsg(4267);
        const CoordinateSystem nad83 = CoordinateSystem::from_epsg(4269);
        const std::vector<DatumTransformation> conus_list =
            nad27.available_transformations(nad83);
        const DatumTransformation* conus = find_code(conus_list, "1241");
        ASSERT_NE(conus, nullptr) << "EPSG:1241 (NAD27 to NAD83 (1)) is not offered";
        EXPECT_FALSE(conus->grids.empty()) << "the NADCON operation must name its grid";
    }
}

// ---------------------------------------------------------------------------
// Selection and pinning
// ---------------------------------------------------------------------------

TEST(Crs, AvailableTransformationsSorted) {
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    ASSERT_GE(list.size(), 2u) << "OSGB36 -> WGS 84 offers several Helmert shifts plus a ballpark";

    // Known accuracy ascending and unknown accuracy last. The ballpark has
    // unknown accuracy, so it must come after every Helmert shift. Nothing is
    // marked preferred: PROJ's automatic transform chooses per point, so no
    // single entry is "the operation PROJ would use".
    bool saw_unknown = false;
    double previous = -1.0;
    for (std::size_t i = 0; i < list.size(); ++i) {
        const double accuracy = list[i].accuracy_m;
        if (accuracy <= 0.0) {
            saw_unknown = true;
            continue;
        }
        EXPECT_FALSE(saw_unknown) << "a known-accuracy entry (" << list[i].name
                                  << ") follows an unknown-accuracy one";
        EXPECT_GE(accuracy, previous) << "entry " << i << " (" << list[i].name
                                      << ") is more accurate than the one before it";
        previous = accuracy;
    }

    // Every entry is describable, and the grid operation is listed with its grid.
    bool saw_ballpark = false;
    bool saw_grid_entry = false;
    for (const DatumTransformation& entry : list) {
        EXPECT_FALSE(entry.name.empty());
        EXPECT_FALSE(describe_transformation(entry).empty());
        if (entry.ballpark) {
            saw_ballpark = true;
            EXPECT_LE(entry.accuracy_m, 0.0) << "a ballpark has no stated accuracy";
        }
        if (!entry.grids.empty()) {
            saw_grid_entry = true;
        }
    }
    EXPECT_TRUE(saw_ballpark) << "OSGB36 -> WGS 84 must list the ballpark fallback";
    EXPECT_TRUE(saw_grid_entry) << "the OSTN15 operation should be listed with its grid";

    // The accuracy ordering PROJ reports here: 1314 (2 m) comes before 1195
    // (21 m), which is what "sorted by accuracy" means in practice.
    const auto index_of = [&list](const std::string& code) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            if (list[i].code == code) {
                return static_cast<long>(i);
            }
        }
        return -1L;
    };
    const long at_1314 = index_of("1314");
    const long at_1195 = index_of("1195");
    ASSERT_GE(at_1314, 0L);
    ASSERT_GE(at_1195, 0L);
    EXPECT_LT(at_1314, at_1195) << "the 2 m shift must precede the 21 m shift";
}

TEST(Crs, PinnedTransformationIsHonoured) {
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    ASSERT_FALSE(list.empty());

    // Pinning an operation and comparing against the unpinned transform is not a
    // meaningful equality test: an unpinned transform lets PROJ choose per
    // point, and for OSGB36 -> WGS 84 that means the Helmert applies inside
    // Great Britain only. What "pinning is honoured" means here is narrower and
    // checkable: the transform built from the pinned CRS runs the pinned
    // operation's pipeline, and its numbers are that operation's.
    // `description()` is the PROJ pipeline text (not the operation's name), so
    // the Helmert parameters are what identifies it.
    const DatumTransformation* helmert = find_code(list, "1314");
    ASSERT_NE(helmert, nullptr);
    const CoordinateSystem pinned = osgb36().with_transformation(*helmert);
    const CrsPoint point{dms(1, 43, 4.5177), dms(52, 39, 27.2531), 0.0};

    const CrsTransform pinned_transform = CrsTransform::create(pinned, wgs84());
    EXPECT_NE(pinned_transform.description().find("x=446.448"), std::string::npos)
        << "the pinned Helmert must be the operation the pipeline used: "
        << pinned_transform.description();
    EXPECT_EQ(pinned_transform.description().find("hgridshift"), std::string::npos)
        << "the OSTN15 grid must not be what the pinned Helmert transform ran";

    // ... and its answer is the §4.1.1/§4.2.3 formula result, not a null shift.
    const std::optional<CrsPoint> got = pinned_transform.forward(point);
    ASSERT_TRUE(got.has_value());
    const CrsPoint want = geocentric_to_geodetic(
        osgb36_helmert_to_wgs84_geocentric(geodetic_to_geocentric(point, kAiry1830)), kWgs84);
    EXPECT_LE(horizontal_gap_m(*got, want, point.y), 1e-3);

    // Pinning a *different* Helmert must change it: 1314 and 1195 are both
    // OSGB36 -> WGS 84 shifts but differ by metres.
    const DatumTransformation* other = find_code(list, "1195");
    ASSERT_NE(other, nullptr);
    const CoordinateSystem pinned_other = osgb36().with_transformation(*other);
    const std::optional<CrsPoint> c = CrsTransform::create(pinned_other, wgs84()).forward(point);
    ASSERT_TRUE(c.has_value());
    EXPECT_GT(horizontal_gap_m(*got, *c, point.y), 1.0)
        << "EPSG:1314 and EPSG:1195 are different shifts, so pinning must matter";
}

TEST(Crs, DerivedAuthorityOperationIsPinnable) {
    // The OSTN15 shift (EPSG:7710) is reported by PROJ under the computed
    // authority "DERIVED_FROM(EPSG)". Pinning must resolve it — the code is an
    // EPSG code, only the authority string is decorated — and the pinned grid
    // shift must then run. Skipped when the OSTN15 grid is absent.
    if (!grid_installed("uk_os_OSTN15_NTv2_OSGBtoETRS.tif")) {
        GTEST_SKIP() << "the OSTN15 grid uk_os_OSTN15_NTv2_OSGBtoETRS.tif is not installed; "
                        "install proj-data to run this case";
    }
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    const DatumTransformation* ostn15 = find_code(list, "7710");
    ASSERT_NE(ostn15, nullptr) << "the OSTN15 operation (EPSG:7710) is not offered";
    EXPECT_EQ(ostn15->auth_name, "DERIVED_FROM(EPSG)");
    EXPECT_FALSE(ostn15->needs_missing_grid());

    const CrsPoint point{dms(1, 43, 4.5177), dms(52, 39, 27.2531), 0.0};
    const CrsTransform pinned =
        CrsTransform::create(osgb36().with_transformation(*ostn15), wgs84());
    EXPECT_NE(pinned.description().find("hgridshift"), std::string::npos)
        << "the pinned OSTN15 transform must run the grid: " << pinned.description();
    EXPECT_EQ(pinned.description().find("x=446.448"), std::string::npos)
        << "the Helmert must not be what the pinned OSTN15 transform ran";

    const std::optional<CrsPoint> got = pinned.forward(point);
    ASSERT_TRUE(got.has_value());
    // The OSTN15 shift differs from the EPSG:1314 Helmert by metres, so this
    // proves the grid operation (not the Helmert) is what ran.
    const CrsPoint helmert_want = geocentric_to_geodetic(
        osgb36_helmert_to_wgs84_geocentric(geodetic_to_geocentric(point, kAiry1830)), kWgs84);
    EXPECT_GT(horizontal_gap_m(*got, helmert_want, point.y), 0.5)
        << "the OSTN15 shift must not coincide with the 7-parameter Helmert";
}

TEST(Crs, PinnedTransformationComposesWithProjectedTarget) {
    // A pinned CRS is a normal CRS: a transform to a projected target must
    // compose the pinned shift with the projection.
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    const DatumTransformation* helmert = find_code(list, "1314");
    ASSERT_NE(helmert, nullptr);
    const CoordinateSystem pinned = osgb36().with_transformation(*helmert);
    const CrsPoint point{dms(1, 43, 4.5177), dms(52, 39, 27.2531), 0.0};

    const CoordinateSystem utm30 = CoordinateSystem::from_epsg(32630);
    const std::optional<CrsPoint> direct =
        CrsTransform::create(pinned, utm30).forward(point);
    ASSERT_TRUE(direct.has_value());

    // The same place the long way round: pinned shift to WGS 84, then the
    // plain geographic -> UTM transform.
    const std::optional<CrsPoint> geographic =
        CrsTransform::create(pinned, wgs84()).forward(point);
    ASSERT_TRUE(geographic.has_value());
    const std::optional<CrsPoint> via_two =
        CrsTransform::create(wgs84(), utm30).forward(*geographic);
    ASSERT_TRUE(via_two.has_value());
    EXPECT_LE(std::hypot(direct->x - via_two->x, direct->y - via_two->y), 1e-6);
}

TEST(Crs, MissingGridThrowsWithName) {
    // A CRS whose datum shift names a grid nobody installed. The list keeps the
    // operation (so the choice stays visible) but reports the grid, and pinning
    // it refuses with the grid's name rather than degrading.
    const std::string absent = "alphashape_absent_grid.gsb";
    const CoordinateSystem source =
        CoordinateSystem::from_proj("+proj=longlat +ellps=clrk66 +nadgrids=" + absent);

    const std::vector<DatumTransformation> list = source.available_transformations(wgs84());
    ASSERT_FALSE(list.empty());

    const DatumTransformation* grid_entry = nullptr;
    for (const DatumTransformation& entry : list) {
        if (std::find(entry.grids.begin(), entry.grids.end(), absent) != entry.grids.end()) {
            grid_entry = &entry;
            break;
        }
    }
    ASSERT_NE(grid_entry, nullptr) << "the nadgrids=" << absent << " operation was not listed";
    EXPECT_TRUE(grid_entry->needs_missing_grid());
    EXPECT_NE(std::find(grid_entry->missing_grids.begin(), grid_entry->missing_grids.end(), absent),
              grid_entry->missing_grids.end());
    EXPECT_NE(missing_grids_summary(*grid_entry).find(absent), std::string::npos);

    try {
        const CoordinateSystem pinned = source.with_transformation(*grid_entry);
        (void)pinned;
        FAIL() << "expected the missing-grid refusal";
    } catch (const Error& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("transformation needs grid " + absent), std::string::npos)
            << "the message must name the grid: " << what;
        EXPECT_NE(what.find("install proj-data"), std::string::npos)
            << "the message must say how to fix it: " << what;
    }
}

TEST(Crs, MissingGridNamedByPlainTransformCreate) {
    // The same diagnosis on the unpinned path. PROJ's own failure for this case
    // is "File not found", which tells a user nothing; the message must name the
    // grid and the fix.
    const std::string absent = "alphashape_absent_grid.gsb";
    const CoordinateSystem source =
        CoordinateSystem::from_proj("+proj=longlat +ellps=clrk66 +nadgrids=" + absent);

    try {
        const CrsTransform transform = CrsTransform::create(source, wgs84());
        (void)transform;
        FAIL() << "expected the missing-grid refusal";
    } catch (const Error& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find(absent), std::string::npos) << what;
        EXPECT_NE(what.find("install proj-data"), std::string::npos) << what;
    }
}

TEST(Crs, GridOperationRunsWhenInstalled) {
    // The complement of the two refusal tests, and the grid half of the spec's
    // NAD27 -> NAD83 case: with the grid present, the pinned grid operation
    // loads and transforms. PROJ's *automatic* choice for this pair is the
    // ballpark (a strict area test rejects every NADCON region), so the grid
    // operation has to be asked for by pinning it. Skipped when the grid is
    // absent, naming it.
    if (!grid_installed("us_noaa_conus.tif")) {
        GTEST_SKIP() << "the NADCON grid us_noaa_conus.tif is not installed; install proj-data "
                        "(or the proj-data package) to run this case";
    }
    const CoordinateSystem nad27 = CoordinateSystem::from_epsg(4267);
    const CoordinateSystem nad83 = CoordinateSystem::from_epsg(4269);
    const std::vector<DatumTransformation> list = nad27.available_transformations(nad83);
    const DatumTransformation* conus = find_code(list, "1241");
    ASSERT_NE(conus, nullptr);
    EXPECT_FALSE(conus->needs_missing_grid()) << "the grid is installed but reported missing";

    const CoordinateSystem pinned = nad27.with_transformation(*conus);
    const CrsTransform pinned_transform = CrsTransform::create(pinned, nad83);
    EXPECT_NE(pinned_transform.description().find("hgridshift"), std::string::npos)
        << "the pinned grid operation must be the one the pipeline used: "
        << pinned_transform.description();

    // Every CONUS sample moves, by the grid's own arcsecond-scale magnitude.
    // NADCON 27->83 CONUS offsets reach about 4" (largest near the west coast),
    // so 10" is a ceiling that still proves a grid shift rather than a null
    // one; a shift of tens of arcseconds would mean a different operation.
    const std::vector<CrsPoint> points = {CrsPoint{-100.0, 40.0, 0.0},
                                          CrsPoint{-90.0, 35.0, 0.0},
                                          CrsPoint{-80.0, 30.0, 0.0},
                                          CrsPoint{-120.0, 45.0, 0.0},
                                          CrsPoint{-75.0, 42.0, 0.0}};
    for (const CrsPoint& point : points) {
        const std::optional<CrsPoint> got = pinned_transform.forward(point);
        ASSERT_TRUE(got.has_value()) << "lon " << point.x;
        const double offset_arcsec =
            std::hypot(got->x - point.x, got->y - point.y) * 3600.0;
        EXPECT_GT(offset_arcsec, 1e-4) << "no shift at lon " << point.x;
        EXPECT_LT(offset_arcsec, 10.0) << "shift above the grid's magnitude at lon " << point.x;
    }
}

TEST(Crs, Towgs84Applied) {
    // A custom WKT with TOWGS84[] is modelled by PROJ as a bound CRS, so its
    // Helmert shift is the pair's only real operation, is not a ballpark, and
    // produces the same coordinates as EPSG:1314, which carries the same
    // parameters.
    const char* wkt =
        "GEOGCS[\"Custom Airy\","
        "DATUM[\"Custom OSGB36\","
        "SPHEROID[\"Airy 1830\",6377563.396,299.3249646],"
        "TOWGS84[446.448,-125.157,542.06,0.15,0.247,0.842,-20.489]],"
        "PRIMEM[\"Greenwich\",0],"
        "UNIT[\"degree\",0.0174532925199433]]";
    const CoordinateSystem custom = CoordinateSystem::from_wkt(wkt);
    ASSERT_TRUE(custom.is_set());

    const std::vector<DatumTransformation> list = custom.available_transformations(wgs84());
    ASSERT_EQ(list.size(), 1u) << "a bound CRS offers exactly its own shift";
    EXPECT_FALSE(list.front().ballpark) << "TOWGS84 is a real shift, not a ballpark";
    EXPECT_FALSE(list.front().needs_missing_grid());

    const CrsPoint point{dms(1, 43, 4.5177), dms(52, 39, 27.2531), 0.0};
    const std::optional<CrsPoint> got = CrsTransform::create(custom, wgs84()).forward(point);
    ASSERT_TRUE(got.has_value());

    // ... and the reference formulas agree with it, so the shift applied really
    // is the TOWGS84 Helmert. The pair is 2D, so only lon/lat move. The
    // reference uses the *WKT's own* rounded parameters (0.15"/0.247"/0.842",
    // -20.489 ppm), not EPSG:1314's four-decimal ones: the two differ by about
    // 3 mm, so comparing against the EPSG figures would test the parameters
    // rather than the engine.
    const Xyz geocentric = geodetic_to_geocentric(point, kAiry1830);
    const CrsPoint want = geocentric_to_geodetic(
        position_vector(geocentric, 446.448, -125.157, 542.06, 0.15, 0.247, 0.842, -20.489),
        kWgs84);
    EXPECT_LE(horizontal_gap_m(*got, want, point.y), 1e-3);
    EXPECT_NEAR(got->z, point.z, 1e-9);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

TEST(Crs, DatumRefusesLocalUnsetAndProjected) {
    const CoordinateSystem local = CoordinateSystem::local();
    EXPECT_THROW((void)local.available_transformations(wgs84()), ValueError);
    EXPECT_THROW((void)wgs84().available_transformations(local), ValueError);

    // A default-constructed value is the local frame.
    const CoordinateSystem unset;
    EXPECT_THROW((void)unset.available_transformations(wgs84()), ValueError);

    // An operation code PROJ does not know is a ValueError naming it.
    DatumTransformation bogus;
    bogus.auth_name = "EPSG";
    bogus.code = "999999";
    bogus.name = "not a real operation";
    EXPECT_THROW((void)wgs84().with_transformation(bogus), ValueError);

    // A projected CRS cannot carry a datum shift: PROJ attaches one to the
    // geodetic CRS, so the message must point at the geographic CRS instead.
    const std::vector<DatumTransformation> utm_list =
        CoordinateSystem::from_epsg(32633).available_transformations(wgs84());
    ASSERT_FALSE(utm_list.empty());
    try {
        (void)CoordinateSystem::from_epsg(32633).with_transformation(utm_list.front());
        FAIL() << "expected a projected source to be refused";
    } catch (const ValueError& error) {
        EXPECT_NE(std::string(error.what()).find("geographic"), std::string::npos) << error.what();
    }

    // An entry with no authority code cannot be pinned, because its hub CRS
    // cannot be recovered; the message says what to use instead.
    DatumTransformation anonymous;
    anonymous.name = "Ballpark geographic offset";
    anonymous.ballpark = true;
    try {
        (void)wgs84().with_transformation(anonymous);
        FAIL() << "expected an anonymous entry to be refused";
    } catch (const ValueError& error) {
        EXPECT_NE(std::string(error.what()).find("allow_ballpark"), std::string::npos)
            << error.what();
    }
}

TEST(Crs, IdenticalCrsHasNoDatumShift) {
    // WGS 84 to itself: PROJ offers one null geographic offset. The list has
    // one entry, it is a no-op, and a transform built from the pair is the
    // identity. The point is that the entry point does not throw for a pair with
    // nothing to do.
    const std::vector<DatumTransformation> list = wgs84().available_transformations(wgs84());
    ASSERT_EQ(list.size(), 1u);
    EXPECT_EQ(list.front().name, "Null geographic offset from WGS 84 to WGS 84");
    EXPECT_FALSE(list.front().ballpark) << "a null offset is not a ballpark";

    const CrsPoint point{9.0, 50.0, 12.5};
    const std::optional<CrsPoint> got = CrsTransform::create(wgs84(), wgs84()).forward(point);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, point);
}

// ---------------------------------------------------------------------------
// The automatic (unpinned) transform is PROJ's per-point choice
//
// The oracle is a bare PROJ pipeline for the same pair: exactly the call the
// engine built before WP-12.4 touched operation selection. Comparing against it
// is the "unchanged from main for every pair" guard, and unlike a literal it
// holds on every platform and PROJ data release, because both sides ask the
// same PROJ the same question.
//
// Every pair is also probed at a point inside the national operation's area of
// use AND one far outside it, because only the outside point distinguishes
// "PROJ chose per point" from "one operation everywhere" — the defect this
// replaced.
// ---------------------------------------------------------------------------

namespace {

/// Assert `got` reproduces `want` to `tolerance` in x/y (z passes through).
void expect_point(const std::optional<CrsPoint>& got, const CrsPoint& want, double tolerance,
                  const char* what) {
    ASSERT_TRUE(got.has_value()) << what;
    EXPECT_NEAR(got->x, want.x, tolerance) << what << " (x)";
    EXPECT_NEAR(got->y, want.y, tolerance) << what << " (y)";
    EXPECT_DOUBLE_EQ(got->z, want.z) << what << " (z must pass through a 2D shift)";
}

/// The reference: a bare `proj_create_crs_to_crs_from_pj` pipeline with no
/// options and no pre-selected operation, normalised to visualisation order.
/// This is deliberately the same call the engine makes, spelled out here so the
/// test fails if the engine starts doing something else.
class BareProj {
public:
    BareProj(const CoordinateSystem& src, const CoordinateSystem& dst) {
        const std::string src_wkt = src.to_wkt();
        const std::string dst_wkt = dst.to_wkt();
        ctx_ = thread_context();
        PJ* src_pj = proj_create(ctx_, src_wkt.c_str());
        PJ* dst_pj = proj_create(ctx_, dst_wkt.c_str());
        pipeline_ = (src_pj != nullptr && dst_pj != nullptr)
                        ? proj_create_crs_to_crs_from_pj(ctx_, src_pj, dst_pj, nullptr, nullptr)
                        : nullptr;
        if (pipeline_ != nullptr) {
            PJ* normalised = proj_normalize_for_visualization(ctx_, pipeline_);
            proj_destroy(pipeline_);
            pipeline_ = normalised;
        }
        proj_destroy(src_pj);
        proj_destroy(dst_pj);
    }
    ~BareProj() {
        if (pipeline_ != nullptr) {
            proj_destroy(pipeline_);
        }
    }
    BareProj(const BareProj&) = delete;
    BareProj& operator=(const BareProj&) = delete;

    bool is_valid() const { return pipeline_ != nullptr; }

    CrsPoint forward(const CrsPoint& point) const {
        proj_errno_reset(pipeline_);
        const PJ_COORD in = proj_coord(point.x, point.y, point.z, 0.0);
        const PJ_COORD out = proj_trans(pipeline_, PJ_FWD, in);
        return CrsPoint{out.xyzt.x, out.xyzt.y, out.xyzt.z};
    }

private:
    PJ_CONTEXT* ctx_ = nullptr;
    PJ* pipeline_ = nullptr;
};

/// The engine's unpinned transform must reproduce `reference` at every point,
/// and both must either succeed or fail together. The comparison is exact: the
/// two are the same PROJ call, so any difference is the bug this guards.
///
/// `allow_ballpark` is on because PROJ's own pick for a pair with only
/// area-limited national operations (NAD 27 -> NAD 83 and friends) is the
/// ballpark, and `create()` refuses that by default — main did too. The flag
/// only lifts the refusal; it does not change the pipeline.
void expect_matches_bare_proj(const CoordinateSystem& src, const CoordinateSystem& dst,
                              const std::vector<CrsPoint>& points, const char* what) {
    const BareProj reference(src, dst);
    ASSERT_TRUE(reference.is_valid()) << what;
    TransformOptions options;
    options.allow_ballpark = true;
    const CrsTransform engine = CrsTransform::create(src, dst, options);
    for (const CrsPoint& point : points) {
        const CrsPoint want = reference.forward(point);
        const std::optional<CrsPoint> got = engine.forward(point);
        const bool want_valid = std::isfinite(want.x) && std::isfinite(want.y) &&
                                std::isfinite(want.z);
        EXPECT_EQ(got.has_value(), want_valid)
            << what << " at " << point.x << "," << point.y << ": validity differs";
        if (!want_valid || !got) {
            continue;
        }
        EXPECT_EQ(got->x, want.x) << what << " (x) at " << point.x << "," << point.y;
        EXPECT_EQ(got->y, want.y) << what << " (y) at " << point.x << "," << point.y;
        EXPECT_EQ(got->z, want.z) << what << " (z) at " << point.x << "," << point.y;
    }
}

}  // namespace

TEST(Crs, AutomaticMatchesBareProjPipeline) {
    const CoordinateSystem nad27 = CoordinateSystem::from_epsg(4267);
    const CoordinateSystem nad83 = CoordinateSystem::from_epsg(4269);
    const CoordinateSystem etrs89 = CoordinateSystem::from_epsg(4258);
    const CoordinateSystem bng = CoordinateSystem::from_epsg(27700);
    const CoordinateSystem rd_new = CoordinateSystem::from_epsg(28992);

    // NAD 27 -> NAD 83. The reported id of the CONUS operation is
    // "DERIVED_FROM(EPSG):1241", which is not a proj.db row — the case that
    // used to build a null shift when an id was handed back to PROJ.
    expect_matches_bare_proj(nad27, nad83,
                             {CrsPoint{-100.0, 40.0, 0.0}, CrsPoint{5.0, 52.0, 0.0},
                              CrsPoint{-72.0, 45.0, 0.0}},
                             "NAD27 -> NAD83");

    // ETRS89 -> British National Grid: OSGB36 to ETRS89 is Great Britain only.
    expect_matches_bare_proj(etrs89, bng,
                             {CrsPoint{-1.0, 52.0, 0.0}, CrsPoint{-100.0, 40.0, 0.0}},
                             "ETRS89 -> BNG");

    // WGS 84 -> RD New: Amersfoort to WGS 84 (4) is the Netherlands only.
    expect_matches_bare_proj(wgs84(), rd_new,
                             {CrsPoint{5.0, 52.0, 0.0}, CrsPoint{-100.0, 40.0, 0.0}},
                             "WGS 84 -> RD New");

    // NAD 83 -> WGS 84 and WGS 84 -> NAD 27 (the computed-id inverse).
    expect_matches_bare_proj(nad83, wgs84(),
                             {CrsPoint{-100.0, 40.0, 0.0}, CrsPoint{5.0, 52.0, 0.0}},
                             "NAD83 -> WGS 84");
    expect_matches_bare_proj(wgs84(), nad27,
                             {CrsPoint{-72.0, 45.0, 0.0}, CrsPoint{-100.0, 40.0, 0.0}},
                             "WGS 84 -> NAD 27");
    expect_matches_bare_proj(osgb36(), wgs84(),
                             {CrsPoint{1.7167, 52.65, 0.0}, CrsPoint{-100.0, 40.0, 0.0}},
                             "OSGB36 -> WGS 84");

    // WGS 84 -> UTM 33N must stay what it always was: no datum shift at all.
    expect_matches_bare_proj(wgs84(), CoordinateSystem::from_epsg(32633),
                             {CrsPoint{9.0, 50.0, 123.0}, CrsPoint{-1.0, 52.0, 0.0}},
                             "WGS 84 -> UTM 33N");
}

TEST(Crs, AutomaticIsNotRefusedForAGridOnlyTheFirstCandidateNeeds) {
    // OSGB36 -> WGS 84's first listed candidate needs the OSTN15 grid, which a
    // `proj-data` package without the `.tif` files does not carry. PROJ's own
    // automatic pipeline still transforms every point: it falls back to a
    // grid-free shift. The missing-grid diagnosis used to read the first listed
    // candidate and refuse the whole pair, rejecting a transform that works.
    //
    // The assertion does not care which operation PROJ picks or whether the grid
    // is installed — only that `create()` succeeds and agrees with bare PROJ,
    // which is exactly the contract the listing must not override. The pair is
    // chosen because its first candidate is the grid operation on every data
    // package seen, so it reproduces the bug wherever a grid is absent.
    const CoordinateSystem source = osgb36();
    const CoordinateSystem target = wgs84();

    const BareProj reference(source, target);
    ASSERT_TRUE(reference.is_valid());

    std::optional<CrsTransform> engine;
    std::string refusal;
    try {
        engine.emplace(CrsTransform::create(source, target));
    } catch (const Error& error) {
        refusal = error.what();
    }
    ASSERT_TRUE(refusal.empty())
        << "the automatic transform must not be refused for a grid only the first listed "
           "candidate needs, but create() threw: "
        << refusal;
    ASSERT_TRUE(engine.has_value());

    // One point inside Great Britain and one far outside every candidate's area:
    // PROJ answers both, and the engine must answer the same.
    for (const CrsPoint& point : {CrsPoint{-0.0016, 51.5, 0.0}, CrsPoint{-100.0, 40.0, 0.0}}) {
        const CrsPoint want = reference.forward(point);
        ASSERT_TRUE(std::isfinite(want.x) && std::isfinite(want.y));
        const std::optional<CrsPoint> got = engine->forward(point);
        ASSERT_TRUE(got.has_value()) << "at " << point.x << "," << point.y;
        EXPECT_EQ(got->x, want.x) << "at " << point.x << "," << point.y;
        EXPECT_EQ(got->y, want.y) << "at " << point.x << "," << point.y;
    }
}

TEST(Crs, AutomaticAppliesNationalOperationsOnlyInsideTheirArea) {
    // The defect this guards was "one listed candidate applied to every point".
    // Reproduce that old behaviour explicitly — take the first candidate from a
    // widened listing, normalise it, and run it everywhere — and require the
    // engine's unpinned transform to differ from it wherever PROJ's own
    // per-point pipeline does, following PROJ rather than the candidate.
    //
    // **This assertion has a premise that is PROJ's, not the engine's.** For it
    // to say anything, PROJ's bare pipeline must actually pick a *different*
    // operation for a point outside the first candidate's area. That depends on
    // the linked PROJ and the installed grids: with PROJ 9.8 plus the full
    // grid set the Netherlands pair diverges by kilometres; with the gridless
    // data package a CI runner gets from `proj-data` (PROJ 9.4, no `.tif`s) the
    // bare pipeline instantiates the area-limited operation itself and agrees
    // with candidate 0 everywhere, so there is nothing to discriminate. Nothing
    // is wrong with the engine in that case — `AutomaticMatchesBareProjPipeline`
    // still requires it to reproduce PROJ's answer exactly.
    //
    // So the samples loop runs on every machine, and only the "at least one
    // pair discriminated" premise skips, naming the condition. The assertions
    // that hold regardless (engine == bare PROJ at every sample; UTM stays a
    // plain projection; an outside-every-area point stays put) live in
    // `AutomaticKeepsTheProjectionAndTheOutsidePoint` so a skip here cannot
    // hide them.
    TransformOptions options;
    options.allow_ballpark = true;

    struct Sample {
        int src_epsg;
        int dst_epsg;
        CrsPoint point;
        const char* what;
    };
    const std::vector<Sample> samples = {
        {4326, 28992, {-100.0, 40.0, 0.0}, "WGS 84 -> RD New outside the Netherlands"},
        {4258, 27700, {-100.0, 40.0, 0.0}, "ETRS89 -> BNG outside Great Britain"},
        {4267, 4269, {5.0, 52.0, 0.0}, "NAD27 -> NAD83 outside North America"},
        {4269, 4326, {5.0, 52.0, 0.0}, "NAD83 -> WGS 84 outside North America"},
    };

    PJ_CONTEXT* ctx = thread_context();
    int discriminating = 0;
    for (const Sample& sample : samples) {
        const CoordinateSystem src = CoordinateSystem::from_epsg(sample.src_epsg);
        const CoordinateSystem dst = CoordinateSystem::from_epsg(sample.dst_epsg);

        // The old behaviour: candidate 0 of the widened listing, everywhere.
        PJ* src_pj = proj_create(ctx, src.to_wkt().c_str());
        PJ* dst_pj = proj_create(ctx, dst.to_wkt().c_str());
        ASSERT_NE(src_pj, nullptr);
        ASSERT_NE(dst_pj, nullptr);
        PJ_OPERATION_FACTORY_CONTEXT* factory =
            proj_create_operation_factory_context(ctx, nullptr);
        ASSERT_NE(factory, nullptr);
        proj_operation_factory_context_set_grid_availability_use(
            ctx, factory, PROJ_GRID_AVAILABILITY_IGNORED);
        proj_operation_factory_context_set_spatial_criterion(
            ctx, factory, PROJ_SPATIAL_CRITERION_PARTIAL_INTERSECTION);
        PJ_OBJ_LIST* candidates = proj_create_operations(ctx, src_pj, dst_pj, factory);
        ASSERT_NE(candidates, nullptr);
        ASSERT_GT(proj_list_get_count(candidates), 0) << sample.what;
        PJ* first = proj_list_get(ctx, candidates, 0);
        ASSERT_NE(first, nullptr);
        PJ* first_normalised = proj_normalize_for_visualization(ctx, first);
        ASSERT_NE(first_normalised, nullptr);

        const PJ_COORD where =
            proj_coord(sample.point.x, sample.point.y, sample.point.z, 0.0);
        const PJ_COORD as_candidate = proj_trans(first_normalised, PJ_FWD, where);
        const bool candidate_valid =
            std::isfinite(as_candidate.xyzt.x) && std::isfinite(as_candidate.xyzt.y);

        const BareProj proj_itself(src, dst);
        ASSERT_TRUE(proj_itself.is_valid());
        const CrsPoint as_proj = proj_itself.forward(sample.point);
        const bool proj_valid = std::isfinite(as_proj.x) && std::isfinite(as_proj.y);
        ASSERT_TRUE(proj_valid) << sample.what << ": bare PROJ must answer this sample";

        // Half one, meaningful on every PROJ version and data package: the
        // engine reproduces bare PROJ exactly.
        const CrsTransform engine = CrsTransform::create(src, dst, options);
        const std::optional<CrsPoint> got = engine.forward(sample.point);
        ASSERT_TRUE(got.has_value()) << sample.what;
        EXPECT_EQ(got->x, as_proj.x) << sample.what << " must reproduce bare PROJ (x)";
        EXPECT_EQ(got->y, as_proj.y) << sample.what << " must reproduce bare PROJ (y)";

        // Half two, measurable only where PROJ itself discriminates: candidate 0
        // of the widened listing is not what PROJ applies for a point outside
        // its area, and the engine must not apply it either.
        if (candidate_valid &&
            std::hypot(as_candidate.xyzt.x - as_proj.x, as_candidate.xyzt.y - as_proj.y) > 1e-6) {
            ++discriminating;
            EXPECT_GT(std::hypot(got->x - as_candidate.xyzt.x, got->y - as_candidate.xyzt.y), 1e-6)
                << sample.what << ": the first listed candidate must not be applied everywhere";
        }

        proj_destroy(first_normalised);
        proj_destroy(first);
        proj_list_destroy(candidates);
        proj_operation_factory_context_destroy(factory);
        proj_destroy(src_pj);
        proj_destroy(dst_pj);
    }
    if (discriminating == 0) {
        // See the comment at the top: this is the linked PROJ's routing, not
        // the engine's. Name what was measured so a reader knows the guard did
        // not run rather than that it passed.
        //
        // A skip would hide any expectation the loop already recorded, so it is
        // only taken when the test is still passing; a real disagreement is
        // reported as a failure. (The engine-matches-bare-PROJ half is asserted
        // unconditionally by `AutomaticMatchesBareProjPipeline` either way.)
        ASSERT_TRUE(!::testing::Test::HasFailure())
            << "the engine disagreed with bare PROJ above; refusing to report a skip over it";
        GTEST_SKIP() << "PROJ " << proj_version() << " with this data package ("
                     << proj_data_info().search_directory.string()
                     << ") applies the first listed candidate everywhere for every sample, so "
                        "candidate-0 and the bare pipeline never differ and there is nothing to "
                        "discriminate. A PROJ data set with the national grids makes this case "
                        "run; the engine's equality with bare PROJ is asserted unconditionally by "
                        "Crs.AutomaticMatchesBareProjPipeline.";
    }
}

TEST(Crs, AutomaticKeepsTheProjectionAndTheOutsidePoint) {
    // The half of the automatic-transform contract that holds on every PROJ and
    // data package, so none of it is skipped where `discriminating` cannot be
    // measured (see the test above).
    TransformOptions options;
    options.allow_ballpark = true;

    // WGS 84 -> UTM 33N: the projection only, unchanged.
    {
        const CrsTransform shift =
            CrsTransform::create(wgs84(), CoordinateSystem::from_epsg(32633));
        expect_point(shift.forward(CrsPoint{9.0, 50.0, 123.0}),
                     CrsPoint{70134.497635423846, 5555901.554208880290, 123.0}, 1e-9,
                     "WGS 84 -> UTM 33N");
        EXPECT_NE(shift.description().find("utm"), std::string::npos) << shift.description();
        EXPECT_EQ(shift.description().find("hgridshift"), std::string::npos)
            << "a UTM conversion must not grow a datum shift: " << shift.description();
    }

    // NAD 27 -> NAD 83 needs a NADCON grid. Where one is installed the CONUS
    // point must move; where none is, the pair must be a clean null shift rather
    // than a crash.
    {
        const CrsTransform shift = CrsTransform::create(
            CoordinateSystem::from_epsg(4267), CoordinateSystem::from_epsg(4269), options);
        const std::optional<CrsPoint> conus = shift.forward(CrsPoint{-100.0, 40.0, 0.0});
        ASSERT_TRUE(conus.has_value());
        if (grid_installed("us_noaa_conus.tif") ||
            grid_installed("us_noaa_nadcon5_nad27_nad83_1986_conus.tif")) {
            EXPECT_GT(std::hypot(conus->x + 100.0, conus->y - 40.0), 1e-7)
                << "the NADCON grid must move a CONUS point";
        }
        // Outside every NAD 27 -> NAD 83 area there is nothing to apply.
        expect_point(shift.forward(CrsPoint{5.0, 52.0, 0.0}), CrsPoint{5.0, 52.0, 0.0}, 1e-12,
                     "NAD27 -> NAD83 outside every area");
    }
}

TEST(Crs, PinnedGridOperationIsNotANullShift) {
    // The pinned form of a grid operation: the shift must move the point by the
    // grid amount, not pass it through. EPSG:1693 is the stored NAD 27 -> WGS 84
    // shift, so pinning the listed *inverse* exercises the orientation logic in
    // `with_transformation()` and the object-based pipeline in transform.cpp —
    // the two places a decorated identifier ("INVERSE(DERIVED_FROM(EPSG)):1693")
    // used to be handed back to PROJ and silently became a null shift.
    if (!grid_installed("ca_nrc_ntv2_0.tif")) {
        GTEST_SKIP() << "the NTv2 grid ca_nrc_ntv2_0.tif is not installed; install proj-data to "
                        "run the pinned WGS 84 -> NAD 27 case";
    }
    const std::vector<DatumTransformation> list =
        wgs84().available_transformations(CoordinateSystem::from_epsg(4267));
    const DatumTransformation* inverse_shift = find_code(list, "1693");
    ASSERT_NE(inverse_shift, nullptr) << "the inverse NAD 27 -> WGS 84 shift is not offered";
    EXPECT_EQ(inverse_shift->auth_name, "INVERSE(DERIVED_FROM(EPSG))");

    const CrsPoint source{-72.0, 45.0, 0.0};
    const CoordinateSystem pinned_crs = wgs84().with_transformation(*inverse_shift);
    const CrsTransform pinned = CrsTransform::create(pinned_crs, CoordinateSystem::from_epsg(4267));
    const std::optional<CrsPoint> got = pinned.forward(source);
    ASSERT_TRUE(got.has_value());
    EXPECT_NE(pinned.description().find("hgridshift"), std::string::npos) << pinned.description();
    EXPECT_GT(std::hypot(got->x - source.x, got->y - source.y), 1e-8)
        << "a pinned grid operation must not be a null shift";

    // The pinned CRS is persisted as WKT and rebuilt on another thread, so a
    // second transform from the same value must reproduce the answer exactly.
    const CrsTransform again = CrsTransform::create(CoordinateSystem::from_wkt(pinned_crs.to_wkt()),
                                                    CoordinateSystem::from_epsg(4267));
    const std::optional<CrsPoint> repeated = again.forward(source);
    ASSERT_TRUE(repeated.has_value());
    EXPECT_EQ(repeated->x, got->x);
    EXPECT_EQ(repeated->y, got->y);

    // ... and the shift applied is the grid's, so a different grid operation
    // (the plain 7-parameter Helmert offered for a different pair) cannot be
    // what ran.
    EXPECT_EQ(pinned.description().find("helmert"), std::string::npos)
        << "the pinned grid operation must not have been replaced by a Helmert";
}

TEST(Crs, WithTransformationRefusesAnOperationFromAnotherPair) {
    // The API takes a DatumTransformation, not a CRS pair, so a hand-built or
    // stale entry can name a shift between other CRSs. Binding it would apply
    // the wrong datum shift; it must be refused, naming the CRSs involved.
    const std::vector<DatumTransformation> osgb =
        osgb36().available_transformations(wgs84());
    const DatumTransformation* helmert = find_code(osgb, "1314");
    ASSERT_NE(helmert, nullptr);

    // EPSG:1314 is OSGB36 -> WGS 84. NAD 27 is neither end of it.
    try {
        (void)CoordinateSystem::from_epsg(4267).with_transformation(*helmert);
        FAIL() << "expected the wrong-pair refusal";
    } catch (const ValueError& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("other CRSs"), std::string::npos) << what;
    }

    // The same operation pinned from either of its own ends is accepted.
    EXPECT_NO_THROW((void)osgb36().with_transformation(*helmert));
    EXPECT_NO_THROW((void)wgs84().with_transformation(*helmert));

    // ... and an identifier proj.db does not know is still a ValueError.
    DatumTransformation bogus;
    bogus.auth_name = "EPSG";
    bogus.code = "999999";
    bogus.name = "not a real operation";
    EXPECT_THROW((void)wgs84().with_transformation(bogus), ValueError);
}

// ---------------------------------------------------------------------------
// A pinned CRS survives persistence
// ---------------------------------------------------------------------------

namespace {

/// A scratch directory, unique across processes (ctest -jN), removed on exit.
class ScratchDir {
public:
    ScratchDir()
        : path_(platform::unique_temp_path("alphashape_pinned_crs_", "")) {
        std::error_code ec;
        std::filesystem::create_directories(path_, ec);
    }
    ~ScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    ScratchDir(const ScratchDir&) = delete;
    ScratchDir& operator=(const ScratchDir&) = delete;

    std::string file(const std::string& name) const {
        return (std::filesystem::path(path_) / name).string();
    }

private:
    std::string path_;
};

}  // namespace

TEST(Crs, PinnedCrsSurvivesDocXmlRoundTrip) {
    // A pinned CRS is stored as its BOUNDCRS WKT, with `epsg` 0 because a
    // BOUNDCRS has no authority code. Saving and loading must preserve both the
    // WKT and the behaviour, and `kind()` must still report the source CRS's
    // kind rather than fall through to the bound-CRS default.
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    const DatumTransformation* helmert = find_code(list, "1314");
    ASSERT_NE(helmert, nullptr);
    const CoordinateSystem pinned = osgb36().with_transformation(*helmert);
    ASSERT_EQ(pinned.epsg(), 0) << "a BOUNDCRS carries no EPSG code";
    EXPECT_EQ(pinned.kind(), CrsKind::Geographic2D)
        << "a bound CRS is as its source CRS is, not a projected/unknown kind";

    const CrsPoint point{dms(1, 43, 4.5177), dms(52, 39, 27.2531), 0.0};
    const CrsTransform before = CrsTransform::create(pinned, wgs84());
    const std::optional<CrsPoint> before_answer = before.forward(point);
    ASSERT_TRUE(before_answer.has_value());

    ScratchDir scratch;
    {
        ::alphashape::Document document;
        ::alphashape::Chunk& chunk = document.workspace().add_chunk("crs");
        document.set_active_chunk_id(chunk.id());
        chunk.set_coordinate_system(pinned);
        document.set_global_coordinate_system(pinned);
        ::alphashape::ProjectIO::save(document, scratch.file("pinned.psx"));
    }

    const std::shared_ptr<::alphashape::Document> loaded =
        ::alphashape::ProjectIO::load(scratch.file("pinned.psx"));
    ASSERT_NE(loaded, nullptr);
    const CoordinateSystem after = loaded->global_coordinate_system();
    EXPECT_EQ(after.to_wkt(), pinned.to_wkt()) << "the BOUNDCRS WKT must round-trip verbatim";
    EXPECT_EQ(after.epsg(), 0);
    EXPECT_EQ(after.kind(), CrsKind::Geographic2D);
    EXPECT_EQ(loaded->workspace().chunks().front()->coordinate_system().to_wkt(),
              pinned.to_wkt())
        << "the chunk slot must round-trip too";

    const std::optional<CrsPoint> after_answer =
        CrsTransform::create(after, wgs84()).forward(point);
    ASSERT_TRUE(after_answer.has_value());
    EXPECT_EQ(after_answer->x, before_answer->x);
    EXPECT_EQ(after_answer->y, before_answer->y);
}

TEST(Crs, KindUnwrapsABoundCrs) {
    // `kind()` on a BOUNDCRS recurses into its source CRS. A 3D source must
    // report Geographic3D, not the 2D default the fallback would give.
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    const DatumTransformation* helmert = find_code(list, "1314");
    ASSERT_NE(helmert, nullptr);
    const CoordinateSystem bound = osgb36().with_transformation(*helmert);

    EXPECT_EQ(bound.kind(), osgb36().kind());
    EXPECT_EQ(bound.kind(), CrsKind::Geographic2D);

    // An equivalent copy parsed afresh reaches the same conclusion, so the kind
    // is a property of the stored WKT rather than of how the value was built.
    const CoordinateSystem reparsed = CoordinateSystem::from_wkt(bound.to_wkt());
    EXPECT_EQ(reparsed.kind(), CrsKind::Geographic2D);
    EXPECT_EQ(reparsed.to_wkt(), bound.to_wkt());
}

}  // namespace
}  // namespace alphashape::crs
