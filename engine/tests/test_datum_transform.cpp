// WP-12.4: datum transformations and grid shifts.
//
// Two layers are tested.
//
// The *selection* layer (`available_transformations`, `with_transformation`) is
// checked against its contract: preferred first, then increasing accuracy with
// unknown accuracy last; pinning an operation must reproduce it; and a grid that
// is not installed must be named rather than silently degrading to a ballpark.
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
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
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

    // Exactly one preferred entry, and it is first: that is the one PROJ would
    // pick, so the ordering contract is "preferred first".
    EXPECT_TRUE(list.front().preferred);
    EXPECT_EQ(std::count_if(list.begin(), list.end(),
                            [](const DatumTransformation& e) { return e.preferred; }),
              1);

    // Behind it, known accuracy ascending and unknown accuracy last. The
    // ballpark has unknown accuracy, so it must come after every Helmert shift.
    bool saw_unknown = false;
    double previous = -1.0;
    for (std::size_t i = 1; i < list.size(); ++i) {
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

TEST(Crs, PreferredTransformationHonoured) {
    const std::vector<DatumTransformation> list = osgb36().available_transformations(wgs84());
    ASSERT_FALSE(list.empty());

    // Pinning an operation and comparing against the unpinned transform is not a
    // meaningful equality test: PROJ's own automatic pick for this pair can be
    // the ballpark (it ranks by area, and a strict area test rejects the
    // national operations), so `CrsTransform::create(osgb36(), wgs84())` may
    // legitimately apply no shift at all. What "pinning is honoured" means here
    // is narrower and checkable: the transform built from the pinned CRS runs
    // the pinned operation's pipeline, and its numbers are that operation's.
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

    // Every CONUS sample moves, by the grid's own sub-arcsecond magnitude.
    const std::vector<CrsPoint> points = {CrsPoint{-100.0, 40.0, 0.0},
                                          CrsPoint{-90.0, 35.0, 0.0},
                                          CrsPoint{-80.0, 30.0, 0.0},
                                          CrsPoint{-120.0, 45.0, 0.0}};
    for (const CrsPoint& point : points) {
        const std::optional<CrsPoint> got = pinned_transform.forward(point);
        ASSERT_TRUE(got.has_value()) << "lon " << point.x;
        const double offset_arcsec =
            std::hypot(got->x - point.x, got->y - point.y) * 3600.0;
        EXPECT_GT(offset_arcsec, 1e-4) << "no shift at lon " << point.x;
        EXPECT_LT(offset_arcsec, 2.0) << "shift above the grid's magnitude at lon " << point.x;
    }
}

TEST(Crs, Towgs84Applied) {
    // A custom WKT with TOWGS84[] is modelled by PROJ as a bound CRS, so its
    // Helmert shift is the preferred operation, is not a ballpark, and produces
    // the same coordinates as EPSG:1314, which carries the same parameters.
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
    ASSERT_FALSE(list.empty());
    EXPECT_TRUE(list.front().preferred);
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
    // WGS 84 to itself: PROJ offers one null geographic offset. The list is
    // short, every entry is a no-op, and a transform built from the pair is the
    // identity. The point is that the entry point does not throw for a pair with
    // nothing to do.
    const std::vector<DatumTransformation> list = wgs84().available_transformations(wgs84());
    ASSERT_FALSE(list.empty());
    EXPECT_TRUE(list.front().preferred);

    const CrsPoint point{9.0, 50.0, 12.5};
    const std::optional<CrsPoint> got = CrsTransform::create(wgs84(), wgs84()).forward(point);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, point);
}

}  // namespace
}  // namespace alphashape::crs
