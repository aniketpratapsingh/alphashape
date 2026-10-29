// WP-12.6: the ECEF-anchored local frame and the per-object CRS slots.
//
// The local frame is checked where it matters most -- numerical exactness. The
// anchor sits at the ECEF magnitude the spec calls out (about 6.4e6 m), the
// points extend 500 km from it, and the round trip must stay inside a
// micrometre, which is what forces doubles throughout. The matrix form is
// checked against the function form at the anchor and at a kilometre offset.
//
// The slot tests cover the two rules that make the feature safe: a slot that
// was never set follows the chunk CRS, and an unchanged project keeps its exact
// bytes (no <camera_crs> invented on save).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/local_frame.hpp"

namespace alphashape::crs {
namespace {

namespace fs = std::filesystem;

/// WGS 84 geographic 3D (lon/lat degrees, ellipsoidal height) as the CRS.
/// 3D rather than 4326 because a 2D geographic CRS has no height to carry: the
/// forward leg would drop it and the round trip could only be exact in x and y.
/// A chunk georeferenced by the SfM pipeline stores cryptic 4326 labels, but
/// the frame itself is unit-agnostic -- it converts whatever the CRS defines.
CoordinateSystem wgs84_3d() { return CoordinateSystem::from_epsg(4979); }

/// Local -> ECEF rotation whose columns are the east/north/up axes at the
/// anchor, rebuilt here independently of the implementation so the anchor's
/// placement is also verified (not merely accepted from enu_at).
Eigen::Matrix3d enu_rotation(double lon_deg, double lat_deg) {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    const double lon = lon_deg * kDeg;
    const double lat = lat_deg * kDeg;
    const double slon = std::sin(lon);
    const double clon = std::cos(lon);
    const double slat = std::sin(lat);
    const double clat = std::cos(lat);
    Eigen::Matrix3d r;
    r.col(0) << -slon, clon, 0.0;
    r.col(1) << -slat * clon, -slat * slon, clat;
    r.col(2) << clat * clon, clat * slon, slat;
    return r;
}

/// A geodetic position on the WGS 84 ellipsoid -> ECEF. Independent of the
/// frame code (this is the textbook closed form), so the anchor is a known
/// value rather than a value the frame produced.
Eigen::Vector3d geodetic_to_ecef(double lon_deg, double lat_deg, double height_m) {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    constexpr double a = 6378137.0;
    constexpr double f = 1.0 / 298.257223563;
    const double e2 = f * (2.0 - f);
    const double lon = lon_deg * kDeg;
    const double lat = lat_deg * kDeg;
    const double slat = std::sin(lat);
    const double clat = std::cos(lat);
    const double n = a / std::sqrt(1.0 - e2 * slat * slat);
    return {(n + height_m) * clat * std::cos(lon), (n + height_m) * clat * std::sin(lon),
            (n * (1.0 - e2) + height_m) * slat};
}

TEST(Crs, LocalFrameRoundTripExact) {
    // The anchor is in the Alps (about 6.37e6 m of ECEF magnitude) and the
    // points spread over +/-500 km of local space, the spec's worst case.
    const Eigen::Vector3d anchor = geodetic_to_ecef(11.4, 47.3, 600.0);
    const LocalFrame frame = LocalFrame::enu_at(wgs84_3d(), anchor);
    ASSERT_FALSE(frame.is_identity());

    // enu_at builds the east/north/up axes from the anchor's geodetic
    // position; the rotation must match the textbook row for an ENU frame,
    // which is what makes `local` below mean east, north, up.
    const Eigen::Matrix3d expected_rotation = enu_rotation(11.4, 47.3);
    EXPECT_LE((frame.rotation() - expected_rotation).cwiseAbs().maxCoeff(), 1e-12);

    std::mt19937_64 rng(20261027);
    // Deliberately draw in a fixed order (z, y, x) and through the portable
    // helpers so the corpus is identical on every standard library (see
    // tests/test_rng_helpers.hpp and base/utils/portable_rng.hpp). Points are
    // pulled inside the spec's 500 km sphere, so a corner of the box cannot
    // test a distance the acceptance criterion never promised.
    double worst = 0.0;
    for (int i = 0; i < 1000; ++i) {
        const double z = rng::uniform_real(rng, -1.0, 1.0);
        const double y = rng::uniform_real(rng, -1.0, 1.0);
        const double x = rng::uniform_real(rng, -1.0, 1.0);
        Eigen::Vector3d local{x, y, z};
        const double radius = local.norm();
        if (radius > 1.0) {
            local /= radius;
        }
        local *= 500000.0;

        const CrsPoint crs = frame.to_crs(local);
        ASSERT_TRUE(std::isfinite(crs.x) && std::isfinite(crs.y) && std::isfinite(crs.z))
            << "point " << i << " outside the transform domain";
        const Eigen::Vector3d back = frame.to_local(crs);
        worst = std::max(worst, (back - local).cwiseAbs().maxCoeff());
    }
    EXPECT_LE(worst, 1e-6) << "worst round-trip error (m): " << worst;
}

TEST(Crs, LocalFrameTransformMatrixMatchesFunction) {
    // Geocentric target: the mapping is affine, so the 4x4 must reproduce it
    // exactly and this pins the axis convention of the matrix columns.
    const Eigen::Vector3d geocentric_anchor = geodetic_to_ecef(9.2, 48.8, 250.0);
    const LocalFrame geocentric =
        LocalFrame::enu_at(CoordinateSystem::from_epsg(4978), geocentric_anchor);
    const Eigen::Matrix4d exact = geocentric.transform();
    const Eigen::Vector3d far{700.0, -430.0, 120.0};
    const Eigen::Vector4d mapped =
        exact * Eigen::Vector4d{far.x(), far.y(), far.z(), 1.0};
    const CrsPoint exact_expected = geocentric.to_crs(far);
    // A micrometre: the central difference divides values near the 6.4e6 m ECEF
    // magnitude by a 200 m span, so it cannot be bit-exact, but it must not
    // drift beyond the round-trip criterion either.
    EXPECT_NEAR(mapped.x(), exact_expected.x, 1e-6);
    EXPECT_NEAR(mapped.y(), exact_expected.y, 1e-6);
    EXPECT_NEAR(mapped.z(), exact_expected.z, 1e-6);

    // Geographic target: the mapping is curved, so the matrix is the anchor
    // linearisation. At a 10 km offset the ellipsoid curvature is ~8 m, far
    // beyond a linearisation; at 10 m it is a tenth of a millimetre, which is
    // the scale the matrix is documented for.
    const LocalFrame frame =
        LocalFrame::enu_at(wgs84_3d(), geodetic_to_ecef(9.2, 48.8, 250.0));
    const Eigen::Matrix4d m = frame.transform();

    // The translation column is the CRS position of the local origin.
    const CrsPoint origin = frame.to_crs(Eigen::Vector3d::Zero());
    EXPECT_NEAR(m(0, 3), origin.x, 1e-9);
    EXPECT_NEAR(m(1, 3), origin.y, 1e-9);
    EXPECT_NEAR(m(2, 3), origin.z, 1e-9);

    const Eigen::Vector3d near{7.0, -4.3, 1.2};
    const Eigen::Vector4d near_mapped = m * Eigen::Vector4d{near.x(), near.y(), near.z(), 1.0};
    const CrsPoint near_expected = frame.to_crs(near);
    // Linearisation error for a 10 m offset: the second-order term is
    // (offset^2 / R) with R ~ 6.4e6 m, so 1e-5 deg in longitude and ~1e-5 m in
    // height. The tolerances below leave a margin for where in the spread the
    // point falls.
    EXPECT_NEAR(near_mapped.x(), near_expected.x, 1e-5);
    EXPECT_NEAR(near_mapped.y(), near_expected.y, 1e-5);
    EXPECT_NEAR(near_mapped.z(), near_expected.z, 1e-3);

    // transformationMatrix() is the inverse.
    const Eigen::Matrix4d back = frame.transformationMatrix();
    const Eigen::Matrix4d product = back * m;
    EXPECT_NEAR((product - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

/// A tiny but real project: one chunk in a known CRS, saved to a scratch file.
struct ScratchProject {
    fs::path dir;
    ScratchProject() {
        dir = fs::temp_directory_path() /
              ("alphashape_local_frame_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
    }
    ~ScratchProject() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    std::string file(const std::string& name) const { return (dir / name).string(); }
};

/// The project.zip bytes of a .psx, for byte-identity comparisons. The archive
/// lives beside the manifest as "<name>.files/project.zip" (project_io.hpp).
std::vector<std::uint8_t> zip_bytes(const std::string& psx) {
    fs::path base = psx;
    base.replace_extension();
    const fs::path zip = fs::path(base.string() + ".files") / "project.zip";
    std::ifstream in(zip, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>());
}

std::string doc_xml(const std::string& psx) {
    return ZipArchive::parse(zip_bytes(psx)).get_text("doc.xml");
}

Document one_chunk_document(CRS crs) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    chunk.crs() = crs;
    return document;
}

CRS utm33n() {
    CRS crs;
    crs.epsg = 32633;
    crs.wkt = "EPSG:32633";
    return crs;
}

CRS wgs84_dto() {
    CRS crs;
    crs.epsg = 4326;
    crs.wkt = "EPSG:4326";
    return crs;
}

TEST(Crs, PerObjectCrsPersist) {
    ScratchProject scratch;
    Document document = one_chunk_document(utm33n());
    Chunk& chunk = *document.active_chunk();
    chunk.set_camera_crs(wgs84_dto());
    chunk.set_marker_crs(utm33n());
    chunk.set_world_crs(wgs84_dto());

    const std::string psx = scratch.file("slots.psx");
    ProjectIO::save(document, psx);
    const std::string xml = doc_xml(psx);
    EXPECT_NE(xml.find("<camera_crs"), std::string::npos);
    EXPECT_NE(xml.find("<marker_crs"), std::string::npos);
    EXPECT_NE(xml.find("<world_crs"), std::string::npos);

    std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    const Chunk& got = *loaded->active_chunk();
    ASSERT_TRUE(got.explicit_camera_crs().has_value());
    ASSERT_TRUE(got.explicit_marker_crs().has_value());
    ASSERT_TRUE(got.explicit_world_crs().has_value());
    EXPECT_EQ(got.camera_crs().epsg, 4326);
    EXPECT_EQ(got.marker_crs().epsg, 32633);
    EXPECT_EQ(got.world_crs().epsg, 4326);
    // The chunk CRS itself is untouched: the slots are labels, not a conversion.
    EXPECT_EQ(got.crs().epsg, 32633);
    EXPECT_TRUE(got == chunk);
}

TEST(Crs, PerObjectCrsDefaults) {
    // A chunk that never set a slot answers every slot with the chunk CRS, and
    // changing the chunk CRS moves all of them: the defaults are a live view,
    // not a copy taken once.
    Chunk chunk;
    EXPECT_EQ(chunk.crs().epsg, 0);

    chunk.crs() = utm33n();
    EXPECT_EQ(chunk.camera_crs().epsg, 32633);
    EXPECT_EQ(chunk.marker_crs().epsg, 32633);
    EXPECT_EQ(chunk.world_crs().epsg, 32633);
    EXPECT_FALSE(chunk.has_explicit_crs_slots());

    chunk.crs() = wgs84_dto();
    EXPECT_EQ(chunk.camera_crs().epsg, 4326);
    EXPECT_EQ(chunk.marker_crs().epsg, 4326);
    EXPECT_EQ(chunk.world_crs().epsg, 4326);

    // An explicit slot survives the chunk CRS moving underneath it, which is
    // the whole point of having one.
    chunk.set_camera_crs(utm33n());
    chunk.crs() = wgs84_dto();
    EXPECT_EQ(chunk.camera_crs().epsg, 32633);
    EXPECT_EQ(chunk.marker_crs().epsg, 4326);
}

TEST(Crs, IndependentCrsSlots) {
    // Setting one slot must leave the others and the chunk CRS alone.
    Chunk chunk;
    chunk.crs() = utm33n();
    chunk.set_marker_crs(wgs84_dto());

    EXPECT_EQ(chunk.crs().epsg, 32633);
    EXPECT_EQ(chunk.marker_crs().epsg, 4326);
    EXPECT_EQ(chunk.camera_crs().epsg, 32633) << "camera slot must follow the chunk CRS";
    EXPECT_EQ(chunk.world_crs().epsg, 32633) << "world slot must follow the chunk CRS";
    EXPECT_FALSE(chunk.explicit_camera_crs().has_value());
    EXPECT_FALSE(chunk.explicit_world_crs().has_value());

    chunk.clear_marker_crs();
    EXPECT_EQ(chunk.marker_crs().epsg, 32633);
    EXPECT_FALSE(chunk.has_explicit_crs_slots());
}

TEST(Crs, OldProjectWithoutCrsSlotsResavesByteIdentically) {
    ScratchProject scratch;
    Document document = one_chunk_document(utm33n());
    const std::string first = scratch.file("a.psx");
    ProjectIO::save(document, first);

    // No slot was ever set, so nothing extra may be written.
    const std::string xml = doc_xml(first);
    EXPECT_EQ(xml.find("<camera_crs"), std::string::npos);
    EXPECT_EQ(xml.find("<marker_crs"), std::string::npos);
    EXPECT_EQ(xml.find("<world_crs"), std::string::npos);

    std::shared_ptr<Document> loaded = ProjectIO::load(first);
    const Chunk& got = *loaded->active_chunk();
    EXPECT_FALSE(got.has_explicit_crs_slots());
    EXPECT_EQ(got.camera_crs().epsg, 32633);
    EXPECT_EQ(got.marker_crs().epsg, 32633);
    EXPECT_EQ(got.world_crs().epsg, 32633);

    const std::string second = scratch.file("b.psx");
    ProjectIO::save(*loaded, second);
    EXPECT_EQ(zip_bytes(first), zip_bytes(second));
}

}  // namespace
}  // namespace alphashape::crs
