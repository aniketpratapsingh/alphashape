#include "common/algorithm/elevation/surface_frame.hpp"

#include <cmath>
#include <string>
#include <utility>

#include "base/utils/error.hpp"
#include "model/geo/geodesy.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "model/crs/local_frame.hpp"
#endif

namespace alphashape::elevation {
namespace {

/// A 4x4 transform is "the identity" for our purposes when it equals the exact
/// identity matrix; Chunk stores Transform::Identity() for unreferenced chunks.
bool is_identity(const Eigen::Matrix4d& m) { return m == Eigen::Matrix4d::Identity(); }

/// The regular UTM zone covering (lon, lat): the 6-degree rule with no
/// Norway/Svalbard exceptions. This is the contract the retired
/// geo::utm_zone_for documented, kept so an existing project's auto-selected
/// DEM lands in exactly the same zone it did before the CRS migration.
int auto_utm_epsg(double longitude_deg, double latitude_deg) {
    double lon = std::remainder(longitude_deg, 360.0);
    if (lon < -180.0) {
        lon += 360.0;
    } else if (lon >= 180.0) {
        lon -= 360.0;
    }
    int zone = static_cast<int>(std::floor((lon + 180.0) / 6.0)) + 1;
    zone = zone < 1 ? 1 : (zone > 60 ? 60 : zone);
    return (latitude_deg >= 0.0 ? 32600 : 32700) + zone;
}

}  // namespace

crs::CoordinateSystem parse_crs_param(const std::string& text) {
    if (text.empty()) {
        return crs::CoordinateSystem::local();
    }
#if defined(ALPHASHAPE_USE_PROJ)
    // "EPSG:n" (the spelling the engine itself emits) and a bare number are the
    // same authority code; anything else is WKT or a PROJ string, both of which
    // PROJ parses from text.
    std::string body = text;
    const std::string epsg_prefix = "EPSG:";
    if (body.size() > epsg_prefix.size() &&
        body.compare(0, epsg_prefix.size(), epsg_prefix) == 0) {
        body = body.substr(epsg_prefix.size());
    }
    const bool all_digits =
        !body.empty() && body.find_first_not_of("0123456789") == std::string::npos;
    if (all_digits) {
        return crs::CoordinateSystem::from_epsg(std::stoi(body));
    }
    if (text.find("+proj") != std::string::npos) {
        return crs::CoordinateSystem::from_proj(text);
    }
    return crs::CoordinateSystem::from_wkt(text);
#else
    throw ValueError("BuildDem/BuildOrthomosaic: the crs parameter needs a build with PROJ "
                     "(ALPHASHAPE_USE_PROJ=ON)");
#endif
}

#if defined(ALPHASHAPE_USE_PROJ)

struct SurfaceFrame::Impl {
    crs::CoordinateSystem crs = crs::CoordinateSystem::local();
    /// Local -> target through the chunk transform's ECEF anchor.
    crs::LocalFrame frame;

    Impl(crs::CoordinateSystem target, const Eigen::Matrix3d& rotation,
         const Eigen::Vector3d& origin_ecef)
        : crs(std::move(target)),
          frame(crs::LocalFrame::with_rotation(crs, origin_ecef, rotation)) {}
};

#else  // !ALPHASHAPE_USE_PROJ

/// Without the CRS engine there is no geographic frame: the type exists only so
/// DEM/ortho code compiles, and every request for a real CRS throws.
struct SurfaceFrame::Impl {
    crs::CoordinateSystem crs;
};

#endif  // ALPHASHAPE_USE_PROJ

SurfaceFrame::SurfaceFrame() = default;
SurfaceFrame::SurfaceFrame(const SurfaceFrame&) = default;
SurfaceFrame& SurfaceFrame::operator=(const SurfaceFrame&) = default;
SurfaceFrame::SurfaceFrame(SurfaceFrame&&) noexcept = default;
SurfaceFrame& SurfaceFrame::operator=(SurfaceFrame&&) noexcept = default;
SurfaceFrame::~SurfaceFrame() = default;

SurfaceFrame SurfaceFrame::for_chunk(const Chunk& chunk, int epsg) {
    if (is_identity(chunk.transform())) {
        // Unreferenced chunk: only the local identity frame has meaning.
        if (epsg != 0) {
            throw ValueError(
                "SurfaceFrame: unreferenced chunk has no projection for EPSG:" +
                std::to_string(epsg));
        }
        return SurfaceFrame{};
    }
    if (epsg == 0) {
        // Auto-select the UTM zone covering the chunk-local origin.
        const geo::Geodetic origin = geo::ecef_to_geodetic(chunk.transform().block<3, 1>(0, 3));
        epsg = auto_utm_epsg(origin.longitude_deg, origin.latitude_deg);
    }
#if defined(ALPHASHAPE_USE_PROJ)
    return for_chunk(chunk, crs::CoordinateSystem::from_epsg(epsg));
#else
    throw ValueError("SurfaceFrame: EPSG:" + std::to_string(epsg) +
                     " needs a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
#endif
}

#if defined(ALPHASHAPE_USE_PROJ)

SurfaceFrame SurfaceFrame::for_chunk(const Chunk& chunk, const crs::CoordinateSystem& target) {
    // An unset target means "the chunk's own CRS" (M12 §4.9); a chunk with no
    // CRS either is the local engineering frame.
    const crs::CoordinateSystem wanted = target.is_set() ? target : chunk.coordinate_system();
    if (!wanted.is_set() || wanted.is_local()) {
        if (!is_identity(chunk.transform())) {
            throw ValueError(
                "SurfaceFrame: a georeferenced chunk has no local plane; request a "
                "projected or geographic CRS");
        }
        return SurfaceFrame{};  // local identity frame
    }
    if (wanted.kind() == crs::CrsKind::Vertical) {
        throw ValueError("SurfaceFrame: a vertical CRS has no plane to write a grid in");
    }
    if (is_identity(chunk.transform())) {
        // The ECEF anchor is unknown, so a real CRS has nothing to attach to.
        throw ValueError(
            "SurfaceFrame: unreferenced chunk has no georeference for the requested CRS");
    }
    SurfaceFrame frame;
    frame.impl_ = std::make_shared<const Impl>(wanted, chunk.transform().block<3, 3>(0, 0),
                                               chunk.transform().block<3, 1>(0, 3));
    return frame;
}

const crs::CoordinateSystem& SurfaceFrame::crs() const {
    static const crs::CoordinateSystem kLocal = crs::CoordinateSystem::local();
    return impl_ ? impl_->crs : kLocal;
}

int SurfaceFrame::epsg() const { return impl_ ? impl_->crs.epsg() : 0; }

Eigen::Vector3d SurfaceFrame::to_plane(const Eigen::Vector3d& local) const {
    if (!impl_) {
        return local;
    }
    const crs::CrsPoint p = impl_->frame.to_crs(local);
    return {p.x, p.y, p.z};
}

Eigen::Vector3d SurfaceFrame::to_local(const Eigen::Vector3d& plane) const {
    if (!impl_) {
        return plane;
    }
    return impl_->frame.to_local(crs::CrsPoint{plane.x(), plane.y(), plane.z()});
}

Eigen::Vector2d SurfaceFrame::metres_per_plane_unit(double plane_y) const {
    if (!impl_) {
        return Eigen::Vector2d(1.0, 1.0);
    }
    const crs::CrsKind kind = impl_->crs.kind();
    if (kind != crs::CrsKind::Geographic2D && kind != crs::CrsKind::Geographic3D) {
        // A projected plane is metric. UTM's point scale is ~0.9996 and the
        // difference is immaterial for a cell size.
        return Eigen::Vector2d(1.0, 1.0);
    }
    // WGS84 radii of curvature, the same constants geodesy.cpp uses.
    constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
    const double e2 = 2.0 * geo::kWgs84F - geo::kWgs84F * geo::kWgs84F;
    const double sin_lat = std::sin(plane_y * kDeg2Rad);
    const double w = 1.0 - e2 * sin_lat * sin_lat;
    const double prime_vertical = geo::kWgs84A / std::sqrt(w);
    const double meridian = geo::kWgs84A * (1.0 - e2) / (w * std::sqrt(w));
    const double cos_lat = std::cos(plane_y * kDeg2Rad);
    return Eigen::Vector2d(prime_vertical * cos_lat * kDeg2Rad, meridian * kDeg2Rad);
}

#else  // !ALPHASHAPE_USE_PROJ

SurfaceFrame SurfaceFrame::for_chunk(const Chunk& chunk, const crs::CoordinateSystem& target) {
    static_cast<void>(chunk);
    static_cast<void>(target);
    throw ValueError(
        "SurfaceFrame: a CRS plane needs a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
}

const crs::CoordinateSystem& SurfaceFrame::crs() const {
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
}

int SurfaceFrame::epsg() const { return 0; }

bool SurfaceFrame::is_local() const { return impl_ == nullptr; }

Eigen::Vector3d SurfaceFrame::to_plane(const Eigen::Vector3d& local) const { return local; }

Eigen::Vector3d SurfaceFrame::to_local(const Eigen::Vector3d& plane) const { return plane; }

Eigen::Vector2d SurfaceFrame::metres_per_plane_unit(double) const {
    return Eigen::Vector2d(1.0, 1.0);
}

#endif  // ALPHASHAPE_USE_PROJ

}  // namespace alphashape::elevation
