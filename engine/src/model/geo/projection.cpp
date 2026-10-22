#include "model/geo/projection.hpp"

#include <string>

#include "base/utils/error.hpp"

namespace alphashape::geo {

Projection Projection::from_epsg(int epsg) {
    Projection p;
    if (epsg == 0) {
        p.kind = ProjectionKind::Local;
        return p;
    }
    if (epsg == 4326) {
        p.kind = ProjectionKind::Geographic;
        return p;
    }
    if (const std::optional<UtmZone> zone = utm_zone_from_epsg(epsg)) {
        p.kind = ProjectionKind::Utm;
        p.zone = *zone;
        return p;
    }
    throw ValueError("unsupported projection EPSG:" + std::to_string(epsg));
}

int Projection::epsg() const {
    switch (kind) {
        case ProjectionKind::Local:
            return 0;
        case ProjectionKind::Geographic:
            return 4326;
        case ProjectionKind::Utm:
            return zone.epsg();
    }
    return 0;
}

Eigen::Vector2d Projection::forward(const Geodetic& g) const {
    switch (kind) {
        case ProjectionKind::Geographic:
            return {g.longitude_deg, g.latitude_deg};
        case ProjectionKind::Utm:
            return geodetic_to_utm(g.longitude_deg, g.latitude_deg, zone);
        case ProjectionKind::Local:
            break;
    }
    throw ValueError("forward() undefined for a Local projection");
}

Geodetic Projection::inverse(double x, double y, double height_m) const {
    switch (kind) {
        case ProjectionKind::Geographic:
            return Geodetic{x, y, height_m};
        case ProjectionKind::Utm: {
            const Eigen::Vector2d ll = utm_to_geodetic(x, y, zone);
            return Geodetic{ll.x(), ll.y(), height_m};
        }
        case ProjectionKind::Local:
            break;
    }
    throw ValueError("inverse() undefined for a Local projection");
}

}  // namespace alphashape::geo
