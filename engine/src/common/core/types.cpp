#include "common/core/types.hpp"

#include <cmath>

#include "base/utils/error.hpp"
#include "common/core/frame.hpp"

namespace alphashape {

#if defined(ALPHASHAPE_USE_PROJ)

// The bridge between the project-format DTO and the PROJ-backed value type.
// Reading prefers the EPSG code because a stored WKT string from an older
// project may predate a datum rename, while the code is stable; writing stores
// both so a reader without PROJ (a script, the v1 reader) still has the code.
CRS::CRS(const crs::CoordinateSystem& cs) {
    if (cs.is_set()) {
        epsg = cs.epsg();
        wkt = cs.to_wkt();
    }
}

CRS::operator crs::CoordinateSystem() const {
    if (!is_set()) {
        return crs::CoordinateSystem::local();
    }
    if (epsg != 0) {
        return crs::CoordinateSystem::from_epsg(epsg);
    }
    return crs::CoordinateSystem::from_wkt(wkt);
}

#else  // !ALPHASHAPE_USE_PROJ

// The legacy three-CRS stub build has no CRS engine to convert to. Declaring
// the conversion and failing at the call is better than a link error naming a
// missing crs:: symbol the caller never referred to.
CRS::CRS(const crs::CoordinateSystem& cs) {
    static_cast<void>(cs);
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
}

CRS::operator crs::CoordinateSystem() const {
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
}

#endif  // ALPHASHAPE_USE_PROJ

namespace {

/// Bitwise double equality, except that two NaNs compare equal. Used for the
/// camera reference's altitude, which is NaN when GNSS reported no altitude —
/// a value that must survive a save/load round-trip unchanged.
bool same_or_both_nan(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || a == b;
}

bool reference_location_equal(const std::optional<Eigen::Vector3d>& a,
                              const std::optional<Eigen::Vector3d>& b) {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    if (!a.has_value()) {
        return true;
    }
    return same_or_both_nan((*a).x(), (*b).x()) &&
           same_or_both_nan((*a).y(), (*b).y()) &&
           same_or_both_nan((*a).z(), (*b).z());
}

}  // namespace

void Meta::set(std::string key, std::string value) {
    for (auto& item : items) {
        if (item.first == key) {
            item.second = std::move(value);
            return;
        }
    }
    items.emplace_back(std::move(key), std::move(value));
}

const std::string* Meta::find(const std::string& key) const {
    for (const auto& item : items) {
        if (item.first == key) {
            return &item.second;
        }
    }
    return nullptr;
}

bool Sensor::operator==(const Sensor& other) const {
    return label == other.label && type == other.type && width == other.width &&
           height == other.height && pixel_size_um == other.pixel_size_um &&
           focal_length_mm == other.focal_length_mm && make == other.make &&
           model == other.model && serial == other.serial &&
           calibration == other.calibration &&
           fixed_params == other.fixed_params;
}

bool CameraReference::operator==(const CameraReference& other) const {
    // location compared with the NaN-equals-NaN rule (see types.hpp); accuracy
    // and enabled are exact compares.
    return reference_location_equal(location, other.location) &&
           accuracy == other.accuracy && enabled == other.enabled;
}

bool Camera::operator==(const Camera& other) const {
    return id == other.id && sensor_id == other.sensor_id &&
           label == other.label && pose == other.pose &&
           enabled == other.enabled && selected == other.selected &&
           reference == other.reference && orientation == other.orientation;
}

bool KeypointSet::operator==(const KeypointSet& other) const {
    return camera_id == other.camera_id && keypoints == other.keypoints &&
           descriptors == other.descriptors;
}

}  // namespace alphashape
