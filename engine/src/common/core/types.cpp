#include "common/core/types.hpp"

#include <algorithm>
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
        vertical = cs.vertical();
    }
}

CRS::operator crs::CoordinateSystem() const {
    if (!is_set()) {
        return crs::CoordinateSystem::local();
    }
    crs::CoordinateSystem cs = epsg != 0 ? crs::CoordinateSystem::from_epsg(epsg)
                                        : crs::CoordinateSystem::from_wkt(wkt);
    // The vertical datum lives beside the horizontal WKT, so it is reattached
    // here: the canonical WKT of a horizontal CRS does not carry it.
    if (vertical && vertical->is_set()) {
        cs = cs.with_vertical(*vertical);
    }
    return cs;
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

bool optional_vector_equal(const std::optional<Eigen::Vector3d>& a,
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

bool reference_location_equal(const std::optional<Eigen::Vector3d>& a,
                              const std::optional<Eigen::Vector3d>& b) {
    return optional_vector_equal(a, b);
}

/// Linear-interpolated percentile of an already-sorted vector, matching the
/// region fit AlignCameras has always used.
double percentile_value(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double pos = q * (static_cast<double>(sorted.size()) - 1.0);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
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
           fixed_params == other.fixed_params &&
           master_sensor_id == other.master_sensor_id && film == other.film &&
           antenna == other.antenna;
}

void validate_sensor_masters(const std::vector<Sensor>& sensors) {
    const auto count = static_cast<std::int32_t>(sensors.size());
    for (std::int32_t i = 0; i < count; ++i) {
        const Sensor& sensor = sensors[static_cast<std::size_t>(i)];
        const std::int32_t master = sensor.master_sensor_id;
        if (master < 0) {
            continue;
        }
        const std::string label = "sensor " + std::to_string(i) + " ('" + sensor.label + "')";
        if (master >= count) {
            throw ValueError(label + " names out-of-range master " + std::to_string(master));
        }
        if (master == i) {
            throw ValueError(label + " cannot be its own master");
        }
        const Sensor& parent = sensors[static_cast<std::size_t>(master)];
        if (parent.master_sensor_id != -1) {
            throw ValueError(label + " names sensor " + std::to_string(master) +
                             ", which is itself a slave");
        }
        if (parent.type != sensor.type || parent.width != sensor.width ||
            parent.height != sensor.height) {
            throw ValueError(label + " and its master sensor " + std::to_string(master) +
                             " have different types or image sizes");
        }
    }
}

std::int32_t effective_sensor_id(const std::vector<Sensor>& sensors,
                                 std::int32_t sensor_id) {
    const auto count = static_cast<std::int32_t>(sensors.size());
    if (sensor_id < 0 || sensor_id >= count) {
        throw ValueError("sensor index " + std::to_string(sensor_id) + " is out of range");
    }
    const std::int32_t master =
        sensors[static_cast<std::size_t>(sensor_id)].master_sensor_id;
    if (master < 0) {
        return sensor_id;
    }
    if (master >= count) {
        throw ValueError("sensor " + std::to_string(sensor_id) +
                         " names out-of-range master " + std::to_string(master));
    }
    return master;
}

Region region_from_points(const std::vector<Eigen::Vector3d>& points,
                          const Region& fallback) {
    if (points.size() < 4) {
        return fallback;
    }
    Region region = fallback;
    std::vector<double> values;
    values.reserve(points.size());
    for (int axis = 0; axis < 3; ++axis) {
        values.clear();
        for (const Eigen::Vector3d& point : points) {
            values.push_back(point[axis]);
        }
        std::sort(values.begin(), values.end());
        region.centre[axis] = percentile_value(values, 0.5);
        const double span = percentile_value(values, 0.95) - percentile_value(values, 0.05);
        region.size[axis] = span * 1.1;
    }
    return region;
}

bool crs_is_geographic(const CRS& crs) {
    const int code = crs.epsg;
    if (code != 0) {
        return code == 4326 || (code >= 4000 && code <= 4999);
    }
    // No EPSG code: a WKT that names a projected CRS is still a refusal, and an
    // empty slot declares nothing.
    const std::string& wkt = crs.wkt;
    if (!wkt.empty()) {
        return wkt.find("PROJCRS") == std::string::npos &&
               wkt.find("PROJECTEDCRS") == std::string::npos;
    }
    return true;
}

bool CameraReference::operator==(const CameraReference& other) const {
    // location and orientation are compared with the NaN-equals-NaN rule (see
    // types.hpp); accuracy and enabled are exact compares.
    return reference_location_equal(location, other.location) &&
           accuracy == other.accuracy &&
           optional_vector_equal(orientation_deg, other.orientation_deg) &&
           optional_vector_equal(rotation_accuracy_deg, other.rotation_accuracy_deg) &&
           enabled == other.enabled;
}

bool Camera::operator==(const Camera& other) const {
    return id == other.id && sensor_id == other.sensor_id &&
           label == other.label && pose == other.pose &&
           enabled == other.enabled && selected == other.selected &&
           reference == other.reference && orientation == other.orientation &&
           location_covariance == other.location_covariance &&
           rotation_covariance == other.rotation_covariance &&
           fiducial_pixels == other.fiducial_pixels &&
           film_transform == other.film_transform;
}

bool KeypointSet::operator==(const KeypointSet& other) const {
    return camera_id == other.camera_id && keypoints == other.keypoints &&
           descriptors == other.descriptors;
}

}  // namespace alphashape
