#pragma once

// The document model.
//
// Shape:
//
//     Document -> Workspace -> Chunk -> Frame
//
// A Document owns workspaces; a Workspace groups chunks; a Chunk owns the
// per-stage containers (cameras, tie points, depth maps, mesh, DEM, ...).
// Frames are the individual input images referenced by cameras.
//
// The container set below is the full doc.xml element vocabulary. v0 populates sensors/cameras/frames/tie_points; the rest are
// declared and empty so the on-disk schema is complete from the start.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "model/camera/calibration.hpp"
#include "model/crs/coordinate_system.hpp"

namespace alphashape {

/// World-space axis-aligned bounding region of a chunk.
struct Region {
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();
    Eigen::Vector3d size = Eigen::Vector3d::Ones();

    bool operator==(const Region& other) const {
        return centre == other.centre && size == other.size;
    }
};

/// A 4x4 affine transform. Used for the chunk's local->world placement.
using Transform = Eigen::Matrix4d;

/// Free-form key/value metadata, mirrored into doc.xml <meta>.
struct Meta {
    std::vector<std::pair<std::string, std::string>> items;

    void set(std::string key, std::string value);
    const std::string* find(const std::string& key) const;

    bool operator==(const Meta& other) const { return items == other.items; }
};

/// Coordinate reference system descriptor.
///
/// Kept as the project-format DTO (`<crs epsg=... wkt=.../>` in doc.xml) and as
/// the bridge to the engine's CRS type. M12 replaced the opaque pair with
/// `crs::CoordinateSystem` (PROJ-backed); `CRS` stays as a deprecated DTO with
/// implicit conversion both ways so existing readers, writers and tests keep
/// compiling until M13 removes it (M12 §4.2, WP-12.2).
///
/// A `CRS` with an EPSG code converts to that authority-coded system; a bare
/// WKT is parsed; an empty DTO is the local frame.
struct CRS {
    CRS() = default;
    std::string wkt;                        ///< WKT2 definition, empty if unset
    int epsg = 0;                           ///< EPSG code, 0 if unknown
    /// Vertical datum (`<vertical kind= name= grid= offset=/>` in doc.xml).
    /// nullopt means ellipsoidal heights; the M12 §4.8 element is written
    /// only when a datum is set, so an older project's XML is unchanged.
    std::optional<crs::VerticalCrs> vertical;

    bool is_set() const { return !wkt.empty() || epsg != 0; }

    bool operator==(const CRS& other) const {
        return wkt == other.wkt && epsg == other.epsg && vertical == other.vertical;
    }

    // Declared unconditionally, even though the definitions need PROJ: a class
    // whose member set depended on a compile definition would be a different
    // type in different translation units (an ODR violation) the moment one
    // consumer included this header through a target that does not link the
    // CRS engine. A `-DALPHASHAPE_USE_PROJ=OFF` build therefore declares these
    // and throws from the bodies rather than hiding them.

    /// DTO -> engine CRS: the EPSG code when there is one (authority wins over
    /// a possibly stale WKT string), else the WKT, else the local frame.
    /// @throws ValueError when both are malformed; Error without PROJ.
    operator crs::CoordinateSystem() const;

    /// Engine CRS -> DTO. A local frame stores epsg 0 and the "LOCAL" WKT; an
    /// authority-coded CRS stores its code plus the canonical WKT2:2019.
    /// @throws Error without PROJ.
    CRS(const crs::CoordinateSystem& cs);  // NOLINT(google-explicit-constructor)
};

/// Camera sensor: the physical imaging device. Several cameras may share one
/// sensor (same body, different shots). Carries the calibration.
struct Sensor {
    std::string label;
    CameraType type = CameraType::Frame;  ///< doc.xml spells this "frame"/"fisheye"
    int width = 0;
    int height = 0;
    double pixel_size_um = 0.0;    ///< micrometres; 0 = unknown
    double focal_length_mm = 0.0;  ///< 0 = unknown
    std::string make, model, serial;  ///< EXIF identity; groups photos into sensors

    /// Initial (from EXIF/spec) or adjusted (from a calibration task).
    Calibration calibration;

    /// Names of coefficients held fixed during adjustment. Any subset of
    /// {"f","cx","cy","b1","b2","k1","k2","k3","k4","p1","p2","p3","p4"}.
    std::vector<std::string> fixed_params;

    bool operator==(const Sensor& other) const;
};

/// The photogrammetric "camera reference": a measured (not adjusted) position
/// from the camera's GNSS log, used as an observation in adjustment.
/// Stored as (longitude_deg, latitude_deg, altitude_m) — longitude first, in
/// x/y/z order. When GNSS reports no altitude the z component is NaN; equality
/// treats NaN as equal to NaN for that field so a reference survives a
/// save/load round-trip and a fully-unknown altitude compares equal to itself.
struct CameraReference {
    std::optional<Eigen::Vector3d> location;

    /// Per-axis 1-sigma accuracy of `location` in metres (east, north, up).
    /// Default 10 m, typical consumer GNSS. Used to weight the reference in
    /// adjustment.
    Eigen::Vector3d accuracy = Eigen::Vector3d::Constant(10.0);

    bool enabled = true;  ///< used as a constraint in adjustment

    bool operator==(const CameraReference& other) const;
};

/// One camera: pose (extrinsics) referencing its sensor and frame.
struct Camera {
    std::int32_t id = -1;
    std::int32_t sensor_id = -1;
    std::string label;

    /// Extrinsics: world -> camera. Identity means "not yet aligned".
    Transform pose = Transform::Identity();

    /// True once SfM/alignment has produced a pose for this camera.
    bool enabled = false;

    /// UI selection state (§4.10). The command layer edits it (WP-11.5b);
    /// persistence of the `selected` attribute is WP-11.8.
    bool selected = false;

    /// Measured position from GPS, the reference for adjustment.
    CameraReference reference;

    /// EXIF orientation tag (1..8). Informational only; pixel geometry is
    /// already expressed in the sensor's own orientation.
    int orientation = 1;

    bool operator==(const Camera& other) const;
};

/// A keypoint: subpixel position, scale, orientation, and a reference into the
/// descriptor store.
struct Keypoint {
    float x = 0.0F;
    float y = 0.0F;
    float scale = 0.0F;         ///< sigma * scale at detection
    float orientation = 0.0F;   ///< dominant orientation, radians

    bool operator==(const Keypoint& other) const {
        return x == other.x && y == other.y && scale == other.scale &&
               orientation == other.orientation;
    }
};

/// A binary descriptor: M-LDB at 63 bytes per keypoint. We keep the byte width a compile-time
/// constant so the on-disk stride is fixed and checkable.
struct Descriptor {
    static constexpr std::size_t kBytes = 63;
    std::array<std::uint8_t, kBytes> bytes{};

    bool operator==(const Descriptor& other) const { return bytes == other.bytes; }
};

/// Per-image keypoint set: the output of feature detection, input to matching.
struct KeypointSet {
    std::int32_t camera_id = -1;
    std::vector<Keypoint> keypoints;
    std::vector<Descriptor> descriptors;

    /// Descriptors must track keypoints one-to-one (fixed 63-byte stride).
    bool consistent() const { return keypoints.size() == descriptors.size(); }

    bool operator==(const KeypointSet& other) const;
};

/// One correspondence of a keypoint index in camera_a's KeypointSet with one in
/// camera_b's.
struct KeypointMatch {
    std::uint32_t index_a = 0;
    std::uint32_t index_b = 0;

    bool operator==(const KeypointMatch& other) const {
        return index_a == other.index_a && index_b == other.index_b;
    }
};

/// Geometrically verified matches between two cameras. Invariant:
/// camera_a < camera_b, and every index is in range of its camera's
/// KeypointSet. Only verified inliers are stored.
struct PairMatches {
    std::int32_t camera_a = -1;
    std::int32_t camera_b = -1;
    std::vector<KeypointMatch> matches;

    bool operator==(const PairMatches& other) const {
        return camera_a == other.camera_a && camera_b == other.camera_b &&
               matches == other.matches;
    }
};

/// A keypoint observed in one camera, as part of a track.
struct Observation {
    std::int32_t camera_id = -1;
    std::uint32_t keypoint_index = 0;

    bool operator==(const Observation& other) const {
        return camera_id == other.camera_id && keypoint_index == other.keypoint_index;
    }
};

/// A tie point: one scene point seen in several cameras. Invariant:
/// observations sorted by camera_id, at most one per camera, size >= 2.
struct Track {
    std::vector<Observation> observations;

    /// Triangulated position in chunk-local coordinates (the frame camera
    /// poses are expressed in), set by AlignCameras. nullopt = not
    /// triangulated. When set, every component is finite.
    std::optional<Eigen::Vector3d> point;

    bool operator==(const Track& other) const {
        return observations == other.observations && point == other.point;
    }
};

}  // namespace alphashape
