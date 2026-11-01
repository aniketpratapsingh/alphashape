#pragma once

/// @file xmp_reference.hpp
/// @brief DJI-style XMP position, orientation and RTK metadata (WP-13.3).
///
/// Cameras flown on DJI-style aircraft write their position, gimbal attitude and
/// RTK solution into the JPEG APP1 XMP packet (or TIFF tag 700) instead of, or
/// in addition to, EXIF GPS. This module extracts the handful of fields the
/// pipeline consumes. It is a deliberately small tag scan, not an XML parser:
/// packets are a few kilobytes and are treated as hostile input, so a malformed
/// or truncated packet yields whatever fields were parsed before the problem
/// and never throws.
///
/// ## Namespace resolution
///
/// Attribute and element names are resolved through the `xmlns:prefix="uri"`
/// declarations in the packet; the prefix spelling is not assumed. Both
/// serialisations are accepted:
///
///     <rdf:Description xmlns:drone-dji="http://ns.dji.com/drone-dji/1.0/"
///                      drone-dji:GpsLatitude="45.1"/>
///
///     <drone-dji:GpsLatitude>45.1</drone-dji:GpsLatitude>
///
/// ## Orientation mapping
///
/// `CameraReference::orientation_deg` is a Tait-Bryan yaw/pitch/roll triple in
/// the chunk's rotation order, interpreted as a body attitude in the local ENU
/// basis with the fixed nadir camera mount documented in `convert_reference`
/// (`body_to_camera_nadir()`). A level body with yaw 0 therefore points the
/// camera straight down with north up; that is the "nadir, north-up" pose.
///
/// DJI's gimbal angles describe the camera itself:
///
///  * `GimbalYawDegree` is a compass heading, clockwise from true north, so it
///    maps to `yaw = -heading` (the engine's yaw is counter-clockwise from
///    north).
///  * `GimbalPitchDegree` is -90 at nadir and 0 at the horizon, while the stored
///    pitch is the body angle whose -90 mount tilt is already in
///    `body_to_camera_nadir()`. The mapping is therefore `pitch = gimbal + 90`:
///    nadir (-90) becomes 0 and the horizon becomes +90.
///  * Roll is carried through unchanged.
///
/// A nadir camera heading north (`yaw 0`, `pitch -90`, `roll 0`) therefore maps
/// to `(0, 0, 0)`. When the gimbal angles are absent the aircraft's
/// `FlightYaw/Pitch/RollDegree` are used instead; those already describe the
/// body, so `yaw = -flightYaw`, `pitch = flightPitch`, `roll = flightRoll`.
///
/// Angles are reported in the default `ZYX` order; the task layer converts them
/// to the chunk's configured rotation order.

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace alphashape {

/// A yaw/pitch/roll triple in degrees, in the engine's stored convention and
/// the `ZYX` order. See the file comment for the mapping from DJI fields.
struct XmpOrientation {
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double roll_deg = 0.0;
};

/// The reference fields read from an XMP packet. Every field is nullopt when
/// absent or unparsable.
struct XmpReference {
    std::optional<double> latitude_deg;   ///< north-positive
    std::optional<double> longitude_deg;  ///< east-positive
    std::optional<double> absolute_altitude_m;
    std::optional<double> relative_altitude_m;
    std::optional<double> gimbal_yaw_deg;
    std::optional<double> gimbal_pitch_deg;
    std::optional<double> gimbal_roll_deg;
    std::optional<double> flight_yaw_deg;
    std::optional<double> flight_pitch_deg;
    std::optional<double> flight_roll_deg;
    std::optional<int> rtk_flag;
    std::optional<double> rtk_std_lon_m;
    std::optional<double> rtk_std_lat_m;
    std::optional<double> rtk_std_hgt_m;
    /// `drone-dji:DewarpData` verbatim. The AddPhotos task records but does not
    /// yet apply it; see the task documentation.
    std::optional<std::string> dewarp_data;

    /// True when both horizontal components are present.
    bool has_location() const {
        return latitude_deg.has_value() && longitude_deg.has_value();
    }

    /// The stored-convention yaw/pitch/roll described in the file comment, or
    /// nullopt when neither the gimbal nor the flight attitude is usable.
    std::optional<XmpOrientation> orientation_zyx() const;

    /// True when `RtkFlag` names a fixed or float RTK solution.
    bool has_rtk_fix() const;

    /// Per-axis 1-sigma accuracy as (east, north, up) in metres. Nullopt unless
    /// `has_rtk_fix()` and all three standard deviations are present and
    /// strictly positive; the caller then leaves the existing accuracy alone.
    std::optional<std::array<double, 3>> rtk_accuracy_m() const;
};

/// Extracts the reference fields from one XMP packet. Never throws: a malformed
/// or truncated packet returns the fields parsed before the problem. An empty
/// packet yields an all-nullopt result.
XmpReference parse_xmp_reference(std::string_view packet);

}  // namespace alphashape
