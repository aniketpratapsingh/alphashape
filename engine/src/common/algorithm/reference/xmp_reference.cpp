#include "common/algorithm/reference/xmp_reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <locale>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#if defined(ALPHASHAPE_USE_PROJ)
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "model/crs/euler.hpp"
#endif

namespace alphashape {
namespace {

constexpr std::string_view kDjiNamespace = "http://ns.dji.com/drone-dji/1.0/";
constexpr std::string_view kXapNamespace = "http://ns.adobe.com/xap/1.0/";
constexpr std::string_view kTiffNamespace = "http://ns.adobe.com/tiff/1.0/";

std::string_view trim(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && (text[begin] == ' ' || text[begin] == '\t' ||
                           text[begin] == '\r' || text[begin] == '\n')) {
        ++begin;
    }
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t' ||
                           text[end - 1] == '\r' || text[end - 1] == '\n')) {
        --end;
    }
    return text.substr(begin, end - begin);
}

/// A classic-locale stream, matching reference_csv.cpp: libc++ (macOS) has no
/// floating-point std::from_chars, and the classic locale is independent of the
/// process's C locale and always uses "." as the decimal point.
std::optional<double> parse_double(std::string_view text) {
    std::string trimmed(trim(text));
    if (trimmed.empty()) {
        return std::nullopt;
    }
    if (trimmed.front() == '+') {
        trimmed.erase(trimmed.begin());
    }
    std::istringstream in(trimmed);
    in.imbue(std::locale::classic());
    double value = 0.0;
    in >> value;
    if (!in || in.peek() != std::char_traits<char>::eof()) {
        return std::nullopt;
    }
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<int> parse_int(std::string_view text) {
    const std::optional<double> value = parse_double(text);
    if (!value.has_value() || std::floor(*value) != *value || std::fabs(*value) > 1.0e9) {
        return std::nullopt;
    }
    return static_cast<int>(*value);
}

/// Collect every `xmlns:prefix="uri"` declaration before interpreting fields,
/// so an attribute may appear before the declaration in the same start tag.
std::map<std::string, std::string> collect_namespaces(std::string_view packet) {
    std::map<std::string, std::string> namespaces;
    std::size_t pos = 0;
    while ((pos = packet.find("xmlns:", pos)) != std::string_view::npos) {
        std::size_t cursor = pos + 6;  // beyond "xmlns:"
        const std::size_t prefix_begin = cursor;
        while (cursor < packet.size() && packet[cursor] != '=' &&
               packet[cursor] != ' ' && packet[cursor] != '\t') {
            ++cursor;
        }
        const std::string prefix(packet.substr(prefix_begin, cursor - prefix_begin));
        while (cursor < packet.size() && (packet[cursor] == ' ' || packet[cursor] == '\t')) {
            ++cursor;
        }
        if (cursor >= packet.size() || packet[cursor] != '=') {
            pos += 6;
            continue;
        }
        ++cursor;
        while (cursor < packet.size() && (packet[cursor] == ' ' || packet[cursor] == '\t')) {
            ++cursor;
        }
        if (cursor >= packet.size() || (packet[cursor] != '"' && packet[cursor] != '\'')) {
            pos += 6;
            continue;
        }
        const char quote = packet[cursor++];
        const std::size_t uri_begin = cursor;
        while (cursor < packet.size() && packet[cursor] != quote) {
            ++cursor;
        }
        if (!prefix.empty()) {
            namespaces[prefix] = std::string(packet.substr(uri_begin, cursor - uri_begin));
        }
        pos = cursor;
    }
    return namespaces;
}

/// The namespace URI an attribute/element qname resolves to, or "" when the
/// prefix is undeclared or unsupported.
std::string_view namespace_of(std::string_view qname,
                              const std::map<std::string, std::string>& namespaces) {
    const std::size_t colon = qname.find(':');
    if (colon == std::string_view::npos || colon == 0) {
        return {};
    }
    const auto found = namespaces.find(std::string(qname.substr(0, colon)));
    if (found == namespaces.end()) {
        return {};
    }
    const std::string_view uri(found->second);
    if (uri == kDjiNamespace) {
        return kDjiNamespace;
    }
    if (uri == kXapNamespace) {
        return kXapNamespace;
    }
    if (uri == kTiffNamespace) {
        return kTiffNamespace;
    }
    return {};
}

std::string_view local_name(std::string_view qname) {
    const std::size_t colon = qname.find(':');
    return colon == std::string_view::npos ? qname : qname.substr(colon + 1);
}

/// Fields accumulated while scanning; the hemisphere refs are applied once the
/// whole packet has been read, because they may follow the coordinate.
struct ScanState {
    XmpReference out;
    std::optional<char> lat_ref;
    std::optional<char> lon_ref;
    /// The innermost start-tag element, for the `<name>value</name>` form.
    /// Empty when the current element is unsupported or none is open.
    std::string element_namespace;
    std::string element_name;
};

void store_field(std::string_view name, std::string_view raw, ScanState& state) {
    XmpReference& out = state.out;
    if (name == "GpsLatitude") {
        out.latitude_deg = parse_double(raw);
    } else if (name == "GpsLongitude" || name == "GpsLongtitude") {
        // GpsLongtitude is the spelling some DJI firmware writes.
        out.longitude_deg = parse_double(raw);
    } else if (name == "AbsoluteAltitude") {
        out.absolute_altitude_m = parse_double(raw);
    } else if (name == "RelativeAltitude") {
        out.relative_altitude_m = parse_double(raw);
    } else if (name == "GimbalYawDegree") {
        out.gimbal_yaw_deg = parse_double(raw);
    } else if (name == "GimbalPitchDegree") {
        out.gimbal_pitch_deg = parse_double(raw);
    } else if (name == "GimbalRollDegree") {
        out.gimbal_roll_deg = parse_double(raw);
    } else if (name == "FlightYawDegree") {
        out.flight_yaw_deg = parse_double(raw);
    } else if (name == "FlightPitchDegree") {
        out.flight_pitch_deg = parse_double(raw);
    } else if (name == "FlightRollDegree") {
        out.flight_roll_deg = parse_double(raw);
    } else if (name == "RtkFlag") {
        out.rtk_flag = parse_int(raw);
    } else if (name == "RtkStdLon") {
        out.rtk_std_lon_m = parse_double(raw);
    } else if (name == "RtkStdLat") {
        out.rtk_std_lat_m = parse_double(raw);
    } else if (name == "RtkStdHgt") {
        out.rtk_std_hgt_m = parse_double(raw);
    } else if (name == "DewarpData") {
        const std::string_view value = trim(raw);
        if (!value.empty()) {
            out.dewarp_data = std::string(value);
        }
    } else if (name == "GpsLatitudeRef" || name == "GPSLatitudeRef") {
        if (!raw.empty()) {
            state.lat_ref = raw[0];
        }
    } else if (name == "GpsLongitudeRef" || name == "GPSLongitudeRef") {
        if (!raw.empty()) {
            state.lon_ref = raw[0];
        }
    }
}

/// Fold the hemisphere refs into the values. A ref of S/W makes the value
/// negative; N/E makes it positive; without a ref the signed value stands.
void apply_hemispheres(ScanState& state) {
    if (state.out.latitude_deg.has_value() && state.lat_ref.has_value()) {
        const char ref = *state.lat_ref;
        const double magnitude = std::fabs(*state.out.latitude_deg);
        state.out.latitude_deg = (ref == 'S' || ref == 's') ? -magnitude : magnitude;
    }
    if (state.out.longitude_deg.has_value() && state.lon_ref.has_value()) {
        const char ref = *state.lon_ref;
        const double magnitude = std::fabs(*state.out.longitude_deg);
        state.out.longitude_deg = (ref == 'W' || ref == 'w') ? -magnitude : magnitude;
    }
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/// Consume one start tag beginning at the '<' at `start`, storing every
/// supported attribute and updating the current-element state. Returns the
/// index just past the tag, or packet.size() when it is malformed.
std::size_t scan_start_tag(std::string_view packet, std::size_t start,
                           const std::map<std::string, std::string>& namespaces,
                           ScanState& state) {
    std::size_t cursor = start + 1;
    const std::size_t name_begin = cursor;
    while (cursor < packet.size() && !is_space(packet[cursor]) &&
           packet[cursor] != '>' && packet[cursor] != '/') {
        ++cursor;
    }
    const std::string_view qname = packet.substr(name_begin, cursor - name_begin);
    const std::string_view element_ns = namespace_of(qname, namespaces);
    const std::string_view element_name = local_name(qname);

    bool self_closing = false;
    while (cursor < packet.size()) {
        while (cursor < packet.size() && is_space(packet[cursor])) {
            ++cursor;
        }
        if (cursor >= packet.size()) {
            return packet.size();
        }
        if (packet[cursor] == '>') {
            ++cursor;
            break;
        }
        if (packet[cursor] == '/' && cursor + 1 < packet.size() &&
            packet[cursor + 1] == '>') {
            self_closing = true;
            cursor += 2;
            break;
        }
        const std::size_t attr_begin = cursor;
        while (cursor < packet.size() && !is_space(packet[cursor]) &&
               packet[cursor] != '=' && packet[cursor] != '>' && packet[cursor] != '/') {
            ++cursor;
        }
        const std::string_view attr_name = packet.substr(attr_begin, cursor - attr_begin);
        while (cursor < packet.size() && is_space(packet[cursor])) {
            ++cursor;
        }
        if (cursor >= packet.size() || packet[cursor] != '=') {
            return packet.size();  // malformed attribute list
        }
        ++cursor;
        while (cursor < packet.size() && is_space(packet[cursor])) {
            ++cursor;
        }
        if (cursor >= packet.size() || (packet[cursor] != '"' && packet[cursor] != '\'')) {
            return packet.size();  // unquoted values are not conforming XML
        }
        const char quote = packet[cursor++];
        const std::size_t value_begin = cursor;
        while (cursor < packet.size() && packet[cursor] != quote) {
            ++cursor;
        }
        if (cursor >= packet.size()) {
            return packet.size();  // truncated value
        }
        const std::string_view value = packet.substr(value_begin, cursor - value_begin);
        ++cursor;  // closing quote
        if (attr_name.substr(0, 6) == "xmlns:") {
            continue;
        }
        if (namespace_of(attr_name, namespaces) == kDjiNamespace) {
            store_field(local_name(attr_name), value, state);
        }
    }

    if (self_closing) {
        state.element_namespace.clear();
        state.element_name.clear();
    } else if (element_ns == kDjiNamespace) {
        state.element_namespace = std::string(element_ns);
        state.element_name = std::string(element_name);
    } else {
        state.element_namespace.clear();
        state.element_name.clear();
    }
    return cursor;
}

}  // namespace

XmpReference parse_xmp_reference(std::string_view packet) {
    ScanState state;
    if (packet.empty()) {
        return state.out;
    }
    const std::map<std::string, std::string> namespaces = collect_namespaces(packet);
    std::size_t pos = 0;
    while (pos < packet.size()) {
        if (packet[pos] != '<') {
            std::size_t next = packet.find('<', pos);
            if (next == std::string_view::npos) {
                next = packet.size();
            }
            if (std::string_view(state.element_namespace) == kDjiNamespace &&
                !state.element_name.empty()) {
                store_field(state.element_name, packet.substr(pos, next - pos), state);
            }
            pos = next;
            continue;
        }
        if (packet.compare(pos, 4, "<!--") == 0) {
            const std::size_t end = packet.find("-->", pos + 4);
            pos = end == std::string_view::npos ? packet.size() : end + 3;
            continue;
        }
        if (packet.compare(pos, 9, "<![CDATA[") == 0) {
            const std::size_t end = packet.find("]]>", pos + 9);
            const std::size_t text_end = end == std::string_view::npos ? packet.size() : end;
            if (std::string_view(state.element_namespace) == kDjiNamespace &&
                !state.element_name.empty()) {
                store_field(state.element_name, packet.substr(pos + 9, text_end - (pos + 9)),
                            state);
            }
            pos = end == std::string_view::npos ? packet.size() : end + 3;
            continue;
        }
        if (packet.compare(pos, 2, "<?") == 0) {
            const std::size_t end = packet.find("?>", pos + 2);
            pos = end == std::string_view::npos ? packet.size() : end + 2;
            continue;
        }
        if (packet.compare(pos, 2, "<!") == 0) {
            const std::size_t end = packet.find('>', pos + 2);
            pos = end == std::string_view::npos ? packet.size() : end + 1;
            continue;
        }
        if (packet.compare(pos, 2, "</") == 0) {
            const std::size_t end = packet.find('>', pos + 2);
            pos = end == std::string_view::npos ? packet.size() : end + 1;
            state.element_namespace.clear();
            state.element_name.clear();
            continue;
        }
        pos = scan_start_tag(packet, pos, namespaces, state);
    }
    apply_hemispheres(state);
    return state.out;
}

#if defined(ALPHASHAPE_USE_PROJ)
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

Eigen::Matrix3d rotation_about(int axis, double degrees) {
    Eigen::Vector3d unit = Eigen::Vector3d::Zero();
    unit[axis] = 1.0;
    return Eigen::AngleAxisd(degrees * kDeg, unit).toRotationMatrix();
}

/// The fixed body-to-camera rotation of a nadir mount, identical to
/// `reference::body_to_camera_nadir()`. Repeated as a 3x3 here so this
/// translation unit does not depend on the CRS conversion module; the geometry
/// tests cross-check the two definitions.
Eigen::Matrix3d nadir_mount() {
    Eigen::Matrix3d mount;
    mount << 0.0, -1.0, 0.0,   //
        -1.0, 0.0, 0.0,        //
        0.0, 0.0, -1.0;
    return mount;
}

/// World-from-body for a stored ZYX triple, per the `convert_reference`
/// contract: `R_wb = Rz(90 deg) * to_matrix({yaw, -pitch, roll})`.
Eigen::Matrix3d world_from_stored(double yaw_deg, double pitch_deg, double roll_deg) {
    const crs::EulerAngles internal{yaw_deg, -pitch_deg, roll_deg, crs::RotationOrder::ZYX};
    return rotation_about(2, 90.0) * crs::to_matrix(internal);
}

/// The inverse of `world_from_stored`: strip the fixed Rz(90) and put the
/// pitch sign back so the triple is in the stored convention again.
XmpOrientation stored_from_world(const Eigen::Matrix3d& world_from_body) {
    const Eigen::Matrix3d internal = rotation_about(2, -90.0) * world_from_body;
    const crs::EulerAngles angles = crs::from_matrix(internal, crs::RotationOrder::ZYX);
    return XmpOrientation{angles.yaw, -angles.pitch, angles.roll};
}

/// World(ENU)-from-camera for a DJI gimbal triple.
///
/// The axes are named explicitly: at heading 0, pitch 0, roll 0 the optical
/// axis points north and image-up points up, so `R0 = [east | down | north]`.
/// Heading is a clockwise rotation about the down/up axis, pitch tilts about
/// the camera's right axis (0 at the horizon, -90 at nadir), and roll spins
/// about the optical axis. Hence
///
///     R_wc = Rz(-heading) * R0 * Rx(pitch) * Rz(roll)
///
/// where each right-hand factor acts on the camera's own axes.
Eigen::Matrix3d world_from_gimbal(double heading_deg, double pitch_deg, double roll_deg) {
    Eigen::Matrix3d base;
    base << 1.0, 0.0, 0.0,   // image right = east
        0.0, 0.0, 1.0,       // image down = down
        0.0, -1.0, 0.0;      // optical axis = north
    return rotation_about(2, -heading_deg) * base * rotation_about(0, pitch_deg) *
           rotation_about(2, roll_deg);
}

}  // namespace
#endif

bool XmpReference::has_rtk_fix() const {
    if (!rtk_flag.has_value()) {
        return false;
    }
    // Only 34 (float) and 50 (fixed) are trusted RTK solutions. Codes 0/16
    // (single-point GNSS) and 1/2 (an unrelated enum some tools emit) are not.
    switch (*rtk_flag) {
        case 34:
        case 50:
            return true;
        default:
            return false;
    }
}

std::optional<std::array<double, 3>> XmpReference::rtk_accuracy_m() const {
    if (!has_rtk_fix() || !rtk_std_lon_m.has_value() || !rtk_std_lat_m.has_value() ||
        !rtk_std_hgt_m.has_value()) {
        return std::nullopt;
    }
    if (!(*rtk_std_lon_m > 0.0) || !(*rtk_std_lat_m > 0.0) || !(*rtk_std_hgt_m > 0.0)) {
        return std::nullopt;
    }
    return std::array<double, 3>{*rtk_std_lon_m, *rtk_std_lat_m, *rtk_std_hgt_m};
}

std::optional<XmpOrientation> XmpReference::orientation_zyx() const {
#if defined(ALPHASHAPE_USE_PROJ)
    if (gimbal_yaw_deg.has_value() && gimbal_pitch_deg.has_value()) {
        // The gimbal angles describe the camera, so build the world-from-camera
        // rotation and undo the fixed nadir mount: R_wb = R_wc * B^T (B is
        // symmetric, so this equals R_wc * B). The result is then decomposed
        // into the stored body-attitude convention.
        const Eigen::Matrix3d world_from_camera = world_from_gimbal(
            *gimbal_yaw_deg, *gimbal_pitch_deg, gimbal_roll_deg.value_or(0.0));
        return stored_from_world(world_from_camera * nadir_mount().transpose());
    }
    if (flight_yaw_deg.has_value() && flight_pitch_deg.has_value()) {
        // Flight angles already describe the body; only the compass heading
        // sign changes. Routing through the same matrix path normalises the
        // triple (and would catch a fully degenerate decomposition).
        return stored_from_world(world_from_stored(
            -*flight_yaw_deg, *flight_pitch_deg, flight_roll_deg.value_or(0.0)));
    }
    return std::nullopt;
#else
    // Without the CRS engine there is no matrix decomposition available. This
    // component-wise mapping is only meaningful in the default ZYX order, so
    // the AddPhotos task skips XMP orientation in such a build.
    if (gimbal_yaw_deg.has_value() && gimbal_pitch_deg.has_value()) {
        return XmpOrientation{-*gimbal_yaw_deg, *gimbal_pitch_deg + 90.0,
                              gimbal_roll_deg.value_or(0.0)};
    }
    if (flight_yaw_deg.has_value() && flight_pitch_deg.has_value()) {
        return XmpOrientation{-*flight_yaw_deg, *flight_pitch_deg,
                              flight_roll_deg.value_or(0.0)};
    }
    return std::nullopt;
#endif
}

}  // namespace alphashape
