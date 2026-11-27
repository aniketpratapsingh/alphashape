// Tests for WP-13.3: XMP reference extraction and its use by AddPhotos.
//
// Every packet is synthesised in-process (a small string builder), and the JPEG
// and TIFF containers are assembled byte-by-byte, so the tests do not depend on
// camera firmware. The AddPhotos integration cases inject an XMP APP1 segment
// into a committed camera JPEG, which keeps a real image header for
// read_image_info.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "common/algorithm/photos/add_photos_task.hpp"
#include "common/algorithm/reference/xmp_reference.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "image/io/exif.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "common/algorithm/reference/convert_reference.hpp"
#include "model/crs/euler.hpp"
#endif

namespace alphashape {
namespace {

namespace fs = std::filesystem;

constexpr const char* kDjiNamespace = "http://ns.dji.com/drone-dji/1.0/";

#ifdef ALPHASHAPE_DATA_DIR
const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

std::string attribute_packet(const std::string& attributes) {
    std::string packet = "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>";
    packet += "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF "
              "xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">";
    packet += "<rdf:Description rdf:about=\"\" xmlns:drone-dji=\"";
    packet += kDjiNamespace;
    packet += "\" ";
    packet += attributes;
    packet += "/></rdf:RDF></x:xmpmeta>";
    return packet;
}

std::string custom_prefix_packet(const std::string& prefix, const std::string& attributes) {
    std::string packet = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" ";
    packet += "xmlns:";
    packet += prefix;
    packet += "=\"";
    packet += kDjiNamespace;
    packet += "\"><rdf:Description ";
    packet += attributes;
    packet += "/></rdf:RDF>";
    return packet;
}

std::string element_packet(const std::string& prefix) {
    std::string packet = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" ";
    packet += "xmlns:";
    packet += prefix;
    packet += "=\"";
    packet += kDjiNamespace;
    packet += "\"><rdf:Description><";
    packet += prefix;
    packet += ":GpsLatitude>45.1</";
    packet += prefix;
    packet += ":GpsLatitude><";
    packet += prefix;
    packet += ":GpsLongitude>-9.25</";
    packet += prefix;
    packet += ":GpsLongitude><";
    packet += prefix;
    packet += ":AbsoluteAltitude>312.5</";
    packet += prefix;
    packet += ":AbsoluteAltitude></rdf:Description></rdf:RDF>";
    return packet;
}

#if defined(ALPHASHAPE_USE_PROJ)
constexpr double kPiTest = 3.14159265358979323846;
constexpr double kRadTest = kPiTest / 180.0;

Eigen::Matrix3d axis_rotation(int axis, double degrees) {
    Eigen::Vector3d unit = Eigen::Vector3d::Zero();
    unit[axis] = 1.0;
    return Eigen::AngleAxisd(degrees * kRadTest, unit).toRotationMatrix();
}

/// The DJI camera mapping, written out independently of the engine: heading
/// clockwise about up, pitch about image-right, roll about the optical axis,
/// with `base = [east | down | north]` at heading/pitch/roll all zero.
Eigen::Matrix3d dji_world_from_camera(double heading, double pitch, double roll) {
    Eigen::Matrix3d base;
    base << 1.0, 0.0, 0.0,   //
        0.0, 0.0, 1.0,       //
        0.0, -1.0, 0.0;
    return axis_rotation(2, -heading) * base * axis_rotation(0, pitch) *
           axis_rotation(2, roll);
}

/// World-from-camera reconstructed from a STORED yaw/pitch/roll triple in the
/// given order, using only the `convert_reference` contract and the fixed nadir
/// mount. This is what a consumer of `orientation_deg` must see.
Eigen::Matrix3d camera_from_stored(const Eigen::Vector3d& stored,
                                   alphashape::crs::RotationOrder order) {
    const alphashape::crs::EulerAngles internal{stored.x(), -stored.y(), stored.z(), order};
    Eigen::Matrix3d rzb;
    rzb << 0.0, -1.0, 0.0,   //
        1.0, 0.0, 0.0,       //
        0.0, 0.0, 1.0;
    return rzb * alphashape::crs::to_matrix(internal) *
           alphashape::reference::body_to_camera_nadir().transpose();
}

/// Inverse of `dji_world_from_camera` for a non-nadir pitch: returns heading,
/// pitch and roll in degrees. The heading comes from requiring the residual
/// `Rx(pitch) Rz(roll)` to have a zero third component on its first row.
std::array<double, 3> recover_dji(const Eigen::Matrix3d& world_from_camera) {
    const double heading = std::atan2(world_from_camera(0, 2), world_from_camera(1, 2));
    Eigen::Matrix3d base;
    base << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
    // atan2 answers in radians; axis_rotation takes degrees.
    const Eigen::Matrix3d residual =
        base.transpose() * axis_rotation(2, heading / kRadTest) * world_from_camera;
    const double pitch =
        std::atan2(-residual(1, 2), std::hypot(residual(1, 0), residual(1, 1)));
    const double roll = std::atan2(-residual(0, 1), residual(0, 0));
    return {heading / kRadTest, pitch / kRadTest, roll / kRadTest};
}

void expect_direction(const Eigen::Vector3d& actual, double x, double y, double z) {
    EXPECT_NEAR(actual.x(), x, 1e-9);
    EXPECT_NEAR(actual.y(), y, 1e-9);
    EXPECT_NEAR(actual.z(), z, 1e-9);
}
#endif

/// A temp file unique to this test process. gtest_discover_tests runs every
/// case in its own process, potentially concurrently, so a per-test name is
/// what keeps two cases from writing the same path.
std::string temp_path(const std::string& extension) {
    const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
    std::string name = info != nullptr ? info->name() : "case";
    for (char& c : name) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
            c = '_';
        }
    }
    return (fs::temp_directory_path() / ("alphashape_xmp_" + name + extension)).string();
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return {};
    }
    const std::streamoff size = in.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

void append_app1(std::vector<std::uint8_t>& bytes, const std::string& packet) {
    const std::string identifier = std::string("http://ns.adobe.com/xap/1.0/") + '\0';
    const std::size_t payload_size = identifier.size() + packet.size();
    ASSERT_LT(payload_size + 2, 65536U);
    const auto length = static_cast<std::uint16_t>(payload_size + 2);
    bytes.push_back(0xFF);
    bytes.push_back(0xE1);
    bytes.push_back(static_cast<std::uint8_t>((length >> 8) & 0xFF));
    bytes.push_back(static_cast<std::uint8_t>(length & 0xFF));
    bytes.insert(bytes.end(), identifier.begin(), identifier.end());
    bytes.insert(bytes.end(), packet.begin(), packet.end());
}

/// Insert an XMP APP1 segment directly after the JPEG SOI.
std::vector<std::uint8_t> jpeg_with_packet(const std::vector<std::uint8_t>& jpeg,
                                           const std::string& packet) {
    std::vector<std::uint8_t> bytes;
    if (jpeg.size() >= 2) {
        bytes.insert(bytes.end(), jpeg.begin(), jpeg.begin() + 2);
    }
    append_app1(bytes, packet);
    if (jpeg.size() > 2) {
        bytes.insert(bytes.end(), jpeg.begin() + 2, jpeg.end());
    }
    return bytes;
}

/// A minimal little-endian TIFF with IFD0 holding one XMLPacket (700) entry.
std::vector<std::uint8_t> tiff_with_packet(const std::string& packet) {
    std::vector<std::uint8_t> bytes(8, 0);
    bytes[0] = 'I';
    bytes[1] = 'I';
    bytes[2] = 42;
    bytes[3] = 0;
    bytes[4] = 8;  // IFD0 offset
    const auto put16 = [&](std::size_t at, std::uint16_t value) {
        bytes[at] = static_cast<std::uint8_t>(value & 0xFF);
        bytes[at + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    };
    const auto put32 = [&](std::size_t at, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            bytes[at + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
        }
    };
    bytes.resize(26, 0);
    put16(8, 1);         // one entry
    put16(10, 700);      // XMLPacket
    put16(12, 1);        // BYTE
    put32(14, static_cast<std::uint32_t>(packet.size()));
    put32(18, 26);       // value lives immediately after the IFD
    put32(22, 0);        // no next IFD
    bytes.insert(bytes.end(), packet.begin(), packet.end());
    return bytes;
}

/// The same one-entry IFD0, but the XMLPacket's count and offset both point far
/// past the end of the 26-byte block: a hostile file must yield no packet and
/// no crash.
std::vector<std::uint8_t> tiff_with_dangling_packet() {
    std::vector<std::uint8_t> bytes(26, 0);
    bytes[0] = 'I';
    bytes[1] = 'I';
    bytes[2] = 42;
    bytes[4] = 8;
    const auto put16 = [&](std::size_t at, std::uint16_t value) {
        bytes[at] = static_cast<std::uint8_t>(value & 0xFF);
        bytes[at + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    };
    const auto put32 = [&](std::size_t at, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            bytes[at + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
        }
    };
    put16(8, 1);       // one entry
    put16(10, 700);    // XMLPacket
    put16(12, 1);      // BYTE
    put32(14, 64);     // claims 64 bytes
    put32(18, 0xFFFF); // ...at an offset past the end of the block
    put32(22, 0);      // no next IFD
    return bytes;
}

Document document_with_chunk() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

void run_add_photos(Document& document, const std::string& path, bool xmp_accuracy,
                    bool xmp_orientation) {
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{path}});
    params.set("load_xmp_accuracy", ParamValue{xmp_accuracy});
    params.set("load_xmp_orientation", ParamValue{xmp_orientation});
    run_task(builtin_registry(), document, "AddPhotos", params);
}

/// Run AddPhotos with the XMP lens-calibration option set, everything else at
/// its default.
void run_add_photos_with_calibration(Document& document, const std::string& path,
                                     bool load_xmp_calibration) {
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{path}});
    params.set("load_xmp_calibration", ParamValue{load_xmp_calibration});
    run_task(builtin_registry(), document, "AddPhotos", params);
}

TEST(Xmp, ParsesDjiLatLonAlt) {
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:GpsLatitude=\"+45.1\" drone-dji:GpsLongitude=\"-9.25\" "
        "drone-dji:AbsoluteAltitude=\"+312.5\" drone-dji:RelativeAltitude=\"120.25\""));
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    ASSERT_TRUE(xmp.longitude_deg.has_value());
    ASSERT_TRUE(xmp.absolute_altitude_m.has_value());
    ASSERT_TRUE(xmp.relative_altitude_m.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 45.1);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, -9.25);
    EXPECT_DOUBLE_EQ(*xmp.absolute_altitude_m, 312.5);
    EXPECT_DOUBLE_EQ(*xmp.relative_altitude_m, 120.25);
    EXPECT_TRUE(xmp.has_location());
}

TEST(Xmp, ParsesElementForm) {
    const XmpReference xmp = parse_xmp_reference(element_packet("drone-dji"));
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    ASSERT_TRUE(xmp.longitude_deg.has_value());
    ASSERT_TRUE(xmp.absolute_altitude_m.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 45.1);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, -9.25);
    EXPECT_DOUBLE_EQ(*xmp.absolute_altitude_m, 312.5);
}

TEST(Xmp, CustomPrefixResolvedByNamespace) {
    // The prefix is "dji", not "drone-dji", and is declared on an ancestor.
    // It also exercises the GpsLongtitude misspelling.
    const XmpReference xmp = parse_xmp_reference(
        custom_prefix_packet("dji", "dji:GpsLatitude=\"12.5\" dji:GpsLongtitude=\"34.25\""));
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    ASSERT_TRUE(xmp.longitude_deg.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 12.5);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, 34.25);
}

TEST(Xmp, FieldsSplitAcrossDescriptionBlocks) {
    const std::string ns = std::string(kDjiNamespace);
    const std::string packet =
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
        "<rdf:Description xmlns:drone-dji=\"" + ns +
        "\" drone-dji:GpsLatitude=\"45.1\" drone-dji:GimbalYawDegree=\"12\"/>"
        "<rdf:Description xmlns:drone-dji=\"" + ns +
        "\" drone-dji:GpsLongitude=\"-9.25\" drone-dji:GimbalPitchDegree=\"-80\"/>"
        "</rdf:RDF>";
    const XmpReference xmp = parse_xmp_reference(packet);
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    ASSERT_TRUE(xmp.longitude_deg.has_value());
    ASSERT_TRUE(xmp.gimbal_yaw_deg.has_value());
    ASSERT_TRUE(xmp.gimbal_pitch_deg.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 45.1);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, -9.25);
    EXPECT_DOUBLE_EQ(*xmp.gimbal_yaw_deg, 12.0);
    EXPECT_DOUBLE_EQ(*xmp.gimbal_pitch_deg, -80.0);
}

TEST(Xmp, HemisphereRefsApplyToCoordinates) {
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:GpsLatitude=\"45.1\" drone-dji:GpsLatitudeRef=\"S\" "
        "drone-dji:GpsLongitude=\"9.25\" drone-dji:GpsLongitudeRef=\"W\""));
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    ASSERT_TRUE(xmp.longitude_deg.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, -45.1);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, -9.25);
}

#if defined(ALPHASHAPE_USE_PROJ)
namespace {
/// Parse one gimbal packet and turn its stored ZYX triple back into the
/// world-from-camera rotation a consumer of `orientation_deg` would build.
Eigen::Matrix3d stored_camera(const std::string& attributes) {
    const std::optional<XmpOrientation> angles =
        parse_xmp_reference(attribute_packet(attributes)).orientation_zyx();
    EXPECT_TRUE(angles.has_value());
    if (!angles.has_value()) {
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Vector3d stored(angles->yaw_deg, angles->pitch_deg, angles->roll_deg);
    return camera_from_stored(stored, alphashape::crs::RotationOrder::ZYX);
}
}  // namespace

TEST(Xmp, GimbalNadirHeadingNorthLooksDownWithNorthUp) {
    const Eigen::Matrix3d r_wc = stored_camera(
        "drone-dji:GimbalYawDegree=\"0\" drone-dji:GimbalPitchDegree=\"-90\" "
        "drone-dji:GimbalRollDegree=\"0\"");
    expect_direction(r_wc.col(2), 0.0, 0.0, -1.0);   // boresight down
    expect_direction(-r_wc.col(1), 0.0, 1.0, 0.0);   // image-up north
}

TEST(Xmp, GimbalNadirHeadingEastLooksDownWithEastUp) {
    const Eigen::Matrix3d r_wc = stored_camera(
        "drone-dji:GimbalYawDegree=\"90\" drone-dji:GimbalPitchDegree=\"-90\" "
        "drone-dji:GimbalRollDegree=\"0\"");
    expect_direction(r_wc.col(2), 0.0, 0.0, -1.0);
    expect_direction(-r_wc.col(1), 1.0, 0.0, 0.0);
}

TEST(Xmp, GimbalPitchMinusSixtyTiltsBoresightDownAndNorth) {
    const Eigen::Matrix3d r_wc = stored_camera(
        "drone-dji:GimbalYawDegree=\"0\" drone-dji:GimbalPitchDegree=\"-60\" "
        "drone-dji:GimbalRollDegree=\"0\"");
    expect_direction(r_wc.col(2), 0.0, 0.5, -0.8660254037844386);
    // A 60 degree tilt is not a rotation about up: the boresight moved, and it
    // must not be reconstructed as a pure yaw/pitch component map.
    EXPECT_GT(std::fabs(r_wc(1, 2)), 0.4);
}

TEST(Xmp, GimbalRollSpinsTheImageAboutTheBoresight) {
    const Eigen::Matrix3d r_wc = stored_camera(
        "drone-dji:GimbalYawDegree=\"0\" drone-dji:GimbalPitchDegree=\"-90\" "
        "drone-dji:GimbalRollDegree=\"90\"");
    // Roll is about the optical axis, so it cannot move the boresight.
    expect_direction(r_wc.col(2), 0.0, 0.0, -1.0);
    // Right-hand rule about the optical axis (pointing north, away from the
    // camera) turns image-up toward image-right, which is east here.
    expect_direction(-r_wc.col(1), 1.0, 0.0, 0.0);
}

TEST(Xmp, GimbalAnglesRoundTripThroughTheStoredTriple) {
    const double heading = 37.0;
    const double pitch = -71.0;
    const double roll = 4.0;
    const std::optional<XmpOrientation> angles = parse_xmp_reference(attribute_packet(
        "drone-dji:GimbalYawDegree=\"37\" drone-dji:GimbalPitchDegree=\"-71\" "
        "drone-dji:GimbalRollDegree=\"4\""))
                                                    .orientation_zyx();
    ASSERT_TRUE(angles.has_value());
    const Eigen::Matrix3d expected = dji_world_from_camera(heading, pitch, roll);
    const Eigen::Matrix3d stored =
        camera_from_stored(Eigen::Vector3d(angles->yaw_deg, angles->pitch_deg, angles->roll_deg),
                           alphashape::crs::RotationOrder::ZYX);
    EXPECT_TRUE(stored.isApprox(expected, 1e-9));
    const std::array<double, 3> recovered = recover_dji(stored);
    EXPECT_NEAR(recovered[0], heading, 1e-9);
    EXPECT_NEAR(recovered[1], pitch, 1e-9);
    EXPECT_NEAR(recovered[2], roll, 1e-9);
}

TEST(Xmp, FlightAnglesAreBodyAttitude) {
    // Without gimbal angles the airframe attitude is used unchanged: it is
    // already a body attitude, so only the compass heading sign flips.
    const std::optional<XmpOrientation> angles = parse_xmp_reference(attribute_packet(
        "drone-dji:FlightYawDegree=\"30\" drone-dji:FlightPitchDegree=\"5\" "
        "drone-dji:FlightRollDegree=\"1\""))
                                                    .orientation_zyx();
    ASSERT_TRUE(angles.has_value());
    EXPECT_NEAR(angles->yaw_deg, -30.0, 1e-9);
    EXPECT_NEAR(angles->pitch_deg, 5.0, 1e-9);
    EXPECT_NEAR(angles->roll_deg, 1.0, 1e-9);
}
#endif

TEST(Xmp, ExtractFromJpegApp1) {
    const std::string packet = attribute_packet("drone-dji:GpsLatitude=\"45.1\"");
    std::vector<std::uint8_t> bytes = {0xFF, 0xD8};
    append_app1(bytes, packet);
    const std::string path = temp_path(".jpg");
    write_bytes(path, bytes);

    const ExifData exif = read_exif(path);
    ASSERT_TRUE(exif.xmp_packet.has_value());
    EXPECT_EQ(*exif.xmp_packet, packet);
    const XmpReference xmp = parse_xmp_reference(*exif.xmp_packet);
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 45.1);
    remove_file(path);
}

TEST(Xmp, ExtractFromTiffTag700) {
    const std::string packet = attribute_packet("drone-dji:GpsLatitude=\"12.5\"");
    const std::string path = temp_path(".tiff");
    write_bytes(path, tiff_with_packet(packet));

    const ExifData exif = read_exif(path);
    ASSERT_TRUE(exif.xmp_packet.has_value());
    EXPECT_EQ(*exif.xmp_packet, packet);
    remove_file(path);
}

TEST(Xmp, XmpApp1AfterExifApp1IsRead) {
    // A TIFF block with one Orientation (0x0112) SHORT entry equal to 6, wrapped
    // in an Exif APP1 that precedes the XMP APP1.
    std::vector<std::uint8_t> tiff(26, 0);
    tiff[0] = 'I';
    tiff[1] = 'I';
    tiff[2] = 42;
    tiff[4] = 8;
    tiff[8] = 1;                   // one entry
    tiff[10] = 0x12;               // 0x0112, little-endian
    tiff[11] = 0x01;
    tiff[12] = 3;                  // SHORT
    tiff[14] = 1;                  // count
    tiff[18] = 6;                  // Orientation value lives in the entry
    std::string exif_payload("Exif\0\0", 6);
    exif_payload.append(reinterpret_cast<const char*>(tiff.data()), tiff.size());

    const std::string packet = attribute_packet("drone-dji:GpsLatitude=\"12.5\"");
    std::vector<std::uint8_t> bytes = {0xFF, 0xD8};
    const auto append_raw_app1 = [&bytes](const std::string& payload) {
        const auto length = static_cast<std::uint16_t>(payload.size() + 2);
        bytes.push_back(0xFF);
        bytes.push_back(0xE1);
        bytes.push_back(static_cast<std::uint8_t>((length >> 8) & 0xFF));
        bytes.push_back(static_cast<std::uint8_t>(length & 0xFF));
        bytes.insert(bytes.end(), payload.begin(), payload.end());
    };
    append_raw_app1(exif_payload);
    append_app1(bytes, packet);
    bytes.push_back(0xFF);
    bytes.push_back(0xD9);

    const std::string path = temp_path(".jpg");
    write_bytes(path, bytes);
    const ExifData exif = read_exif(path);
    // The EXIF block was read, and the later XMP APP1 was not lost with it.
    ASSERT_TRUE(exif.orientation.has_value());
    EXPECT_EQ(*exif.orientation, 6);
    ASSERT_TRUE(exif.xmp_packet.has_value());
    const XmpReference xmp = parse_xmp_reference(*exif.xmp_packet);
    ASSERT_TRUE(xmp.latitude_deg.has_value());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 12.5);
    remove_file(path);
}

TEST(Xmp, TruncatedApp1YieldsNoPacket) {
    const std::string packet = attribute_packet("drone-dji:GpsLatitude=\"45.1\"");
    std::vector<std::uint8_t> bytes = {0xFF, 0xD8};
    append_app1(bytes, packet);
    bytes.resize(bytes.size() - packet.size() / 2);  // segment now runs past EOF

    const std::string path = temp_path(".jpg");
    write_bytes(path, bytes);
    const ExifData exif = read_exif(path);
    EXPECT_FALSE(exif.xmp_packet.has_value());
    remove_file(path);
}

TEST(Xmp, TiffPacketOffsetPastEndIsIgnored) {
    const std::string path = temp_path(".tiff");
    write_bytes(path, tiff_with_dangling_packet());
    const ExifData exif = read_exif(path);
    EXPECT_FALSE(exif.xmp_packet.has_value());
    remove_file(path);
}

TEST(Xmp, MissingPacketIsNoOp) {
    const XmpReference xmp = parse_xmp_reference("");
    EXPECT_FALSE(xmp.has_location());
    EXPECT_FALSE(xmp.latitude_deg.has_value());
    EXPECT_FALSE(xmp.longitude_deg.has_value());
    EXPECT_FALSE(xmp.orientation_zyx().has_value());
    EXPECT_FALSE(xmp.has_rtk_fix());
    EXPECT_FALSE(xmp.rtk_accuracy_m().has_value());
    EXPECT_FALSE(xmp.dewarp.has_value());
    EXPECT_FALSE(xmp.calibrated_focal_length.has_value());
    EXPECT_FALSE(xmp.dewarp_flag.has_value());
}

TEST(Xmp, ParsesDewarpDataWithLeadingSpace) {
    // The exact shape a Phantom 4 RTK writes: a leading space, a date, then the
    // nine OpenCV coefficients.
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:DewarpData=\" 2018-09-04;3670.280000000000,3664.660000000000,"
        "2.680000000000,1.640000000000,-0.268152000000,0.113388000000,"
        "0.000041654400,0.000017653600,-0.033956100000\""));
    ASSERT_TRUE(xmp.dewarp.has_value());
    const XmpDewarpCalibration& d = *xmp.dewarp;
    EXPECT_DOUBLE_EQ(d.fx, 3670.28);
    EXPECT_DOUBLE_EQ(d.fy, 3664.66);
    EXPECT_DOUBLE_EQ(d.cx, 2.68);
    EXPECT_DOUBLE_EQ(d.cy, 1.64);
    EXPECT_DOUBLE_EQ(d.k1, -0.268152);
    EXPECT_DOUBLE_EQ(d.k2, 0.113388);
    EXPECT_DOUBLE_EQ(d.p1, 0.0000416544);
    EXPECT_DOUBLE_EQ(d.p2, 0.0000176536);
    EXPECT_DOUBLE_EQ(d.k3, -0.0339561);
    // The raw string is kept too, trimmed.
    ASSERT_TRUE(xmp.dewarp_data.has_value());
    EXPECT_EQ(xmp.dewarp_data->front(), '2');
}

TEST(Xmp, ParsesDewarpDataWithoutDate) {
    // Some firmware omits the date part; the coefficient list is the whole value.
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:DewarpData=\"1000,1000,0,0,0.1,0.2,0.3,0.4,0.5\""));
    ASSERT_TRUE(xmp.dewarp.has_value());
    EXPECT_DOUBLE_EQ(xmp.dewarp->fx, 1000.0);
    EXPECT_DOUBLE_EQ(xmp.dewarp->k1, 0.1);
    EXPECT_DOUBLE_EQ(xmp.dewarp->p1, 0.3);
    EXPECT_DOUBLE_EQ(xmp.dewarp->p2, 0.4);
    EXPECT_DOUBLE_EQ(xmp.dewarp->k3, 0.5);
}

TEST(Xmp, MalformedDewarpDataIsIgnored) {
    // Fewer than nine numbers, a non-numeric token, and an empty list all yield
    // no typed calibration (the raw string may still be recorded).
    for (const std::string& value :
         {std::string("2018-09-04;1,2,3,4,5,6,7,8"),
          std::string("2018-09-04;1,2,3,4,5,6,7,8,notanumber"),
          std::string("2018-09-04;")}) {
        const XmpReference xmp = parse_xmp_reference(
            attribute_packet("drone-dji:DewarpData=\"" + value + "\""));
        EXPECT_FALSE(xmp.dewarp.has_value()) << value;
    }
}

TEST(Xmp, ParsesCalibratedFocalAndOpticalCentreAndFlag) {
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:CalibratedFocalLength=\"3666.666504\" "
        "drone-dji:CalibratedOpticalCenterX=\"2736.000000\" "
        "drone-dji:CalibratedOpticalCenterY=\"1824.000000\" "
        "drone-dji:DewarpFlag=\"0\""));
    ASSERT_TRUE(xmp.calibrated_focal_length.has_value());
    EXPECT_DOUBLE_EQ(*xmp.calibrated_focal_length, 3666.666504);
    ASSERT_TRUE(xmp.calibrated_optical_center_x.has_value());
    EXPECT_DOUBLE_EQ(*xmp.calibrated_optical_center_x, 2736.0);
    ASSERT_TRUE(xmp.calibrated_optical_center_y.has_value());
    EXPECT_DOUBLE_EQ(*xmp.calibrated_optical_center_y, 1824.0);
    ASSERT_TRUE(xmp.dewarp_flag.has_value());
    EXPECT_EQ(*xmp.dewarp_flag, 0);
}

TEST(Xmp, MalformedPacketDoesNotThrow) {
    const std::string truncated_value =
        "<rdf:Description xmlns:drone-dji=\"" + std::string(kDjiNamespace) +
        "\" drone-dji:GpsLatitude=\"45.1";
    const std::string truncated_element = "<drone-dji:GpsLatitude>45.1";
    for (const std::string& packet :
         {std::string("<"), truncated_value, truncated_element, std::string("<rdf:")}) {
        XmpReference xmp;
        EXPECT_NO_THROW(xmp = parse_xmp_reference(packet));
        EXPECT_FALSE(xmp.has_location());
    }
}

TEST(Xmp, RtkAccuracyApplied) {
    const std::string packet = attribute_packet(
        "drone-dji:RtkFlag=\"50\" drone-dji:RtkStdLon=\"0.01\" "
        "drone-dji:RtkStdLat=\"0.02\" drone-dji:RtkStdHgt=\"0.03\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos(document, path, /*xmp_accuracy=*/true, /*xmp_orientation=*/false);
    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.cameras().size(), 1U);
    const CameraReference& reference = chunk.cameras().front().reference;
    EXPECT_DOUBLE_EQ(reference.accuracy.x(), 0.01);
    EXPECT_DOUBLE_EQ(reference.accuracy.y(), 0.02);
    EXPECT_DOUBLE_EQ(reference.accuracy.z(), 0.03);
    ASSERT_TRUE(reference.rtk_flag.has_value());
    EXPECT_EQ(*reference.rtk_flag, 50);
    remove_file(path);
}

TEST(Xmp, NonFixedRtkAccuracyIsInflated) {
    const std::string packet = attribute_packet(
        "drone-dji:RtkFlag=\"0\" drone-dji:RtkStdLon=\"0.01\" "
        "drone-dji:RtkStdLat=\"0.02\" drone-dji:RtkStdHgt=\"0.30\"");
    const XmpReference parsed = parse_xmp_reference(packet);
    EXPECT_FALSE(parsed.has_rtk_fix());
    const std::optional<std::array<double, 3>> accuracy = parsed.rtk_accuracy_m();
    ASSERT_TRUE(accuracy.has_value());
    // A single-point solution keeps its position but cannot be weighted at
    // RTK precision: every axis is raised to at least the 0.5 m floor.
    EXPECT_DOUBLE_EQ((*accuracy)[0], 0.5);
    EXPECT_DOUBLE_EQ((*accuracy)[1], 0.5);
    EXPECT_DOUBLE_EQ((*accuracy)[2], 0.5);

    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));
    Document document = document_with_chunk();
    run_add_photos(document, path, /*xmp_accuracy=*/true, /*xmp_orientation=*/false);
    const CameraReference& reference =
        document.active_chunk()->cameras().front().reference;
    EXPECT_DOUBLE_EQ(reference.accuracy.x(), 0.5);
    EXPECT_DOUBLE_EQ(reference.accuracy.y(), 0.5);
    EXPECT_DOUBLE_EQ(reference.accuracy.z(), 0.5);
    ASSERT_TRUE(reference.rtk_flag.has_value());
    EXPECT_EQ(*reference.rtk_flag, 0);
    remove_file(path);
}

TEST(Xmp, NonFixedRtkKeepsLargerReportedDeviation) {
    // The floor is a minimum, not a replacement: a single-point solution that
    // already reports a metre-level deviation keeps it.
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:RtkFlag=\"16\" drone-dji:RtkStdLon=\"0.8\" "
        "drone-dji:RtkStdLat=\"1.2\" drone-dji:RtkStdHgt=\"2.5\""));
    const std::optional<std::array<double, 3>> accuracy = xmp.rtk_accuracy_m();
    ASSERT_TRUE(accuracy.has_value());
    EXPECT_DOUBLE_EQ((*accuracy)[0], 0.8);
    EXPECT_DOUBLE_EQ((*accuracy)[1], 1.2);
    EXPECT_DOUBLE_EQ((*accuracy)[2], 2.5);
}

TEST(Xmp, AutelRtkPacketGivesPerAxisAccuracy) {
    // The packet shape this block writes: longitude standard deviation first,
    // then latitude, then height, all in metres, with RtkFlag 50 (fixed).
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:GpsLatitude=\"47.64368732\" "
        "drone-dji:GpsLongtitude=\"16.47592717\" "
        "drone-dji:AbsoluteAltitude=\"+513.00\" "
        "drone-dji:RtkFlag=\"50\" drone-dji:RtkStdLon=\"0.01370\" "
        "drone-dji:RtkStdLat=\"0.01439\" drone-dji:RtkStdHgt=\"0.02969\""));
    ASSERT_TRUE(xmp.has_location());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 47.64368732);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, 16.47592717);
    EXPECT_TRUE(xmp.has_rtk_fix());
    ASSERT_TRUE(xmp.rtk_flag.has_value());
    EXPECT_EQ(*xmp.rtk_flag, 50);
    const std::optional<std::array<double, 3>> accuracy = xmp.rtk_accuracy_m();
    ASSERT_TRUE(accuracy.has_value());
    EXPECT_DOUBLE_EQ((*accuracy)[0], 0.01370);  // east
    EXPECT_DOUBLE_EQ((*accuracy)[1], 0.01439);  // north
    EXPECT_DOUBLE_EQ((*accuracy)[2], 0.02969);  // up
}

TEST(Xmp, AcceptsTheVendorNamespaceUri) {
    // Aircraft firmware declares http://www.dji.com/..., not the
    // http://ns.dji.com/ documentation form. Accepting only the latter made a
    // real packet parse as empty; this pins both.
    const std::string packet =
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" "
        "xmlns:drone=\"http://www.dji.com/drone-dji/1.0/\">"
        "<rdf:Description drone:GpsLatitude=\"47.64368732\" "
        "drone:GpsLongtitude=\"16.47592717\" drone:AbsoluteAltitude=\"+513.00\" "
        "drone:RtkFlag=\"50\" drone:RtkStdLon=\"0.01370\" "
        "drone:RtkStdLat=\"0.01439\" drone:RtkStdHgt=\"0.02969\"/>"
        "</rdf:RDF>";
    const XmpReference xmp = parse_xmp_reference(packet);
    ASSERT_TRUE(xmp.has_location());
    EXPECT_DOUBLE_EQ(*xmp.latitude_deg, 47.64368732);
    EXPECT_DOUBLE_EQ(*xmp.longitude_deg, 16.47592717);
    EXPECT_TRUE(xmp.has_rtk_fix());
    const std::optional<std::array<double, 3>> accuracy = xmp.rtk_accuracy_m();
    ASSERT_TRUE(accuracy.has_value());
    EXPECT_DOUBLE_EQ((*accuracy)[0], 0.01370);
    EXPECT_DOUBLE_EQ((*accuracy)[1], 0.01439);
    EXPECT_DOUBLE_EQ((*accuracy)[2], 0.02969);
}

TEST(Xmp, VendorNamespaceAccuracyReachesAddPhotos) {
    // The full real-world path: depot firmware packet -> AddPhotos with
    // load_xmp_accuracy -> per-camera reference accuracy. The 10 m default
    // here would mean the packet was silently ignored.
    const std::string packet =
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" "
        "xmlns:drone=\"http://www.dji.com/drone-dji/1.0/\">"
        "<rdf:Description drone:GpsLatitude=\"47.64368732\" "
        "drone:GpsLongtitude=\"16.47592717\" drone:AbsoluteAltitude=\"+513.00\" "
        "drone:RtkFlag=\"50\" drone:RtkStdLon=\"0.01370\" "
        "drone:RtkStdLat=\"0.01439\" drone:RtkStdHgt=\"0.02969\"/>"
        "</rdf:RDF>";
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));
    Document document = document_with_chunk();
    run_add_photos(document, path, /*xmp_accuracy=*/true, /*xmp_orientation=*/false);
    const CameraReference& reference =
        document.active_chunk()->cameras().front().reference;
    EXPECT_DOUBLE_EQ(reference.accuracy.x(), 0.01370);
    EXPECT_DOUBLE_EQ(reference.accuracy.y(), 0.01439);
    EXPECT_DOUBLE_EQ(reference.accuracy.z(), 0.02969);
    ASSERT_TRUE(reference.rtk_flag.has_value());
    EXPECT_EQ(*reference.rtk_flag, 50);
    remove_file(path);
}

TEST(Xmp, RtkStdsWithoutStatusAreUsedAsIs) {
    // No status field: the packet cannot be classified, so the reported
    // deviations stand (they are still better than the consumer default).
    const XmpReference xmp = parse_xmp_reference(attribute_packet(
        "drone-dji:RtkStdLon=\"0.02\" drone-dji:RtkStdLat=\"0.03\" "
        "drone-dji:RtkStdHgt=\"0.04\""));
    EXPECT_FALSE(xmp.has_rtk_fix());
    const std::optional<std::array<double, 3>> accuracy = xmp.rtk_accuracy_m();
    ASSERT_TRUE(accuracy.has_value());
    EXPECT_DOUBLE_EQ((*accuracy)[0], 0.02);
    EXPECT_DOUBLE_EQ((*accuracy)[1], 0.03);
    EXPECT_DOUBLE_EQ((*accuracy)[2], 0.04);
}

TEST(Xmp, RtkFlagOnlyAcceptsFloatAndFixed) {
    for (const int code : {34, 50}) {
        const XmpReference xmp = parse_xmp_reference(
            attribute_packet("drone-dji:RtkFlag=\"" + std::to_string(code) + "\""));
        EXPECT_TRUE(xmp.has_rtk_fix()) << "RtkFlag " << code;
    }
    // 1/2 are not RTK states, and 0/16 are single-point GNSS.
    for (const int code : {0, 1, 2, 16}) {
        const XmpReference xmp = parse_xmp_reference(
            attribute_packet("drone-dji:RtkFlag=\"" + std::to_string(code) + "\""));
        EXPECT_FALSE(xmp.has_rtk_fix()) << "RtkFlag " << code;
    }
}

TEST(Xmp, DisabledByDefault) {
    const std::string packet = attribute_packet(
        "drone-dji:GimbalYawDegree=\"90\" drone-dji:GimbalPitchDegree=\"-90\" "
        "drone-dji:RtkFlag=\"50\" drone-dji:RtkStdLon=\"0.01\" "
        "drone-dji:RtkStdLat=\"0.02\" drone-dji:RtkStdHgt=\"0.03\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos(document, path, /*xmp_accuracy=*/false, /*xmp_orientation=*/false);
    const CameraReference& reference = document.active_chunk()->cameras().front().reference;
    EXPECT_FALSE(reference.orientation_deg.has_value());
    EXPECT_DOUBLE_EQ(reference.accuracy.x(), 10.0);
    EXPECT_DOUBLE_EQ(reference.accuracy.y(), 10.0);
    EXPECT_DOUBLE_EQ(reference.accuracy.z(), 10.0);
    remove_file(path);
}

#if defined(ALPHASHAPE_USE_PROJ)
TEST(Xmp, OrientationAppliedWhenEnabled) {
    const std::string packet = attribute_packet(
        "drone-dji:GimbalYawDegree=\"90\" drone-dji:GimbalPitchDegree=\"-90\" "
        "drone-dji:GimbalRollDegree=\"0\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos(document, path, /*xmp_accuracy=*/false, /*xmp_orientation=*/true);
    const std::optional<Eigen::Vector3d>& orientation =
        document.active_chunk()->cameras().front().reference.orientation_deg;
    ASSERT_TRUE(orientation.has_value());
    const Eigen::Matrix3d r_wc =
        camera_from_stored(*orientation, alphashape::crs::RotationOrder::ZYX);
    expect_direction(r_wc.col(2), 0.0, 0.0, -1.0);
    expect_direction(-r_wc.col(1), 1.0, 0.0, 0.0);
    remove_file(path);
}

TEST(Xmp, OrientationConvertedToChunkRotationOrderKeepsGeometry) {
    // The same physical pose as above, stored in XYZ order: the triple changes
    // order, but the camera-to-world rotation a consumer rebuilds must not.
    const std::string packet = attribute_packet(
        "drone-dji:GimbalYawDegree=\"90\" drone-dji:GimbalPitchDegree=\"-90\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    document.active_chunk()->meta().set("reference/rotation_order", "XYZ");
    run_add_photos(document, path, /*xmp_accuracy=*/false, /*xmp_orientation=*/true);
    const std::optional<Eigen::Vector3d>& orientation =
        document.active_chunk()->cameras().front().reference.orientation_deg;
    ASSERT_TRUE(orientation.has_value());
    const Eigen::Matrix3d r_wc =
        camera_from_stored(*orientation, alphashape::crs::RotationOrder::XYZ);
    expect_direction(r_wc.col(2), 0.0, 0.0, -1.0);
    expect_direction(-r_wc.col(1), 1.0, 0.0, 0.0);
    remove_file(path);
}

TEST(Xmp, GenericGimbalPoseIsOrderIndependent) {
    const std::string packet = attribute_packet(
        "drone-dji:GimbalYawDegree=\"37\" drone-dji:GimbalPitchDegree=\"-71\" "
        "drone-dji:GimbalRollDegree=\"4\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    document.active_chunk()->meta().set("reference/rotation_order", "XYZ");
    run_add_photos(document, path, /*xmp_accuracy=*/false, /*xmp_orientation=*/true);
    const std::optional<Eigen::Vector3d>& orientation =
        document.active_chunk()->cameras().front().reference.orientation_deg;
    ASSERT_TRUE(orientation.has_value());
    const Eigen::Matrix3d r_wc =
        camera_from_stored(*orientation, alphashape::crs::RotationOrder::XYZ);
    EXPECT_TRUE(r_wc.isApprox(dji_world_from_camera(37.0, -71.0, 4.0), 1e-9));
    remove_file(path);
}
#endif

TEST(Xmp, LocationFillsWhenExifGpsAbsent) {
    const std::string packet = attribute_packet(
        "drone-dji:GpsLatitude=\"45.1\" drone-dji:GpsLongitude=\"-9.25\" "
        "drone-dji:AbsoluteAltitude=\"312.5\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    // cam_b1 has no EXIF GPS.
    ASSERT_FALSE(read_exif(fixture("cam_b1.jpg")).gps.has_value());
    run_add_photos(document, path, /*xmp_accuracy=*/false, /*xmp_orientation=*/false);
    const CameraReference& reference = document.active_chunk()->cameras().front().reference;
    ASSERT_TRUE(reference.location.has_value());
    EXPECT_DOUBLE_EQ(reference.location->x(), -9.25);
    EXPECT_DOUBLE_EQ(reference.location->y(), 45.1);
    EXPECT_DOUBLE_EQ(reference.location->z(), 312.5);
    EXPECT_TRUE(reference.enabled);
    remove_file(path);
}

TEST(Xmp, ExifGpsIsPrimary) {
    const ExifData base = read_exif(fixture("cam_a1.jpg"));
    ASSERT_TRUE(base.gps.has_value());
    const std::string packet = attribute_packet(
        "drone-dji:GpsLatitude=\"10\" drone-dji:GpsLongitude=\"20\" "
        "drone-dji:AbsoluteAltitude=\"5\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_a1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos(document, path, /*xmp_accuracy=*/false, /*xmp_orientation=*/false);
    const CameraReference& reference = document.active_chunk()->cameras().front().reference;
    ASSERT_TRUE(reference.location.has_value());
    EXPECT_DOUBLE_EQ(reference.location->x(), base.gps->longitude_deg);
    EXPECT_DOUBLE_EQ(reference.location->y(), base.gps->latitude_deg);
    // cam_a1 has no GPS altitude, so the z placeholder is NaN.
    EXPECT_TRUE(std::isnan(reference.location->z()));
    remove_file(path);
}

/// The example DewarpData a Phantom 4 RTK writes, as test constants.
constexpr const char* kDewarpData =
    " 2018-09-04;3670.280000000000,3664.660000000000,2.680000000000,1.640000000000,"
    "-0.268152000000,0.113388000000,0.000041654400,0.000017653600,-0.033956100000";

TEST(Xmp, AddPhotosAppliesDewarpCalibrationWhenEnabled) {
    const std::string packet = attribute_packet(
        std::string("drone-dji:DewarpData=\"") + kDewarpData + "\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos_with_calibration(document, path, /*load_xmp_calibration=*/true);
    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.sensors().size(), 1U);
    const Calibration& c = chunk.sensors().front().calibration;
    const ImageInfo info = read_image_info(path);
    EXPECT_EQ(c.width, info.width);
    EXPECT_EQ(c.height, info.height);
    // f = fy, b1 = fx - fy, principal point offsets direct, radial direct.
    EXPECT_DOUBLE_EQ(c.f, 3664.66);
    EXPECT_DOUBLE_EQ(c.b1, 3670.28 - 3664.66);
    EXPECT_DOUBLE_EQ(c.cx, 2.68);
    EXPECT_DOUBLE_EQ(c.cy, 1.64);
    EXPECT_DOUBLE_EQ(c.k1, -0.268152);
    EXPECT_DOUBLE_EQ(c.k2, 0.113388);
    EXPECT_DOUBLE_EQ(c.k3, -0.0339561);
    // The tangential terms are SWAPPED: engine p1 = OpenCV p2, engine p2 = OpenCV p1.
    EXPECT_DOUBLE_EQ(c.p1, 0.0000176536);
    EXPECT_DOUBLE_EQ(c.p2, 0.0000416544);
    remove_file(path);
}

TEST(Xmp, AddPhotosKeepsExifCalibrationWhenDisabled) {
    const std::string packet = attribute_packet(
        std::string("drone-dji:DewarpData=\"") + kDewarpData + "\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos_with_calibration(document, path, /*load_xmp_calibration=*/false);
    const Calibration& c = document.active_chunk()->sensors().front().calibration;
    // The EXIF-seeded calibration: a focal length but no distortion, and a
    // principal point at the image centre.
    EXPECT_GT(c.f, 0.0);
    EXPECT_NE(c.f, 3664.66);
    EXPECT_DOUBLE_EQ(c.cx, 0.0);
    EXPECT_DOUBLE_EQ(c.cy, 0.0);
    EXPECT_DOUBLE_EQ(c.k1, 0.0);
    EXPECT_DOUBLE_EQ(c.p1, 0.0);
    EXPECT_DOUBLE_EQ(c.p2, 0.0);
    remove_file(path);
}

TEST(Xmp, AddPhotosDewarpFlagOneKeepsFocalButZeroesDistortion) {
    // DewarpFlag=1 means the camera already undistorted the JPEG: the focal and
    // principal point still describe the image, but the distortion must not be
    // applied.
    const std::string packet = attribute_packet(
        std::string("drone-dji:DewarpData=\"") + kDewarpData +
        "\" drone-dji:CalibratedFocalLength=\"3666.666504\" "
        "drone-dji:CalibratedOpticalCenterX=\"100.0\" "
        "drone-dji:CalibratedOpticalCenterY=\"75.0\" drone-dji:DewarpFlag=\"1\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos_with_calibration(document, path, /*load_xmp_calibration=*/true);
    const Calibration& c = document.active_chunk()->sensors().front().calibration;
    const ImageInfo info = read_image_info(path);
    EXPECT_DOUBLE_EQ(c.f, 3666.666504);
    // The optical centre is absolute; the engine stores the offset from centre.
    EXPECT_DOUBLE_EQ(c.cx, 100.0 - static_cast<double>(info.width) / 2.0);
    EXPECT_DOUBLE_EQ(c.cy, 75.0 - static_cast<double>(info.height) / 2.0);
    EXPECT_DOUBLE_EQ(c.k1, 0.0);
    EXPECT_DOUBLE_EQ(c.k2, 0.0);
    EXPECT_DOUBLE_EQ(c.k3, 0.0);
    EXPECT_DOUBLE_EQ(c.p1, 0.0);
    EXPECT_DOUBLE_EQ(c.p2, 0.0);
    remove_file(path);
}

TEST(Xmp, AddPhotosCalibratedFocalOnlyGivesZeroDistortion) {
    // No DewarpData, only CalibratedFocalLength and the optical centre.
    const std::string packet = attribute_packet(
        "drone-dji:CalibratedFocalLength=\"3666.666504\" "
        "drone-dji:CalibratedOpticalCenterX=\"100.0\" "
        "drone-dji:CalibratedOpticalCenterY=\"75.0\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos_with_calibration(document, path, /*load_xmp_calibration=*/true);
    const Calibration& c = document.active_chunk()->sensors().front().calibration;
    const ImageInfo info = read_image_info(path);
    EXPECT_DOUBLE_EQ(c.f, 3666.666504);
    EXPECT_DOUBLE_EQ(c.cx, 100.0 - static_cast<double>(info.width) / 2.0);
    EXPECT_DOUBLE_EQ(c.cy, 75.0 - static_cast<double>(info.height) / 2.0);
    EXPECT_DOUBLE_EQ(c.k1, 0.0);
    EXPECT_DOUBLE_EQ(c.p1, 0.0);
    remove_file(path);
}

TEST(Xmp, AddPhotosWithoutXmpCalibrationKeepsExifSeed) {
    // The option is on but the packet carries no lens calibration: the sensor
    // keeps the EXIF-seeded focal length.
    const std::string packet = attribute_packet("drone-dji:GpsLatitude=\"45.1\"");
    const std::string path = temp_path(".jpg");
    write_bytes(path, jpeg_with_packet(read_bytes(fixture("cam_b1.jpg")), packet));

    Document document = document_with_chunk();
    run_add_photos_with_calibration(document, path, /*load_xmp_calibration=*/true);
    const Calibration& c = document.active_chunk()->sensors().front().calibration;
    EXPECT_GT(c.f, 0.0);
    EXPECT_DOUBLE_EQ(c.cx, 0.0);
    EXPECT_DOUBLE_EQ(c.k1, 0.0);
    remove_file(path);
}

/// An independent OpenCV pinhole + Brown-Conrady projection, written out here so
/// the engine's mapped calibration is checked against the convention the DJI
/// coefficients are defined in, not against the engine's own model.
Eigen::Vector2d opencv_project(const XmpDewarpCalibration& d, int width, int height, double x,
                               double y) {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2;
    const double xd = x * radial + 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y;
    return Eigen::Vector2d(static_cast<double>(width) / 2.0 + d.cx + xd * d.fx,
                           static_cast<double>(height) / 2.0 + d.cy + yd * d.fy);
}

TEST(Xmp, MappedCalibrationMatchesOpenCvProjection) {
    // The engine's Calibration, built from the same DewarpData, must project a
    // camera-frame point to the same pixel an independent OpenCV projection
    // does. This pins the p1/p2 swap and the f/b1 split.
    const XmpReference xmp = parse_xmp_reference(
        attribute_packet(std::string("drone-dji:DewarpData=\"") + kDewarpData + "\""));
    ASSERT_TRUE(xmp.dewarp.has_value());
    const XmpDewarpCalibration& d = *xmp.dewarp;

    const int width = 5472;
    const int height = 3648;
    // The engine mapping, written out here (the task's helper is internal).
    Calibration c = Calibration::initial(CameraType::Frame, width, height, d.fy);
    c.cx = d.cx;
    c.cy = d.cy;
    c.b1 = d.fx - d.fy;
    c.k1 = d.k1;
    c.k2 = d.k2;
    c.k3 = d.k3;
    c.p1 = d.p2;
    c.p2 = d.p1;

    // A handful of off-axis points, including one with a strong tangential
    // contribution so a wrong p1/p2 mapping would show.
    const std::array<std::array<double, 2>, 4> points = {
        std::array<double, 2>{0.0, 0.0}, std::array<double, 2>{0.3, -0.2},
        std::array<double, 2>{-0.5, 0.4}, std::array<double, 2>{0.1, 0.6}};
    for (const auto& xy : points) {
        const double x = xy[0];
        const double y = xy[1];
        const Eigen::Vector2d expected = opencv_project(d, width, height, x, y);
        const Eigen::Vector2d actual = c.project(Eigen::Vector3d(x, y, 1.0));
        EXPECT_NEAR(actual.x(), expected.x(), 1e-6) << "x at (" << x << ", " << y << ")";
        EXPECT_NEAR(actual.y(), expected.y(), 1e-6) << "y at (" << x << ", " << y << ")";
    }
}

}  // namespace
}  // namespace alphashape
