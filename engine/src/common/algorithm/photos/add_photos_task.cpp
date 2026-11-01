#include "common/algorithm/photos/add_photos_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <Eigen/Core>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/reference/xmp_reference.hpp"
#include "common/core/frame.hpp"
#include "image/io/exif.hpp"
#include "image/io/image_io.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "common/algorithm/reference/convert_reference.hpp"
#endif

namespace alphashape {

namespace {

namespace fs = std::filesystem;

/// The focal length in pixels that an unknown lens is assumed to have, as a
/// multiple of the frame's long edge (see the header for why 1.2).
constexpr double kDefaultFocalFactor = 1.2;

/// The identity of a sensor: everything that must match for two photos to be
/// considered shots through the same imaging device. focal_length_mm is
/// compared exactly: it comes from one EXIF RATIONAL, so two photos of the same
/// lens and focal setting produce bit-identical doubles, and a camera body that
/// reports 35.0 vs 35.0000001 genuinely reports two different focal lengths.
///
/// BodySerialNumber (0xA431) is deliberately NOT part of the key: two bodies of
/// the same model at the same focal setting start from one shared calibration
/// (the user can split them later). The sensor keeps the serial only when every
/// photo in the group agrees on it; a mixed group stores an empty serial rather
/// than claiming the first photo's.
struct SensorKey {
    std::string make, model;
    double focal_length_mm = 0.0;
    int width = 0;
    int height = 0;

    bool operator==(const SensorKey& other) const {
        return make == other.make && model == other.model &&
               focal_length_mm == other.focal_length_mm && width == other.width &&
               height == other.height;
    }
};

SensorKey key_of(const Sensor& sensor) {
    return SensorKey{sensor.make, sensor.model, sensor.focal_length_mm, sensor.width,
                     sensor.height};
}

std::string or_default(const std::optional<std::string>& value) {
    return value.has_value() ? *value : std::string{};
}

/// The label shown for a sensor: "<model> (<focal> mm)", or "Unknown" when we
/// know neither. A half-known sensor (a model but no focal, say) still gets the
/// half it has, because "AO-100 (unknown mm)" is more useful than "Unknown" —
/// the point of the fallback is only to avoid the empty string.
std::string sensor_label(const std::string& model, double focal_mm) {
    const bool has_model = !model.empty();
    const bool has_focal = focal_mm > 0.0;
    if (!has_model && !has_focal) {
        return "Unknown";
    }
    std::string focal_text = has_focal ? std::to_string(focal_mm) : "unknown";
    // Trim the trailing zeros std::to_string leaves on an integral focal length,
    // so 35 mm reads "35 mm" and not "35.000000 mm".
    const std::size_t dot = focal_text.find('.');
    if (dot != std::string::npos) {
        const std::size_t last = focal_text.find_last_not_of('0');
        if (last == dot) {
            focal_text.erase(dot);
        } else {
            focal_text.erase(last + 1);
        }
    }
    return (has_model ? model : std::string("Unknown")) + " (" + focal_text + " mm)";
}

/// Warn once per process when XMP orientation cannot be stored safely. The
/// condition is per photo, so a plain log line per file would flood the console.
void warn_orientation_skipped(const char* reason) {
    static std::once_flag once;
    std::call_once(once, [reason] {
        log(LogLevel::Warn) << "AddPhotos: XMP orientation skipped: " << reason;
    });
}

/// Derive the pixel pitch in micrometres, corrected for a JPEG that was scaled
/// after the EXIF was written.
///
/// FocalPlaneXResolution is "the number of pixels in the image width (X)
/// direction per FocalPlaneResolutionUnit". That sentence is about the image
/// the tag was written for, whose width is PixelXDimension (0xA002). If a
/// downstream tool resized the file, the tag still describes the ORIGINAL
/// grid: the pitch it implies is the original pixel pitch. The pitch that
/// belongs to the pixels we actually have is that value scaled by
/// original_width / actual_width — a smaller resized image has larger pixels,
/// so the pitch grows. Comparing the two widths is exactly the check for
/// "was this file resized".
std::optional<double> corrected_pixel_size_um(const ExifData& exif, int width) {
    std::optional<double> size_um = exif.pixel_size_um();
    if (!size_um.has_value()) {
        return std::nullopt;
    }
    if (exif.pixel_x_dimension.has_value() && width > 0 &&
        *exif.pixel_x_dimension != width) {
        *size_um *= static_cast<double>(*exif.pixel_x_dimension) /
                    static_cast<double>(width);
    }
    return size_um;
}

/// The three-route focal length, documented in the header.
double initial_focal_pixels(const ExifData& exif, int width, int height,
                            const std::optional<double>& pixel_size_um) {
    const bool have_focal_mm =
        exif.focal_length_mm.has_value() && *exif.focal_length_mm > 0.0;
    if (have_focal_mm && pixel_size_um.has_value() && *pixel_size_um > 0.0) {
        return *exif.focal_length_mm * 1000.0 / *pixel_size_um;
    }
    const bool have_focal_35mm =
        exif.focal_length_35mm.has_value() && *exif.focal_length_35mm > 0.0;
    if (have_focal_35mm) {
        const double w = static_cast<double>(width);
        const double h = static_cast<double>(height);
        // The 35mm tag is the focal length that would give the same field of
        // view on a 36x24 frame, so the diagonal ratio converts it to pixels:
        // the field of view is 2*atan(half_frame_diagonal / f), and equal field
        // of view means f_px / frame_diagonal_px == f35 / 36x24 diagonal.
        return *exif.focal_length_35mm * std::hypot(w, h) / std::hypot(36.0, 24.0);
    }
    return kDefaultFocalFactor * static_cast<double>(std::max(width, height));
}

#if defined(ALPHASHAPE_USE_PROJ)
/// True when the chunk's camera-reference slot is a projected CRS, i.e. one
/// whose grid north differs from true north by the meridian convergence.
bool camera_slot_is_projected(const Chunk& chunk) {
    const CRS& slot = chunk.camera_crs();
    if (!slot.is_set()) {
        return false;  // unset means the geographic default AddPhotos assumes
    }
    const crs::CoordinateSystem cs = slot;
    return cs.kind() == crs::CrsKind::Projected;
}

/// Turn the XMP parser's stored ZYX triple into the triple the chunk stores.
///
/// The stored convention is `R_wb = Rz(90) * to_matrix({yaw, -pitch, roll})`,
/// so the pitch sign is flipped before the decomposition and back after it.
/// When the camera slot is projected the body attitude also has to be
/// re-expressed from true north (the gimbal heading's frame) to the slot's grid
/// north, which is exactly `convert_orientation`'s world-frame relabelling.
Eigen::Vector3d orientation_in_chunk_order(const XmpOrientation& angles, const Chunk& chunk,
                                           const std::optional<Eigen::Vector2d>& lonlat) {
    const crs::RotationOrder order = reference::rotation_order_of(chunk);
    Eigen::Matrix3d body_attitude =
        crs::to_matrix(crs::EulerAngles{angles.yaw_deg, -angles.pitch_deg, angles.roll_deg,
                                        crs::RotationOrder::ZYX});
    bool converged = false;
    if (camera_slot_is_projected(chunk) && lonlat.has_value()) {
        const crs::CoordinateSystem geographic = crs::CoordinateSystem::from_epsg(4326);
        const crs::CoordinateSystem slot = chunk.camera_crs();
        const crs::EulerAngles internal{angles.yaw_deg, -angles.pitch_deg, angles.roll_deg,
                                        crs::RotationOrder::ZYX};
        const crs::EulerAngles moved = reference::convert_orientation(
            internal, geographic, slot, crs::CrsPoint{lonlat->x(), lonlat->y(), 0.0});
        body_attitude = crs::to_matrix(moved);
        converged = true;
    }
    const crs::EulerAngles out =
        (!converged && order == crs::RotationOrder::ZYX)
            ? crs::EulerAngles{angles.yaw_deg, -angles.pitch_deg, angles.roll_deg, order}
            : crs::from_matrix(body_attitude, order);
    return Eigen::Vector3d(out.yaw, -out.pitch, out.roll);
}
#endif

/// Everything one imported photo contributes to the chunk. Built up in locals
/// so a failure anywhere in the loop leaves the chunk untouched.
struct PendingCamera {
    Camera camera;
    std::shared_ptr<Frame> frame;
    SensorKey key;
    Sensor sensor;       ///< only meaningful when `needs_sensor` is true
    bool needs_sensor = false;
    std::size_t sensor_index = 0;  ///< index into known_keys == final sensor index
    std::string serial;
};

std::int32_t next_camera_id(const Chunk& chunk) {
    std::int32_t max_id = -1;
    for (const Camera& camera : chunk.cameras()) {
        max_id = std::max(max_id, camera.id);
    }
    return max_id + 1;
}

std::int32_t next_frame_id(const Chunk& chunk) {
    std::int32_t max_id = -1;
    for (const auto& frame : chunk.frames()) {
        max_id = std::max(max_id, frame->id());
    }
    return max_id + 1;
}

bool same_path(const std::string& a, const std::string& b) {
    return fs::path(a).lexically_normal() == fs::path(b).lexically_normal();
}

}  // namespace

void AddPhotosTask::declare_params(Params& params) const {
    params.declare({"filenames", ParamType::StringList, std::monostate{},
                    "Image files to import, as a list of paths."});
    params.declare({"load_reference", ParamType::Bool, true,
                    "Load each photo's EXIF GPS position into its camera reference, "
                    "falling back to the XMP position when EXIF GPS is absent."});
    params.declare({"strip_extensions", ParamType::Bool, true,
                    "Label each camera with the file name without its extension."});
    params.declare({"load_xmp_accuracy", ParamType::Bool, false,
                    "Use a fixed/float RTK standard deviation from XMP as the reference "
                    "accuracy."});
    params.declare({"load_xmp_orientation", ParamType::Bool, false,
                    "Store the XMP gimbal (or flight) yaw/pitch/roll as the camera "
                    "reference orientation."});
    params.declare({"load_xmp_calibration", ParamType::Bool, false,
                    "Reserved for XMP lens calibration (DewarpData). Accepted but not yet "
                    "applied."});
}

std::vector<Stage> AddPhotosTask::stages() const {
    return {Stage{"import", 1}};  // one file per work item
}

void AddPhotosTask::apply(Document& document, Params& params, const Progress& progress) {
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("AddPhotos needs an active chunk");
    }

    const std::vector<std::string> filenames = params.get_string_list("filenames");
    if (filenames.empty()) {
        throw ValueError("AddPhotos: 'filenames' must not be empty");
    }
    const bool load_reference = params.get_bool("load_reference");
    const bool strip_extensions = params.get_bool("strip_extensions");
    const bool load_xmp_accuracy = params.get_bool("load_xmp_accuracy");
    const bool load_xmp_orientation = params.get_bool("load_xmp_orientation");
    // Declared so the schema and Python front end accept it; DewarpData is not
    // applied yet (see the parameter description and the PR notes).
    const bool load_xmp_calibration = params.get_bool("load_xmp_calibration");
    // Any of these reads XMP; with all of them off the packet scan is skipped.
    const bool want_xmp =
        load_reference || load_xmp_accuracy || load_xmp_orientation || load_xmp_calibration;

    // -- resolve and de-duplicate before reading anything --------------------
    std::vector<std::string> absolute;
    absolute.reserve(filenames.size());
    for (const std::string& name : filenames) {
        if (name.empty()) {
            throw ValueError("AddPhotos: an empty file name was given");
        }
        // A relative path is resolved against the process's current directory,
        // which is what the user typed; storing the absolute form means the
        // reference stays valid no matter where the project is later opened.
        std::error_code ec;
        fs::path resolved = fs::absolute(fs::path(name), ec);
        if (ec) {
            resolved = fs::path(name);
        }
        absolute.push_back(resolved.lexically_normal().string());
    }
    for (std::size_t i = 0; i < absolute.size(); ++i) {
        for (std::size_t j = i + 1; j < absolute.size(); ++j) {
            if (same_path(absolute[i], absolute[j])) {
                throw ValueError("AddPhotos: '" + absolute[j] +
                                 "' appears twice in the list");
            }
        }
        for (const auto& frame : chunk->frames()) {
            if (same_path(frame->path(), absolute[i])) {
                throw ValueError("AddPhotos: '" + absolute[i] +
                                 "' is already in this chunk");
            }
        }
    }

    // -- read every file into locals ----------------------------------------
    const auto total = static_cast<double>(absolute.size());
    progress.update("import", 0.0);

    std::vector<PendingCamera> pending;
    pending.reserve(absolute.size());
    // Sensors discovered during this call, plus the ones the chunk already has,
    // so a second AddPhotos call reuses an existing sensor.
    std::vector<SensorKey> known_keys;
    known_keys.reserve(chunk->sensors().size() + absolute.size());
    for (const Sensor& sensor : chunk->sensors()) {
        known_keys.push_back(key_of(sensor));
    }

    std::int32_t camera_id = next_camera_id(*chunk);
    std::int32_t frame_id = next_frame_id(*chunk);

    for (std::size_t i = 0; i < absolute.size(); ++i) {
        const std::string& path = absolute[i];
        if (progress.is_cancelled()) {
            throw CancelledError("AddPhotos was cancelled");
        }

        PendingCamera item;
        try {
            // Header-only: we need the dimensions, not the pixels.
            const ImageInfo info = read_image_info(path);
            const ExifData exif = read_exif(path);
            // XMP is read from the same file: a JPEG APP1 packet or TIFF tag
            // 700. Parsing never throws; an absent packet is an empty result.
            const XmpReference xmp =
                want_xmp && exif.xmp_packet.has_value() ? parse_xmp_reference(*exif.xmp_packet)
                                                        : XmpReference{};

            const std::optional<double> pixel_size_um = corrected_pixel_size_um(exif, info.width);
            const double focal_pixels = initial_focal_pixels(exif, info.width, info.height, pixel_size_um);

            item.key = SensorKey{or_default(exif.make), or_default(exif.model),
                                 exif.focal_length_mm.value_or(0.0), info.width, info.height};

            item.serial = or_default(exif.body_serial);
            auto existing = std::find(known_keys.begin(), known_keys.end(), item.key);
            if (existing != known_keys.end()) {
                item.sensor_index = static_cast<std::size_t>(existing - known_keys.begin());
            } else {
                item.sensor_index = known_keys.size();
                known_keys.push_back(item.key);
                item.needs_sensor = true;
                item.sensor.label = sensor_label(item.key.model, item.key.focal_length_mm);
                item.sensor.type = CameraType::Frame;
                item.sensor.width = info.width;
                item.sensor.height = info.height;
                item.sensor.pixel_size_um = pixel_size_um.value_or(0.0);
                item.sensor.focal_length_mm = item.key.focal_length_mm;
                item.sensor.make = item.key.make;
                item.sensor.model = item.key.model;
                item.sensor.serial = item.serial;
                item.sensor.calibration = Calibration::initial(CameraType::Frame, info.width,
                                                              info.height, focal_pixels);
            }

            item.camera.id = camera_id++;
            item.camera.label = strip_extensions
                                    ? fs::path(path).stem().string()
                                    : fs::path(path).filename().string();
            item.camera.orientation = exif.orientation.value_or(1);
            item.camera.enabled = false;  // not aligned yet

            if (load_reference && exif.gps.has_value()) {
                // EXIF GPS stays primary; XMP position is only the fallback.
                const GpsPosition& gps = *exif.gps;
                item.camera.reference.location = Eigen::Vector3d(
                    gps.longitude_deg, gps.latitude_deg,
                    gps.altitude_m.value_or(std::numeric_limits<double>::quiet_NaN()));
                item.camera.reference.enabled = true;
            } else if (load_reference && xmp.has_location()) {
                // AbsoluteAltitude is an ellipsoidal/MSL height we cannot
                // distinguish; RelativeAltitude is not a georeferenced height,
                // so it is deliberately not used as the location's z.
                item.camera.reference.location = Eigen::Vector3d(
                    *xmp.longitude_deg, *xmp.latitude_deg,
                    xmp.absolute_altitude_m.value_or(
                        std::numeric_limits<double>::quiet_NaN()));
                item.camera.reference.enabled = true;
            }

            if (load_xmp_accuracy) {
                if (const auto accuracy = xmp.rtk_accuracy_m()) {
                    item.camera.reference.accuracy =
                        Eigen::Vector3d((*accuracy)[0], (*accuracy)[1], (*accuracy)[2]);
                }
            }

            // A projected camera slot needs the position to turn a true-north
            // heading into grid north; either EXIF GPS or the XMP fallback is
            // enough. Without it the stored heading would be mislabelled.
            std::optional<Eigen::Vector2d> lonlat;
            if (xmp.has_location()) {
                lonlat = Eigen::Vector2d(*xmp.longitude_deg, *xmp.latitude_deg);
            } else if (exif.gps.has_value()) {
                lonlat = Eigen::Vector2d(exif.gps->longitude_deg, exif.gps->latitude_deg);
            }

            if (load_xmp_orientation) {
                if (const auto angles = xmp.orientation_zyx()) {
#if defined(ALPHASHAPE_USE_PROJ)
                    if (camera_slot_is_projected(*chunk) && !lonlat.has_value()) {
                        warn_orientation_skipped(
                            "the camera slot is projected but the photo has no position, "
                            "so the true-north heading cannot be turned into grid north");
                    } else {
                        item.camera.reference.orientation_deg =
                            orientation_in_chunk_order(*angles, *chunk, lonlat);
                    }
#else
                    // Without the CRS engine the chunk's rotation order and the
                    // meridian convergence cannot be resolved; storing the ZYX
                    // triple anyway would mislabel it in a non-ZYX chunk.
                    (void)angles;
                    (void)lonlat;
                    warn_orientation_skipped(
                        "this build has no CRS engine, so the rotation order cannot be resolved");
#endif
                }
            }

            // load_xmp_calibration is accepted for forward compatibility. DJI
            // DewarpData mixes a capture date, an image size and a coefficient
            // list whose exact layout varies by firmware, and this task always
            // seeds a sensor calibration from EXIF focal length, so applying it
            // here would be a guess. It is parsed into XmpReference::dewarp_data
            // but not written to the sensor.

            item.frame = std::make_shared<Frame>(path, info.width, info.height);
            item.frame->set_id(frame_id++);
            item.frame->set_camera_id(item.camera.id);
        } catch (const std::exception& e) {
            throw Error("AddPhotos: '" + path + "': " + e.what());
        }

        pending.push_back(std::move(item));
        progress.update("import", static_cast<double>(i + 1) / total);
    }

    // -- commit: every file succeeded, so the chunk may change ----------------
    // known_keys began as the chunk's sensors in order and new keys were
    // appended in discovery order, which is exactly the order new sensors are
    // pushed below; so each item's sensor_index is already its final sensor_id
    // and nothing here can fail part-way.
    for (PendingCamera& item : pending) {
        if (item.needs_sensor) {
            chunk->sensors().push_back(std::move(item.sensor));
        } else if (chunk->sensors()[item.sensor_index].serial != item.serial) {
            chunk->sensors()[item.sensor_index].serial.clear();  // mixed bodies
        }
        item.camera.sensor_id = static_cast<std::int32_t>(item.sensor_index);
        chunk->cameras().push_back(std::move(item.camera));
        chunk->frames().push_back(std::move(item.frame));
    }
    progress.update("import", 1.0);
}

}  // namespace alphashape
