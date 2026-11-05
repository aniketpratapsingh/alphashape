#include "common/algorithm/feature/detect_features_task.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/feature/mldb_descriptor.hpp"
#include "common/algorithm/match/tracks.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/io/camera_image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/film_geometry.hpp"

namespace alphashape {

void DetectFeaturesTask::declare_params(Params& params) const {
    // Names follow Metashape's matchPhotos() parameters so scripts
    // read the same; the defaults are its publicly documented API defaults.
    params.declare({"keypoint_limit", ParamType::Int, std::int64_t{40000},
                    "Maximum keypoints per image, strongest first. 0 = no limit."});
    params.declare({"keypoint_limit_per_mpx", ParamType::Int, std::int64_t{1000},
                    "Maximum keypoints per megapixel. 0 = no limit."});
    params.declare({"descriptor_type", ParamType::String, std::string{"mldb"},
                    "Descriptor to compute. Only 'mldb' in this version."});
    params.declare({"fiducial_exclusion_mm", ParamType::Float, 5.0,
                    "Drop keypoints this close to a fiducial mark or to the "
                    "normalised image border on scanned-film cameras."});
}

std::vector<Stage> DetectFeaturesTask::stages() const {
    return {Stage{"detect", 1}};  // one image per work item
}

namespace {

/// The keypoint cap from the two limits: 0 (or negative) means "no limit" for
/// each. Kept as a free helper over plain integers so MatchPhotos can pass its
/// own (already validated) values into detect_keypoints_for_frame.
std::size_t keypoint_cap_from_limits(std::int64_t limit, std::int64_t per_mpx, int width,
                                     int height) {
    if (limit < 0 || per_mpx < 0) {
        throw ValueError("keypoint limits must be non-negative");
    }
    std::size_t cap = std::numeric_limits<std::size_t>::max();
    if (limit > 0) cap = static_cast<std::size_t>(limit);
    if (per_mpx > 0) {
        const double mpx = static_cast<double>(width) * static_cast<double>(height) / 1.0e6;
        cap = std::min(cap, static_cast<std::size_t>(static_cast<double>(per_mpx) * mpx));
    }
    return cap;
}

}  // namespace

KeypointSet detect_keypoints_for_frame(const Document& document, const Chunk& chunk,
                                       const Frame& frame,
                                       std::int64_t keypoint_limit,
                                       std::int64_t keypoint_limit_per_mpx,
                                       double fiducial_exclusion_mm) {
    if (frame.camera_id() < 0) {
        throw Error("frame '" + frame.path() + "' is not linked to a camera");
    }
    const Camera* camera = nullptr;
    for (const Camera& candidate : chunk.cameras()) {
        if (candidate.id == frame.camera_id()) {
            camera = &candidate;
            break;
        }
    }
    if (camera == nullptr) {
        throw Error("frame '" + frame.path() + "' names a camera that is not in "
                    "the chunk");
    }
    // Reads the raw scan for a digital camera and the film-normalised image for
    // a scanned film (WP-13.18b), so detection always sees pipeline pixels.
    const Image image = load_camera_image(document, chunk, *camera);
    const DogDetector detector;
    const ScaleSpace space = detector.build_scale_space(image.to_luminance());
    std::vector<Detection> detections = detector.detect(space);

    // Fiducial marks are at the same film position in every frame and would
    // form zero-parallax matches. A general border/mask belongs to the masks
    // milestone; this only covers the film frame's own geometry.
    if (fiducial_exclusion_mm > 0.0) {
        const Sensor& sensor = effective_sensor(chunk, *camera);
        const FilmSettings& film = sensor.film;
        if (film.enabled && film_geometry_complete(film)) {
            const FilmSize size = film_normalised_size(film);
            const double radius =
                fiducial_exclusion_mm / film.resolution_mm_per_px;
            const double max_x = static_cast<double>(size.first) - radius;
            const double max_y = static_cast<double>(size.second) - radius;
            std::vector<Eigen::Vector2d> marks;
            marks.reserve(film.fiducials.size());
            for (const Fiducial& fiducial : film.fiducials) {
                marks.push_back(
                    normalised_from_film_mm(film, fiducial.film_mm));
            }
            const auto excluded = [&](const Detection& detection) {
                // Detector coordinates put the top-left pixel centre at (0, 0);
                // the normalised frame follows the Calibration convention.
                const Eigen::Vector2d point(
                    static_cast<double>(detection.keypoint.x) + 0.5,
                    static_cast<double>(detection.keypoint.y) + 0.5);
                if (point.x() < radius || point.y() < radius ||
                    point.x() > max_x || point.y() > max_y) {
                    return true;
                }
                for (const Eigen::Vector2d& mark : marks) {
                    if ((point - mark).norm() < radius) {
                        return true;
                    }
                }
                return false;
            };
            detections.erase(
                std::remove_if(detections.begin(), detections.end(), excluded),
                detections.end());
        }
    }

    // Strongest first; ties broken by position so output is deterministic
    // regardless of thread scheduling.
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        if (a.response != b.response) return a.response > b.response;
        if (a.keypoint.y != b.keypoint.y) return a.keypoint.y < b.keypoint.y;
        if (a.keypoint.x != b.keypoint.x) return a.keypoint.x < b.keypoint.x;
        return a.keypoint.orientation < b.keypoint.orientation;
    });
    const std::size_t cap =
        keypoint_cap_from_limits(keypoint_limit, keypoint_limit_per_mpx, image.width(), image.height());
    if (detections.size() > cap) detections.resize(cap);

    KeypointSet set;
    set.camera_id = frame.camera_id();
    set.descriptors = compute_mldb(space, detections);
    set.keypoints.reserve(detections.size());
    for (const Detection& d : detections) set.keypoints.push_back(d.keypoint);
    return set;
}

void DetectFeaturesTask::apply(Document& document, Params& params, const Progress& progress) {
    if (params.get_string("descriptor_type") != "mldb") {
        throw ValueError("descriptor_type '" + params.get_string("descriptor_type") +
                         "' is not supported (only 'mldb')");
    }
    // Validate limits before spending time on images.
    keypoint_cap_from_limits(params.get_int("keypoint_limit"),
                             params.get_int("keypoint_limit_per_mpx"), 1, 1);
    if (!(params.get_float("fiducial_exclusion_mm") >= 0.0)) {
        throw ValueError("fiducial_exclusion_mm must be non-negative");
    }

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("DetectFeatures needs an active chunk");
    }
    const auto& frames = chunk->frames();
    const auto total = static_cast<std::ptrdiff_t>(frames.size());
    std::vector<std::optional<KeypointSet>> results(frames.size());
    std::vector<std::string> errors(frames.size());

    if (total == 0) {
        progress.update("detect", 1.0);
        return;
    }

    // progress.report / progress.cancelled are user callbacks (later: Python)
    // and may throw, and nothing may unwind out of an OpenMP region. So the
    // workers below touch no callback at all: they only fill `results` and
    // `errors`. Progress is reported between batches, on the calling thread.
    // A batch is one thread's worth of frames, so each batch keeps all cores
    // busy while a cancelled flag can be honoured at batch granularity.
#if defined(ALPHASHAPE_HAVE_OPENMP)
    const std::ptrdiff_t batch_size =
        std::max<std::ptrdiff_t>(1, static_cast<std::ptrdiff_t>(omp_get_max_threads()));
#else
    const std::ptrdiff_t batch_size = 1;
#endif

    progress.update("detect", 0.0);
    std::ptrdiff_t processed = 0;
    while (processed < total) {
        const std::ptrdiff_t batch_end = std::min(total, processed + batch_size);
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t i = processed; i < batch_end; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            try {
                results[idx] = detect_keypoints_for_frame(
                    document, *chunk, *frames[idx],
                    params.get_int("keypoint_limit"),
                    params.get_int("keypoint_limit_per_mpx"),
                    params.get_float("fiducial_exclusion_mm"));
            } catch (const std::exception& e) {
                errors[idx] = "frame '" + frames[idx]->path() + "': " + e.what();
            }
        }
        processed = batch_end;
        progress.update("detect", static_cast<double>(processed) / static_cast<double>(total));
        if (progress.is_cancelled()) {
            throw CancelledError("DetectFeatures was cancelled");
        }
    }

    for (const std::string& error : errors) {
        if (!error.empty()) throw Error(error);
    }

    // Two frames linked to the same camera would collide below: either the
    // second silently overwrites the first, or the chunk ends up with two sets
    // for one camera. Fail before touching the chunk.
    std::set<std::int32_t> camera_ids;
    for (const auto& result : results) {
        if (!camera_ids.insert(result->camera_id).second) {
            throw Error("two frames are linked to camera " + std::to_string(result->camera_id));
        }
    }

    // Commit only after every image succeeded: a failed run leaves the chunk
    // exactly as it was.
    TiePoints& tie_points = chunk->tie_points();
    std::set<std::int32_t> replaced;
    for (auto& result : results) {
        if (KeypointSet* existing = tie_points.find_by_camera(result->camera_id)) {
            replaced.insert(result->camera_id);
            *existing = std::move(*result);
        } else {
            tie_points.keypoint_sets.push_back(std::move(*result));
        }
    }

    // Matches and tracks index into keypoint sets. A re-detected camera's old
    // indices are meaningless, so drop every pair touching it and rebuild the
    // tracks from the pairs that remain valid. (The MatchPhotos tie-point limit
    // is not re-applied here; re-running MatchPhotos restores it.)
    if (!replaced.empty()) {
        auto& matches = tie_points.matches;
        const auto stale = [&replaced](const PairMatches& pair) {
            return replaced.count(pair.camera_a) != 0 || replaced.count(pair.camera_b) != 0;
        };
        const auto before = matches.size();
        matches.erase(std::remove_if(matches.begin(), matches.end(), stale), matches.end());
        if (matches.size() != before || !tie_points.tracks.empty()) {
            tie_points.tracks = match::build_tracks(matches);
        }
    }
    progress.update("detect", 1.0);
}

}  // namespace alphashape
