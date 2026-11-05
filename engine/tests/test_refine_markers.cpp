// WP-13.9: RefineMarkers, multi-view consensus refinement of marker projections.
//
// The scene is rendered analytically by inverse-mapping every subpixel through
// the ground plane (y = 0): a deterministic textured surface plus printed
// targets, seen by 4-6 calibrated pinhole cameras with known poses. Blur and
// noise are added afterwards, so the "truth" projection of each target centre
// is exactly project_marker(centre). The task never sees those truth pixels:
// marker projections are perturbed before the run.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/algorithm/markers/refine_markers_task.hpp"
#include "common/commands/command.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "common/tasks/registry.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "make_circular_target.hpp"
#include "make_marker_sheet.hpp"

namespace alphashape {
namespace {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;

struct SceneTarget {
    bool cross = false;
    bool painted = true;
    Vec3 centre = Vec3::Zero();  ///< on the ground plane, y = 0
    double radius = 0.07;        ///< physical outer radius, metres
    double rotation = 0.0;
    std::uint32_t word = 0b101101001011U;
    int parity = 1;  ///< 1 prints a dark centre dot, 0 a white one.
};

struct BlankBox {
    double x0 = 0.0;
    double x1 = 0.0;
    double z0 = 0.0;
    double z1 = 0.0;
};

struct SceneSpec {
    int marker_count = 6;
    int camera_count = 5;
    double ring_radius = 1.5;
    double height = 1.0;
    double spread_x = 0.28;
    double spread_z = 0.32;
    double start_angle = 0.0;
    bool include_blank_target = false;
    std::vector<BlankBox> blank_boxes;
    /// Target index printed with parity bit 0 (white centre), or -1 for none.
    int even_code_target = -1;
    /// Physical outer radius of every printed target, metres.
    double target_radius = 0.075;
    /// Target index that receives an extra small dark speck near its centre,
    /// or -1. Used to prove a speck cannot win over the printed centre dot.
    int speck_target = -1;
    double speck_offset_px = 8.0;
};

/// Deterministic, continuous ground texture. The value depends only on the
/// world position, so every camera sees the same physical surface.
float ground_shade(double x, double z, const std::vector<BlankBox>& blanks) {
    for (const BlankBox& box : blanks) {
        if (x >= box.x0 && x <= box.x1 && z >= box.z0 && z <= box.z1) {
            return 0.85F;
        }
    }
    const double wave = 0.11 * std::sin(37.0 * x + 0.6) +
                        0.09 * std::cos(29.0 * z - 0.3) +
                        0.06 * std::sin(53.0 * (x + z));
    const auto hash = [](int ix, int iz) {
        std::uint32_t value = static_cast<std::uint32_t>(ix) * 73856093U ^
                              static_cast<std::uint32_t>(iz) * 19349663U;
        value ^= value >> 13;
        value *= 0x5bd1e995U;
        value ^= value >> 15;
        return static_cast<double>(value & 0xffffU) / 65535.0 - 0.5;
    };
    const double grain = 0.10 * hash(static_cast<int>(std::floor(x * 45.0)),
                                     static_cast<int>(std::floor(z * 45.0)));
    return static_cast<float>(std::clamp(0.5 + wave + grain, 0.08, 0.92));
}

/// Target intensity at a world point, or nullopt when the point is background.
std::optional<float> target_shade(const SceneTarget& target, double x, double z) {
    if (!target.painted) {
        return std::nullopt;
    }
    const double dx = x - target.centre.x();
    const double dz = z - target.centre.z();
    const double extent = target.radius * 1.5;
    if (std::abs(dx) > extent || std::abs(dz) > extent) {
        return std::nullopt;
    }
    const double c = std::cos(target.rotation);
    const double s = std::sin(target.rotation);
    const double u = dx * c + dz * s;
    const double v = -dx * s + dz * c;
    if (target.cross) {
        const double half = target.radius * 0.9;
        if (std::abs(u) > half || std::abs(v) > half) {
            return std::nullopt;
        }
        return (u > 0.0) == (v > 0.0) ? 0.95F : 0.05F;
    }
    const double rho = std::sqrt(u * u + v * v) / target.radius;
    if (rho > 1.0) {
        return std::nullopt;
    }
    test::CircularTruth truth;
    truth.bits = 12;
    truth.word = target.word;
    truth.parity = target.parity;
    const Eigen::Vector2d local(u / target.radius, v / target.radius);
    const double sector_width =
        2.0 * std::numbers::pi_v<double> / static_cast<double>(truth.bits);
    return test::circular_black_at(local, truth, sector_width) ? 0.05F : 0.95F;
}

/// Look-at pose for a camera at `centre` aimed at `target`. The engine
/// convention is +X right, +Y down, +Z forward with a right-handed frame.
Transform look_at(const Vec3& centre, const Vec3& target) {
    const Vec3 forward = (target - centre).normalized();
    Vec3 reference = Vec3::UnitX();
    if (std::abs(forward.dot(reference)) > 0.9) {
        reference = Vec3::UnitZ();
    }
    const Vec3 right = forward.cross(reference).normalized();
    const Vec3 down = forward.cross(right);
    Eigen::Matrix3d rotation;
    rotation.row(0) = right.transpose();
    rotation.row(1) = down.transpose();
    rotation.row(2) = forward.transpose();
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = rotation;
    pose.block<3, 1>(0, 3) = -rotation * centre;
    return pose;
}

std::vector<Transform> ring_poses(const SceneSpec& spec) {
    std::vector<Transform> poses;
    for (int i = 0; i < spec.camera_count; ++i) {
        const double angle = spec.start_angle +
                             2.0 * std::numbers::pi_v<double> *
                                 static_cast<double>(i) /
                                 static_cast<double>(spec.camera_count);
        const Vec3 centre(spec.ring_radius * std::cos(angle), -spec.height,
                          spec.ring_radius * std::sin(angle));
        poses.push_back(look_at(centre, Vec3(0.0, 0.0, 0.0)));
    }
    return poses;
}

FloatImage render_view(const Transform& pose, const Calibration& calibration,
                       const SceneSpec& spec,
                       const std::vector<SceneTarget>& targets) {
    FloatImage image(calibration.width, calibration.height, 0.5F);
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    const Vec3 centre = -rotation.transpose() * pose.block<3, 1>(0, 3);
    for (int y = 0; y < calibration.height; ++y) {
        for (int x = 0; x < calibration.width; ++x) {
            double sum = 0.0;
            for (int sy = 0; sy < 2; ++sy) {
                for (int sx = 0; sx < 2; ++sx) {
                    const Vec2 pixel(static_cast<double>(x) + 0.25 + 0.5 * sx,
                                     static_cast<double>(y) + 0.25 + 0.5 * sy);
                    const Vec3 ray =
                        rotation.transpose() * calibration.unproject(pixel);
                    double shade = 0.5;
                    if (ray.y() > 1e-9) {
                        const double distance = -centre.y() / ray.y();
                        const Vec3 world = centre + distance * ray;
                        shade = static_cast<double>(ground_shade(
                            world.x(), world.z(), spec.blank_boxes));
                        for (const SceneTarget& target : targets) {
                            if (const auto value =
                                    target_shade(target, world.x(), world.z())) {
                                shade = static_cast<double>(*value);
                                break;
                            }
                        }
                    }
                    sum += shade;
                }
            }
            image.at(x, y) = static_cast<float>(sum / 4.0);
        }
    }
    return image;
}

/// A 5 cm lattice of ground points around every target. This is the tie-point
/// cloud the task fits the local plane and the depth range to.
void add_ground_cloud(Chunk& chunk, const std::vector<SceneTarget>& targets) {
    PointCloud cloud;
    for (const SceneTarget& target : targets) {
        for (int ix = -7; ix <= 7; ++ix) {
            for (int iz = -7; iz <= 7; ++iz) {
                cloud.points.emplace_back(target.centre.x() + 0.05 * ix, 0.0,
                                          target.centre.z() + 0.05 * iz);
            }
        }
    }
    chunk.point_cloud() = cloud;
}

void write_gray_png(const std::string& path, const FloatImage& image) {
    Image out(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const double value = static_cast<double>(image.at(x, y)) * 255.0;
            out.at(x, y, 0) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    write_png(path, out);
}

struct RefineScene {
    Document document;
    Chunk* chunk = nullptr;
    std::vector<std::string> paths;
    std::vector<SceneTarget> targets;
    std::vector<Transform> poses;

    RefineScene() = default;
    RefineScene(RefineScene&&) = default;
    RefineScene& operator=(RefineScene&&) = default;
    RefineScene(const RefineScene&) = delete;
    RefineScene& operator=(const RefineScene&) = delete;
    ~RefineScene() {
        for (const std::string& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }

    const Camera& camera(std::size_t index) const {
        return chunk->cameras()[index];
    }
    const Calibration& calibration(std::size_t index) const {
        return chunk->sensors()[static_cast<std::size_t>(camera(index).sensor_id)]
            .calibration;
    }
    std::optional<Vec2> truth(std::size_t target_index,
                              std::size_t camera_index) const {
        return project_marker(targets[target_index].centre, camera(camera_index),
                              calibration(camera_index), false);
    }
};

RefineScene make_scene(SceneSpec spec, std::uint32_t seed) {
    RefineScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("Chunk 1");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    for (int i = 0; i < spec.marker_count; ++i) {
        SceneTarget target;
        target.cross = (i % 2) == 1;
        const int column = i % 3;
        const int row = i / 3;
        target.centre = Vec3((static_cast<double>(column) - 1.0) * spec.spread_x,
                             0.0,
                             (static_cast<double>(row) - 0.5) * spec.spread_z);
        target.radius = spec.target_radius;
        target.rotation = 0.17 * static_cast<double>(i);
        target.word = 0b101101001011U ^ (static_cast<std::uint32_t>(i) << 2);
        target.parity = i == spec.even_code_target ? 0 : 1;
        scene.targets.push_back(target);
    }
    if (spec.include_blank_target) {
        SceneTarget blank;
        blank.painted = false;
        blank.centre = Vec3(0.48, 0.0, -0.34);
        blank.radius = 0.075;
        scene.targets.push_back(blank);
        BlankBox box;
        box.x0 = 0.34;
        box.x1 = 0.62;
        box.z0 = -0.48;
        box.z1 = -0.20;
        spec.blank_boxes.push_back(box);
    }
    scene.poses = ring_poses(spec);
    for (std::size_t i = 0; i < scene.poses.size(); ++i) {
        const int width = 800;
        const int height = 600;
        const double focal = 700.0;
        Sensor sensor;
        sensor.label = "sensor" + std::to_string(i);
        sensor.type = CameraType::Frame;
        sensor.width = width;
        sensor.height = height;
        sensor.calibration =
            Calibration::initial(CameraType::Frame, width, height, focal);
        chunk.sensors().push_back(sensor);
        Camera camera_record;
        camera_record.id = chunk.allocate_id();
        camera_record.sensor_id =
            static_cast<std::int32_t>(chunk.sensors().size()) - 1;
        camera_record.label = "cam" + std::to_string(i);
        camera_record.enabled = true;
        camera_record.pose = scene.poses[i];
        chunk.cameras().push_back(camera_record);
        const std::string path =
            platform::unique_temp_path("alphashape_refine_", ".png");
        auto frame = std::make_shared<Frame>(path, width, height);
        frame->set_id(chunk.allocate_id());
        frame->set_camera_id(camera_record.id);
        chunk.frames().push_back(frame);
        scene.paths.push_back(path);
        FloatImage image =
            render_view(scene.poses[i], sensor.calibration, spec, scene.targets);
        if (spec.speck_target >= 0 &&
            static_cast<std::size_t>(spec.speck_target) < scene.targets.size()) {
            const auto centre = project_marker(
                scene.targets[static_cast<std::size_t>(spec.speck_target)]
                    .centre,
                camera_record, sensor.calibration);
            if (centre) {
                const int speck_x = static_cast<int>(
                    std::lround(centre->x() + spec.speck_offset_px));
                const int speck_y =
                    static_cast<int>(std::lround(centre->y()));
                for (int y = speck_y - 1; y <= speck_y + 1; ++y) {
                    for (int x = speck_x - 1; x <= speck_x + 1; ++x) {
                        if (x < 0 || y < 0 || x >= width || y >= height) {
                            continue;
                        }
                        image.at(x, y) = 0.02F;
                    }
                }
            }
        }
        image = gaussian_blur(image, 0.8F);
        test::add_noise(image, 0.008, seed + static_cast<std::uint32_t>(i));
        write_gray_png(path, image);
    }
    add_ground_cloud(chunk, scene.targets);
    return scene;
}

std::size_t camera_index(const RefineScene& scene, std::int32_t camera_id) {
    for (std::size_t i = 0; i < scene.chunk->cameras().size(); ++i) {
        if (scene.chunk->cameras()[i].id == camera_id) {
            return i;
        }
    }
    return scene.chunk->cameras().size();
}

/// Attach a marker to the scene with one projection per visible camera.
/// `perturbation` is the largest random offset in pixels; 0 leaves the truth.
std::int32_t add_scene_marker(RefineScene& scene, std::size_t target_index,
                              const std::string& label, double perturbation,
                              std::mt19937& engine,
                              const std::vector<std::size_t>& only = {}) {
    Marker marker;
    marker.id = scene.chunk->allocate_id();
    marker.label = label;
    marker.type = MarkerType::Fiducial;
    marker.valid = true;
    // Record the printed kind, exactly as DetectMarkers does, so refinement
    // can tell a coded circular target from a cross.
    marker.meta.set("target_type", scene.targets[target_index].cross
                                       ? "Cross"
                                       : "Circular12");
    for (std::size_t ci = 0; ci < scene.chunk->cameras().size(); ++ci) {
        if (!only.empty() &&
            std::find(only.begin(), only.end(), ci) == only.end()) {
            continue;
        }
        const auto truth = scene.truth(target_index, ci);
        if (!truth || truth->x() < 30.0 || truth->y() < 30.0 ||
            truth->x() > 770.0 || truth->y() > 570.0) {
            continue;
        }
        Vec2 pixel = *truth;
        if (perturbation > 0.0) {
            const double distance =
                rng::uniform_real(engine, 1.0, perturbation);
            const double angle =
                rng::uniform_real(engine, 0.0, 2.0 * std::numbers::pi_v<double>);
            pixel += distance * Vec2(std::cos(angle), std::sin(angle));
        }
        marker.projections.push_back(
            {scene.chunk->cameras()[ci].id, pixel, true});
    }
    scene.chunk->markers().push_back(std::move(marker));
    return scene.chunk->markers().back().id;
}

Marker* find_marker(Chunk& chunk, std::int32_t marker_id) {
    for (Marker& marker : chunk.markers()) {
        if (marker.id == marker_id) {
            return &marker;
        }
    }
    return nullptr;
}

void set_projection(RefineScene& scene, std::int32_t marker_id,
                    std::int32_t camera_id, const Vec2& pixel) {
    Marker* marker = find_marker(*scene.chunk, marker_id);
    if (marker == nullptr) {
        return;
    }
    for (MarkerProjection& projection : marker->projections) {
        if (projection.camera_id == camera_id) {
            projection.pixel = pixel;
        }
    }
}

/// RMS distance between a marker's projections and the target centre's truth.
double rms_to_truth(const RefineScene& scene, const Marker& marker,
                    std::size_t target_index, double* maximum = nullptr) {
    double squared = 0.0;
    std::size_t count = 0;
    double worst = 0.0;
    for (const MarkerProjection& projection : marker.projections) {
        const auto truth =
            scene.truth(target_index, camera_index(scene, projection.camera_id));
        if (!truth) {
            continue;
        }
        const double error = (projection.pixel - *truth).norm();
        squared += error * error;
        worst = std::max(worst, error);
        ++count;
    }
    if (maximum != nullptr) {
        *maximum = worst;
    }
    return count == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(count));
}

/// Projected semi-major radius of the printed target in one camera, in
/// full-resolution pixels. Two ground-plane offsets span the target's ellipse,
/// so the larger projection approximates the detector's radius.
double apparent_radius(const RefineScene& scene, std::size_t target_index,
                       std::size_t camera_index) {
    const SceneTarget& target = scene.targets[target_index];
    const auto truth = scene.truth(target_index, camera_index);
    if (!truth) {
        return 0.0;
    }
    double best = 0.0;
    for (const Vec3& offset : {Vec3(target.radius, 0.0, 0.0),
                               Vec3(0.0, 0.0, target.radius)}) {
        const auto edge = project_marker(
            target.centre + offset, scene.camera(camera_index),
            scene.calibration(camera_index), false);
        if (edge) {
            best = std::max(best, (*edge - *truth).norm());
        }
    }
    return best;
}

RefineMarkersStats run_refine(RefineScene& scene,
                              const RefineMarkersOptions& options) {
    RefineMarkersPlan plan =
        plan_refine_markers(scene.document, scene.chunk->id(), options);
    const RefineMarkersStats stats = plan.stats;
    if (plan.command) {
        plan.command->apply(scene.document);
    }
    return stats;
}

std::vector<std::int32_t> add_all_markers(RefineScene& scene,
                                          double perturbation,
                                          std::mt19937& engine) {
    std::vector<std::int32_t> ids;
    for (std::size_t i = 0; i < scene.targets.size(); ++i) {
        ids.push_back(add_scene_marker(scene, i, "M" + std::to_string(i),
                                       perturbation, engine));
    }
    return ids;
}

TEST(Markers, RefineReducesError) {
    SceneSpec spec;
    spec.marker_count = 6;
    RefineScene scene = make_scene(spec, 7U);
    std::mt19937 engine(42U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 3.0, engine);

    std::vector<double> before;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        before.push_back(rms_to_truth(scene, *find_marker(*scene.chunk, ids[i]), i));
    }
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    double circular_squared = 0.0;
    double cross_squared = 0.0;
    int circular = 0;
    int cross = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const double after =
            rms_to_truth(scene, *find_marker(*scene.chunk, ids[i]), i);
        EXPECT_LE(after, 0.1) << "marker " << i << " RMS " << after;
        if (scene.targets[i].cross) {
            cross_squared += after * after;
            ++cross;
        } else {
            circular_squared += after * after;
            ++circular;
        }
    }
    const double circular_rms =
        std::sqrt(circular_squared / static_cast<double>(circular));
    const double cross_rms = std::sqrt(cross_squared / static_cast<double>(cross));
    double before_squared = 0.0;
    for (const double value : before) {
        before_squared += value * value;
    }
    const double before_rms =
        std::sqrt(before_squared / static_cast<double>(before.size()));
    std::printf(
        "[refine rms] before %.4f px, after circular %.4f px, "
        "cross %.4f px; refined %zu unchanged %zu failed %zu\n",
        before_rms, circular_rms, cross_rms, stats.refined, stats.unchanged,
        stats.failed);
    EXPECT_LT(circular_rms, 0.1);
    EXPECT_LT(cross_rms, 0.1);
}

TEST(Markers, RefineStableOnExactProjection) {
    SceneSpec spec;
    spec.marker_count = 6;
    RefineScene scene = make_scene(spec, 11U);
    std::mt19937 engine(3U);
    add_all_markers(scene, 0.0, engine);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    double worst = 0.0;
    for (const Marker& marker : scene.chunk->markers()) {
        const std::size_t target = static_cast<std::size_t>(
            std::stoul(marker.label.substr(1)));
        rms_to_truth(scene, marker, target, &worst);
    }
    std::printf("[refine stable] worst %.4f px, refined %zu\n",
                worst, stats.refined);
    EXPECT_LE(worst, 0.02);
    // Exact projections must be processed, not silently skipped.
    EXPECT_EQ(stats.skipped, 0U);
    EXPECT_EQ(stats.refined + stats.unchanged, scene.chunk->markers().size());
    EXPECT_GT(stats.refined + stats.unchanged, 0U);
}

TEST(Markers, RefineEvenCodeMarker) {
    // Parity-0 circular targets print a white centre, so the dark-dot blob
    // seed has nothing to find. The light-interior seed from the second pass
    // must carry the circular refiner instead.
    // A single printed target keeps every camera's crop around it free of a
    // neighbouring outline, so the parity-0 outer-border fit is unambiguous.
    SceneSpec spec;
    spec.marker_count = 1;
    spec.even_code_target = 0;
    RefineScene scene = make_scene(spec, 171U);
    ASSERT_EQ(scene.targets[0].parity, 0);
    std::mt19937 engine(83U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.failed, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine even] marker %zu parity %d RMS %.4f px\n", i,
                    scene.targets[i].parity, rms);
        // Parity-0 has no fiducial dot and keeps the border-ellipse centre;
        // the decoder documents a 0.38 px first-edge bias for such targets at
        // 2 px blur, several times smaller than the input perturbation here.
        // Half a pixel is the meaningful bound for this parity.
        EXPECT_LE(rms, 0.6) << "marker " << i;
    }
    EXPECT_GE(stats.refined, 1U);
}

TEST(Markers, RefineLargeTargetUsesBorderFit) {
    // A target whose projected radius is well over the 48 px minimum window:
    // the crop and blob scale must follow the apparent size, or the interior
    // is clipped and the border fit silently falls back to a biased centre.
    // The camera looks straight down because, for a strongly tilted target,
    // the image of the printed circle is an ellipse whose centre is a few
    // percent of R away from the projected circle centre (a documented
    // limitation of the border-ellipse centre); a single-view measurement
    // against the projected centre is only well posed at (near) normal
    // incidence, where the two coincide.
    SceneSpec spec;
    spec.marker_count = 1;
    spec.even_code_target = 0;
    spec.target_radius = 0.18;
    spec.ring_radius = 0.0;
    spec.height = 2.0;
    RefineScene scene = make_scene(spec, 191U);
    ASSERT_EQ(scene.targets[0].parity, 0);
    const double apparent = apparent_radius(scene, 0, 0);
    ASSERT_GT(apparent, 48.0);
    std::mt19937 engine(29U);
    // One projection keeps the 2-D path, so the reported pixel is the window's
    // own border-fit centre rather than a consensus of several views.
    const std::vector<std::int32_t> ids = {
        add_scene_marker(scene, 0, "M0", 0.0, engine,
                         std::vector<std::size_t>{0})};
    ASSERT_FALSE(ids.empty());
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.failed, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, 0);
        std::printf("[refine large] apparent %.2f px RMS %.4f px\n", apparent,
                    rms);
        EXPECT_LE(rms, 0.5) << "marker " << i;
    }
}

TEST(Markers, RefineIgnoresSpeckNearCentreDot) {
    // A 2 px dark speck inside the anchor gate must not beat the printed
    // centre dot: the dot nearest the seed with a plausible radius wins.
    SceneSpec spec;
    spec.marker_count = 1;
    spec.target_radius = 0.05;
    spec.speck_target = 0;
    spec.speck_offset_px = 8.0;
    RefineScene scene = make_scene(spec, 233U);
    ASSERT_EQ(scene.targets[0].parity, 1);
    std::mt19937 engine(97U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    ASSERT_FALSE(ids.empty());
    RefineMarkersOptions options;
    // A wider patch grows the anchor gate past the speck offset, so the test
    // exercises the choice between two candidate blobs.
    options.patch_size = 65;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.failed, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, 0);
        std::printf("[refine speck] RMS %.4f px\n", rms);
        EXPECT_LE(rms, 0.6) << "marker " << i;
    }
}

RefineScene make_contradicted_scene(std::vector<std::int32_t>& ids,
                                    std::int32_t& outlier_camera,
                                    double* displacement = nullptr,
                                    double* apparent = nullptr) {
    SceneSpec spec;
    spec.marker_count = 6;
    RefineScene scene = make_scene(spec, 21U);
    std::mt19937 engine(9U);
    ids = add_all_markers(scene, 3.0, engine);
    const auto truth = scene.truth(0, 0);
    outlier_camera = scene.chunk->cameras()[0].id;
    if (truth) {
        // Move along the more foreshortened axis, so 30 px is guaranteed to
        // leave the printed target and land on unrelated ground texture.
        const Vec3 centre_point = scene.targets[0].centre;
        const auto along_x = project_marker(
            centre_point + Vec3(0.075, 0.0, 0.0), scene.camera(0),
            scene.calibration(0), false);
        const auto along_z = project_marker(
            centre_point + Vec3(0.0, 0.0, 0.075), scene.camera(0),
            scene.calibration(0), false);
        Vec2 direction(1.0, 0.0);
        if (along_x && along_z) {
            const Vec2 dx = *along_x - *truth;
            const Vec2 dz = *along_z - *truth;
            if (dx.norm() > 0.0 && dz.norm() > 0.0) {
                direction = dx.norm() < dz.norm() ? dx.normalized()
                                                  : dz.normalized();
            }
        }
        // Three and a half apparent radii: unmistakably beyond the one-radius
        // gate, while still landing inside the image.
        const double radius = std::max(1.0, apparent_radius(scene, 0, 0));
        const double moved = 3.5 * radius;
        if (apparent != nullptr) {
            *apparent = radius;
        }
        if (displacement != nullptr) {
            *displacement = moved;
        }
        set_projection(scene, ids[0], outlier_camera, *truth + moved * direction);
    }
    return scene;
}

TEST(Markers, RefineRemovesContradictedProjection) {
    std::vector<std::int32_t> ids;
    std::int32_t outlier_camera = -1;
    double displacement = 0.0;
    double apparent = 0.0;
    {
        RefineScene scene =
            make_contradicted_scene(ids, outlier_camera, &displacement, &apparent);
        Marker* marker = find_marker(*scene.chunk, ids[0]);
        ASSERT_NE(marker, nullptr);
        const std::size_t before = marker->projections.size();
        ASSERT_GT(before, 0U);
        // The contradiction must clear the one-radius gate by a wide margin.
        EXPECT_GE(displacement, 3.0 * apparent);
        EXPECT_GT(displacement, std::max(3.0, apparent));
        RefineMarkersOptions options;
        run_refine(scene, options);
        marker = find_marker(*scene.chunk, ids[0]);
        EXPECT_EQ(marker->projections.size(), before - 1U);
        for (const MarkerProjection& projection : marker->projections) {
            EXPECT_NE(projection.camera_id, outlier_camera);
        }
        EXPECT_LE(rms_to_truth(scene, *marker, 0), 0.1);
    }
    {
        RefineScene scene = make_contradicted_scene(ids, outlier_camera);
        Marker* marker = find_marker(*scene.chunk, ids[0]);
        ASSERT_NE(marker, nullptr);
        Vec2 kept = Vec2::Zero();
        for (const MarkerProjection& projection : marker->projections) {
            if (projection.camera_id == outlier_camera) {
                kept = projection.pixel;
            }
        }
        const std::size_t before = marker->projections.size();
        RefineMarkersOptions options;
        options.remove_invalid = false;
        run_refine(scene, options);
        marker = find_marker(*scene.chunk, ids[0]);
        EXPECT_EQ(marker->projections.size(), before);
        for (const MarkerProjection& projection : marker->projections) {
            if (projection.camera_id == outlier_camera) {
                EXPECT_LE((projection.pixel - kept).norm(), 1e-12);
            } else {
                const auto truth = scene.truth(
                    0, camera_index(scene, projection.camera_id));
                ASSERT_TRUE(truth.has_value());
                EXPECT_LE((projection.pixel - *truth).norm(), 0.1);
            }
        }
    }
}

TEST(Markers, RefineSkipsLowTexture) {
    SceneSpec spec;
    spec.marker_count = 6;
    spec.include_blank_target = true;
    RefineScene scene = make_scene(spec, 31U);
    std::mt19937 engine(17U);
    std::vector<std::int32_t> ids = add_all_markers(scene, 3.0, engine);
    ASSERT_EQ(ids.size(), 7U);
    const std::vector<MarkerProjection> blank_before =
        find_marker(*scene.chunk, ids.back())->projections;
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_TRUE(find_marker(*scene.chunk, ids.back())->projections == blank_before);
    EXPECT_GE(stats.failed, 1U);
}

TEST(Markers, RefineObliqueViews) {
    SceneSpec spec;
    spec.marker_count = 3;
    spec.camera_count = 4;
    spec.ring_radius = 1.3;
    spec.height = 0.95;
    // Keep the targets near the optical axis so their rays really are at the
    // 50-60 degree obliquity this test is about; an off-centre target adds its
    // own angular offset on top of the camera tilt.
    spec.spread_x = 0.16;
    spec.spread_z = 0.0;
    RefineScene scene = make_scene(spec, 41U);
    for (const Transform& pose : scene.poses) {
        const Vec3 forward = pose.block<3, 3>(0, 0).row(2).transpose();
        const double obliquity =
            std::acos(std::clamp(std::abs(forward.dot(Vec3(0.0, 1.0, 0.0))),
                                 -1.0, 1.0)) *
            180.0 / std::numbers::pi_v<double>;
        EXPECT_GE(obliquity, 50.0);
        EXPECT_LE(obliquity, 60.0);
    }
    std::mt19937 engine(23U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 3.0, engine);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine oblique] marker %zu rms %.4f px, %zu projections\n",
                    i, rms, marker->projections.size());
        EXPECT_LE(rms, 0.15) << "marker " << i << " RMS " << rms;
    }
    std::printf(
        "[refine oblique] refined %zu unchanged %zu failed %zu removed %zu\n",
        stats.refined, stats.unchanged, stats.failed,
        stats.removed_projections);
    EXPECT_EQ(stats.failed, 0U);
}

TEST(Markers, Refine2DFallbackSingleProjection) {
    SceneSpec spec;
    spec.marker_count = 2;
    RefineScene scene = make_scene(spec, 51U);
    std::mt19937 engine(29U);
    const std::vector<std::int32_t> ids = {
        add_scene_marker(scene, 0, "A", 1.5, engine, {0}),
        add_scene_marker(scene, 1, "B", 1.5, engine, {1})};
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        ASSERT_EQ(marker->projections.size(), 1U);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine 2d fallback] marker %zu RMS %.4f px\n", i, rms);
        EXPECT_LE(rms, 0.5);
    }
    EXPECT_EQ(stats.removed_projections, 0U);
    EXPECT_EQ(stats.refined, 2U);
}

TEST(Markers, RefineIsUndoable) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 61U);
    std::mt19937 engine(31U);
    add_all_markers(scene, 2.0, engine);
    const std::vector<Marker> original = scene.chunk->markers();
    RefineMarkersOptions options;
    RefineMarkersPlan plan =
        plan_refine_markers(scene.document, scene.chunk->id(), options);
    ASSERT_TRUE(plan.command);
    UndoStack stack(&scene.document);
    stack.execute(std::move(plan.command));
    EXPECT_EQ(stack.size(), 1U);
    ASSERT_EQ(stack.command_names().size(), 1U);
    EXPECT_EQ(stack.command_names()[0], "Refine Markers");
    EXPECT_FALSE(scene.chunk->markers() == original);
    stack.undo();
    EXPECT_TRUE(scene.chunk->markers() == original);
    stack.redo();
    EXPECT_FALSE(scene.chunk->markers() == original);
}

TEST(Markers, RefineUndoRestoresRemovedProjection) {
    std::vector<std::int32_t> ids;
    std::int32_t outlier_camera = -1;
    RefineScene scene = make_contradicted_scene(ids, outlier_camera);
    const Marker* before = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(before, nullptr);
    const Marker original = *before;
    ASSERT_GE(original.projections.size(), 3U);

    RefineMarkersOptions options;
    RefineMarkersPlan plan =
        plan_refine_markers(scene.document, scene.chunk->id(), options);
    ASSERT_TRUE(plan.command);
    UndoStack stack(&scene.document);
    stack.execute(std::move(plan.command));
    const Marker* removed = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(removed, nullptr);
    ASSERT_EQ(removed->projections.size(), original.projections.size() - 1U);

    // The single staged snapshot restores order, enabled flags and metadata.
    stack.undo();
    const Marker* restored = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(*restored == original);
    EXPECT_TRUE(restored->projections == original.projections);

    stack.redo();
    EXPECT_EQ(find_marker(*scene.chunk, ids[0])->projections.size(),
              original.projections.size() - 1U);
}

TEST(Markers, RefineParamsFilter) {
    SceneSpec spec;
    spec.marker_count = 6;
    RefineScene scene = make_scene(spec, 71U);
    std::mt19937 engine(37U);
    add_all_markers(scene, 2.5, engine);
    const std::vector<Marker> original = scene.chunk->markers();
    RefineMarkersOptions options;
    options.markers = {"M1"};
    options.cameras = {scene.chunk->cameras()[0].label,
                       scene.chunk->cameras()[1].label};
    const RefineMarkersStats stats = run_refine(scene, options);
    ASSERT_EQ(scene.chunk->markers().size(), original.size());
    for (std::size_t m = 0; m < scene.chunk->markers().size(); ++m) {
        const Marker& marker = scene.chunk->markers()[m];
        const Marker& before = original[m];
        ASSERT_EQ(marker.projections.size(), before.projections.size());
        for (std::size_t p = 0; p < marker.projections.size(); ++p) {
            const bool selected_camera =
                marker.projections[p].camera_id == scene.chunk->cameras()[0].id ||
                marker.projections[p].camera_id == scene.chunk->cameras()[1].id;
            if (marker.label != "M1" || !selected_camera) {
                EXPECT_TRUE(marker.projections[p] == before.projections[p])
                    << "marker " << marker.label << " projection " << p;
            }
        }
    }
    EXPECT_EQ(stats.refined, 1U);
}

TEST(Markers, RefineIgnoresDisabledProjections) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 81U);
    std::mt19937 engine(43U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    ASSERT_GE(marker->projections.size(), 2U);
    // A grossly wrong disabled measurement must stay exactly as it was: not
    // used for the consensus, not moved, not removed.
    marker->projections.back().enabled = false;
    marker->projections.back().pixel += Vec2(120.0, 90.0);
    const std::vector<MarkerProjection> before = marker->projections;

    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);

    marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    ASSERT_EQ(marker->projections.size(), before.size());
    for (std::size_t p = 0; p < marker->projections.size(); ++p) {
        if (!before[p].enabled) {
            EXPECT_TRUE(marker->projections[p] == before[p])
                << "projection " << p << " was touched";
        }
    }
    EXPECT_GE(stats.refined, 1U);
}

TEST(Markers, RefineRejectsNonFiniteInput) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 91U);
    std::mt19937 engine(47U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    ASSERT_GE(marker->projections.size(), 3U);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    // Two unusable measurements and two unusable tie points: none of them may
    // reach a floor/lround cast or a sort comparator.
    marker->projections[0].pixel = Vec2(nan, 5.0);
    marker->projections[1].pixel = Vec2(inf, 7.0);
    const std::vector<MarkerProjection> before = marker->projections;
    scene.chunk->point_cloud().points.emplace_back(nan, nan, nan);
    scene.chunk->point_cloud().points.emplace_back(inf, 0.0, 0.0);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    ASSERT_EQ(marker->projections.size(), before.size());
    // NaN never compares equal to itself, so compare the fields that prove
    // the measurement was neither moved nor replaced.
    EXPECT_EQ(marker->projections[0].camera_id, before[0].camera_id);
    EXPECT_EQ(marker->projections[1].camera_id, before[1].camera_id);
    EXPECT_EQ(marker->projections[0].enabled, before[0].enabled);
    EXPECT_EQ(marker->projections[1].enabled, before[1].enabled);
    EXPECT_FALSE(std::isfinite(marker->projections[0].pixel.x()));
    EXPECT_EQ(marker->projections[0].pixel.y(), 5.0);
    EXPECT_FALSE(std::isfinite(marker->projections[1].pixel.x()));
    EXPECT_EQ(marker->projections[1].pixel.y(), 7.0);
    EXPECT_GE(stats.refined + stats.unchanged, 1U);
}

TEST(Markers, RefineStatsCountEverySelectedMarkerOnce) {
    SceneSpec spec;
    spec.marker_count = 4;
    spec.include_blank_target = true;
    RefineScene scene = make_scene(spec, 111U);
    std::mt19937 engine(59U);
    add_all_markers(scene, 3.0, engine);
    // A single-projection marker takes the 2-D fallback; a marker with no
    // projection at all is skipped, not failed.
    add_scene_marker(scene, 0, "Single", 1.5, engine, {0});
    Marker empty;
    empty.id = scene.chunk->allocate_id();
    empty.label = "Empty";
    empty.type = MarkerType::Fiducial;
    scene.chunk->markers().push_back(std::move(empty));
    const std::size_t selected = scene.chunk->markers().size();

    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.refined + stats.unchanged + stats.failed + stats.skipped,
              selected);
    EXPECT_GE(stats.skipped, 1U);
}

TEST(Markers, RefineSkipsUnreadableCameraUnlessExplicit) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 131U);
    std::mt19937 engine(61U);
    add_all_markers(scene, 2.0, engine);
    ASSERT_FALSE(scene.chunk->frames().empty());
    Frame* broken = scene.chunk->frames().front().get();
    ASSERT_NE(broken, nullptr);
    broken->set_path(broken->path() + ".missing");

    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.unreadable_cameras, 1U);
    EXPECT_EQ(stats.failed, 0U);

    // An explicit camera list must not silently drop the broken view.
    options.cameras = {scene.chunk->cameras()[0].label};
    EXPECT_THROW(plan_refine_markers(scene.document, scene.chunk->id(), options),
                 Error);
}

TEST(Markers, RefineAcceptsCandidateWhenAViewPatchLeavesTheImage) {
    SceneSpec spec;
    spec.marker_count = 1;
    spec.camera_count = 4;
    spec.ring_radius = 1.2;
    spec.height = 0.8;
    spec.spread_x = 0.5;
    spec.spread_z = 0.32;
    RefineScene scene = make_scene(spec, 101U);
    std::mt19937 engine(53U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    ASSERT_EQ(marker->projections.size(), 4U);
    // The last ring camera sees the target 33 px above the bottom edge while
    // the sampled patch extends past it: that view is a non-contributor.
    const std::int32_t border_camera = scene.chunk->cameras()[3].id;
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    EXPECT_EQ(stats.failed, 0U);
    EXPECT_EQ(marker->projections.size(), 4U);
    for (const MarkerProjection& projection : marker->projections) {
        if (projection.camera_id == border_camera) {
            continue;  // non-contributor: never a reason to remove it
        }
        const auto truth =
            scene.truth(0, camera_index(scene, projection.camera_id));
        ASSERT_TRUE(truth.has_value());
        // Three usable views carry less redundancy than the four-to-six view
        // scenes, so this new test keeps a looser bound.
        EXPECT_LE((projection.pixel - *truth).norm(), 0.2)
            << "camera " << projection.camera_id;
    }
}

TEST(Markers, RefineUpdatesEstimatedPosition) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 151U);
    std::mt19937 engine(73U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.failed, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        ASSERT_TRUE(marker->position.has_value());
        EXPECT_LE((*marker->position - scene.targets[i].centre).norm(), 0.05)
            << "marker " << i;
        // The geodetic reference is not invented or modified.
        EXPECT_FALSE(marker->reference.has_value());
    }
}

TEST(Markers, RefineDownscaleTwoStaysAccurate) {
    SceneSpec spec;
    spec.marker_count = 6;
    RefineScene scene = make_scene(spec, 121U);
    std::mt19937 engine(67U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    RefineMarkersOptions options;
    options.downscale = 2;
    run_refine(scene, options);
    double worst = 0.0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        worst = std::max(worst, rms_to_truth(scene, *marker, i));
    }
    std::printf("[refine downscale2] worst %.4f px at full resolution\n", worst);
    EXPECT_LE(worst, 0.15);
}

TEST(Markers, RefineWithoutTiePointsUsesFallbackRange) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 141U);
    std::mt19937 engine(71U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    // Neither a dense cloud nor marker positions: the symmetric +/-search_steps
    // window is the only depth range.
    scene.chunk->point_cloud() = PointCloud{};
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.failed, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine no cloud] marker %zu RMS %.4f px\n", i, rms);
        EXPECT_LE(rms, 0.2);
    }
}

}  // namespace
}  // namespace alphashape
