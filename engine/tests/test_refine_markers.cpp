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
    truth.parity = 1;
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
        target.radius = 0.075;
        target.rotation = 0.17 * static_cast<double>(i);
        target.word = 0b101101001011U ^ (static_cast<std::uint32_t>(i) << 2);
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

RefineMarkersStats run_refine(RefineScene& scene,
                              const RefineMarkersOptions& options) {
    RefineMarkersPlan plan =
        plan_refine_markers(scene.document, scene.chunk->id(), options);
    const RefineMarkersStats stats = plan.stats;
    if (plan.command) {
        plan.command->apply(scene.document);
    }
    std::printf(
        "[refine stats] refined %zu unchanged %zu failed %zu skipped %zu "
        "removed %zu rounds %zu peak %zu bytes (%.2f MiB)\n",
        stats.refined, stats.unchanged, stats.failed, stats.skipped,
        stats.removed_projections, stats.rounds, stats.peak_bytes,
        static_cast<double>(stats.peak_bytes) / (1024.0 * 1024.0));
    return stats;
}

/// Apparent radius of a target in one camera, in full-resolution pixels, from
/// the projection of the radius along the two ground-plane axes.
double apparent_radius_px(const RefineScene& scene, std::size_t target_index,
                          std::size_t camera_index) {
    const auto truth = scene.truth(target_index, camera_index);
    if (!truth) {
        return 0.0;
    }
    const Vec3 centre = scene.targets[target_index].centre;
    const double radius = scene.targets[target_index].radius;
    double best = 0.0;
    for (const Vec3& axis : {Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)}) {
        const auto offset = project_marker(centre + radius * axis,
                                           scene.camera(camera_index),
                                           scene.calibration(camera_index),
                                           false);
        if (offset) {
            best = std::max(best, (*offset - *truth).norm());
        }
    }
    return best;
}

void set_projection_enabled(RefineScene& scene, std::int32_t marker_id,
                            std::int32_t camera_id, bool enabled) {
    Marker* marker = find_marker(*scene.chunk, marker_id);
    if (marker == nullptr) {
        return;
    }
    for (MarkerProjection& projection : marker->projections) {
        if (projection.camera_id == camera_id) {
            projection.enabled = enabled;
        }
    }
}

/// RMS over a marker's projections excluding one camera.
double rms_to_truth_excluding(const RefineScene& scene, const Marker& marker,
                              std::size_t target_index,
                              std::int32_t excluded_camera) {
    double squared = 0.0;
    std::size_t count = 0;
    for (const MarkerProjection& projection : marker.projections) {
        if (projection.camera_id == excluded_camera) {
            continue;
        }
        const auto truth =
            scene.truth(target_index, camera_index(scene, projection.camera_id));
        if (!truth) {
            continue;
        }
        const double error = (projection.pixel - *truth).norm();
        squared += error * error;
        ++count;
    }
    return count == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(count));
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
        "cross %.4f px; refined %zu unchanged %zu failed %zu skipped %zu "
        "rounds %zu peak %zu bytes\n",
        before_rms, circular_rms, cross_rms, stats.refined, stats.unchanged,
        stats.failed, stats.skipped, stats.rounds, stats.peak_bytes);
    EXPECT_LT(circular_rms, 0.1);
    EXPECT_LT(cross_rms, 0.1);
    EXPECT_EQ(stats.refined + stats.unchanged + stats.failed + stats.skipped,
              scene.chunk->markers().size());
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
        double marker_worst = 0.0;
        const double rms = rms_to_truth(scene, marker, target, &marker_worst);
        worst = std::max(worst, marker_worst);
        std::printf("[refine stable] marker %zu rms %.4f worst %.4f px\n",
                    target, rms, marker_worst);
    }
    std::printf("[refine stable] worst %.4f px, refined %zu rounds %zu peak "
                "%zu bytes\n",
                worst, stats.refined, stats.rounds, stats.peak_bytes);
    EXPECT_LE(worst, 0.02);
    EXPECT_EQ(stats.failed, 0U);
    EXPECT_EQ(stats.skipped, 0U);
    EXPECT_EQ(stats.refined + stats.unchanged, scene.chunk->markers().size());
    EXPECT_EQ(stats.refined + stats.unchanged + stats.failed + stats.skipped,
              scene.chunk->markers().size());
}

/// Build a scene whose first projection is displaced by three apparent radii
/// (plus margin), well beyond the one-radius gate, and report both the applied
/// offset and the gate so the test can assert the relationship.
RefineScene make_contradicted_scene(std::vector<std::int32_t>& ids,
                                    std::int32_t& outlier_camera,
                                    double& applied_offset, double& gate) {
    SceneSpec spec;
    spec.marker_count = 6;
    RefineScene scene = make_scene(spec, 21U);
    std::mt19937 engine(9U);
    ids = add_all_markers(scene, 3.0, engine);
    const auto truth = scene.truth(0, 0);
    outlier_camera = scene.chunk->cameras()[0].id;
    const double radius = apparent_radius_px(scene, 0, 0);
    gate = std::max(3.0, radius);
    applied_offset = 3.0 * radius + 6.0;
    if (truth) {
        // Move along the more foreshortened axis, so the displacement leaves
        // the printed target and lands on unrelated ground texture.
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
        Vec2 outlier = *truth + applied_offset * direction;
        if (outlier.x() < 2.0 || outlier.x() > 798.0 || outlier.y() < 2.0 ||
            outlier.y() > 598.0) {
            Vec2 inward = Vec2(400.0, 300.0) - *truth;
            if (inward.norm() > 1e-9) {
                inward.normalize();
                outlier = *truth + applied_offset * inward;
            }
        }
        set_projection(scene, ids[0], outlier_camera, outlier);
    }
    return scene;
}

TEST(Markers, RefineRemovesContradictedProjection) {
    std::vector<std::int32_t> ids;
    std::int32_t outlier_camera = -1;
    double applied_offset = 0.0;
    double gate = 0.0;
    {
        RefineScene scene =
            make_contradicted_scene(ids, outlier_camera, applied_offset, gate);
        EXPECT_GT(applied_offset, gate)
            << "the test displacement must exceed the projection gate";
        Marker* marker = find_marker(*scene.chunk, ids[0]);
        ASSERT_NE(marker, nullptr);
        const std::size_t before = marker->projections.size();
        ASSERT_GT(before, 0U);
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
        RefineScene scene =
            make_contradicted_scene(ids, outlier_camera, applied_offset, gate);
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

TEST(Markers, RefineAtDownscaleTwo) {
    SceneSpec spec;
    spec.marker_count = 4;
    RefineScene scene = make_scene(spec, 81U);
    std::mt19937 engine(43U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    RefineMarkersOptions options;
    options.downscale = 2;
    const RefineMarkersStats stats = run_refine(scene, options);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine downscale2] marker %zu rms %.4f px\n", i, rms);
        EXPECT_LE(rms, 0.15) << "marker " << i << " RMS " << rms;
    }
    EXPECT_EQ(stats.failed, 0U);
}

TEST(Markers, RefineWithoutTiePointsUsesFallbackRange) {
    SceneSpec spec;
    spec.marker_count = 4;
    // Keep every target away from the image border so its printed disc is
    // fully visible in every view; this test is about the depth range.
    spec.spread_x = 0.16;
    spec.spread_z = 0.16;
    RefineScene scene = make_scene(spec, 82U);
    scene.chunk->point_cloud().points.clear();
    std::mt19937 engine(47U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine no-tie-points] marker %zu rms %.4f px\n", i, rms);
        EXPECT_LE(rms, 0.3) << "marker " << i << " RMS " << rms;
    }
    EXPECT_EQ(stats.failed, 0U);
    EXPECT_EQ(stats.skipped, 0U);
}

TEST(Markers, RefineUndoRestoresRemovedProjectionExactly) {
    std::vector<std::int32_t> ids;
    std::int32_t outlier_camera = -1;
    double applied_offset = 0.0;
    double gate = 0.0;
    RefineScene scene =
        make_contradicted_scene(ids, outlier_camera, applied_offset, gate);
    const std::vector<Marker> original = scene.chunk->markers();
    RefineMarkersOptions options;
    RefineMarkersPlan plan =
        plan_refine_markers(scene.document, scene.chunk->id(), options);
    ASSERT_TRUE(plan.command);
    UndoStack stack(&scene.document);
    stack.execute(std::move(plan.command));
    Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    const std::size_t removed_size = marker->projections.size();
    EXPECT_EQ(removed_size + 1U, original[0].projections.size());
    stack.undo();
    EXPECT_TRUE(scene.chunk->markers() == original)
        << "undo must restore every projection, in order";
    stack.redo();
    marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    EXPECT_EQ(marker->projections.size(), removed_size);
}

TEST(Markers, RefineNeverTouchesDisabledProjections) {
    SceneSpec spec;
    spec.marker_count = 3;
    RefineScene scene = make_scene(spec, 83U);
    std::mt19937 engine(53U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    const std::int32_t disabled_camera = scene.chunk->cameras()[1].id;
    Vec2 disabled_pixel = Vec2::Zero();
    bool found_disabled = false;
    for (const std::int32_t id : ids) {
        set_projection_enabled(scene, id, disabled_camera, false);
    }
    for (const MarkerProjection& projection :
         find_marker(*scene.chunk, ids[0])->projections) {
        if (projection.camera_id == disabled_camera) {
            disabled_pixel = projection.pixel;
            found_disabled = true;
        }
    }
    ASSERT_TRUE(found_disabled);
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    for (const std::int32_t id : ids) {
        const Marker* marker = find_marker(*scene.chunk, id);
        ASSERT_NE(marker, nullptr);
        const MarkerProjection* projection = nullptr;
        for (const MarkerProjection& candidate : marker->projections) {
            if (candidate.camera_id == disabled_camera) {
                projection = &candidate;
            }
        }
        ASSERT_NE(projection, nullptr) << "a disabled projection was removed";
        EXPECT_FALSE(projection->enabled);
        if (id == ids[0]) {
            EXPECT_LE((projection->pixel - disabled_pixel).norm(), 1e-12);
        }
    }
    EXPECT_EQ(stats.refined + stats.unchanged + stats.failed + stats.skipped,
              scene.chunk->markers().size());
}

TEST(Markers, RefineBorderViewIsNonContributor) {
    SceneSpec spec;
    spec.marker_count = 2;
    // Six cameras give the marker enough well-centred views to triangulate
    // and refine while one projection sits at the image border.
    spec.camera_count = 6;
    RefineScene scene = make_scene(spec, 84U);
    std::mt19937 engine(59U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 1.0, engine);
    const auto truth = scene.truth(0, 0);
    ASSERT_TRUE(truth.has_value());
    const std::int32_t border_camera = scene.chunk->cameras()[0].id;
    const int width = scene.calibration(0).width;
    const bool left_is_far = truth->x() >= 0.5 * width;
    const Vec2 border = left_is_far
                            ? Vec2(2.0, truth->y())
                            : Vec2(static_cast<double>(width - 3), truth->y());
    EXPECT_GT((border - *truth).norm(), 256.0);
    set_projection(scene, ids[0], border_camera, border);
    RefineMarkersOptions options;
    options.markers = {"M0"};
    const RefineMarkersStats stats = run_refine(scene, options);
    const Marker* marker = find_marker(*scene.chunk, ids[0]);
    ASSERT_NE(marker, nullptr);
    const MarkerProjection* border_projection = nullptr;
    for (const MarkerProjection& projection : marker->projections) {
        if (projection.camera_id == border_camera) {
            border_projection = &projection;
        }
    }
    ASSERT_NE(border_projection, nullptr)
        << "an out-of-window projection must not be removed";
    EXPECT_LE((border_projection->pixel - border).norm(), 1e-12);
    EXPECT_GE(marker->projections.size(), 3U);
    const double rms =
        rms_to_truth_excluding(scene, *marker, 0, border_camera);
    std::printf("[refine border] remaining rms %.4f px, %zu projections\n", rms,
                marker->projections.size());
    EXPECT_LE(rms, 0.2);
    EXPECT_EQ(stats.failed, 0U);
}

TEST(Markers, RefineRemovesCommonModeLateralBias) {
    SceneSpec spec;
    spec.marker_count = 4;
    RefineScene scene = make_scene(spec, 85U);
    std::mt19937 engine(61U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 0.0, engine);
    // The same 3-D displacement in the target plane, seen by every camera: a
    // 0.6-radius lateral error that no single view can detect.
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Vec3 displaced =
            scene.targets[i].centre +
            Vec3(0.6 * scene.targets[i].radius, 0.0, 0.0);
        for (const MarkerProjection& projection :
             find_marker(*scene.chunk, ids[i])->projections) {
            const std::size_t ci = camera_index(scene, projection.camera_id);
            const auto pixel = project_marker(displaced, scene.camera(ci),
                                              scene.calibration(ci), false);
            if (pixel) {
                set_projection(scene, ids[i], projection.camera_id, *pixel);
            }
        }
    }
    RefineMarkersOptions options;
    // Cross targets carry an unambiguous absolute anchor, so they isolate the
    // common-mode behaviour of the rounds from target-detection noise.
    options.markers = {"M1", "M3"};
    const RefineMarkersStats stats = run_refine(scene, options);
    for (const std::size_t i : {1U, 3U}) {
        const Marker* marker = find_marker(*scene.chunk, ids[i]);
        ASSERT_NE(marker, nullptr);
        const double rms = rms_to_truth(scene, *marker, i);
        std::printf("[refine common-mode] marker %zu rms %.4f px\n", i, rms);
        EXPECT_LE(rms, 0.1) << "marker " << i << " RMS " << rms;
    }
    EXPECT_EQ(stats.failed, 0U);
    EXPECT_GT(stats.rounds, 0U);
}

TEST(Markers, RefineStatsCountEverySelectedMarkerOnce) {
    SceneSpec spec;
    spec.marker_count = 4;
    RefineScene scene = make_scene(spec, 86U);
    std::mt19937 engine(67U);
    const std::vector<std::int32_t> ids = add_all_markers(scene, 2.0, engine);
    for (const MarkerProjection& projection :
         find_marker(*scene.chunk, ids[2])->projections) {
        set_projection_enabled(scene, ids[2], projection.camera_id, false);
    }
    RefineMarkersOptions options;
    const RefineMarkersStats stats = run_refine(scene, options);
    EXPECT_EQ(stats.skipped, 1U);
    EXPECT_EQ(stats.refined + stats.unchanged + stats.failed + stats.skipped,
              scene.chunk->markers().size());
}

}  // namespace
}  // namespace alphashape
