#include "apps/gui/viewer/scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "apps/gui/theme/viewport_colors.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/mesh_ops.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape::gui {

namespace {

/// "#RRGGBB" from the generated token header -> sRGB components in 0..1.
Eigen::Vector3f rgb_of(const char* hex) {
    // The tokens are generated from tokens.json, which the theme test already validates as
    // #RRGGBB; the defensive default keeps a malformed value from reading past the string.
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return 0;
    };
    const std::size_t length = hex != nullptr ? std::char_traits<char>::length(hex) : 0U;
    if (length < 7U || hex[0] != '#') {
        return Eigen::Vector3f::Ones();
    }
    const int r = nibble(hex[1]) * 16 + nibble(hex[2]);
    const int g = nibble(hex[3]) * 16 + nibble(hex[4]);
    const int b = nibble(hex[5]) * 16 + nibble(hex[6]);
    return Eigen::Vector3f(static_cast<float>(r) / 255.0F, static_cast<float>(g) / 255.0F,
                           static_cast<float>(b) / 255.0F);
}

Eigen::Vector3f to_float3(const Eigen::Vector3d& value) {
    return Eigen::Vector3f(static_cast<float>(value.x()), static_cast<float>(value.y()),
                           static_cast<float>(value.z()));
}

void add_line(std::vector<SceneLineVertex>& lines, std::vector<LineKind>& kinds,
              std::vector<std::int32_t>& cameras, const Eigen::Vector3d& a,
              const Eigen::Vector3d& b, const Eigen::Vector3f& colour, LineKind kind,
              std::int32_t camera_id = -1) {
    lines.push_back({to_float3(a), colour});
    lines.push_back({to_float3(b), colour});
    kinds.push_back(kind);
    kinds.push_back(kind);
    cameras.push_back(camera_id);
    cameras.push_back(camera_id);
}

/// Nice grid spacing: 1, 2 or 5 times a power of ten, sized so the grid has roughly
/// `target_divisions` lines across the scene.
double nice_step(double extent, int target_divisions) {
    if (!(extent > 0.0)) {
        return 1.0;
    }
    const double raw = extent / static_cast<double>(target_divisions);
    const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
    for (const double multiple : {1.0, 2.0, 5.0, 10.0}) {
        if (raw <= multiple * magnitude) {
            return multiple * magnitude;
        }
    }
    return 10.0 * magnitude;
}

struct Bounds {
    Eigen::Vector3d min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

    void add(const Eigen::Vector3d& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }
    bool valid() const { return min.allFinite() && max.allFinite() && (min.array() <= max.array()).all(); }
};

/// A camera's centre and the four image-corner directions (chunk-local), or nullopt when the
/// pose is still the identity (not aligned) or degenerate.
struct Frustum {
    Eigen::Vector3d apex = Eigen::Vector3d::Zero();
    std::array<Eigen::Vector3d, 4> corners{};  ///< TL, TR, BR, BL in chunk-local units
};

const Sensor* sensor_for(const Chunk& chunk, const Camera& camera) {
    if (camera.sensor_id < 0 ||
        static_cast<std::size_t>(camera.sensor_id) >= chunk.sensors().size()) {
        return nullptr;
    }
    return &chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)];
}

std::optional<Frustum> frustum_of(const Chunk& chunk, const Camera& camera, double depth) {
    if (camera.pose == Transform::Identity()) {
        return std::nullopt;
    }
    const Eigen::Matrix3d rotation = camera.pose.topLeftCorner<3, 3>();
    const Eigen::Vector3d translation = camera.pose.topRightCorner<3, 1>();
    const Eigen::Matrix3d rotation_t = rotation.transpose();

    // A camera pixel -> a unit ray in the camera frame (Calibration::unproject), then to the
    // chunk frame: p_chunk = R^T (p_cam - t), because the stored pose is world -> camera.
    const Sensor* sensor = sensor_for(chunk, camera);
    const bool sensor_calibrated =
        sensor != nullptr && sensor->calibration.f > 0.0 && sensor->calibration.width > 0 &&
        sensor->calibration.height > 0;
    const int sensor_width = sensor != nullptr && sensor->width > 0 ? sensor->width : 1500;
    const int sensor_height = sensor != nullptr && sensor->height > 0 ? sensor->height : 1000;
    const Calibration fallback =
        Calibration::initial(CameraType::Frame, sensor_width, sensor_height,
                             1.2 * static_cast<double>(sensor_width));
    const Calibration& calibration =
        sensor_calibrated ? sensor->calibration : fallback;
    // The corner pixels must come from the same geometry the rays are unprojected with.
    const double width = static_cast<double>(calibration.width);
    const double height = static_cast<double>(calibration.height);

    Frustum frustum;
    frustum.apex = rotation_t * (-translation);
    const std::array<Eigen::Vector2d, 4> pixels = {
        Eigen::Vector2d(0.0, 0.0),
        Eigen::Vector2d(width, 0.0),
        Eigen::Vector2d(width, height),
        Eigen::Vector2d(0.0, height),
    };
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const Eigen::Vector3d ray = calibration.unproject(pixels[i]);
        if (!ray.allFinite() || ray.norm() <= 0.0) {
            return std::nullopt;
        }
        frustum.corners[i] = frustum.apex + rotation_t * (ray.normalized() * depth);
    }
    return frustum;
}

/// True when all three corners of `face` are real vertices.
bool valid_face(const Eigen::Vector3i& face, std::size_t vertex_count) {
    return face.x() >= 0 && face.y() >= 0 && face.z() >= 0 &&
           static_cast<std::size_t>(face.x()) < vertex_count &&
           static_cast<std::size_t>(face.y()) < vertex_count &&
           static_cast<std::size_t>(face.z()) < vertex_count;
}

/// The size the display copy of `page` gets: the page itself, or a box-filtered downscale that
/// fits SceneTexture::kMaxTextureSize on both sides. A preview larger than the cap costs GPU
/// memory without adding detail at any reasonable zoom level.
Eigen::Vector2i display_size_of(const Image& page) {
    const int longest = std::max(page.width(), page.height());
    if (longest <= SceneTexture::kMaxTextureSize) {
        return Eigen::Vector2i(page.width(), page.height());
    }
    const double scale = static_cast<double>(SceneTexture::kMaxTextureSize) /
                         static_cast<double>(longest);
    const auto scaled = [&](int value) {
        return std::max(1, static_cast<int>(std::lround(static_cast<double>(value) * scale)));
    };
    return Eigen::Vector2i(scaled(page.width()), scaled(page.height()));
}

/// Expand `page` (1 or 3 channels) to RGBA8 and, if it is oversized, downscale it to
/// SceneTexture::kMaxTextureSize with a box filter. Copies whole rows where it can: the source
/// is tightly packed, so the common (no-downscale) case is one row loop over plain memcpy-ish
/// channel expansion rather than a per-pixel accessor call.
SceneTexture copy_page(const Image& page) {
    const Eigen::Vector2i size = display_size_of(page);
    SceneTexture texture;
    texture.width = size.x();
    texture.height = size.y();
    texture.rgba.resize(static_cast<std::size_t>(texture.width) *
                        static_cast<std::size_t>(texture.height) * 4U);

    const int channels = page.channels();
    const std::uint8_t* source = page.data();

    if (size.x() == page.width() && size.y() == page.height()) {
        // No resample: expand one row at a time from the tightly packed source, so the common
        // path never calls the per-pixel accessor or walks a pixel twice.
        for (int y = 0; y < texture.height; ++y) {
            const std::uint8_t* in = source + static_cast<std::size_t>(y) *
                                                  static_cast<std::size_t>(page.width()) *
                                                  static_cast<std::size_t>(channels);
            std::uint8_t* out = texture.rgba.data() +
                                static_cast<std::size_t>(y) *
                                    static_cast<std::size_t>(texture.width) * 4U;
            for (int x = 0; x < texture.width; ++x) {
                if (channels >= 3) {
                    std::memcpy(out, in, 3U);
                } else {
                    out[0] = in[0];
                    out[1] = in[0];
                    out[2] = in[0];
                }
                out[3] = 255;
                out += 4;
                in += channels;
            }
        }
        return texture;
    }

    // Box filter: average every source pixel of the block that maps to one output texel.
    for (int y = 0; y < texture.height; ++y) {
        const int y0 = static_cast<int>(static_cast<long long>(y) * page.height() /
                                        texture.height);
        int y1 = static_cast<int>(static_cast<long long>(y + 1) * page.height() /
                                  texture.height);
        y1 = std::clamp(y1, y0 + 1, page.height());
        for (int x = 0; x < texture.width; ++x) {
            const int x0 = static_cast<int>(static_cast<long long>(x) * page.width() /
                                            texture.width);
            int x1 = static_cast<int>(static_cast<long long>(x + 1) * page.width() /
                                      texture.width);
            x1 = std::clamp(x1, x0 + 1, page.width());
            unsigned sum[3] = {0U, 0U, 0U};
            unsigned count = 0U;
            for (int sy = y0; sy < y1; ++sy) {
                const std::uint8_t* row = source +
                                          static_cast<std::size_t>(sy) *
                                              static_cast<std::size_t>(page.width()) *
                                              static_cast<std::size_t>(channels);
                for (int sx = x0; sx < x1; ++sx) {
                    const std::uint8_t* texel = row + static_cast<std::size_t>(sx) *
                                                          static_cast<std::size_t>(channels);
                    if (channels >= 3) {
                        sum[0] += texel[0];
                        sum[1] += texel[1];
                        sum[2] += texel[2];
                    } else {
                        sum[0] += texel[0];
                        sum[1] += texel[0];
                        sum[2] += texel[0];
                    }
                    ++count;
                }
            }
            std::uint8_t* out = texture.rgba.data() +
                                (static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(texture.width) +
                                 static_cast<std::size_t>(x)) *
                                    4U;
            const unsigned divisor = std::max(count, 1U);
            out[0] = static_cast<std::uint8_t>((sum[0] + divisor / 2U) / divisor);
            out[1] = static_cast<std::uint8_t>((sum[1] + divisor / 2U) / divisor);
            out[2] = static_cast<std::uint8_t>((sum[2] + divisor / 2U) / divisor);
            out[3] = 255;
        }
    }
    return texture;
}

/// Copy the mesh into the scene. Textured meshes emit one vertex per face corner (the uv pool
/// is per corner), untextured meshes share vertices through the index buffer.
void add_mesh(const Chunk& chunk, const Mesh& mesh, const SceneStyle& style, Scene& scene) {
    if (mesh.empty() || mesh.vertices.empty()) {
        return;
    }
    const std::vector<Eigen::Vector3f> normals = vertex_normals(mesh);
    const bool has_vertex_colours = mesh.colours.size() == mesh.vertices.size() * 3U;
    const bool has_corner_uvs = !mesh.uvs.empty() && mesh.uv_faces.size() == mesh.faces.size();

    const Image* page = nullptr;
    if (has_corner_uvs && !chunk.model_textures().empty() &&
        !chunk.model_textures().front().empty()) {
        page = &chunk.model_textures().front();
    }

    if (page != nullptr) {
        scene.texture = copy_page(*page);
        scene.textured = true;
    }
    // A textured mesh is drawn from its flat grey times the page; the vertex colours are not
    // used, so a theme change has to re-derive the grey.
    scene.style_colours_mesh = scene.textured || !has_vertex_colours;

    const auto colour_of = [&](std::size_t vertex) -> Eigen::Vector3f {
        if (scene.textured || !has_vertex_colours) {
            return style.mesh_base;
        }
        const std::size_t base = vertex * 3U;
        return Eigen::Vector3f(static_cast<float>(mesh.colours[base + 0]) / 255.0F,
                               static_cast<float>(mesh.colours[base + 1]) / 255.0F,
                               static_cast<float>(mesh.colours[base + 2]) / 255.0F);
    };
    const auto normal_of = [&](std::size_t vertex) -> Eigen::Vector3f {
        if (vertex < normals.size() && normals[vertex].allFinite() &&
            normals[vertex].squaredNorm() > 1e-12F) {
            return normals[vertex];
        }
        return Eigen::Vector3f::UnitZ().eval();
    };

    if (scene.textured) {
        scene.mesh_vertices.reserve(mesh.faces.size() * 3U);
        scene.mesh_indices.reserve(mesh.faces.size() * 3U);
        for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
            const Eigen::Vector3i& face = mesh.faces[face_index];
            const Eigen::Vector3i& uv_face = mesh.uv_faces[face_index];
            if (!valid_face(face, mesh.vertices.size())) {
                // A malformed face is dropped whole rather than read out of bounds or emitted as
                // a partial triangle; Chunk::validate() reports such a chunk separately.
                continue;
            }
            for (int corner = 0; corner < 3; ++corner) {
                const int vertex = face[corner];
                const int uv_index = uv_face[corner];
                SceneMeshVertex output;
                output.position = to_float3(mesh.vertices[static_cast<std::size_t>(vertex)]);
                output.normal = normal_of(static_cast<std::size_t>(vertex));
                output.colour = colour_of(static_cast<std::size_t>(vertex));
                if (uv_index >= 0 && static_cast<std::size_t>(uv_index) < mesh.uvs.size()) {
                    output.uv = mesh.uvs[static_cast<std::size_t>(uv_index)];
                }
                scene.mesh_indices.push_back(static_cast<std::uint32_t>(scene.mesh_vertices.size()));
                scene.mesh_vertices.push_back(output);
            }
        }
        return;
    }

    scene.mesh_vertices.reserve(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        SceneMeshVertex vertex;
        vertex.position = to_float3(mesh.vertices[i]);
        vertex.normal = normal_of(i);
        vertex.colour = colour_of(i);
        scene.mesh_vertices.push_back(vertex);
    }
    scene.mesh_indices.reserve(mesh.faces.size() * 3U);
    for (const Eigen::Vector3i& face : mesh.faces) {
        if (!valid_face(face, mesh.vertices.size())) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            scene.mesh_indices.push_back(static_cast<std::uint32_t>(face[corner]));
        }
    }
}

/// Every `stride`-th element, so a huge cloud still uploads a representative subset.
std::size_t stride_for(std::size_t count, std::size_t limit) {
    if (limit == 0U || count <= limit) {
        return 1U;
    }
    return count / limit + 1U;
}

}  // namespace

SceneStyle scene_style_for(ThemeMode mode) {
    namespace vc = viewport_colors;
    const bool dark = mode == ThemeMode::Dark;
    SceneStyle style;
    style.background_top = rgb_of(dark ? vc::dark::background : vc::light::background);
    style.background_bottom = rgb_of(dark ? vc::dark::gradient_to : vc::light::gradient_to);
    style.label = rgb_of(dark ? vc::dark::label : vc::light::label);
    style.camera = rgb_of(dark ? "#F2C94C" : "#8A6A00");
    style.camera_active = rgb_of(dark ? "#FFE07A" : "#5C4700");
    style.grid = rgb_of(dark ? "#3A4A54" : "#B4C1C7");
    style.bounding_box = rgb_of(dark ? "#3B736A" : "#0E7C72");
    style.tie_point = rgb_of(dark ? "#7F9AA8" : "#4A5A65");
    style.point_cloud = rgb_of(dark ? "#B9C8D2" : "#2C3A42");
    style.point_highlight = rgb_of(dark ? "#FFFFFF" : "#000000");
    style.selection = rgb_of(dark ? "#35BDAF" : "#0E7C72");
    // Axis colours are geometry markers rather than theme surfaces: the same red / green / blue
    // reads on both canvases, so they are not token-driven.
    style.axis_x = Eigen::Vector3f(0.85F, 0.35F, 0.32F);
    style.axis_y = Eigen::Vector3f(0.42F, 0.75F, 0.36F);
    style.axis_z = Eigen::Vector3f(0.35F, 0.55F, 0.90F);
    style.mesh_base = Eigen::Vector3f::Constant(0.72F);
    return style;
}

Scene build_scene(const Chunk& chunk, const SceneStyle& style, const BuildOptions& options) {
    Scene scene;
    scene.total_cameras = chunk.cameras().size();

    // --- points: tie points first, then the dense cloud ------------------------
    Bounds bounds;
    for (const Track& track : chunk.tie_points().tracks) {
        if (track.point) {
            bounds.add(*track.point);
        }
    }
    const std::size_t tie_limit = options.max_tie_points;
    const std::size_t tie_count = chunk.tie_points().tracks.size();
    const std::size_t tie_stride = stride_for(tie_count, tie_limit);
    scene.tie_points.reserve(std::min(tie_count, tie_limit));
    for (std::size_t i = 0; i < tie_count; i += tie_stride) {
        const Track& track = chunk.tie_points().tracks[i];
        if (!track.point) {
            continue;
        }
        scene.tie_points.push_back({to_float3(*track.point), style.tie_point});
    }

    const PointCloud& cloud = chunk.point_cloud();
    for (const Eigen::Vector3d& point : cloud.points) {
        bounds.add(point);
    }
    const bool cloud_has_colours = cloud.colours.size() == cloud.points.size() * 3U;
    const std::size_t dense_stride = stride_for(cloud.points.size(), options.max_dense_points);
    scene.dense_points.reserve(
        std::min(cloud.points.size(), options.max_dense_points > 0 ? options.max_dense_points
                                                                   : cloud.points.size()));
    scene.style_colours_points = !cloud.points.empty() && !cloud_has_colours;
    for (std::size_t i = 0; i < cloud.points.size(); i += dense_stride) {
        ScenePointVertex vertex;
        vertex.position = to_float3(cloud.points[i]);
        if (cloud_has_colours) {
            const std::size_t base = i * 3U;
            vertex.colour = Eigen::Vector3f(
                static_cast<float>(cloud.colours[base + 0]) / 255.0F,
                static_cast<float>(cloud.colours[base + 1]) / 255.0F,
                static_cast<float>(cloud.colours[base + 2]) / 255.0F);
        } else {
            vertex.colour = style.point_cloud;
        }
        scene.dense_points.push_back(vertex);
    }

    // --- mesh ------------------------------------------------------------------
    const Mesh& mesh = chunk.model();
    for (const Eigen::Vector3d& vertex : mesh.vertices) {
        bounds.add(vertex);
    }
    add_mesh(chunk, mesh, style, scene);

    // Cameras participate in the framing even before anything was reconstructed.
    for (const Camera& camera : chunk.cameras()) {
        if (camera.pose == Transform::Identity()) {
            continue;
        }
        const Eigen::Matrix3d rotation = camera.pose.topLeftCorner<3, 3>();
        const Eigen::Vector3d translation = camera.pose.topRightCorner<3, 1>();
        const Eigen::Vector3d centre = rotation.transpose() * (-translation);
        bounds.add(centre);
        ++scene.aligned_cameras;
    }

    // --- frustums, grid, axes, bounding box ------------------------------------
    double diagonal = 0.0;
    if (bounds.valid()) {
        scene.bounds_min = bounds.min;
        scene.bounds_max = bounds.max;
        scene.has_bounds = true;
        diagonal = (bounds.max - bounds.min).norm();
    }
    if (!(diagonal > 0.0)) {
        diagonal = 1.0;
    }
    // Small pyramids: big enough to read at the default framing, small enough not to swamp the
    // scene they sit in.
    const double frustum_depth = diagonal * 0.04;

    for (const Camera& camera : chunk.cameras()) {
        if (camera.pose == Transform::Identity()) {
            continue;
        }
        const std::optional<Frustum> frustum = frustum_of(chunk, camera, frustum_depth);
        if (!frustum) {
            continue;
        }
        const bool highlighted =
            options.highlight_camera && *options.highlight_camera == camera.id;
        const Eigen::Vector3f colour = highlighted ? style.camera_active : style.camera;
        for (const Eigen::Vector3d& corner : frustum->corners) {
            add_line(scene.lines, scene.line_kinds, scene.line_cameras, frustum->apex, corner,
                     colour, LineKind::Camera, camera.id);
        }
        for (std::size_t i = 0; i < frustum->corners.size(); ++i) {
            add_line(scene.lines, scene.line_kinds, scene.line_cameras,
                     frustum->corners[i], frustum->corners[(i + 1U) % frustum->corners.size()],
                     colour, LineKind::Camera, camera.id);
        }
    }

    if (scene.has_bounds) {
        if (options.grid) {
            const double step = nice_step(std::max(bounds.max.x() - bounds.min.x(),
                                                   bounds.max.y() - bounds.min.y()),
                                          20);
            const double z = bounds.min.z();
            const double x0 = std::floor(bounds.min.x() / step) * step;
            const double y0 = std::floor(bounds.min.y() / step) * step;
            const double x1 = std::ceil(bounds.max.x() / step) * step;
            const double y1 = std::ceil(bounds.max.y() / step) * step;
            // nice_step() targets ~20 divisions, but a pathological extent can still overflow
            // an int. A hostile chunk must never turn into millions of line segments.
            constexpr int kMaxGridLines = 512;
            const int columns = std::clamp(
                static_cast<int>(std::lround((x1 - x0) / step)), 0, kMaxGridLines);
            const int rows =
                std::clamp(static_cast<int>(std::lround((y1 - y0) / step)), 0, kMaxGridLines);
            for (int i = 0; i <= columns; ++i) {
                const double x = x0 + static_cast<double>(i) * step;
                add_line(scene.lines, scene.line_kinds, scene.line_cameras,
                         Eigen::Vector3d(x, y0, z), Eigen::Vector3d(x, y1, z), style.grid,
                         LineKind::Grid);
            }
            for (int i = 0; i <= rows; ++i) {
                const double y = y0 + static_cast<double>(i) * step;
                add_line(scene.lines, scene.line_kinds, scene.line_cameras,
                         Eigen::Vector3d(x0, y, z), Eigen::Vector3d(x1, y, z), style.grid,
                         LineKind::Grid);
            }
        }
        if (options.axes) {
            const double length = diagonal * 0.15;
            const Eigen::Vector3d origin = bounds.min;
            add_line(scene.lines, scene.line_kinds, scene.line_cameras, origin,
                     origin + Eigen::Vector3d::UnitX() * length, style.axis_x, LineKind::Axis);
            add_line(scene.lines, scene.line_kinds, scene.line_cameras, origin,
                     origin + Eigen::Vector3d::UnitY() * length, style.axis_y, LineKind::Axis);
            add_line(scene.lines, scene.line_kinds, scene.line_cameras, origin,
                     origin + Eigen::Vector3d::UnitZ() * length, style.axis_z, LineKind::Axis);
        }
        const Eigen::Vector3d& lo = bounds.min;
        const Eigen::Vector3d& hi = bounds.max;
        const std::array<Eigen::Vector3d, 8> corners = {
            Eigen::Vector3d(lo.x(), lo.y(), lo.z()), Eigen::Vector3d(hi.x(), lo.y(), lo.z()),
            Eigen::Vector3d(hi.x(), hi.y(), lo.z()), Eigen::Vector3d(lo.x(), hi.y(), lo.z()),
            Eigen::Vector3d(lo.x(), lo.y(), hi.z()), Eigen::Vector3d(hi.x(), lo.y(), hi.z()),
            Eigen::Vector3d(hi.x(), hi.y(), hi.z()), Eigen::Vector3d(lo.x(), hi.y(), hi.z()),
        };
        constexpr std::array<std::array<int, 2>, 12> kEdges = {{
            {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
            {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
            {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
        }};
        for (const auto& edge : kEdges) {
            add_line(scene.lines, scene.line_kinds, scene.line_cameras,
                     corners[static_cast<std::size_t>(edge[0])],
                     corners[static_cast<std::size_t>(edge[1])], style.bounding_box,
                     LineKind::BoundingBox);
        }
    }

    return scene;
}

Scene build_scene(const Document& document, const SceneStyle& style, const BuildOptions& options) {
    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        return Scene{};
    }
    return build_scene(*chunk, style, options);
}

void Scene::restyle(const SceneStyle& style, std::optional<std::int32_t> highlight_camera) {
    // The geometry stays where it is; only the colours that came from the theme are rewritten.
    const std::size_t line_count = std::min(lines.size(), line_kinds.size());
    for (std::size_t i = 0; i < line_count; ++i) {
        const std::int32_t camera_id =
            i < line_cameras.size() ? line_cameras[i] : std::int32_t{-1};
        switch (line_kinds[i]) {
            case LineKind::Grid:
                lines[i].colour = style.grid;
                break;
            case LineKind::BoundingBox:
                lines[i].colour = style.bounding_box;
                break;
            case LineKind::Camera:
                lines[i].colour = highlight_camera && *highlight_camera == camera_id
                                      ? style.camera_active
                                      : style.camera;
                break;
            case LineKind::Axis:
                // Axis colours mark the world directions (X red, Y green, Z blue), not a theme
                // surface: the same three read on either canvas, so there is nothing to redo.
                break;
        }
    }
    // Tie points are always drawn in the theme's tie-point colour, so they always follow it.
    if (!tie_points.empty()) {
        for (ScenePointVertex& vertex : tie_points) {
            vertex.colour = style.tie_point;
        }
    }
    if (style_colours_points) {
        for (ScenePointVertex& vertex : dense_points) {
            vertex.colour = style.point_cloud;
        }
    }
    if (style_colours_mesh) {
        for (SceneMeshVertex& vertex : mesh_vertices) {
            vertex.colour = style.mesh_base;
        }
    }
}

}  // namespace alphashape::gui
