#pragma once

// Scene: the flat, GPU-ready description of what the 3-D view draws.
//
// Pure data (Eigen + the engine model), no Qt and no graphics API: build_scene() is a plain
// function of a Document, so the counts and bounds are unit-tested without a render target. The
// viewport uploads the vectors as-is; the renderer only knows about these structs.
//
// Coordinates are chunk-local. Nothing here reads the file system or takes a lock: the caller
// hands in a const reference while it holds the document lock.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "apps/gui/theme/theme_mode.hpp"

namespace alphashape {

class Chunk;
class Document;

}  // namespace alphashape

namespace alphashape::gui {

/// Theme colours for the view, in sRGB, components 0..1.
struct SceneStyle {
    Eigen::Vector3f background_top = Eigen::Vector3f(0.11F, 0.14F, 0.16F);
    Eigen::Vector3f background_bottom = Eigen::Vector3f(0.08F, 0.11F, 0.13F);
    Eigen::Vector3f label = Eigen::Vector3f(0.73F, 0.78F, 0.82F);
    Eigen::Vector3f grid = Eigen::Vector3f(0.23F, 0.29F, 0.33F);
    Eigen::Vector3f bounding_box = Eigen::Vector3f(0.23F, 0.45F, 0.42F);
    Eigen::Vector3f camera = Eigen::Vector3f(0.95F, 0.79F, 0.30F);
    Eigen::Vector3f camera_active = Eigen::Vector3f(1.0F, 0.88F, 0.48F);
    Eigen::Vector3f tie_point = Eigen::Vector3f(0.50F, 0.60F, 0.66F);
    Eigen::Vector3f point_cloud = Eigen::Vector3f(0.73F, 0.78F, 0.82F);
    Eigen::Vector3f point_highlight = Eigen::Vector3f(1.0F, 1.0F, 1.0F);
    Eigen::Vector3f selection = Eigen::Vector3f(0.21F, 0.74F, 0.69F);
};

/// The style of the viewport mode, resolved from the generated colour tokens
/// (apps/gui/theme/viewport_colors.hpp, itself generated from design/system/tokens.json).
SceneStyle scene_style_for(ThemeMode mode);

struct ScenePointVertex {
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Vector3f colour = Eigen::Vector3f::Ones();
};

struct SceneLineVertex {
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Vector3f colour = Eigen::Vector3f::Ones();
};

struct SceneMeshVertex {
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f colour = Eigen::Vector3f::Ones();
    Eigen::Vector2f uv = Eigen::Vector2f::Zero();
};

/// An RGBA8 texture page, row 0 = v = 0 (the mesh UV convention in model/mesh/mesh.hpp).
struct SceneTexture {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;

    bool empty() const {
        return width <= 0 || height <= 0 ||
               rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U;
    }
};

/// Everything one frame draws. Vectors are empty when the chunk has no such product.
struct Scene {
    std::vector<ScenePointVertex> tie_points;
    std::vector<ScenePointVertex> dense_points;

    std::vector<SceneMeshVertex> mesh_vertices;
    std::vector<std::uint32_t> mesh_indices;
    SceneTexture texture;

    /// Camera frustums, the ground grid, the axes and the bounding box, as line segments.
    std::vector<SceneLineVertex> lines;

    Eigen::Vector3d bounds_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d bounds_max = Eigen::Vector3d::Zero();
    bool has_bounds = false;

    std::size_t aligned_cameras = 0;
    std::size_t total_cameras = 0;
    /// True when the mesh was drawn with its texture page instead of its vertex colours.
    bool textured = false;

    bool has_texture() const { return !texture.empty(); }
    std::size_t mesh_triangles() const { return mesh_indices.size() / 3U; }
    bool has_points() const { return !tie_points.empty() || !dense_points.empty(); }
    bool has_mesh() const { return !mesh_indices.empty(); }
    bool empty() const { return !has_points() && !has_mesh() && lines.empty(); }
};

/// Knobs the viewport sets from its own state; the defaults are what the tests use.
struct BuildOptions {
    /// More points than this are stride-sampled (Phase 1 keeps the upload bounded).
    std::size_t max_dense_points = 2000000;
    std::size_t max_tie_points = 1000000;
    bool grid = true;
    bool axes = true;
    /// Drawn in the active-camera colour when the tree selects a camera.
    std::optional<std::int32_t> highlight_camera;
};

/// Scene for `chunk` (never throws; an empty chunk yields an empty scene).
Scene build_scene(const Chunk& chunk, const SceneStyle& style, const BuildOptions& options = {});

/// Scene for the document's active chunk (empty scene when there is none).
Scene build_scene(const Document& document, const SceneStyle& style, const BuildOptions& options = {});

}  // namespace alphashape::gui
