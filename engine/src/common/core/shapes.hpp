#pragma once

// Shapes: vector annotation features overlaid on a chunk (WP-11.1, M11 §4.1).
// Points, polylines and polygons, with user attributes and a layer name used
// for styling/export grouping.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace alphashape {

/// Geometry kind of a shape. Vertices are always `Shape::vertices`; a Point has
/// exactly one, a Polyline two or more, a Polygon three or more.
enum class ShapeType { Point, Polyline, Polygon };

/// One vector feature. `closed` is only meaningful for polylines (drawn as a
/// ring) and is implied by `Polygon`.
struct Shape {
    std::int32_t id = -1;
    ShapeType type = ShapeType::Point;
    std::string label;
    std::vector<Eigen::Vector3d> vertices;
    bool closed = false;
    std::map<std::string, std::string> attributes;
    std::string layer = "Default";
    /// When false, vertices are chunk-local; when true the Z component is an
    /// absolute altitude and X/Y stay chunk-local.
    bool has_altitude = false;

    bool operator==(const Shape& other) const {
        if (id != other.id || type != other.type || label != other.label ||
            closed != other.closed || attributes != other.attributes ||
            layer != other.layer || has_altitude != other.has_altitude) {
            return false;
        }
        if (vertices.size() != other.vertices.size()) {
            return false;
        }
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            if (!(vertices[i] == other.vertices[i])) {
                return false;
            }
        }
        return true;
    }
};

/// A labelled group of shapes.
struct ShapeGroup {
    std::int32_t id = -1;
    std::string label;
    std::vector<std::int32_t> members;

    bool operator==(const ShapeGroup&) const = default;
};

/// All vector content of a chunk. Features are sorted by id; groups reference
/// features by id.
struct Shapes {
    std::vector<Shape> features;
    std::vector<ShapeGroup> groups;

    bool operator==(const Shapes&) const = default;
};

/// A named layer is simply the sub-vector of features sharing `Shape::layer`.
using ShapeLayer = std::vector<Shape>;

}  // namespace alphashape
