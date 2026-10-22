#pragma once

// Scale bars (WP-11.1, M11 §4.2). A scale bar constrains the distance between
// two objects — markers or cameras — to a measured length, which fixes the
// overall scale of an otherwise free-network adjustment.

#include <cstdint>
#include <string>

namespace alphashape {

/// One end of a scale bar. `id` names a marker in `Chunk::markers()` when
/// `kind == Kind::Marker`, or a camera in `Chunk::cameras()` when
/// `kind == Kind::Camera`.
struct ScalebarEndpoint {
    enum class Kind { Marker, Camera };

    Kind kind = Kind::Marker;
    std::int32_t id = -1;

    bool operator==(const ScalebarEndpoint&) const = default;
};

/// A measured distance between two endpoints. `reference_length_m` is the
/// field measurement; `estimated_length_m` is filled in after adjustment.
struct Scalebar {
    std::int32_t id = -1;
    std::string label;
    ScalebarEndpoint from;
    ScalebarEndpoint to;
    double reference_length_m = 0.0;
    double reference_accuracy_m = 0.001;
    double estimated_length_m = 0.0;

    bool operator==(const Scalebar&) const = default;
};

}  // namespace alphashape
