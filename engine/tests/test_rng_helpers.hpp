#pragma once

// test_rng_helpers.hpp -- sequenced random draws for seeded test scenes.
//
// Writing Eigen::Vector3d(f(rng), f(rng), f(rng)) leaves the evaluation order
// of the arguments UNSPECIFIED: clang evaluates them left to right, g++ right
// to left. The same seed therefore consumes the engine's bit stream in a
// different order and builds a different scene on macOS and Linux, so bounds
// calibrated on one platform do not hold on the other (see LIMITATIONS.md,
// "Argument evaluation order").
//
// draw2/draw3/draw4 sequence the draws explicitly, in g++'s right-to-left
// order, so a seed selects the same scene on every implementation AND keeps the
// scenes the existing bounds were calibrated on. Any fixture that draws more
// than once for a value must go through these (tools/check_rng_order.py
// enforces it).

#include <Eigen/Core>

namespace test_rng {

/// x, y drawn in g++'s order (y first).
template <class Engine, class DrawX, class DrawY>
Eigen::Vector2d draw2(Engine& rng, DrawX dx, DrawY dy) {
    const double y = dy(rng);
    const double x = dx(rng);
    return {x, y};
}

/// x, y, z drawn in g++'s order (z first, then y, then x).
template <class Engine, class DrawX, class DrawY, class DrawZ>
Eigen::Vector3d draw3(Engine& rng, DrawX dx, DrawY dy, DrawZ dz) {
    const double z = dz(rng);
    const double y = dy(rng);
    const double x = dx(rng);
    return {x, y, z};
}

/// x, y, z, w drawn in g++'s order (w first, then z, y, x). The natural use is
/// a quaternion, whose constructor takes (x, y, z, w).
template <class Engine, class DrawX, class DrawY, class DrawZ, class DrawW>
Eigen::Vector4d draw4(Engine& rng, DrawX dx, DrawY dy, DrawZ dz, DrawW dw) {
    const double w = dw(rng);
    const double z = dz(rng);
    const double y = dy(rng);
    const double x = dx(rng);
    return {x, y, z, w};
}

}  // namespace test_rng
