#pragma once

/// @file light_seed_hints.hpp
/// @brief Border-radius hints for a light-on-dark coded-target interior seed.
///
/// The bright interior enclosed by the dark border of an even (parity-0)
/// target has a fitted radius between roughly 0.5 R and 0.9 R. The circular
/// decoder takes the seed radius as a border-radius hint and accepts a located
/// border only within [hint/1.5, hint*1.5]. Scaling the interior radius by the
/// values below places the hint so that every interior fraction in that range
/// falls inside the gate of at least one scale. Detection decodes every scale
/// and keeps the outermost border that is still a plausible multiple of the
/// interior radius, because a short search window can make the first decoding
/// scale lock onto the 0.85 R sector edge instead of the true border.

#include <array>

namespace alphashape {

inline constexpr std::array<double, 3> kLightSeedHintScales = {1.2, 1.5, 1.9};

}  // namespace alphashape
