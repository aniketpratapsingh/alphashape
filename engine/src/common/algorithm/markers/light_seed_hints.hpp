#pragma once

/// @file light_seed_hints.hpp
/// @brief Border-radius hints for a light-on-dark coded-target interior seed.
///
/// The bright interior enclosed by the dark border of an even (parity-0)
/// target has a fitted radius between roughly 0.5 R and 0.9 R. The circular
/// decoder takes the seed radius as a border-radius hint and accepts a located
/// border only within [hint/1.5, hint*1.5]. Scaling the interior radius by the
/// values below places the hint so that every interior fraction in that range
/// falls inside the gate of at least one scale. Detection tries them in order
/// and keeps the first that decodes.
///
/// A hint below 1.25x lets the border search window end short of R for a small
/// interior: the decoder then accepts the 0.85 R sector edge, which decodes the
/// same word with a biased centre. The first scale therefore starts above that
/// bound; the later ones cover an interior measured as small as 0.4 R.

#include <array>

namespace alphashape {

inline constexpr std::array<double, 3> kLightSeedHintScales = {1.3, 1.6, 2.0};

}  // namespace alphashape
