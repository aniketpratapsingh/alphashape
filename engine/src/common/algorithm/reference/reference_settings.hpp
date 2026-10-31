#pragma once

/// @file reference_settings.hpp
/// @brief The chunk reference settings, re-exported for the algorithms layer.
///
/// `ReferenceSettings` is owned by `Chunk`, so it lives in `common/core` to
/// keep the core layer free of an upward include. Code in this layer includes
/// this header for symmetry with the other reference headers.

#include "common/core/reference_settings.hpp"
