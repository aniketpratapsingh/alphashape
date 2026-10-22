#pragma once

// Box downsampling. The inverse of the depth-map "downscale" factor: the
// dense-stereo stage works on reduced-resolution luminance, and the reduced
// calibration is derived with scaled_calibration (model/camera/calibration.hpp).
//
// The reduced image covers only the top-left floor(W/f) x floor(H/f) block-aligned
// region of the source; any trailing strip of fewer than f rows/columns is
// dropped. This matches scaled_calibration's convention exactly (width' = W / f
// by integer division), so a pixel at reduced coordinate u' corresponds to
// full-res u' * f.

#include "image/image.hpp"

namespace alphashape {

/// Mean over each `factor` x `factor` block. `factor` must be a power of two
/// >= 1; factor 1 returns a copy. Output is (W / factor) x (H / factor).
/// Throws ValueError when `factor` is not a power of two >= 1.
FloatImage downsample_box(const FloatImage& src, int factor);

/// Per-channel box mean, each channel rounded to the nearest integer. Same
/// power-of-two >= 1 requirement and output geometry as the FloatImage overload.
Image downsample_box(const Image& src, int factor);

}  // namespace alphashape
