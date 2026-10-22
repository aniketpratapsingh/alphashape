#pragma once

// Texture baking: turn a mesh with texture coordinates plus a set of views into
// one RGB texture page.
//
// The input is the mesh's uv layout (uvs / uv_faces, produced by the atlas
// packer) and the per-face view assignment (from select_face_views). For every
// texel inside a chart the baker finds the covering face, interpolates the
// surface point, projects it into the chosen view, samples the photograph and
// writes the colour.
//
// This header declares the interface only. bake_texture() is a stub until the
// texturing milestone lands its implementation.

#include <string>

#include "image/image.hpp"
#include "model/mesh/mesh.hpp"

#include "common/algorithm/texture/face_views.hpp"

namespace alphashape::texturing {

/// How colours from several views are combined on a texel.
enum class BlendingMode {
    /// Take the colour from each face's single best view; no cross-face
    /// blending, so chart seams can show. Fastest.
    Mosaic,
    /// Average the samples from every visible view, weighted by view angle.
    Average,
    /// Best view per face, exactly like Mosaic, but with colour correction
    /// forced OFF: the per-view gains stay at 1, so the page carries the raw
    /// photograph colour. On the same inputs the output is byte-identical to
    /// Mosaic with color_correction = false, and an explicit color_correction =
    /// true is ignored. Use it to exercise the atlas / sampling path without
    /// paying for the gain solve.
    /// (An earlier revision skipped every texel here and baked a flat black
    /// page; that was a bug -- see colour_texels() in bake.cpp.)
    Disabled,
};

/// Parse a baking mode from its lower-case name: "mosaic", "average" or
/// "disabled". Throws ValueError on any other text. Throws
/// Error("parse_blending_mode: not implemented") until implemented.
BlendingMode parse_blending_mode(const std::string& text);

/// The lower-case name of `mode`, the inverse of parse_blending_mode.
/// Throws Error("to_string: not implemented") until implemented.
std::string to_string(BlendingMode mode);

/// Parameters for the baking pass.
struct BakeOptions {
    /// How multiple views combine on a texel.
    BlendingMode blending = BlendingMode::Mosaic;
    /// Fill texels that no view covers (self-occluded or seen by no camera)
    /// from the surrounding covered texels.
    bool fill_holes = true;
    /// Level each view's overall gain before mixing, so seams between views do
    /// not show as brightness steps.
    bool color_correction = true;
    /// Gutter width, in pixels, filled outward beyond each chart's border so
    /// bilinear filtering at the seam does not pull in an uncovered texel.
    int dilation_px = 4;
};

/// Bake one RGB texture page of `texture_size` x `texture_size` pixels for a
/// mesh with uvs / uv_faces, sampling from `views` according to `face_views`.
/// Throws Error("bake_texture: not implemented") until implemented.
Image bake_texture(const Mesh& mesh, const std::vector<TexView>& views,
                   const FaceViews& face_views, int texture_size, const BakeOptions& options);

}  // namespace alphashape::texturing
