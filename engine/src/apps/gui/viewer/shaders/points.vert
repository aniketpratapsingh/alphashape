// Point clouds, drawn as instanced screen-facing quads so the sprite size is in pixels on every
// backend (Vulkan and D3D do not honour gl_PointSize).
//
// The uniform block is identical in mesh.vert, points.vert and lines.vert.

#version 440

layout(location = 0) in vec2 in_corner;    // per-vertex quad corner, both components in [-1, 1]
layout(location = 1) in vec3 in_position;  // per-instance
layout(location = 2) in vec3 in_colour;    // per-instance

layout(location = 0) out vec3 v_colour;

layout(std140, binding = 0) uniform Uniforms {
    mat4 mvp;
    mat4 normal_xform;
    vec4 light_dir;
    vec4 viewport;    // xy: pixel size, z: point size in pixels
};

void main() {
    vec4 centre = mvp * vec4(in_position, 1.0);
    // A constant pixel offset: the NDC extent of one pixel is 2 / viewport, scaled by w so the
    // perspective divide does not shrink distant points to nothing.
    vec2 offset = in_corner * (viewport.z * 2.0 / viewport.xy);
    gl_Position = centre + vec4(offset * centre.w, 0.0, 0.0);
    v_colour = in_colour;
}
