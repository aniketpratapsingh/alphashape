// Frustums, the ground grid, the axes and the bounding box.
//
// The uniform block is identical in mesh.vert, points.vert and lines.vert.

#version 440

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_colour;

layout(location = 0) out vec3 v_colour;

layout(std140, binding = 0) uniform Uniforms {
    mat4 mvp;
    mat4 normal_xform;
    vec4 light_dir;
    vec4 viewport;
};

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_colour = in_colour;
}
