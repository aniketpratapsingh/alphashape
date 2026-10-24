// Shaded, optionally textured mesh. The uniform block is identical in every shader in this
// directory (see the note in points.vert).
//
// Attribute locations follow SceneMeshVertex (position, normal, colour, uv).

#version 440

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_colour;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec3 v_colour;
layout(location = 1) out vec2 v_uv;

layout(std140, binding = 0) uniform Uniforms {
    mat4 mvp;         // clip from world, already carrying the backend clip-space correction
    mat4 normal_xform; // rotation part of the model transform
    vec4 light_dir;   // xyz: unit vector toward the light
    vec4 viewport;    // xy: pixel size, z: point size in pixels, w: unused
};

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    vec3 normal = normalize(mat3(normal_xform) * in_normal);
    // A soft wrap (0.3 ambient, 0.7 lambert) so surfaces facing away stay readable.
    float diffuse = max(dot(normal, normalize(light_dir.xyz)), 0.0);
    v_colour = in_colour * (0.30 + 0.70 * diffuse);
    v_uv = in_uv;
}
