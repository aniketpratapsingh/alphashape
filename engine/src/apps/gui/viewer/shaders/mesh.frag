#version 440

layout(location = 0) in vec3 v_colour;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec4 frag_colour;

layout(binding = 1) uniform sampler2D page;

void main() { frag_colour = vec4(v_colour * texture(page, v_uv).rgb, 1.0); }
