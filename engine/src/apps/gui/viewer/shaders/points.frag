#version 440

layout(location = 0) in vec3 v_colour;

layout(location = 0) out vec4 frag_colour;

void main() { frag_colour = vec4(v_colour, 1.0); }
