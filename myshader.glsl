#version 330 core
out vec4 fragColor;
uniform float iTime;
uniform vec2  iResolution;

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution;
    float r = 0.5 + 0.5 * sin(iTime + uv.x * 10.0);
    float g = 0.5 + 0.5 * sin(iTime + uv.y * 10.0);
    fragColor = vec4(r, g, 0.5, 1.0);
}
