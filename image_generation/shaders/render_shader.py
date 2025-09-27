import moderngl, numpy as np
from PIL import Image
from time import time

# Windowless context
ctx = moderngl.create_standalone_context()

# Read your shader
with open("myshader.glsl") as f:
    frag_shader = f.read()

prog = ctx.program(
    vertex_shader="""
        #version 330 core
        in vec2 in_vert;
        void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }
    """,
    fragment_shader=frag_shader,
)

# Quad to cover the screen
quad = ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4').tobytes())
vao = ctx.simple_vertex_array(prog, quad, 'in_vert')

# Output size
width, height = 512, 512
fbo = ctx.simple_framebuffer((width, height))
fbo.use()

# Set uniforms if present
if "iResolution" in prog:
    prog["iResolution"].value = (width, height)   # two floats if vec2
if "iTime" in prog:
    prog["iTime"].value = time()

vao.render(moderngl.TRIANGLE_STRIP)

# Save image

for t in np.linspace(0, 6.28, 60):  # 60 frames
    prog['iTime'].value = t
    vao.render(moderngl.TRIANGLE_STRIP)
    data = fbo.read(components=4)
    Image.frombytes('RGBA', fbo.size, data).save(f'frame_{t:.2f}.png')
print("Saved output.png")

