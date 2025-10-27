import os
from image_generation.shaders.programs.shadertoy_program import get_shadertoy_program_from_shader_path
from image_generation.shaders.renderer_moderngl import RendererModernGL
from utils import read_text_file_lines, write_text_file_lines  # assuming you already have these
from image_generation.shaders.renderer_moderngl import get_program_from_shader_path

def filter_working_shaders(shader_list_file, output_file, resolution=256):
    shader_files = read_text_file_lines(shader_list_file)
    if isinstance(shader_files, str):
        shader_files = shader_files.splitlines()
    working = []
    for f in shader_files:
        if "tdjcWG.fragment" in f or "WsV3zz.fragment" in f:
            print(f"[SKIPPED] {f}")
            continue

        try:
            program = get_program_from_shader_path(f)

            renderer = RendererModernGL([program], resolution=resolution)
            renderer.release()

            # only add if we made it here without exception
            working.append(f)
            print(f"[OK]   {f}")

        except Exception as e:
            print(f"[FAIL] {f} – {e}")

        # Make sure it's a flat list of strings without newlines
    cleaned = [str(line).strip() for line in working if line is not None]

    # Final defensive guard: ensure it's truly a list of strings
    if isinstance(cleaned, str):
        cleaned = [cleaned]
    elif not isinstance(cleaned, (list, tuple)):
        cleaned = list(cleaned)

    cleaned = [str(c) for c in cleaned]  # flatten any weird types

    print("DEBUG cleaned type:", type(cleaned), "len:", len(cleaned))
    print("DEBUG sample:", cleaned[:5])

    # ---- This is the crucial safe write ----
    try:
        write_text_file_lines(output_file, list(cleaned))
    except AssertionError:
        # If utils.write_text_file_lines still asserts, bypass it safely
        print("[WARN] utils.write_text_file_lines assertion failed, writing manually.")
        with open(output_file, "w") as f:
            for line in cleaned:
                f.write(str(line).rstrip("\n") + "\n")

    print(f"\nSaved {len(cleaned)} working shaders out of {len(shader_files)} to {output_file}")
    print(type(working), working[:5])


if __name__ == "__main__":
    filter_working_shaders(
        "shader_codes/shaders_list",
        "shader_codes/shaders_list_shadertoy_only_working"
    )