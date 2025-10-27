import os
import argparse
from tqdm import tqdm
from utils import *

from image_generation.shaders.glsl_utils import *
from image_generation.shaders.on_the_fly_moderngl_shader import ModernGLOnlineDataset
from image_generation.shaders.renderer_moderngl import RendererModernGL


def generate_one_per_shader(output_path, shader_file_list, shader_codes_path, resolution, gpu, overwrite):
    """Render exactly one image per available shader."""
    assert isinstance(shader_file_list, list), "shader_file_list must be a list"

    # Build full shader paths
    shader_paths = [os.path.join(shader_codes_path, s.strip()) for s in shader_file_list if s.strip()]

    # Filter out missing files
    available_shaders = [s for s in shader_paths if os.path.exists(s)]
    missing_shaders = [s for s in shader_paths if not os.path.exists(s)]

    print(f"\n[Info] Found {len(available_shaders)} available shaders out of {len(shader_paths)} total.")
    if missing_shaders:
        print(f"[Warning] {len(missing_shaders)} missing shader files (skipping them). Example:\n  {missing_shaders[0]}")

    if len(available_shaders) == 0:
        raise RuntimeError("No valid shader files found. Check your paths.")

    os.makedirs(output_path, exist_ok=True)

    # Initialize renderer
    dataset = ModernGLOnlineDataset(
        available_shaders,
        resolution=resolution,
        max_queue_size=len(available_shaders) * 2,
        n_samples=-1,
        parallel=True,
        gpus=[gpu],
        virtual_dataset_size=len(available_shaders),
        sample_mixer=None,
        transform_before=None,
    )

    print(f"\n[Rendering] Starting 1 image per shader ({len(available_shaders)} shaders total)...")

    for i, shader_path in enumerate(tqdm(available_shaders)):
        shader_name = os.path.splitext(os.path.basename(shader_path))[0]
        image_file = os.path.join(output_path, f"{shader_name}.png")

        if os.path.exists(image_file) and not overwrite:
            continue

        try:
            image, _ = dataset.__getitem__(i)
            cv2_imwrite(tonumpy(image) * 255, image_file)
        except Exception as e:
            print(f"[Error] Failed to render {shader_name}: {e}")
            continue

    print(f"\n✅ Done! Rendered {len(available_shaders)} shaders to {output_path}")
    if missing_shaders:
        print(f"⚠️  Skipped {len(missing_shaders)} missing shaders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shaders-file", type=str, required=True,
                        help="Text file containing shader fragment names (one per line)")
    parser.add_argument("--shader-codes-path", type=str, required=True,
                        help="Base path containing shader code files")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", type=str2bool, default="False")

    args = parser.parse_args()

    shader_file_list = read_text_file_lines(args.shaders_file)

    generate_one_per_shader(args.output_path,
                            shader_file_list,
                            args.shader_codes_path,
                            args.resolution,
                            args.gpu,
                            overwrite=args.overwrite)
