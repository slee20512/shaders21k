import argparse
import os
import multiprocessing as mp
from tqdm import tqdm
from torchvision import transforms
from image_generation.shaders.on_the_fly_moderngl_shader import ModernGLOnlineDataset
from image_generation.shaders.samples_mixer import ConvexSamplesMixer

def sanitize_glsl(shader_code):
    """
    Automatically fix common Shadertoy → desktop GLSL issues.
    """
    fixed = shader_code
    fixed = fixed.replace("varying", "in")
    fixed = fixed.replace("attribute", "in")
    fixed = fixed.replace("texture2D", "texture")

    # Replace deprecated gl_FragColor with fragColor
    if "gl_FragColor" in fixed:
        fixed = fixed.replace("gl_FragColor", "fragColor")
        fixed = "out vec4 fragColor;\n" + fixed

    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed shader images using ModernGLOnlineDataset + ConvexSamplesMixer."
    )
    parser.add_argument("--shader-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--n-mix", type=int, default=6)
    parser.add_argument("--mixing-type", choices=["convex", "cutmix", "convex_cutmix"], default="convex")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # safer multiprocessing start method
    # -------------------------------------------------------------------------
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # -------------------------------------------------------------------------
    # load shader list
    # -------------------------------------------------------------------------
    shader_list_file = "/ccn2/u/seojinl/shaders21k/shader_codes/shaders_list_working_shadertoy"
    with open(shader_list_file, "r") as f:
        shader_paths = [
            os.path.join("/ccn2/u/seojinl/shaders21k", line.strip())
            for line in f
            if line.strip()
        ]

    print(f"Loaded {len(shader_paths)} verified shaders from {shader_list_file}")

    os.makedirs(args.output_path, exist_ok=True)

    # -------------------------------------------------------------------------
    # initialize mixer
    # -------------------------------------------------------------------------
    sample_mixer = ConvexSamplesMixer(
        n_samples_mix=args.n_mix,
        convex_combination_type="dirichlet",
        dirichlet_alpha=args.dirichlet_alpha,
    )

    # -------------------------------------------------------------------------
    # create dataset
    # -------------------------------------------------------------------------
    dataset = ModernGLOnlineDataset(
        shader_paths=shader_paths,
        resolution=args.resolution,
        gpus=args.gpus,
        n_samples=args.n_samples,
        sample_mixer=sample_mixer,
        parallel=True
    )


    # -------------------------------------------------------------------------
    # render and save samples
    # -------------------------------------------------------------------------
    success, failed = 0, 0
    for i, (img_tensor, _) in enumerate(tqdm(dataset, total=args.n_samples)):
        try:
            img_pil = transforms.ToPILImage()(img_tensor)
            img_pil.save(os.path.join(args.output_path, f"mix_{i:05d}.png"))
            success += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed at sample {i}: {e}")
            continue
        if i + 1 >= args.n_samples:
            break

    print(f"\n✅ Done. Saved {success} images (skipped {failed}) → {args.output_path}")


if __name__ == "__main__":
    main()
