import os
import random
from torchvision import transforms
from PIL import Image
from image_generation.shaders.renderer_moderngl import get_renderer_from_shader_path
from image_generation.shaders.samples_mixer import ConvexSamplesMixer
import numpy as np
# === SETTINGS ===
shader_list_file = "/ccn2/u/seojinl/shaders21k/shader_codes/shaders_list_working_shadertoy"
base_output = "/ccn2/u/seojinl/shaders21k/outputs/mixup_example_manual/"
os.makedirs(base_output, exist_ok=True)

# === LOAD SHADER PATHS ===
with open(shader_list_file, "r") as f:
    shader_paths_all = [os.path.join("/ccn2/u/seojinl/shaders21k", line.strip()) for line in f if line.strip()]

# === MIXER ===
mixer = ConvexSamplesMixer(
    n_samples_mix=6,
    convex_combination_type="dirichlet",
    dirichlet_alpha=1.0
)

to_pil = transforms.ToPILImage()

# === GENERATE MULTIPLE EXAMPLES ===
for ex_id in range(1, 11):
    print(f"\n=== Generating example {ex_id} ===")
    output_dir = os.path.join(base_output, str(ex_id))
    os.makedirs(output_dir, exist_ok=True)

    shader_subset = random.sample(shader_paths_all, 6)
    images = []

    # Step 2. Render each shader separately
    for i, shader_path in enumerate(shader_subset):
        try:
            renderer = get_renderer_from_shader_path(shader_path, renderer_kwargs={"resolution": 224, "gpu": 0})
            imgs, _ = renderer.render(show_progress=False)

            # Validate image format
            if not isinstance(imgs, list) or len(imgs) == 0:
                raise ValueError("Renderer returned no frames.")
            img_array = imgs[0]
            if img_array.shape[0] != 3:
                raise ValueError(f"Unexpected shape {img_array.shape}, skipping shader.")

            # Convert to tensor
            img_pil = Image.fromarray(img_array.transpose(1, 2, 0))
            img_tensor = transforms.ToTensor()(img_pil)
            images.append(img_tensor)

            img_pil.save(os.path.join(output_dir, f"orig_{i+1}.png"))

        except Exception as e:
            print(f"⚠️ Shader {shader_path} failed: {e}")

    # Step 2b. Refill missing shaders if necessary
    while len(images) < mixer.n_samples_mix:
        extra_shader = random.choice(shader_paths_all)
        try:
            renderer = get_renderer_from_shader_path(extra_shader, renderer_kwargs={"resolution": 224, "gpu": 0})
            imgs, _ = renderer.render(show_progress=False)
            img_array = imgs[0]
            if img_array.shape[0] != 3:
                continue
            img_pil = Image.fromarray(img_array.transpose(1, 2, 0))
            img_tensor = transforms.ToTensor()(img_pil)
            images.append(img_tensor)
        except Exception:
            continue

    # Ensure exactly 6 samples
    images = images[:mixer.n_samples_mix]

    # Step 3. Mix them manually
     # Step 3. Prepare clean numpy images for the mixer
    image_arrays = []
    for img_t in images:
        img_np = (img_t.numpy().transpose(1, 2, 0) * 255).astype("uint8")  # (H, W, 3)
        image_arrays.append(img_np)

    # Mix using numpy arrays instead of tensors
    mixed_pil, _ = mixer.mix_samples(
        [Image.fromarray(img) for img in image_arrays],
        list(range(len(image_arrays)))
    )

    # print(f"[DEBUG] Mixed result shape: {getattr(mixed_np, 'shape', None)}  dtype: {getattr(mixed_np, 'dtype', None)}")
    # if mixed_np is None or not hasattr(mixed_np, 'shape'):
    #     raise RuntimeError("❌ Mixer returned None — check ConvexSamplesMixer implementation.")

    # # Ensure correct shape (some mixers output C,H,W)
    # if mixed_np.shape[0] == 3 and mixed_np.ndim == 3:
    #     mixed_np = mixed_np.transpose(1, 2, 0)

    # # Clip and convert to uint8 just in case
    # mixed_np = np.clip(mixed_np, 0, 255).astype("uint8")

    # Save final mixed image
    # mixed_img = Image.fromarray(mixed_pil)
    # mixed_img.save(os.path.join(output_dir, "mixed.png"))
    mixed_pil.save(os.path.join(output_dir, "mixed.png"))

    print(f"✅ Saved mixed + originals to {output_dir}")

print("\n🎉 Finished generating 10 MixUp examples manually.")
