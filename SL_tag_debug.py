import os
import json
import shutil
import subprocess
import pandas as pd
from collections import defaultdict
import glob

# === PATH SETTINGS ===
base_dir = "/ccn2/u/seojinl/shaders21k"
json_dir = os.path.join(base_dir, "shader_codes", "shadertoy_jsons")
frag_base = os.path.join(base_dir, "shader_codes", "shadertoy")
output_tag_dir = os.path.join(base_dir, "outputs", "shadertoy_by_tags")
os.makedirs(output_tag_dir, exist_ok=True)

# === LOAD TAGS ===
csv_path = os.path.join(base_dir, "shader_tag_frequencies.csv")
df = pd.read_csv(csv_path)
all_tags = set(df["tag"].str.lower())
print(f"Found {len(all_tags)} total tags.")

# === CREATE TAG FOLDERS ===
for tag in all_tags:
    tag = str(tag).strip().lower()
    os.makedirs(os.path.join(output_tag_dir, tag), exist_ok=True)

# === STORAGE STRUCTURE (tag → list of fragment paths) ===
tag_to_fragpaths = defaultdict(list)

# === LOOP OVER JSON FILES ===
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
total = len(json_files)
for i, fname in enumerate(json_files, 1):
    print(f"[{i}/{total}] Parsing {fname}", flush=True)
    try:
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            shader_data = json.load(f)

        info = shader_data.get("info", {})
        shader_id = info.get("id")
        tags = info.get("tags", [])
        if not shader_id or not isinstance(tags, list):
            continue

        first_letter = shader_id[0]
        frag_path = os.path.join(frag_base, first_letter, f"{shader_id}.fragment")
        if not os.path.exists(frag_path):
            print(f"[SKIP] Missing fragment file for {shader_id}")
            continue

        for tag in tags:
            tag = str(tag).lower().strip()
            if tag in all_tags:
                tag_to_fragpaths[tag].append(frag_path)

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")

print(f"\nProcessed {total} shaders.")
print(f"Found {len(tag_to_fragpaths)} tags with at least one valid fragment.")

# === SAVE TAG→FRAGPATHS MAPPING ===
mapping_json = os.path.join(base_dir, "outputs", "tag_to_fragpaths.json")
with open(mapping_json, "w", encoding="utf-8") as f:
    json.dump(dict(tag_to_fragpaths), f, indent=2)
print(f"Saved mapping: {mapping_json}")

# === STEP 2: RENDER ALL SHADERS FOR TAGS WITH >10 SHADERS ===
visual_dir = os.path.join(base_dir, "outputs", "tag_over_10_images")
os.makedirs(visual_dir, exist_ok=True)

total_tags = len(tag_to_fragpaths)
for i, (tag, frag_paths) in enumerate(tag_to_fragpaths.items(), 1):
    num_shaders = len(frag_paths)
    print(f"\n=== [{i}/{total_tags}] Rendering tag '{tag}' ({num_shaders} shaders) ===", flush=True)

    # Skip tags with <10 shaders
    if num_shaders < 10:
        continue

    tag_out_dir = os.path.join(visual_dir, tag)
    os.makedirs(tag_out_dir, exist_ok=True)

    for j, frag_path in enumerate(frag_paths, 1):
        shader_id = os.path.splitext(os.path.basename(frag_path))[0]
        output_path = os.path.join(tag_out_dir, f"{shader_id}.png")

        # Skip if already rendered
        if os.path.exists(output_path):
            continue

        cmd = [
            "python", "-m", "image_generation.shaders.generate_data_single_shader",
            "--shader-file", frag_path,
            "--n-samples", "1",
            "--resolution", "224",
            "--output-path", tag_out_dir,
        ]

        # === NEW: Timeout protection ===
        try:
            subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                timeout=15  # seconds
            )
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {shader_id} took too long, skipping.")
            continue

        # === Post-processing ===
        matches = glob.glob(os.path.join(tag_out_dir, "**", "*.png"), recursive=True) + \
                  glob.glob(os.path.join(tag_out_dir, "**", "*.jpg"), recursive=True)
        if matches:
            src = matches[0]
            os.rename(src, output_path)
            src_dir = os.path.dirname(src)
            if src_dir != tag_out_dir:
                try:
                    shutil.rmtree(src_dir)
                except Exception:
                    pass

        if j % 10 == 0 or j == num_shaders:
            print(f"  [{j}/{num_shaders}] {shader_id}", flush=True)

print(f"\n✅ Completed rendering previews for all tags with >10 shaders.")
print(f"Images saved under: {visual_dir}")
