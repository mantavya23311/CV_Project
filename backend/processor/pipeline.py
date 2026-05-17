import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import sys
import glob
import requests

from config import (
    MAST3R_PATH, GAUSSIAN_PATH, OPENSPLAT_PATH,
    UPLOAD_DIR, OUTPUT_DIR, DEVICE, ITERATIONS, BACKEND_URL
)

from vlm_gateway import VLMGateway
from device import get_device


def report_progress(job_id, status, progress, stage):
    try:
        url = f"{BACKEND_URL}/projects/{job_id}/progress"
        requests.post(url, json={"status": status,
                                  "progressPercentage": progress,
                                  "currentStage": stage}, timeout=5)
        print(f"📡 Progress: {stage} ({progress}%)")
    except Exception as e:
        print(f"⚠️ Failed to report progress: {e}")


def run_command(command, description):
    print(f"\n===== Running {description} =====\n")
    print("Command:", " ".join(str(x) for x in command))
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\n❌ {description} failed.")
        sys.exit(1)
    print(f"\n✅ {description} completed successfully.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: merge per-label edited point clouds back into the main points3D.txt
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_edited_labels_into_main(colmap_dir: str,
                                    multi_seg_output: str,
                                    scene_json_path: str):
    """
    After multi-seg + editing, the per-label files (points3D_bottle.txt, etc.)
    contain the edited colours/geometry.  The main points3D.txt in colmap_dir
    still has the OLD colours for those points.

    Strategy
    ────────
    1. Read scene_objects.json to find every label and its points file.
    2. Re-read the main points3D.txt.
    3. For every labelled point, find the nearest neighbour in the main cloud
       (KD-tree on XYZ) and replace its RGB with the edited value.
    4. Write the merged result back to points3D.txt (backup original first).
    5. Regenerate .bin so COLMAP / OpenSplat picks up the change.

    Point IDs across the two files may differ (multi_seg uses its own numbering),
    so we match on nearest-neighbour in 3-D space.
    """
    import json
    import shutil
    import numpy as np

    main_pts_path = os.path.join(colmap_dir, "points3D.txt")
    if not os.path.exists(main_pts_path):
        print(f"  ⚠️  merge: {main_pts_path} not found — skipping merge")
        return

    if not os.path.exists(scene_json_path):
        print(f"  ⚠️  merge: {scene_json_path} not found — skipping merge")
        return

    with open(scene_json_path) as f:
        meta = json.load(f)

    if not meta:
        print("  ⚠️  merge: scene_objects.json is empty — skipping merge")
        return

    # Import point-cloud I/O from sibling module
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from segment_object_pointcloud import read_points3D_txt, write_points3D_txt

    print("\n🔀 Merging edited label clouds into main points3D.txt …")

    main_pts = read_points3D_txt(main_pts_path)
    if not main_pts:
        print("  ⚠️  merge: main cloud is empty — skipping")
        return

    main_xyz = np.array([p['xyz'] for p in main_pts])   # (N, 3)

    # Backup original main cloud once
    backup_path = main_pts_path.replace(".txt", "_pre_merge.txt")
    if not os.path.exists(backup_path):
        shutil.copy2(main_pts_path, backup_path)
        print(f"  💾 Backup → {backup_path}")

    from scipy.spatial import cKDTree

    replaced_total = 0

    for label, info in meta.items():
        pts_file = info.get("points_file", "")
        if not pts_file or not os.path.exists(pts_file):
            print(f"  ⚠️  merge: '{label}' points file missing — skipping label")
            continue

        label_pts = read_points3D_txt(pts_file)
        if not label_pts:
            continue

        label_xyz = np.array([p['xyz'] for p in label_pts])   # (M, 3)
        label_rgb = np.array([p['rgb'] for p in label_pts])   # (M, 3)

        # Compute a tight matching radius = 3× median NN distance in label cloud
        if len(label_xyz) > 1:
            self_tree             = cKDTree(label_xyz)
            self_dists, _         = self_tree.query(label_xyz, k=2)
            median_nn_dist        = float(np.median(self_dists[:, 1]))
            radius                = max(median_nn_dist * 3.0, 1e-4)
        else:
            radius = 1e-2

        tree           = cKDTree(main_xyz)
        dists, idxs    = tree.query(label_xyz, k=1)

        n_replaced = 0
        for i, (dist, idx) in enumerate(zip(dists, idxs)):
            if dist <= radius:
                main_pts[idx]['rgb'] = label_rgb[i].tolist()
                n_replaced += 1

        replaced_total += n_replaced
        print(f"  ✅ '{label}': replaced {n_replaced}/{len(label_pts)} pts "
              f"(radius={radius:.5f})")

    # Write merged cloud back
    write_points3D_txt(main_pts_path, main_pts)
    print(f"  💾 Merged cloud → {main_pts_path}  ({replaced_total} pts recoloured)")

    # Regenerate .bin
    _HERE = os.path.dirname(os.path.abspath(__file__))
    up = os.path.join(_HERE, "../../pipeline/gaussian-splatting/utils")
    if up not in sys.path:
        sys.path.append(up)
    try:
        import read_write_model as rwm
        c2, im2, p2 = rwm.read_model(path=colmap_dir, ext=".txt")
        rwm.write_model(c2, im2, p2, path=colmap_dir, ext=".bin")
        print("  ✅ .bin regenerated after merge")
    except Exception as e:
        print(f"  ⚠️  .bin regen skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: extract target RGB from an EditPlan (rule-based, no LLM needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_edit_colour_from_prompt(prompt: str) -> tuple | None:
    """
    Re-run the same rule-based parser used by object_editor_3d.py and return
    the *final* RGB that will be painted onto the object.

    Returns (R, G, B) uint8 tuple, or None if no colour op was found.
    The logic mirrors execute_edit_plan() so the image colour matches the
    point-cloud colour exactly.
    """
    import re

    NAMED_COLOURS = {
        'red': (220, 30, 30), 'green': (30, 180, 30), 'blue': (30, 80, 220),
        'yellow': (240, 220, 20), 'orange': (240, 120, 10), 'purple': (140, 30, 200),
        'pink': (240, 80, 160), 'cyan': (20, 200, 220), 'white': (255, 255, 255),
        'black': (0, 0, 0), 'grey': (128, 128, 128), 'gray': (128, 128, 128),
        'brown': (120, 70, 30), 'gold': (212, 175, 55), 'silver': (192, 192, 192),
        'bronze': (205, 127, 50), 'navy': (10, 30, 100), 'teal': (10, 140, 140),
        'magenta': (210, 20, 180), 'lime': (100, 220, 20), 'maroon': (128, 0, 0),
        'coral': (255, 80, 60), 'crimson': (220, 20, 60), 'turquoise': (64, 224, 208),
        'beige': (245, 245, 220), 'ivory': (255, 255, 240),
    }
    METALLIC_BASE = {
        'gold': (212, 175, 55), 'silver': (192, 192, 192),
        'bronze': (205, 127, 50), 'copper': (184, 115, 51),
    }

    text = prompt.lower().strip()

    # Metallic (e.g. "metallic gold")
    m = re.search(r'\bmetallic\s+(\w+)|\b(\w+)\s+metallic\b', text)
    if m:
        base_name = (m.group(1) or m.group(2) or 'silver').strip()
        return METALLIC_BASE.get(base_name, NAMED_COLOURS.get(base_name, (192, 192, 192)))
    if re.search(r'\bmetallic\b', text):
        return (192, 192, 192)

    # Hex colour
    hex_m = re.search(r'#([0-9a-f]{6})', text)
    if hex_m:
        hx = hex_m.group(1)
        return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))

    # Named colour
    for name, rgb in NAMED_COLOURS.items():
        if re.search(rf'\b{name}\b', text):
            return rgb

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: recolour training images using per-label masks so OpenSplat sees the
#         edited colour during training, not just the original photo colour.
#         Without this step OpenSplat will fit Gaussians to the pink bottle
#         pixels and ignore the gold colour we put in points3D.txt.
# ═══════════════════════════════════════════════════════════════════════════════

def _recolour_training_images(images_dir: str,
                               multi_seg_output: str,
                               scene_json_path: str,
                               edit_target: str,
                               edit_prompt: str):
    """
    For every label that has an edit prompt, load the per-label masked images
    produced by multi_object_segmentation (images_masked_<label>/), derive the
    target colour from the edit prompt, and paint that colour onto the
    corresponding region in the training images in `images_dir`.

    The mask images are black-background PNGs where the label pixels are the
    original image colour — we use them purely as binary masks (non-black = label).

    Parameters
    ──────────
    images_dir      : the training images directory OpenSplat reads
    multi_seg_output: directory containing images_masked_<label>/ subdirs
    scene_json_path : scene_objects.json
    edit_target     : label name, "single", or "batch"
    edit_prompt     : edit instruction string (or JSON for batch)
    """
    import json
    import shutil
    from PIL import Image as PILImage
    import numpy as np

    if not os.path.exists(scene_json_path):
        print("  ⚠️  recolour: scene_objects.json not found — skipping image recolouring")
        return

    with open(scene_json_path) as f:
        meta = json.load(f)

    # Build label → prompt mapping
    if edit_target == "batch":
        try:
            label_prompts = json.loads(edit_prompt)
        except Exception:
            print("  ⚠️  recolour: could not parse batch edit JSON — skipping")
            return
    elif edit_target == "single" or not edit_target:
        # single mode — no per-label masks available, skip
        print("  ℹ️  recolour: single-object mode — image recolouring skipped")
        return
    else:
        # Specific label
        label_prompts = {edit_target: edit_prompt}

    if not label_prompts:
        return

    # Backup training images once before we modify them
    backup_dir = images_dir + "_pre_recolour"
    if not os.path.isdir(backup_dir):
        shutil.copytree(images_dir, backup_dir)
        print(f"  💾 Training images backup → {backup_dir}")

    for label, prompt in label_prompts.items():
        target_rgb = _get_edit_colour_from_prompt(prompt)
        if target_rgb is None:
            print(f"  ⚠️  recolour: could not extract colour from prompt '{prompt}' "
                  f"for label '{label}' — skipping")
            continue

        safe_lbl   = label.replace(" ", "_").replace("/", "_")
        masked_dir = os.path.join(multi_seg_output, f"images_masked_{safe_lbl}")
        if not os.path.isdir(masked_dir):
            print(f"  ⚠️  recolour: masked dir not found: {masked_dir} — skipping '{label}'")
            continue

        n_recoloured = 0
        target = np.array(target_rgb, dtype=np.float32)

        for fname in sorted(os.listdir(masked_dir)):
            mask_path  = os.path.join(masked_dir, fname)
            train_path = os.path.join(images_dir,  fname)
            if not os.path.exists(train_path):
                continue

            # Load the masked image — non-black pixels are the label region
            mask_img   = np.array(PILImage.open(mask_path).convert("RGB"))
            train_img  = np.array(PILImage.open(train_path).convert("RGB")).astype(np.float32)

            # Binary mask: pixel belongs to label if any channel > 10
            label_mask = (mask_img.max(axis=2) > 10)   # (H, W) bool

            if not label_mask.any():
                continue

            # Paint: 85% target colour + 15% original (matches recolour op blend)
            train_img[label_mask] = (
                train_img[label_mask] * 0.15 +
                target * 0.85
            )
            train_img = np.clip(train_img, 0, 255).astype(np.uint8)

            PILImage.fromarray(train_img).save(train_path)
            n_recoloured += 1

        print(f"  🎨 '{label}': painted {n_recoloured} training images → "
              f"RGB{target_rgb}")

    print(f"  ✅ Training image recolouring complete")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(job_id):

    device     = get_device()
    iterations = int(os.getenv("VOK_ITERATIONS", ITERATIONS))

    input_folder  = os.path.join(UPLOAD_DIR, job_id)
    output_folder = os.path.join(OUTPUT_DIR,  job_id)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n🚀 Starting Pipeline for: {job_id}")
    print(f"Device: {device} | Input: {input_folder} | Output: {output_folder}")

    # ── PHASE 1: Collect images ───────────────────────────────────────────────
    image_paths = []
    for ext in ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    print(f"📸 Found {len(image_paths)} images")

    valid_image_paths = image_paths
    if os.getenv("VOK_SKIP_VLM","0") != "1":
        try:
            gateway    = VLMGateway()
            rejections = gateway.audit_images(image_paths)
            rejected   = {r['path'] for r in rejections}
            valid_image_paths = [p for p in image_paths if p not in rejected]
        except Exception as e:
            print(f"⚠️ VLM failed: {e}")

    # ── PHASE 2: User-supplied masks ─────────────────────────────────────────
    user_mask_dir = os.path.join(input_folder, "masks")
    if os.path.isdir(user_mask_dir) and len(os.listdir(user_mask_dir)) > 0:
        print(f"\n✅ User-supplied masks: {user_mask_dir}")
        segmented_folder = user_mask_dir
    else:
        print("\n===== No user masks (auto-segmentation will be used) =====")
        segmented_folder = None

    # ── STEP 2b: Pre-edit images BEFORE reconstruction ───────────────────────
    # When VOK_EDIT_BEFORE_RECONSTRUCT=1:
    #   1. Segment the object in every input image using GroundingDINO+GrabCut
    #   2. Paint the desired colour onto those pixels
    #   3. Write the modified images to images_pre_edit/
    #   4. Point MASt3R at the edited images (input_folder = images_pre_edit/)
    #
    # This is the CORRECT approach because MASt3R reads pixel colours to build
    # the initial point-cloud RGB values, and OpenSplat fits Gaussians to match
    # the training images. If the images show a gold bottle, both the point cloud
    # and the final splat will be gold — by construction, with no post-hoc patching.
    _edit_before_reconstruct = os.getenv("VOK_EDIT_BEFORE_RECONSTRUCT", "0") == "1"
    _pre_edit_target = os.getenv("VOK_EDIT_TARGET", "").strip()
    _pre_edit_prompt = os.getenv("VOK_EDIT_PROMPT", "").strip()
    if not _pre_edit_target and not _pre_edit_prompt:
        _sp = os.getenv("VOK_EDIT_SINGLE_PROMPT", "").strip()
        if _sp:
            _pre_edit_target = "single"
            _pre_edit_prompt = _sp

    if _edit_before_reconstruct and _pre_edit_target and _pre_edit_prompt:
        print("\n===== Pre-editing Input Images Before Reconstruction =====")
        _editor_2d = os.path.join(os.path.dirname(__file__), "image_editor_2d.py")
        if not os.path.exists(_editor_2d):
            print("  ⚠️  image_editor_2d.py not found — skipping pre-edit")
        else:
            import torch as _tch_pre
            import subprocess as _sp_pre
            _pre_dev = "cuda" if _tch_pre.cuda.is_available() else "cpu"
            _pre_edit_out  = os.path.join(output_folder, "images_pre_edit")
            _pre_mask_cache = os.path.join(output_folder, "pre_edit_masks")
            os.makedirs(_pre_edit_out, exist_ok=True)

            # Build prompt string: "label: prompt" or plain for whole-image edit
            if _pre_edit_target in ("single", ""):
                _prompt_str = f"__all__: {_pre_edit_prompt}"
            else:
                _prompt_str = f"{_pre_edit_target}: {_pre_edit_prompt}"

            _pe_cmd = [
                sys.executable, _editor_2d,
                "--images_dir",  input_folder,    # READ from original uploads
                "--output_dir",  _pre_edit_out,   # WRITE to separate edited dir
                "--prompt",      _prompt_str,
                "--device",      _pre_dev,
                "--mask_cache",  _pre_mask_cache,
            ]
            _pe_result = _sp_pre.run(_pe_cmd)
            if _pe_result.returncode == 0 and os.path.isdir(_pre_edit_out):
                _edited_count = len([f for f in os.listdir(_pre_edit_out)
                                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
                if _edited_count > 0:
                    input_folder = _pre_edit_out   # ← MASt3R reads edited images
                    print(f"  ✅ MASt3R will reconstruct from EDITED images "
                          f"({_edited_count} images): {input_folder}")
                else:
                    print("  ⚠️  No edited images produced — using original images")
            else:
                print("  ⚠️  Pre-edit failed — using original images for reconstruction")

    # ── STEP 3a: MASt3R ───────────────────────────────────────────────────────
    print("\n===== Running MASt3R =====")
    mast3r_script = os.path.join(os.path.dirname(__file__), "mast3r_reconstruct.py")
    if not os.path.exists(mast3r_script):
        raise FileNotFoundError(f"MASt3R script not found: {mast3r_script}")

    mast3r_cmd = [
        sys.executable, mast3r_script,
        "--input_dir",        input_folder,
        "--output_dir",       output_folder,
        "--image_size",       os.getenv("VOK_IMAGE_SIZE",       "512"),
        "--conf_thresh",      os.getenv("VOK_CONF_THRESH",      "1.5"),
        "--max_pts_per_view", os.getenv("VOK_MAX_PTS_VIEW",     "100000"),
    ]
    if segmented_folder is not None:
        mast3r_cmd += ["--mask_dir", segmented_folder]
    run_command(mast3r_cmd, "MASt3R Reconstruction")

    # ── Resolve COLMAP + images dirs (used by all downstream steps) ───────────
    colmap_dir = os.path.join(output_folder, "dataset", "sparse", "0")
    images_dir = os.path.join(output_folder, "dataset", "images")

    # ── STEP 3b: Segmentation ─────────────────────────────────────────────────
    segment_object = os.getenv("VOK_SEGMENT_OBJECT", "1") == "1"

    if segment_object:
        print("\n===== Running Object Point-Cloud Isolation =====")

        vote_thresh = float(os.getenv("VOK_VOTE_THRESH",  "0.65"))
        mask_images = os.getenv("VOK_MASK_IMAGES",  "1") == "1"
        seg_method  = os.getenv("VOK_SEG_METHOD",   "auto")

        import torch
        sam_device = "cuda" if torch.cuda.is_available() else "cpu"

        seg_script = os.path.join(os.path.dirname(__file__),
                                  "segment_object_pointcloud.py")
        if not os.path.exists(seg_script):
            print(f"⚠️  seg script not found — skipping")
        else:
            seg_cmd = [
                sys.executable, seg_script,
                "--colmap_dir",  colmap_dir,
                "--images_dir",  images_dir,
                "--output_dir",  colmap_dir,
                "--vote_thresh", str(vote_thresh),
                "--device",      sam_device,
                "--method",      seg_method,
            ]
            if segmented_folder is not None:
                seg_cmd += ["--mask_dir", segmented_folder]
            if mask_images:
                seg_cmd.append("--mask_images")
            run_command(seg_cmd, "Object Point-Cloud Isolation")

            if mask_images:
                masked_dir = os.path.join(output_folder, "dataset", "images_masked")
                if os.path.isdir(masked_dir):
                    # When edit-before-reconstruct is active, images already show
                    # the edited colour on the object. We do NOT swap to the masked
                    # (black-background) images because:
                    #  a) OpenSplat needs to see the full scene including background
                    #     to converge properly (isolated object Gaussians perform poorly)
                    #  b) The edited colour is already baked into every pixel
                    # We still keep the masked dir as a backup, just don't swap.
                    _ebr_active = os.getenv("VOK_EDIT_BEFORE_RECONSTRUCT","0") == "1"
                    if _ebr_active:
                        import shutil
                        orig_backup = images_dir + "_orig"
                        if not os.path.exists(orig_backup):
                            shutil.copytree(images_dir, orig_backup)
                        print("  ℹ️  edit-before-reconstruct active — "
                              "keeping full-scene images (not swapping to masked).")
                    else:
                        import shutil
                        orig_backup = images_dir + "_orig"
                        if not os.path.exists(orig_backup):
                            shutil.copytree(images_dir, orig_backup)
                        shutil.rmtree(images_dir)
                        shutil.copytree(masked_dir, images_dir)
                        print("🔄 Swapped to masked images for OpenSplat.")
    else:
        print("\n===== Segmentation SKIPPED =====")

    # ── STEP 3c: Multi-object segmentation ────────────────────────────────────
    multi_prompt     = os.getenv("VOK_MULTI_SEG_PROMPT", "").strip()
    multi_auto       = (os.getenv("VOK_MULTI_SEG_AUTO",  "0") == "1" or
                        os.getenv("VOK_MULTI_SEG",       "0") == "1")
    multi_seg_output = os.path.join(output_folder, "multi_seg")
    multi_seg_ran    = False   # tracked so the merge step below knows whether to run

    if multi_prompt or multi_auto:
        print("\n===== Running Multi-Object Segmentation =====")
        multi_seg_script = os.path.join(os.path.dirname(__file__),
                                        "multi_object_segmentation.py")
        if not os.path.exists(multi_seg_script):
            print("⚠️  multi_object_segmentation.py not found — skipping")
        else:
            import torch
            # CRITICAL: multi-seg needs FULL ORIGINAL images (all objects visible).
            # Step 3b may have swapped images/ → images_masked/ (bottle only).
            # If images_orig backup exists, use it; else use images_dir as-is.
            orig_backup = images_dir + "_orig"
            multi_seg_images = orig_backup if os.path.isdir(orig_backup) else images_dir
            print(f"  🖼  Multi-seg images source: {multi_seg_images}")

            sam_device = "cuda" if torch.cuda.is_available() else "cpu"
            multi_cmd = [
                sys.executable, multi_seg_script,
                "--colmap_dir",  colmap_dir,
                "--images_dir",  images_dir,
                "--output_dir",  multi_seg_output,
                "--vote_thresh", os.getenv("VOK_MULTI_VOTE_THRESH", "0.40"),
                "--device",      sam_device,
            ]
            if multi_prompt:
                multi_cmd += ["--prompt", multi_prompt]
            else:
                multi_cmd += [
                    "--auto_discover",
                    "--n_clusters", os.getenv("VOK_MULTI_N_CLUSTERS", "6"),
                ]
            run_command(multi_cmd, "Multi-Object Segmentation")
            multi_seg_ran = True
            print(f"  📄 Multi-seg outputs → {multi_seg_output}")
    else:
        print("\n===== Multi-Object Segmentation SKIPPED "
              "(set VOK_MULTI_SEG=1, VOK_MULTI_SEG_AUTO=1, "
              "or VOK_MULTI_SEG_PROMPT='label1,label2') =====")

    # ── STEP 3d: Text-prompted 3D object editing ──────────────────────────────
    edit_target = os.getenv("VOK_EDIT_TARGET", "").strip()
    edit_prompt = os.getenv("VOK_EDIT_PROMPT", "").strip()

    if not edit_target and not edit_prompt:
        single_prompt = os.getenv("VOK_EDIT_SINGLE_PROMPT", "").strip()
        if single_prompt:
            edit_target = "single"
            edit_prompt = single_prompt

    if edit_target and edit_prompt:
        print("\n===== Running 3D Object Editor =====")
        editor_script = os.path.join(os.path.dirname(__file__),
                                     "object_editor_3d.py")
        if not os.path.exists(editor_script):
            print("⚠️  object_editor_3d.py not found — skipping")
        else:
            no_llm_flag = [] if os.getenv("ANTHROPIC_API_KEY") else ["--no_llm"]

            if edit_target == "single":
                pts_file = os.path.join(colmap_dir, "points3D.txt")
                if not os.path.exists(pts_file):
                    print(f"⚠️  points3D.txt not found at {pts_file} — skipping editor")
                else:
                    edit_cmd = ([sys.executable, editor_script, "edit",
                                 "--points_file", pts_file,
                                 "--prompt", edit_prompt] + no_llm_flag)
                    run_command(edit_cmd, "3D Object Edit (single)")

            elif edit_target == "batch":
                scene_json = os.path.join(multi_seg_output, "scene_objects.json")
                if not os.path.exists(scene_json):
                    print("⚠️  scene_objects.json missing — run multi-seg first "
                          "(set VOK_MULTI_SEG=1)")
                else:
                    edit_cmd = ([sys.executable, editor_script, "batch",
                                 "--scene_json", scene_json,
                                 "--edits", edit_prompt] + no_llm_flag)
                    run_command(edit_cmd, "3D Object Batch Edit")

            else:
                safe_lbl = edit_target.replace(" ", "_").replace("/", "_")
                # Prefer multi-seg label cloud if available; fall back to the
                # main isolated object cloud from step 3b (colmap_dir/points3D.txt).
                # This is the correct behaviour when --edit-before-reconstruct
                # is used: the images were already recoloured, and step 3b has
                # produced the clean isolated point cloud we want to recolour too.
                pts_file_multiseg = os.path.join(multi_seg_output,
                                                  f"points3D_{safe_lbl}.txt")
                pts_file_main     = os.path.join(colmap_dir, "points3D.txt")

                if os.path.exists(pts_file_multiseg):
                    pts_file = pts_file_multiseg
                elif os.path.exists(pts_file_main):
                    pts_file = pts_file_main
                    print(f"  ℹ️  Using main points3D.txt (multi-seg '{edit_target}' "
                          f"cloud not found — this is normal when multi-seg is skipped)")
                else:
                    pts_file = None

                if not pts_file:
                    print(f"⚠️  No points file found for '{edit_target}' — skipping 3D edit")
                else:
                    edit_cmd = ([sys.executable, editor_script, "edit",
                                 "--points_file", pts_file,
                                 "--prompt", edit_prompt] + no_llm_flag)
                    run_command(edit_cmd, f"3D Object Edit ({edit_target})")
    else:
        print("\n===== Object Editing SKIPPED "
              "(set VOK_EDIT_TARGET + VOK_EDIT_PROMPT, "
              "or VOK_EDIT_SINGLE_PROMPT for single-object edit) =====")

    # ── STEP 3e: Merge edited label clouds → main points3D.txt ───────────────
    scene_json_path = os.path.join(multi_seg_output, "scene_objects.json")
    if multi_seg_ran and os.path.exists(scene_json_path):
        _merge_edited_labels_into_main(colmap_dir, multi_seg_output, scene_json_path)
    elif multi_seg_ran:
        print("\n⚠️  scene_objects.json not found — merge skipped")

    # ── STEP 3f: Restore ORIGINAL images before OpenSplat ────────────────────
    # The single-object segmentation step (3b) swapped images/ → images_masked/
    # (bottle only).  When multi-seg is active the table and background must
    # also be visible so OpenSplat can train all objects.  We restore the
    # original images so OpenSplat sees the full scene, while the merged
    # point cloud (now with edited colours) guides colour initialisation.
    if multi_seg_ran:
        orig_backup = images_dir + "_orig"
        if os.path.isdir(orig_backup):
            import shutil
            print("\n🔄 Restoring original (full-scene) images for OpenSplat …")
            if os.path.isdir(images_dir):
                shutil.rmtree(images_dir)
            shutil.copytree(orig_backup, images_dir)
            print(f"  ✅ Restored {len(os.listdir(images_dir))} images")
        else:
            print("\n  ℹ️  No images_orig backup found — OpenSplat will use "
                  "current images directory as-is.")

    # ── STEP 3g: Recolour training images using image_editor_2d ─────────────
    # THE KEY STEP for correct Gaussian splat colours.
    #
    # OpenSplat minimises photometric loss against training images. If we only
    # change colours in points3D.txt, OpenSplat reverts to the original image
    # colours within ~300 steps. We must paint the desired colour onto the
    # training image pixels so OpenSplat fits Gaussians to the edited colour.
    #
    # This step uses image_editor_2d.py which:
    #   1. Detects the object in each image via GroundingDINO (cached, 1 load)
    #   2. Gets a pixel mask via GrabCut or SAM2
    #   3. Paints the target colour/texture onto those pixels
    #   4. Saves the modified images back to images_dir (in-place)
    #
    # Runs whenever an edit target+prompt is set, regardless of whether
    # multi-seg was active. This means --edit-before-reconstruct AND the
    # normal post-reconstruction editing both result in correctly coloured images.
    _cur_edit_target = os.getenv("VOK_EDIT_TARGET", "").strip()
    _cur_edit_prompt = os.getenv("VOK_EDIT_PROMPT", "").strip()
    if not _cur_edit_target and not _cur_edit_prompt:
        _cur_edit_prompt = os.getenv("VOK_EDIT_SINGLE_PROMPT", "").strip()
        if _cur_edit_prompt:
            _cur_edit_target = "single"

    # Skip if images were already edited before reconstruction — they already
    # contain the right colours and we would be double-applying the effect.
    _already_edited = os.getenv("VOK_EDIT_BEFORE_RECONSTRUCT", "0") == "1"

    if _cur_edit_target and _cur_edit_prompt and not _already_edited:
        print("\n===== Recolouring Training Images (image_editor_2d) =====")
        editor_2d = os.path.join(os.path.dirname(__file__), "image_editor_2d.py")
        if not os.path.exists(editor_2d):
            print("  ⚠️  image_editor_2d.py not found — skipping image recolouring")
        else:
            import torch as _torch2d
            _dev2d = "cuda" if _torch2d.cuda.is_available() else "cpu"
            _mask_cache_2d = os.path.join(output_folder, "edit_mask_cache_3g")

            # Build prompt: "label: edit_prompt" or plain prompt for single/all
            if _cur_edit_target in ("single", ""):
                _prompt_2d = f"__all__: {_cur_edit_prompt}"
            else:
                _prompt_2d = f"{_cur_edit_target}: {_cur_edit_prompt}"

            import subprocess as _sp2d
            _edit_cmd_2d = [
                sys.executable, editor_2d,
                "--images_dir",  images_dir,
                "--output_dir",  images_dir,    # in-place; backup auto-created
                "--prompt",      _prompt_2d,
                "--device",      _dev2d,
                "--mask_cache",  _mask_cache_2d,
            ]
            result_2d = _sp2d.run(_edit_cmd_2d)
            if result_2d.returncode == 0:
                print("  ✅ Training images recoloured successfully")
            else:
                print("  ⚠️  image_editor_2d returned error — check logs above")

    elif _cur_edit_target and _cur_edit_prompt and _already_edited:
        print("\n===== Training Image Recolouring SKIPPED "
              "(images already edited before reconstruction) =====")
    else:
        print("\n===== Training Image Recolouring SKIPPED "
              "(no VOK_EDIT_TARGET/VOK_EDIT_PROMPT set) =====")

    # ── STEP 4: OpenSplat ─────────────────────────────────────────────────────
    print("\n===== Running OpenSplat =====")

    dataset_output = os.path.join(output_folder, "dataset")
    model_output   = os.path.join(output_folder, "model.ply")

    # ── Pre-flight: detect PyTorch ABI mismatch before running OpenSplat ─────
    # The error "undefined symbol: _ZNK3c105Error4whatEv" means OpenSplat was
    # compiled against a different PyTorch version than the one currently in the
    # venv.  We detect this with a quick ldd / nm check and emit a clear rebuild
    # message instead of a cryptic symbol error.
    def _check_opensplat_abi():
        import subprocess as _sp
        if not os.path.exists(OPENSPLAT_PATH):
            return True, ""
        try:
            # Check for the missing symbol using nm
            result = _sp.run(
                ["nm", "-D", OPENSPLAT_PATH],
                capture_output=True, text=True, timeout=10)
            # UND = undefined symbol that must be resolved at runtime
            undef_torch = [l for l in result.stdout.splitlines()
                           if "UND" in l and ("c10" in l or "torch" in l or "at::" in l)]
            # Try a quick launch to see if it actually crashes
            test = _sp.run([OPENSPLAT_PATH, "--version"],
                           capture_output=True, text=True, timeout=5)
            if "undefined symbol" in (test.stderr or ""):
                return False, test.stderr.strip()
            return True, ""
        except Exception:
            return True, ""   # can't check, assume ok

    _splat_ok, _splat_err = _check_opensplat_abi()
    if not _splat_ok:
        print("\n❌ OpenSplat ABI MISMATCH DETECTED")
        print(f"   Error: {_splat_err}")
        print("\n   OpenSplat was compiled against a different PyTorch version.")
        print("   To fix, rebuild OpenSplat against your current venv's PyTorch:")
        print()
        print("   1. Find your venv's torch cmake dir:")
        print('  python -c "import torch; print(torch.utils.cmake_prefix_path)"')
        print()
        print("   2. Rebuild OpenSplat:")
        print("      cd pipeline/opensplat")
        print("      rm -rf build && mkdir build && cd build")
        print("      cmake .. -DCMAKE_PREFIX_PATH=$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")")
        print("      cmake --build . --config Release -j$(nproc)")
        print()
        print("   Your training images and point cloud are ready in:")
        print(f"   {dataset_output}")
        print("   Once OpenSplat is rebuilt, run it manually:")
        print(f"   {OPENSPLAT_PATH} {dataset_output} -o {model_output} -n {iterations}")
        # Don't sys.exit — let the pipeline produce what it can
        print("\n⚠️  Skipping OpenSplat due to ABI mismatch. "
              "All other pipeline outputs are complete.")
        print("\n🎉 PIPELINE COMPLETED (OpenSplat skipped — rebuild needed)\n")
        return

    gaussian_cmd = [
        OPENSPLAT_PATH,
        dataset_output,
        "-o", model_output,
        "-n", str(iterations),
        "--resolution-schedule", "2000",
        "--densify-grad-thresh", "0.0008",
        "--refine-every",        "150",
        "--warmup-length",       "500",
        "--reset-alpha-every",   "3000",
    ]
    if str(device) == "cpu":
        gaussian_cmd.append("--cpu")

    # Wrap OpenSplat run to give a helpful message if ABI crash happens at runtime
    import subprocess as _sp_splat
    print("Command:", " ".join(str(x) for x in gaussian_cmd))
    _splat_result = _sp_splat.run(gaussian_cmd, capture_output=False)
    if _splat_result.returncode != 0:
        print("\n❌ OpenSplat failed.")
        print("\n   If you see 'undefined symbol' above, this is a PyTorch ABI mismatch.")
        print("   Rebuild OpenSplat against your current venv:")
        print("      cd pipeline/opensplat && rm -rf build && mkdir build && cd build")
        print("      cmake .. -DCMAKE_PREFIX_PATH=$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")")
        print("      cmake --build . --config Release -j$(nproc)")
        print(f"\n   Edited images are at: {dataset_output}/images")
        print(f"   Point cloud is at:    {dataset_output}/sparse/0/points3D.txt")
        sys.exit(1)

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!\n")