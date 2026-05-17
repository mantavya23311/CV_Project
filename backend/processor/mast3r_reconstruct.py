import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pillow_heif import register_heif_opener

register_heif_opener()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../pipeline/mast3r")))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../pipeline/mast3r/dust3r")))

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


def main():
    parser = argparse.ArgumentParser(description="MASt3R SfM + COLMAP Export")
    parser.add_argument("--input_dir",        required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--mask_dir",         default=None)
    parser.add_argument("--image_size",       type=int,   default=512)
    parser.add_argument("--iterations",       type=int,   default=2500)
    parser.add_argument("--conf_thresh",      type=float, default=1.5,
                        help="Min confidence to keep a point (1.5 = top ~60%%).")
    parser.add_argument("--max_pts_per_view", type=int,   default=100_000,
                        help="Max points per view kept (top-confidence ranked).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"🚀 MASt3R SfM | device={device} | size={args.image_size} | "
          f"conf≥{args.conf_thresh}")

    # ── Collect images ────────────────────────────────────────────────────────
    filelist = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        filelist.extend([str(p) for p in Path(args.input_dir).glob(ext)])
    filelist = sorted(filelist)
    print(f"📸 Found {len(filelist)} images")
    if not filelist:
        print("❌ No images found"); sys.exit(1)

    imgs        = load_images(filelist, size=args.image_size)
    TARGET_SIZE = args.image_size

    # ── Load segmentation masks ───────────────────────────────────────────────
    masks = [None] * len(filelist)
    if args.mask_dir:
        from PIL import Image as PILImage
        n_loaded = 0
        for idx, path in enumerate(filelist):
            mp = os.path.join(args.mask_dir, os.path.basename(path))
            if os.path.exists(mp):
                m = PILImage.open(mp).convert('L')
                m = m.resize((TARGET_SIZE, TARGET_SIZE), PILImage.NEAREST)
                masks[idx] = np.array(m) < 250
                n_loaded += 1
        print(f"🎭 Loaded {n_loaded} masks")

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # ── Make pairs ────────────────────────────────────────────────────────────
    if len(imgs) <= 40:
        print("🤝 Complete graph pairs …")
        pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    else:
        print("🤝 Sliding-window pairs …")
        pairs = make_pairs(imgs, scene_graph="swin-3-noncyclic", prefilter=None, symmetrize=True)

    # ── Global alignment ──────────────────────────────────────────────────────
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"🧩 Global alignment ({args.iterations} iters) …")
    scene = sparse_global_alignment(filelist, pairs, cache_dir, model,
                                    device=device,
                                    niter1=args.iterations,
                                    niter2=args.iterations)

    # ── Extract ───────────────────────────────────────────────────────────────
    focals = to_numpy(scene.get_focals())
    poses  = to_numpy(scene.get_im_poses())
    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=False)
    pts3d = [to_numpy(p) for p in pts3d]
    confs = [to_numpy(c) for c in confs]

    # ── Output dirs ───────────────────────────────────────────────────────────
    dataset_dir = os.path.join(args.output_dir, "dataset")
    colmap_dir  = os.path.join(dataset_dir, "sparse/0")
    os.makedirs(colmap_dir, exist_ok=True)

    import shutil as _sh
    images_dir = os.path.join(dataset_dir, "images")
    if os.path.islink(images_dir):   os.remove(images_dir)
    elif os.path.isdir(images_dir):  _sh.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    for src in filelist:
        _sh.copy2(src, os.path.join(images_dir, Path(src).name))
    print(f"📁 Copied {len(filelist)} images → {images_dir}")

    # ── cameras.txt ───────────────────────────────────────────────────────────
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        for i, focal in enumerate(focals):
            h, w = imgs[i]['true_shape'][0]
            f.write(f"{i+1} PINHOLE {w} {h} {focal:.6f} {focal:.6f} "
                    f"{w/2:.2f} {h/2:.2f}\n")

    # ── images.txt ────────────────────────────────────────────────────────────
    # COLMAP images.txt format: every image entry is TWO lines:
    #   Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   Line 2: 2D point observations (empty line = no tracked points written)
    # The read_images_txt parser does i+=2 to skip line 2.
    # We must write BOTH lines or the parser will read every other image only.
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        for i, c2w in enumerate(poses):
            w2c = np.linalg.inv(c2w)
            q   = R.from_matrix(w2c[:3, :3]).as_quat()  # x,y,z,w
            t   = w2c[:3, 3]
            # Line 1: pose
            f.write(f"{i+1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i+1} "
                    f"{Path(filelist[i]).name}\n")
            # Line 2: empty 2D point track line (required by COLMAP format)
            f.write("\n")

    # ── points3D.txt ──────────────────────────────────────────────────────────
    print(f"💾 Writing points3D.txt (conf≥{args.conf_thresh}) …")
    from PIL import Image as PILImage

    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        point_id     = 1
        total        = 0
        skipped_conf = 0
        skipped_mask = 0

        for i, (pts, conf) in enumerate(zip(pts3d, confs)):
            h_c, w_c  = conf.shape[:2]
            conf_flat = conf.ravel()
            pts_flat  = pts.reshape(-1, 3)

            # Confidence gate
            conf_ok = conf_flat >= args.conf_thresh
            skipped_conf += int((~conf_ok).sum())

            # Segmentation mask gate
            if masks[i] is not None:
                mask_seg = masks[i]
                if mask_seg.shape != (h_c, w_c):
                    mp       = PILImage.fromarray(mask_seg.astype("uint8") * 255)
                    mp       = mp.resize((w_c, h_c), PILImage.NEAREST)
                    mask_seg = np.array(mp) > 0
                final_mask = conf_ok & mask_seg.ravel()
                skipped_mask += int((conf_ok & ~mask_seg.ravel()).sum())
            else:
                final_mask = conf_ok

            valid_pts   = pts_flat[final_mask]
            valid_confs = conf_flat[final_mask]
            if not len(valid_pts):
                continue

            # Colours from MASt3R tensor — resize to conf resolution
            img_t = imgs[i]['img']
            if img_t.ndim == 4: img_t = img_t.squeeze(0)
            arr = (img_t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)      # (H_t, W_t, 3)
            if arr.shape[:2] != (h_c, w_c):
                arr = np.array(
                    PILImage.fromarray(arr).resize((w_c, h_c), PILImage.BILINEAR))
            colors_flat = arr.reshape(-1, 3)[final_mask]

            # Top-confidence subsampling
            if len(valid_pts) > args.max_pts_per_view:
                rank        = np.argsort(valid_confs)[::-1][:args.max_pts_per_view]
                valid_pts   = valid_pts[rank]
                colors_flat = colors_flat[rank]

            for p, c in zip(valid_pts, colors_flat):
                f.write(f"{point_id} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")
                point_id += 1
                total    += 1

    print(f"  ✅ Wrote {total:,} pts  "
          f"(skipped {skipped_conf:,} low-conf, {skipped_mask:,} masked-out)")

    # ── .txt → .bin ───────────────────────────────────────────────────────────
    print("🔄 Converting .txt → .bin …")
    up = os.path.join(os.path.dirname(__file__),
                      "../../pipeline/gaussian-splatting/utils")
    if up not in sys.path: sys.path.append(up)
    try:
        import read_write_model as rwm
        c2, im2, p2 = rwm.read_model(path=colmap_dir, ext=".txt")
        rwm.write_model(c2, im2, p2, path=colmap_dir, ext=".bin")
        print("✅ .bin done")
    except Exception as e:
        print(f"⚠️  .bin skipped: {e}")

    print("\n✅ MASt3R complete!")


if __name__ == "__main__":
    main()