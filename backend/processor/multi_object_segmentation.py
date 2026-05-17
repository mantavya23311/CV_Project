"""
multi_object_segmentation.py  (v1)
═══════════════════════════════════════════════════════════════════════════════

Multi-Object 3D Point-Cloud Segmentation with Text Prompting
─────────────────────────────────────────────────────────────
Given a reconstructed COLMAP scene (cameras.txt / images.txt / points3D.txt),
this module segments EVERY distinguishable object in the scene and assigns
each 3D point a semantic label, OR isolates a specific object/region
described by a free-text prompt.

Architecture
────────────
1. GROUNDING DINO  — open-vocabulary 2D object detection from text prompt.
   Falls back to DINO-v2 + K-means clustering when GroundingDINO is absent.
2. SAM2             — segment anything from bounding-box prompts.
   Falls back to GrabCut multi-scale when SAM2 is absent.
3. MULTI-VIEW VOTE  — project each 3D point into every camera, accumulate
   per-object mask votes, assign the object label with the highest weighted
   vote score.
4. DBSCAN REFINEMENT — per-object spatial clustering to remove cross-label
   contamination.

Outputs
───────
• points3D_<label>.txt   — one COLMAP points3D file per segmented object.
• scene_objects.json     — metadata: label → point count, bbox, centroid.
• images_masked_<label>/ — (optional) masked input images per object.

Usage
─────
    python multi_object_segmentation.py \
        --colmap_dir  path/to/sparse/0 \
        --images_dir  path/to/images \
        --output_dir  path/to/output \
        --prompt      "bottle, table, background" \
        --device      cuda

    # Auto-discover all objects (no prompt):
    python multi_object_segmentation.py \
        --colmap_dir  ... --images_dir ... --output_dir ... \
        --auto_discover

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Re-use geometry helpers from the existing segmentation module
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from segment_object_pointcloud import (
    read_cameras_txt, read_images_txt, read_points3D_txt, write_points3D_txt,
    qvec_to_rotmat, project_points,
    sanitise_points, statistical_outlier_removal, voxel_downsample,
    depth_aware_dbscan, try_write_bin, _check_mask_quality,
)


# ═════════════════════════════════════════════════════════════════════════════
# DETECTION BACKENDS  (GroundingDINO → DINO-v2 fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_gdino_paths():
    """
    Resolve GroundingDINO config + checkpoint paths.
    Search order:
      1. $GDINO_CONFIG / $GDINO_CKPT  env vars
      2. <repo_root>/GroundingDINO/  — installed alongside vok-vision-main
      3. ~/.cache/groundingdino/
    Returns (cfg, ckpt) or raises FileNotFoundError.
    """
    cfg  = os.environ.get("GDINO_CONFIG",  "")
    ckpt = os.environ.get("GDINO_CKPT",    "")
    if cfg and ckpt and os.path.exists(cfg) and os.path.exists(ckpt):
        return cfg, ckpt

    # Walk up from this file to find repo root containing GroundingDINO/
    for root in [Path(_HERE), Path(_HERE).parent,
                 Path(_HERE).parent.parent, Path(_HERE).parent.parent.parent]:
        gdino_root = root / "GroundingDINO"
        if not gdino_root.is_dir():
            continue
        for cfg_rel in [
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "groundingdino/config/GroundingDINO_SwinB_cfg.py",
            "GroundingDINO_SwinT_OGC.py",
        ]:
            c = gdino_root / cfg_rel
            if c.exists(): cfg = str(c); break
        for ckpt_rel in [
            "weights/groundingdino_swint_ogc.pth",
            "weights/groundingdino_swinb_cogcoor.pth",
            "groundingdino_swint_ogc.pth",
        ]:
            k = gdino_root / ckpt_rel
            if k.exists(): ckpt = str(k); break
        if cfg and ckpt and os.path.exists(cfg) and os.path.exists(ckpt):
            # Add the GroundingDINO root to sys.path so imports work
            gdino_str = str(gdino_root)
            if gdino_str not in sys.path:
                sys.path.insert(0, gdino_str)
            return cfg, ckpt

    # ~/.cache/groundingdino/
    cache = Path.home() / ".cache" / "groundingdino"
    cfg_c  = str(cache / "GroundingDINO_SwinT_OGC.py")
    ckpt_c = str(cache / "groundingdino_swint_ogc.pth")
    if os.path.exists(cfg_c) and os.path.exists(ckpt_c):
        return cfg_c, ckpt_c

    raise FileNotFoundError("GroundingDINO weights not found")


def _detect_grounding_dino(img_np: np.ndarray, labels: list[str],
                            device: str = 'cpu',
                            box_thresh: float = 0.30,
                            text_thresh: float = 0.25):
    """
    Run GroundingDINO on a single image.
    Returns list of dicts: {label, box:[x1,y1,x2,y2], score}
    box coords are in pixel space of img_np.
    """
    try:
        cfg, ckpt = _resolve_gdino_paths()
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
        import torch
        from torchvision.ops import box_convert

        model = load_model(cfg, ckpt, device=device)
        caption = " . ".join(labels) + " ."

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        img_pil = Image.fromarray(img_np)
        img_t, _ = transform(img_pil, None)

        boxes, logits, phrases = predict(
            model=model, image=img_t, caption=caption,
            box_threshold=box_thresh, text_threshold=text_thresh,
            device=device)

        H, W = img_np.shape[:2]
        boxes_px = box_convert(boxes * torch.tensor([W, H, W, H]),
                               in_fmt='cxcywh', out_fmt='xyxy').numpy()
        detections = []
        for box, score, phrase in zip(boxes_px, logits.numpy(), phrases):
            # Match phrase to closest label
            matched = min(labels, key=lambda l: _phrase_distance(phrase, l))
            detections.append({'label': matched,
                                'box': box.tolist(),
                                'score': float(score)})
        return detections

    except Exception as e:
        print(f"  ℹ️  GroundingDINO unavailable: {e}")
        return None


def _phrase_distance(phrase: str, label: str) -> float:
    """Simple word-overlap distance between a detected phrase and a label."""
    p_words = set(phrase.lower().split())
    l_words = set(label.lower().split())
    overlap = len(p_words & l_words)
    return 1.0 / (overlap + 1)


def _detect_dino_v2_kmeans(img_np: np.ndarray, n_clusters: int,
                            device: str = 'cpu'):
    """
    Fallback: DINO-v2 patch features + K-means → approximate object regions.
    Returns list of dicts: {label:'object_N', box:[x1,y1,x2,y2], score:1.0}
    """
    try:
        import torch
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                               pretrained=True).to(device).eval()

        from torchvision import transforms as tvT
        tfm = tvT.Compose([
            tvT.Resize(448),
            tvT.CenterCrop(448),
            tvT.ToTensor(),
            tvT.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        img_pil = Image.fromarray(img_np)
        x = tfm(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = model.get_intermediate_layers(x, n=1)[0][0]  # (patches, C)

        feats_np = feats.cpu().numpy()
        P = int(feats_np.shape[0] ** 0.5)   # patch grid size

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels_flat = km.fit_predict(feats_np)
        label_map = labels_flat.reshape(P, P)

        H, W = img_np.shape[:2]
        detections = []
        for k in range(n_clusters):
            rows, cols = np.where(label_map == k)
            if not len(rows): continue
            # Convert patch indices to pixel coords
            ph = H / P; pw = W / P
            y1 = int(rows.min() * ph); y2 = int((rows.max()+1) * ph)
            x1 = int(cols.min() * pw); x2 = int((cols.max()+1) * pw)
            # Skip tiny or full-image clusters
            area_frac = (y2-y1)*(x2-x1) / (H*W)
            if area_frac < 0.02 or area_frac > 0.92: continue
            detections.append({'label': f'object_{k}',
                                'box': [x1, y1, x2, y2],
                                'score': 1.0})
        return detections

    except Exception as e:
        print(f"  ℹ️  DINO-v2 fallback failed: {e}")
        # Last resort: divide image into a 2×2 grid
        H, W = img_np.shape[:2]
        detections = []
        for i, (r1, r2, c1, c2) in enumerate([
                (0,H//2,0,W//2),(0,H//2,W//2,W),
                (H//2,H,0,W//2),(H//2,H,W//2,W)]):
            detections.append({'label': f'region_{i}',
                                'box': [c1,r1,c2,r2], 'score':1.0})
        return detections


# ═════════════════════════════════════════════════════════════════════════════
# MASK BACKENDS  (SAM2 → GrabCut fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _sam2_mask_from_box(img_np: np.ndarray, box, sam2_predictor) -> np.ndarray:
    """Use SAM2 predictor to get a mask from a bounding box prompt."""
    import torch
    sam2_predictor.set_image(img_np)
    box_arr = np.array(box, dtype=np.float32)
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None,
        box=box_arr[None, :], multimask_output=False)
    best = masks[np.argmax(scores)]
    return best.astype(bool)


def _grabcut_from_box(img_np: np.ndarray, box) -> np.ndarray:
    """GrabCut mask from a bounding-box prompt."""
    import cv2
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W), bool)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    msk = np.zeros((H, W),  np.uint8)
    try:
        cv2.grabCut(img_bgr, msk, (x1, y1, x2-x1, y2-y1),
                    bgd, fgd, 7, cv2.GC_INIT_WITH_RECT)
        fg = ((msk == cv2.GC_FGD) | (msk == cv2.GC_PR_FGD)).astype(np.uint8)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        return fg.astype(bool)
    except cv2.error:
        result = np.zeros((H, W), bool)
        result[y1:y2, x1:x2] = True
        return result


def _load_sam2_predictor(device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    ckpt = os.environ.get("SAM2_CHECKPOINT",
                          os.path.expanduser("~/.cache/sam2/sam2.1_hiera_small.pt"))
    cfg  = os.environ.get("SAM2_CONFIG", "sam2.1/sam2.1_hiera_small.yaml")
    sam2 = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
    return SAM2ImagePredictor(sam2)


# ═════════════════════════════════════════════════════════════════════════════
# REPRESENTATIVE FRAME SELECTOR
# ═════════════════════════════════════════════════════════════════════════════

def _pick_representative_frames(image_paths: list[str],
                                 n: int = 5) -> list[str]:
    """
    Pick up to `n` representative frames by selecting every k-th path,
    preferring frames from different viewing angles.  Simple uniform stride
    is used; replace with DINO-v2 diversity sampling if available.
    """
    if len(image_paths) <= n:
        return image_paths
    step = max(1, len(image_paths) // n)
    return [image_paths[i] for i in range(0, len(image_paths), step)][:n]


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-OBJECT MASK GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_multi_object_masks(image_paths: list[str],
                                  prompt_labels: list[str] | None,
                                  mask_cache: str,
                                  device: str = 'cpu',
                                  n_auto_clusters: int = 6,
                                  rep_frames: int = 6):
    """
    For each image produce a dict:  image_name → {label: bool_mask}

    If prompt_labels is given, GroundingDINO detects each label in each frame.
    If prompt_labels is None (auto-discover), DINO-v2 K-means is used.

    Returns
    ───────
    masks_by_label : dict[label → dict[image_name → np.ndarray(H,W,bool)]]
    all_labels     : list[str]  ordered list of discovered labels
    """
    os.makedirs(mask_cache, exist_ok=True)

    # Try to load SAM2 predictor once
    sam2_pred = None
    try:
        sam2_pred = _load_sam2_predictor(device)
        print("  ✅ SAM2 predictor loaded")
    except Exception as e:
        print(f"  ℹ️  SAM2 not available ({e}), using GrabCut fallback")

    # Pick representative frames for detection (speed optimisation)
    rep_paths = _pick_representative_frames(image_paths, rep_frames)
    print(f"  🖼  Running detection on {len(rep_paths)} representative frames …")

    # Collect detections across rep frames
    # label → list of (image_name, box, score)
    label_detections: dict[str, list] = {}

    for img_path in rep_paths:
        name    = Path(img_path).name
        img_np  = np.array(Image.open(img_path).convert('RGB'))

        if prompt_labels:
            dets = _detect_grounding_dino(img_np, prompt_labels, device)
            if dets is None:
                # GroundingDINO unavailable — fall back to colour-based per-label
                dets = _colour_based_detection(img_np, prompt_labels)
        else:
            dets = _detect_dino_v2_kmeans(img_np, n_auto_clusters, device)

        if dets is None:
            dets = []

        for det in dets:
            lbl = det['label']
            if lbl not in label_detections:
                label_detections[lbl] = []
            label_detections[lbl].append((name, det['box'], det['score']))

    all_labels = sorted(label_detections.keys())
    print(f"  🏷  Discovered labels: {all_labels}")

    if not all_labels:
        print("  ⚠️  No objects detected — returning empty masks")
        return {}, []

    # ── Build per-frame boxes ────────────────────────────────────────────────
    # For GroundingDINO: re-run per frame (it is fast and frame-aware).
    # For colour/GrabCut fallback: run detection once per rep-frame, then
    # RE-USE those boxes for ALL non-rep frames.  This avoids 22× redundant
    # GrabCut calls and — critically — avoids the old bug where the slab
    # fallback gave different (wrong) boxes to different frames.
    rep_name_set = {Path(p).name for p in rep_paths}

    # Boxes from rep-frames (already computed above in label_detections)
    # Build: label → best (box, score) from rep frames
    rep_best_boxes: dict[str, tuple] = {}
    for lbl, entries in label_detections.items():
        best = max(entries, key=lambda e: e[2])   # (name, box, score)
        rep_best_boxes[lbl] = (best[1], best[2])  # (box, score)

    # Now generate masks for EVERY image for each label
    masks_by_label: dict[str, dict[str, np.ndarray]] = {l: {} for l in all_labels}

    for img_path in image_paths:
        name   = Path(img_path).name
        img_np = np.array(Image.open(img_path).convert('RGB'))
        H, W   = img_np.shape[:2]

        # Determine boxes for this frame
        frame_boxes: dict[str, tuple] = {}   # label → (box, score)

        if prompt_labels:
            # Try GroundingDINO first (fast, frame-aware)
            dets = _detect_grounding_dino(img_np, prompt_labels, device)
            if dets is not None:
                for det in dets:
                    lbl = det['label']
                    if lbl not in frame_boxes or det['score'] > frame_boxes[lbl][1]:
                        frame_boxes[lbl] = (det['box'], det['score'])
            else:
                # GroundingDINO unavailable.
                # Re-run colour detection only for rep-frames;
                # all other frames reuse the best rep-frame box for each label.
                if name in rep_name_set:
                    cdets = _colour_based_detection(img_np, prompt_labels)
                    for det in (cdets or []):
                        lbl = det['label']
                        if lbl not in frame_boxes or det['score'] > frame_boxes[lbl][1]:
                            frame_boxes[lbl] = (det['box'], det['score'])
                else:
                    # Reuse rep-frame boxes unchanged — no detection call
                    frame_boxes = dict(rep_best_boxes)
        else:
            dets = _detect_dino_v2_kmeans(img_np, n_auto_clusters, device)
            for det in (dets or []):
                lbl = det['label']
                if lbl not in frame_boxes or det['score'] > frame_boxes[lbl][1]:
                    frame_boxes[lbl] = (det['box'], det['score'])

        # Fill any missing labels from rep-frame fallback
        for lbl in all_labels:
            if lbl not in frame_boxes and lbl in rep_best_boxes:
                frame_boxes[lbl] = rep_best_boxes[lbl]

        for lbl in all_labels:
            cache_path = os.path.join(mask_cache, f"{lbl}__{name}.png")
            if os.path.exists(cache_path):
                m = np.array(Image.open(cache_path).convert('L')) > 127
                masks_by_label[lbl][name] = m
                continue

            if lbl in frame_boxes:
                box = frame_boxes[lbl][0]
                if sam2_pred is not None:
                    try:
                        m = _sam2_mask_from_box(img_np, box, sam2_pred)
                    except Exception:
                        m = _grabcut_from_box(img_np, box)
                else:
                    m = _grabcut_from_box(img_np, box)
            else:
                m = np.zeros((H, W), bool)

            masks_by_label[lbl][name] = m
            Image.fromarray(m.astype(np.uint8) * 255).save(cache_path)

        q_str = " | ".join(
            f"{l}:{100*masks_by_label[l].get(name, np.zeros((1,1))).mean():.1f}%"
            for l in all_labels)
        print(f"    ✓ {name}  [{q_str}]")

    return masks_by_label, all_labels


def _colour_based_detection(img_np: np.ndarray,
                              labels: list[str]) -> list[dict]:
    """
    Colour-contrast based detection fallback when GroundingDINO is absent.

    Strategy
    ────────
    For each label we try to find a semantically plausible bounding box using
    two signals:
      1. Foreground isolation — GrabCut with progressively tighter init-rects
         gives us a binary FG map.  The LARGEST connected component is the
         dominant foreground object (bottle / primary subject).
      2. Complementary regions — for a second label we use the pixels that
         GrabCut classified as background (or a different cluster).

    For the special case of exactly TWO labels the heuristic is:
      • Run GrabCut to find the dominant FG blob → label 0
      • Remaining image region (BG of GrabCut) → label 1
    For three or more labels we fall back to K-means on HSV colour space.

    This is far better than the previous horizontal-slab fallback which
    gave every label a fixed portion of the image regardless of content.
    """
    import cv2
    H, W = img_np.shape[:2]
    n    = len(labels)

    if n == 1:
        return [{'label': labels[0], 'box': [0, 0, W, H], 'score': 0.9}]

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ── GrabCut foreground mask ───────────────────────────────────────────────
    def _grabcut_fg(frac=0.50):
        rw = int(W * frac); rh = int(H * frac)
        rx = (W - rw)//2;   ry = (H - rh)//2
        bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
        msk = np.zeros((H, W), np.uint8)
        try:
            cv2.grabCut(img_bgr, msk, (rx, ry, rw, rh), bgd, fgd, 5,
                        cv2.GC_INIT_WITH_RECT)
            return ((msk == cv2.GC_FGD) | (msk == cv2.GC_PR_FGD)).astype(np.uint8)
        except cv2.error:
            return np.ones((H, W), np.uint8)

    fg = _grabcut_fg(0.45)

    # Find largest connected FG component
    num_lbl, cc_map, cc_stats, _ = cv2.connectedComponentsWithStats(fg)
    if num_lbl > 1:
        areas = cc_stats[1:, cv2.CC_STAT_AREA]
        best_cc = int(np.argmax(areas)) + 1
        fg_main = (cc_map == best_cc).astype(np.uint8)
    else:
        fg_main = fg

    # Tight bounding box of dominant FG
    rows = np.where(fg_main.any(axis=1))[0]
    cols = np.where(fg_main.any(axis=0))[0]
    if len(rows) > 0 and len(cols) > 0:
        y1_fg, y2_fg = int(rows[0]), int(rows[-1])
        x1_fg, x2_fg = int(cols[0]), int(cols[-1])
    else:
        x1_fg, y1_fg, x2_fg, y2_fg = W//4, H//4, 3*W//4, 3*H//4

    dets = []

    if n == 2:
        # Label 0 → dominant FG (primary object, e.g. bottle)
        dets.append({'label': labels[0],
                     'box': [x1_fg, y1_fg, x2_fg, y2_fg], 'score': 0.75})
        # Label 1 → complementary region (background object, e.g. table)
        # Use the bounding box of the entire non-FG region
        bg = (1 - fg_main)
        bg_rows = np.where(bg.any(axis=1))[0]
        bg_cols = np.where(bg.any(axis=0))[0]
        if len(bg_rows) > 0 and len(bg_cols) > 0:
            x1_bg = int(bg_cols[0]); y1_bg = int(bg_rows[0])
            x2_bg = int(bg_cols[-1]); y2_bg = int(bg_rows[-1])
        else:
            x1_bg, y1_bg, x2_bg, y2_bg = 0, 0, W, H
        dets.append({'label': labels[1],
                     'box': [x1_bg, y1_bg, x2_bg, y2_bg], 'score': 0.60})
        return dets

    # n >= 3: HSV K-means to produce n distinct region proposals
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)
    crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, km_labels, _ = cv2.kmeans(pixels, n, None, crit, 5,
                                      cv2.KMEANS_PP_CENTERS)
        km_map = km_labels.flatten().reshape(H, W)
        for k in range(n):
            region = (km_map == k).astype(np.uint8)
            r_rows = np.where(region.any(axis=1))[0]
            r_cols = np.where(region.any(axis=0))[0]
            if not len(r_rows) or not len(r_cols): continue
            area_frac = region.mean()
            if area_frac < 0.02: continue
            dets.append({'label': labels[k % len(labels)],
                         'box': [int(r_cols[0]), int(r_rows[0]),
                                 int(r_cols[-1]), int(r_rows[-1])],
                         'score': 0.55})
    except cv2.error:
        # Ultimate fallback: quadrant grid
        for i, (r1, r2, c1, c2) in enumerate([
                (0, H//2, 0, W//2), (0, H//2, W//2, W),
                (H//2, H, 0, W//2), (H//2, H, W//2, W)])[:n]:
            dets.append({'label': labels[i], 'box': [c1,r1,c2,r2], 'score':0.4})

    return dets


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-CUE VOTE  (multi-label version)
# ═════════════════════════════════════════════════════════════════════════════

def multi_label_vote(points, images, cameras,
                     masks_by_label: dict[str, dict[str, np.ndarray]],
                     all_labels: list[str],
                     vote_thresh: float = 0.40,
                     min_votes:   int   = 2):
    """
    For each 3D point, accumulate per-label mask votes across all cameras.
    Assign the label with the highest weighted vote, provided it clears
    vote_thresh.  Points that don't clear any threshold are labelled 'bg'.

    Returns
    ───────
    label_assignments : np.ndarray[str] of shape (N,)
    label_scores      : np.ndarray[float] of shape (N, n_labels)
    """
    N         = len(points)
    L         = len(all_labels)
    xyz       = np.array([p['xyz'] for p in points])
    score_sum = np.zeros((N, L), np.float32)
    vis_cnt   = np.zeros(N, np.int32)

    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    for img_id, img_data in images.items():
        name = img_data['name']
        cam  = cameras[img_data['camera_id']]
        W_c, H_c = cam['w'], cam['h']
        R, t = qvec_to_rotmat(img_data['qvec']), img_data['tvec']

        uv, inf, _ = project_points(xyz, R, t, cam)
        u_i = uv[:, 0].astype(np.int32)
        v_i = uv[:, 1].astype(np.int32)
        ib  = inf & (u_i >= 0) & (u_i < W_c) & (v_i >= 0) & (v_i < H_c)
        idx = np.where(ib)[0]
        if not len(idx): continue

        vis_cnt[idx] += 1

        for li, lbl in enumerate(all_labels):
            mask_dict = masks_by_label.get(lbl, {})
            raw = mask_dict.get(name)
            if raw is None: continue

            # Resize mask to camera resolution
            if raw.shape != (H_c, W_c):
                raw = np.array(
                    Image.fromarray(raw.astype(np.uint8) * 255)
                         .resize((W_c, H_c), Image.NEAREST)) > 127

            vals = raw[v_i[idx], u_i[idx]].astype(np.float32)
            score_sum[idx, li] += vals

    # Normalise
    vis_safe  = np.maximum(vis_cnt, 1)
    norm      = score_sum / vis_safe[:, None]

    # Assignment
    enough       = vis_cnt >= min_votes
    best_label_i = np.argmax(norm, axis=1)
    best_score   = norm[np.arange(N), best_label_i]

    assignments = np.where(
        enough & (best_score >= vote_thresh),
        np.array(all_labels)[best_label_i],
        'bg')

    # Score distribution
    for li, lbl in enumerate(all_labels):
        pts_lbl = np.sum(assignments == lbl)
        print(f"  🏷  '{lbl}': {pts_lbl} pts  "
              f"(mean score={norm[assignments==lbl, li].mean():.3f})"
              if pts_lbl else f"  🏷  '{lbl}': 0 pts")

    print(f"  🏷  'bg' (unassigned): {np.sum(assignments=='bg')} pts")
    return assignments, norm


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SEGMENTATION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_multi_segmentation(colmap_dir: str,
                            images_dir: str,
                            output_dir: str,
                            prompt: str | None = None,
                            auto_discover: bool = False,
                            n_auto_clusters: int = 6,
                            vote_thresh: float = 0.40,
                            min_votes:   int   = 2,
                            mask_images: bool  = True,
                            device: str = 'cpu',
                            rep_frames: int = 6):
    """
    Full multi-object segmentation pipeline.

    Parameters
    ──────────
    colmap_dir      : path to sparse/0 (cameras.txt, images.txt, points3D.txt)
    images_dir      : path to images/
    output_dir      : where to write outputs
    prompt          : comma-separated object labels, e.g. "bottle, table, floor"
                      or None if auto_discover=True
    auto_discover   : if True, use DINO-v2 K-means to find objects automatically
    n_auto_clusters : number of clusters for auto-discovery
    vote_thresh     : min weighted vote fraction to assign a label
    min_votes       : min number of cameras that must have seen a point
    mask_images     : write masked image sets per label
    device          : 'cpu' | 'cuda' | 'mps'
    rep_frames      : number of representative frames for detection pass
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load COLMAP ───────────────────────────────────────────────────────────
    print("\n📂 Reading COLMAP model …")
    cameras = read_cameras_txt(os.path.join(colmap_dir, "cameras.txt"))
    images  = read_images_txt (os.path.join(colmap_dir, "images.txt"))
    points  = read_points3D_txt(os.path.join(colmap_dir, "points3D.txt"))
    print(f"   {len(cameras)} cams | {len(images)} imgs | {len(points)} pts")

    points = sanitise_points(points)

    image_paths = sorted({
        os.path.join(images_dir, d['name'])
        for d in images.values()
        if os.path.exists(os.path.join(images_dir, d['name']))})
    print(f"   {len(image_paths)} image files on disk")

    # ── Parse labels ─────────────────────────────────────────────────────────
    if prompt:
        prompt_labels = [p.strip() for p in prompt.split(',') if p.strip()]
        print(f"\n🔍 Text prompt labels: {prompt_labels}")
    elif auto_discover:
        prompt_labels = None
        print(f"\n🔍 Auto-discover mode: {n_auto_clusters} clusters")
    else:
        raise ValueError("Provide --prompt or --auto_discover")

    # ── Generate 2D masks per label ───────────────────────────────────────────
    mask_cache = os.path.join(output_dir, "multi_masks_cache")
    print("\n🎭 Generating per-label 2D masks …")
    masks_by_label, all_labels = generate_multi_object_masks(
        image_paths, prompt_labels, mask_cache,
        device=device,
        n_auto_clusters=n_auto_clusters,
        rep_frames=rep_frames)

    if not all_labels:
        print("❌ No labels found — aborting"); return {}

    # ── Multi-label 3D vote ───────────────────────────────────────────────────
    print("\n🗳️  Multi-label 3D vote …")
    min_v = max(min_votes, int(len(images) * 0.15))
    assignments, scores = multi_label_vote(
        points, images, cameras, masks_by_label, all_labels,
        vote_thresh=vote_thresh, min_votes=min_v)

    # ── Per-label post-processing and write ───────────────────────────────────
    results_meta = {}
    for lbl in all_labels:
        mask_lbl = assignments == lbl
        lbl_pts  = [p for p, m in zip(points, mask_lbl) if m]
        if not lbl_pts:
            print(f"  ⚠️  '{lbl}': 0 pts — skipping"); continue

        print(f"\n🔧 Post-processing '{lbl}' ({len(lbl_pts)} pts) …")

        # DBSCAN to remove isolated noise
        lbl_pts = depth_aware_dbscan(lbl_pts, images, cameras)
        lbl_pts = statistical_outlier_removal(lbl_pts, k=15, std_ratio=2.0)
        lbl_pts = sanitise_points(lbl_pts, pct=99.5)

        if not lbl_pts:
            print(f"  ⚠️  '{lbl}': 0 pts after cleanup — skipping"); continue

        # Write points3D file
        safe_lbl   = lbl.replace(' ', '_').replace('/', '_')
        out_pts    = os.path.join(output_dir, f"points3D_{safe_lbl}.txt")
        write_points3D_txt(out_pts, lbl_pts)
        # try_write_bin needs cameras.txt + images.txt in the same dir.
        # These live in colmap_dir, not output_dir, so we skip the bin
        # conversion here — the pipeline merge step regenerates .bin from
        # the main colmap_dir after merging all edited clouds back in.

        # Compute metadata
        xyz_arr  = np.array([p['xyz'] for p in lbl_pts])
        centroid = xyz_arr.mean(axis=0).tolist()
        bbox_min = xyz_arr.min(axis=0).tolist()
        bbox_max = xyz_arr.max(axis=0).tolist()

        results_meta[lbl] = {
            'n_points': len(lbl_pts),
            'centroid': centroid,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'points_file': out_pts,
        }
        print(f"  ✅ '{lbl}': {len(lbl_pts)} pts → {out_pts}")

        # Masked images
        if mask_images:
            mdir = os.path.join(output_dir, f"images_masked_{safe_lbl}")
            _write_per_label_masked_images(
                images_dir, masks_by_label[lbl], mdir)
            results_meta[lbl]['masked_images_dir'] = mdir

    # Write scene metadata
    meta_path = os.path.join(output_dir, "scene_objects.json")
    with open(meta_path, 'w') as f:
        json.dump(results_meta, f, indent=2)
    print(f"\n📄 Metadata → {meta_path}")
    print(f"\n🎉 Multi-object segmentation complete!  Labels: {list(results_meta)}")
    return results_meta


# ═════════════════════════════════════════════════════════════════════════════
# MASKED IMAGE WRITER (per label)
# ═════════════════════════════════════════════════════════════════════════════

def _write_per_label_masked_images(images_dir: str,
                                    mask_dict: dict[str, np.ndarray],
                                    output_dir: str):
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(images_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for name in all_imgs:
        src  = os.path.join(images_dir, name)
        dst  = os.path.join(output_dir, name)
        mask = mask_dict.get(name)
        if mask is not None:
            img = np.array(Image.open(src).convert('RGB'))
            if mask.shape[:2] != img.shape[:2]:
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255)
                         .resize((img.shape[1], img.shape[0]),
                                 Image.NEAREST)) > 127
            # Erode slightly to clean up jagged edges
            ek   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask.astype(np.uint8), ek, iterations=2).astype(bool)
            img[~mask] = [0, 0, 0]
            Image.fromarray(img).save(dst)
        else:
            orig = Image.open(src)
            Image.new("RGB", orig.size, (0, 0, 0)).save(dst)
    print(f"  📸 Masked images → {output_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Multi-object 3D segmentation with text prompting  (v1)")
    ap.add_argument("--colmap_dir",      required=True,
                    help="Path to COLMAP sparse/0 directory")
    ap.add_argument("--images_dir",      required=True,
                    help="Path to images directory")
    ap.add_argument("--output_dir",      required=True,
                    help="Where to write output files")
    ap.add_argument("--prompt",          default=None,
                    help="Comma-separated object labels, e.g. 'bottle, table'")
    ap.add_argument("--auto_discover",   action="store_true",
                    help="Auto-discover objects without a text prompt")
    ap.add_argument("--n_clusters",      type=int, default=6,
                    help="Number of clusters for auto-discovery (default 6)")
    ap.add_argument("--vote_thresh",     type=float, default=0.40,
                    help="Min vote fraction to assign a label (default 0.40)")
    ap.add_argument("--min_votes",       type=int,   default=2,
                    help="Min cameras that must see a point (default 2)")
    ap.add_argument("--rep_frames",      type=int,   default=6,
                    help="Representative frames for detection pass (default 6)")
    ap.add_argument("--no_mask_images",  action="store_true",
                    help="Skip writing masked image directories")
    ap.add_argument("--device",          default="cpu",
                    choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    if not args.prompt and not args.auto_discover:
        ap.error("Provide --prompt 'label1, label2' or --auto_discover")

    run_multi_segmentation(
        colmap_dir      = args.colmap_dir,
        images_dir      = args.images_dir,
        output_dir      = args.output_dir,
        prompt          = args.prompt,
        auto_discover   = args.auto_discover,
        n_auto_clusters = args.n_clusters,
        vote_thresh     = args.vote_thresh,
        min_votes       = args.min_votes,
        mask_images     = not args.no_mask_images,
        device          = args.device,
        rep_frames      = args.rep_frames,
    )
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()