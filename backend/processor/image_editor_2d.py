"""
image_editor_2d.py  (v1)
═══════════════════════════════════════════════════════════════════════════════

Image-Space Object Editor  —  the RIGHT approach for colour/texture edits.
────────────────────────────────────────────────────────────────────────────
WHY IMAGE-SPACE IS CORRECT
  Gaussian Splatting optimises Gaussian colours by minimising photometric
  loss against the training images.  If we only change colours in the 3D
  point cloud (points3D.txt), OpenSplat will simply override those colours
  within a few hundred training steps to match the original pink-bottle pixels.

  The ONLY way to get a reliably gold bottle out of OpenSplat is to:
    1. Segment the object in every training image (get a per-pixel mask)
    2. Paint the desired colour/texture onto those pixels
    3. Run MASt3R + OpenSplat on the MODIFIED images

  This module does exactly that.

PIPELINE POSITION
  Run BEFORE mast3r_reconstruct.py (or between MASt3R and OpenSplat if you
  want to keep the original reconstruction geometry).

  Typical call from pipeline.py:
    image_editor_2d.py edit_images \
        --images_dir  path/to/images \
        --output_dir  path/to/edited_images \
        --prompt      "bottle: make it metallic gold; table: wood texture"
        --device      cuda

SEGMENTATION BACKENDS (in priority order)
  1. GroundingDINO + SAM2  — best quality, requires weights
  2. GroundingDINO + GrabCut — good, requires only GroundingDINO weights
  3. GrabCut foreground      — no external weights needed, works well for
                               centred objects against distinct backgrounds
  4. Colour-based K-means    — last resort, always produces something

GROUNDING DINO PATH RESOLUTION
  Searches (in order):
    a. $GDINO_CONFIG / $GDINO_CKPT  environment variables
    b. <repo_root>/GroundingDINO/groundingdino/config/
       <repo_root>/GroundingDINO/weights/
    c. ~/.cache/groundingdino/
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, sys, re, json, argparse, warnings, shutil
import numpy as np
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Module-level GroundingDINO model cache — load once, reuse for all images.
# Without this, load_model() is called 22× (once per image) which takes ~30s
# each time and prints the BertModel warning 22 times.
# ─────────────────────────────────────────────────────────────────────────────
_GDINO_MODEL_CACHE: dict = {}   # key: (cfg, ckpt, device) → model


# ═════════════════════════════════════════════════════════════════════════════
# COLOUR / TEXTURE TABLES  (same as object_editor_3d.py)
# ═════════════════════════════════════════════════════════════════════════════

NAMED_COLOURS: dict[str, tuple[int, int, int]] = {
    'red':      (220,  30,  30), 'green':   ( 30, 180,  30),
    'blue':     ( 30,  80, 220), 'yellow':  (240, 220,  20),
    'orange':   (240, 120,  10), 'purple':  (140,  30, 200),
    'pink':     (240,  80, 160), 'cyan':    ( 20, 200, 220),
    'white':    (255, 255, 255), 'black':   (  0,   0,   0),
    'grey':     (128, 128, 128), 'gray':    (128, 128, 128),
    'brown':    (120,  70,  30), 'gold':    (212, 175,  55),
    'silver':   (192, 192, 192), 'bronze':  (205, 127,  50),
    'navy':     ( 10,  30, 100), 'teal':    ( 10, 140, 140),
    'magenta':  (210,  20, 180), 'lime':    (100, 220,  20),
    'maroon':   (128,   0,   0), 'olive':   (128, 128,   0),
    'coral':    (255,  80,  60), 'crimson': (220,  20,  60),
    'turquoise':(  64, 224, 208),'beige':   (245, 245, 220),
    'indigo':   ( 75,   0, 130), 'violet':  (140,  60, 200),
}

METALLIC_BASE = {
    'gold':   (212, 175,  55),
    'silver': (180, 180, 190),
    'bronze': (180, 110,  40),
    'copper': (160,  90,  40),
}

# Sheen is a DARKER, richer version of the base — not brighter.
# Bright sheen on an already-bright surface = white wash.
# Darker sheen = depth + visible metallic colour.
METALLIC_SHEEN = {
    'gold':   (255, 210,  40),   # saturated warm gold, not white
    'silver': (210, 215, 225),
    'bronze': (220, 140,  50),
    'copper': (200, 110,  50),
}

TEXTURE_PALETTES = {
    'wood':     [(0.35,(160, 90, 40)),(0.30,(140, 75, 30)),
                 (0.20,(180,110, 55)),(0.15,(200,130, 70))],
    'marble':   [(0.45,(240,238,232)),(0.25,(200,195,185)),
                 (0.20,(160,155,148)),(0.10,( 80, 80, 80))],
    'rust':     [(0.40,(150, 55, 15)),(0.30,(180, 80, 20)),
                 (0.20,(100, 40, 10)),(0.10,( 60, 30,  5))],
    'concrete': [(0.50,(150,150,148)),(0.30,(130,130,128)),
                 (0.20,(170,168,165))],
    'glass':    [(0.60,(200,220,240)),(0.40,(180,200,220))],
    'ceramic':  [(0.55,(245,240,235)),(0.30,(230,225,218)),
                 (0.15,(215,210,200))],
    'leather':  [(0.45,( 80, 45, 20)),(0.35,( 60, 35, 15)),
                 (0.20,(100, 60, 30))],
}


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT PARSER
# ═════════════════════════════════════════════════════════════════════════════

def parse_multi_label_prompt(prompt: str) -> dict[str, str]:
    """
    Parse "label1: edit1; label2: edit2" into {label: edit_prompt}.
    Also handles plain edit prompts (no label) → {'__all__': prompt}.
    """
    result = {}
    parts  = [p.strip() for p in prompt.split(';') if p.strip()]
    for part in parts:
        if ':' in part:
            lbl, ep = part.split(':', 1)
            result[lbl.strip().lower()] = ep.strip()
        else:
            result['__all__'] = part.strip()
    return result if result else {'__all__': prompt.strip()}


def parse_colour_from_prompt(prompt: str) -> tuple | None:
    """Extract target (R,G,B) from an edit prompt string."""
    text = prompt.lower().strip()

    # Metallic — highest priority
    m = re.search(r'\bmetallic\s+(\w+)|\b(\w+)\s+metallic\b', text)
    if m:
        base = (m.group(1) or m.group(2) or 'silver').strip()
        return METALLIC_BASE.get(base, NAMED_COLOURS.get(base, (192, 192, 192)))
    if re.search(r'\bmetallic\b', text):
        return METALLIC_BASE['silver']

    # Hex
    hex_m = re.search(r'#([0-9a-f]{6})', text)
    if hex_m:
        hx = hex_m.group(1)
        return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))

    # Named colour
    for name, rgb in NAMED_COLOURS.items():
        if re.search(rf'\b{name}\b', text):
            return rgb

    return None


def parse_texture_from_prompt(prompt: str) -> str | None:
    """Return texture palette name if detected in prompt, else None."""
    text = prompt.lower()
    for tex in TEXTURE_PALETTES:
        if re.search(rf'\b{tex}\b', text):
            return tex
    return None


def parse_brightness_from_prompt(prompt: str) -> float | None:
    """Return brightness multiplier if detected, else None."""
    text = prompt.lower()
    m = re.search(r'\b(bright(?:en)?|lighten)\s*(?:by\s*)?([\d.]+)?', text)
    if m:
        return float(m.group(2)) if m.group(2) else 1.35
    m = re.search(r'\b(dark(?:en)?)\s*(?:by\s*)?([\d.]+)?', text)
    if m:
        return float(m.group(2)) if m.group(2) else 0.65
    return None

def parse_desaturate_from_prompt(prompt: str) -> bool:
    text = prompt.lower()
    return bool(re.search(r'\b(grey|gray|greyscale|grayscale|desatur|monochrome|remove colou?r)\b', text))


# ═════════════════════════════════════════════════════════════════════════════
# GROUNDINGDINO PATH RESOLVER
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_gdino_paths() -> tuple[str | None, str | None]:
    """
    Return (config_path, ckpt_path) for GroundingDINO, or (None, None).

    Search order:
      1. Environment variables  GDINO_CONFIG, GDINO_CKPT
      2. <repo_root>/GroundingDINO/  (installed alongside vok-vision-main)
      3. ~/.cache/groundingdino/
    """
    # 1. Env vars
    cfg  = os.environ.get("GDINO_CONFIG",  "")
    ckpt = os.environ.get("GDINO_CKPT",    "")
    if cfg and ckpt and os.path.exists(cfg) and os.path.exists(ckpt):
        return cfg, ckpt

    # 2. Walk up from _HERE to find repo root (contains GroundingDINO/ dir)
    #    The processor lives at <repo>/backend/processor/
    #    GroundingDINO lives at <repo>/GroundingDINO/
    candidate_roots = [
        Path(_HERE),
        Path(_HERE).parent,          # backend/
        Path(_HERE).parent.parent,   # repo root  ← most likely
        Path(_HERE).parent.parent.parent,
    ]
    for root in candidate_roots:
        gdino_root = root / "GroundingDINO"
        if not gdino_root.is_dir():
            continue

        # Config: look in groundingdino/config/ or top-level
        for cfg_rel in [
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "groundingdino/config/GroundingDINO_SwinB_cfg.py",
            "GroundingDINO_SwinT_OGC.py",
        ]:
            cfg_cand = gdino_root / cfg_rel
            if cfg_cand.exists():
                cfg = str(cfg_cand)
                break

        # Checkpoint: look in weights/ or top-level
        for ckpt_rel in [
            "weights/groundingdino_swint_ogc.pth",
            "weights/groundingdino_swinb_cogcoor.pth",
            "groundingdino_swint_ogc.pth",
            "groundingdino_swinb_cogcoor.pth",
        ]:
            ckpt_cand = gdino_root / ckpt_rel
            if ckpt_cand.exists():
                ckpt = str(ckpt_cand)
                break

        if cfg and ckpt and os.path.exists(cfg) and os.path.exists(ckpt):
            return cfg, ckpt

    # 3. ~/.cache/groundingdino/
    cache = Path.home() / ".cache" / "groundingdino"
    cfg_c  = str(cache / "GroundingDINO_SwinT_OGC.py")
    ckpt_c = str(cache / "groundingdino_swint_ogc.pth")
    if os.path.exists(cfg_c) and os.path.exists(ckpt_c):
        return cfg_c, ckpt_c

    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

def _get_gdino_box(img_np: np.ndarray, label: str, device: str,
                   box_thresh: float = 0.30,
                   text_thresh: float = 0.25) -> list | None:
    """
    Run GroundingDINO for a single label on a single image.
    Returns [x1, y1, x2, y2] in pixel coords, or None.
    Model is cached in _GDINO_MODEL_CACHE so it is loaded ONCE across all images.
    """
    cfg, ckpt = _resolve_gdino_paths()
    if not cfg or not ckpt:
        return None
    try:
        # Add GroundingDINO package to path if needed
        gdino_root = str(Path(cfg).parent.parent.parent)
        if gdino_root not in sys.path:
            sys.path.insert(0, gdino_root)

        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
        import torch
        from torchvision.ops import box_convert

        # Load model once and cache it — avoids 22× slow BertModel init
        cache_key = (cfg, ckpt, device)
        if cache_key not in _GDINO_MODEL_CACHE:
            print(f"  🤖 Loading GroundingDINO model (one-time) …")
            _GDINO_MODEL_CACHE[cache_key] = load_model(cfg, ckpt, device=device)
        model = _GDINO_MODEL_CACHE[cache_key]
        caption = label.strip() + " ."
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_pil = Image.fromarray(img_np)
        img_t, _ = transform(img_pil, None)

        boxes, logits, _ = predict(
            model=model, image=img_t, caption=caption,
            box_threshold=box_thresh, text_threshold=text_thresh,
            device=device)

        if not len(boxes):
            return None

        H, W = img_np.shape[:2]
        boxes_px = box_convert(
            boxes * torch.tensor([W, H, W, H], dtype=torch.float32),
            in_fmt='cxcywh', out_fmt='xyxy').numpy()

        # Return highest-scoring box
        best = int(logits.argmax())
        return boxes_px[best].tolist()

    except Exception as e:
        print(f"    ℹ️  GroundingDINO: {e}")
        return None


def _sam2_mask_from_box(img_np: np.ndarray, box: list, device: str) -> np.ndarray | None:
    """SAM2 mask from a bounding box. Returns bool mask or None."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = os.environ.get("SAM2_CHECKPOINT",
                              str(Path.home()/".cache"/"sam2"/"sam2.1_hiera_small.pt"))
        cfg  = os.environ.get("SAM2_CONFIG", "sam2.1/sam2.1_hiera_small.yaml")
        sam2 = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
        pred = SAM2ImagePredictor(sam2)
        pred.set_image(img_np)
        masks, scores, _ = pred.predict(
            point_coords=None, point_labels=None,
            box=np.array(box, dtype=np.float32)[None, :],
            multimask_output=False)
        return masks[np.argmax(scores)].astype(bool)
    except Exception:
        return None


def _grabcut_mask_from_box(img_np: np.ndarray, box: list) -> np.ndarray:
    """GrabCut mask initialised from a bounding box."""
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


def _grabcut_fg_mask(img_np: np.ndarray) -> np.ndarray:
    """
    GrabCut foreground extraction with no bounding-box hint.
    Uses the image centre as the dominant-object seed.
    """
    import cv2
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    # Try three progressively wider init rects
    best_mask = np.zeros((H, W), np.uint8)
    for frac in [0.40, 0.55, 0.65]:
        rw = int(W * frac); rh = int(H * frac)
        rx = (W - rw)//2;   ry = (H - rh)//2
        bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
        msk = np.zeros((H, W), np.uint8)
        try:
            cv2.grabCut(img_bgr, msk, (rx, ry, rw, rh), bgd, fgd, 7,
                        cv2.GC_INIT_WITH_RECT)
            fg = ((msk == cv2.GC_FGD) | (msk == cv2.GC_PR_FGD)).astype(np.uint8)
            if fg.sum() > best_mask.sum():
                best_mask = fg
        except cv2.error:
            pass

    if best_mask.sum() == 0:
        cx, cy = W//2, H//2
        cv2.ellipse(best_mask, (cx, cy), (int(W*0.25), int(H*0.35)), 0, 0, 360, 1, -1)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, k, iterations=3)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN,  k, iterations=1)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best_mask = cv2.erode(best_mask, ek, iterations=2)
    return best_mask.astype(bool)


def get_object_mask(img_np: np.ndarray, label: str, device: str) -> np.ndarray:
    """
    Return a bool mask (H, W) for `label` in `img_np`.
    Tries GroundingDINO+SAM2, GroundingDINO+GrabCut, then pure GrabCut.
    """
    H, W = img_np.shape[:2]

    # ── Try GroundingDINO for a bounding box ──────────────────────────────────
    box = _get_gdino_box(img_np, label, device)

    if box is not None:
        # Try SAM2 first for precise mask
        mask = _sam2_mask_from_box(img_np, box, device)
        if mask is None:
            mask = _grabcut_mask_from_box(img_np, box)
        return mask

    # ── Fallback: GrabCut foreground (works for centred objects) ─────────────
    print(f"    ℹ️  No detection for '{label}' — using GrabCut foreground")
    return _grabcut_fg_mask(img_np)


# ═════════════════════════════════════════════════════════════════════════════
# PIXEL-LEVEL COLOUR / TEXTURE APPLICATOR
# ═════════════════════════════════════════════════════════════════════════════

def _apply_colour_to_pixels(img_np: np.ndarray,
                              mask: np.ndarray,
                              edit_prompt: str) -> np.ndarray:
    """
    Apply colour/texture edit to the masked pixels of img_np.
    Returns a new uint8 RGB array of the same shape.
    """
    out = img_np.copy().astype(np.float32)
    px  = out[mask]                         # (N, 3)

    if px.size == 0:
        return img_np

    # ── Desaturate ────────────────────────────────────────────────────────────
    if parse_desaturate_from_prompt(edit_prompt):
        lum = px @ np.array([0.299, 0.587, 0.114])
        px  = np.stack([lum, lum, lum], axis=1)
        out[mask] = np.clip(px, 0, 255)
        return out.astype(np.uint8)

    # ── Texture palette ───────────────────────────────────────────────────────
    tex_name = parse_texture_from_prompt(edit_prompt)
    if tex_name:
        palette = TEXTURE_PALETTES[tex_name]
        rng     = np.random.default_rng(42)
        N       = px.shape[0]
        weights = np.array([w for w, _ in palette], np.float32)
        weights /= weights.sum()
        assigned = rng.choice(len(palette), size=N, p=weights)
        result   = np.zeros_like(px)
        for i, (_, col) in enumerate(palette):
            m_i = assigned == i
            c   = np.array(col, np.float32)
            noise = rng.normal(0, 0.06 * 255, (int(m_i.sum()), 3))
            result[m_i] = np.clip(c + noise, 0, 255)
        out[mask] = result
        return out.astype(np.uint8)

    # ── Metallic ──────────────────────────────────────────────────────────────
    text = edit_prompt.lower()
    metallic_m = re.search(r'\bmetallic\s+(\w+)|\b(\w+)\s+metallic\b', text)
    is_metallic = metallic_m or re.search(r'\bmetallic\b', text)
    if is_metallic:
        if metallic_m:
            base_name = (metallic_m.group(1) or metallic_m.group(2) or 'silver').strip()
        else:
            base_name = 'silver'
        base_rgb  = np.array(METALLIC_BASE.get(base_name,  (192, 192, 192)), np.float32)
        sheen_rgb = np.array(METALLIC_SHEEN.get(base_name, (235, 235, 245)), np.float32)

        # ── Correct metallic gold tinting ─────────────────────────────────────
        # The key insight: we want pixels to look gold, not to look physically
        # metallic (that requires ray-tracing).  The correct image-space operation:
        #
        # 1. Compute original luminance (brightness) of each pixel
        # 2. Replace hue with gold — set all pixels to gold base colour
        # 3. Scale the gold by original luminance so dark→dark gold, bright→bright gold
        # 4. Add a SUBTLE darker-gold sheen on the brightest pixels for depth
        #
        # This preserves the bottle's shape/shading while making it look gold.

        # Step 1: original luminance [0..255]
        orig_lum = (px @ np.array([0.299, 0.587, 0.114], np.float32))   # (N,)
        orig_lum_n = orig_lum / 255.0   # normalise to [0,1]

        # Step 2: scale gold base by original luminance
        # dark bottle areas → dark gold; bright areas → bright gold
        # This preserves all the original 3D shading/reflectance information
        tinted = base_rgb[None, :] * orig_lum_n[:, None]   # (N,3) — luminance-scaled gold

        # Step 3: very subtle sheen on mid-to-high luminance pixels only
        # sheen_mask = pixels brighter than median (simulate specular highlight)
        sheen_mask = orig_lum_n > 0.60
        if sheen_mask.any():
            sheen_strength = 0.20   # 20% sheen, 80% base gold — keeps colour visible
            sheen_factor = (orig_lum_n[sheen_mask] - 0.60) / 0.40   # ramp 0→1
            tinted[sheen_mask] = (
                tinted[sheen_mask] * (1 - sheen_strength * sheen_factor[:, None]) +
                sheen_rgb[None, :] * sheen_strength * sheen_factor[:, None]
            )

        # Step 4: blend 85% tinted gold + 15% original (preserves subtle surface detail)
        px = tinted * 0.85 + px * 0.15

        out[mask] = np.clip(px, 0, 255)
        return out.astype(np.uint8)

    # ── Named / hex colour → luminance-preserving recolour ──────────────────
    target_rgb = parse_colour_from_prompt(edit_prompt)
    if target_rgb is not None:
        target = np.array(target_rgb, np.float32)
        # Luminance-preserving tint: scale target by original pixel brightness.
        # This keeps the bottle's 3D shading so the result looks painted, not flat.
        orig_lum_n = (px @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
        tinted = target[None, :] * orig_lum_n[:, None]   # dark→dark colour, bright→bright
        # 85% tinted colour + 15% original for subtle detail preservation
        px = tinted * 0.85 + px * 0.15
        out[mask] = np.clip(px, 0, 255)
        return out.astype(np.uint8)

    # ── Brightness only ───────────────────────────────────────────────────────
    factor = parse_brightness_from_prompt(edit_prompt)
    if factor is not None:
        out[mask] = np.clip(px * factor, 0, 255)
        return out.astype(np.uint8)

    # Nothing matched — return original
    return img_np


# ═════════════════════════════════════════════════════════════════════════════
# MAIN: edit all images in a directory
# ═════════════════════════════════════════════════════════════════════════════

def edit_images(images_dir: str,
                output_dir: str,
                label_prompts: dict[str, str],
                device: str = 'cpu',
                mask_cache_dir: str | None = None,
                backup: bool = True) -> str:
    """
    For every image in `images_dir`, apply the colour/texture edits described
    in `label_prompts` and write the modified images to `output_dir`.

    Parameters
    ──────────
    images_dir    : source images directory
    output_dir    : destination for edited images (may == images_dir for in-place)
    label_prompts : {label_name: edit_prompt} or {'__all__': prompt} for whole image
    device        : 'cpu' | 'cuda' | 'mps'
    mask_cache_dir: if set, save/load per-image masks here to avoid re-computing
    backup        : if output_dir == images_dir, back up originals first

    Returns output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    in_place = os.path.realpath(images_dir) == os.path.realpath(output_dir)

    if in_place and backup:
        bk = images_dir + "_pre_edit"
        if not os.path.isdir(bk):
            shutil.copytree(images_dir, bk)
            print(f"  💾 Backup → {bk}")
        # Work from the backup
        images_dir = bk

    if mask_cache_dir:
        os.makedirs(mask_cache_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"  🖼  Editing {len(img_files)} images  |  labels: {list(label_prompts)}")

    gdino_cfg, gdino_ckpt = _resolve_gdino_paths()
    if gdino_cfg and gdino_ckpt:
        print(f"  ✅ GroundingDINO: cfg={gdino_cfg}")
        print(f"                    ckpt={gdino_ckpt}")
    else:
        print("  ℹ️  GroundingDINO weights not found — using GrabCut foreground")

    for fname in img_files:
        src_path = os.path.join(images_dir,  fname)
        dst_path = os.path.join(output_dir,  fname)
        ext      = Path(fname).suffix.lower()

        img_np = np.array(Image.open(src_path).convert('RGB'))
        edited = img_np.copy()

        for label, edit_prompt in label_prompts.items():
            if label == '__all__':
                # Edit entire image (no segmentation)
                mask = np.ones(img_np.shape[:2], bool)
            else:
                # Load cached mask if available
                if mask_cache_dir:
                    cache_path = os.path.join(mask_cache_dir, f"{label}__{fname}.npy")
                    if os.path.exists(cache_path):
                        mask = np.load(cache_path)
                    else:
                        mask = get_object_mask(img_np, label, device)
                        np.save(cache_path, mask)
                else:
                    mask = get_object_mask(img_np, label, device)

            if not mask.any():
                print(f"    ⚠️  {fname}: no pixels found for '{label}' — skipping")
                # Still write the unmodified image so output_dir has all N images
                continue

            coverage = 100.0 * mask.mean()
            edited   = _apply_colour_to_pixels(edited, mask, edit_prompt)
            print(f"    ✓ {fname}  '{label}' cov={coverage:.1f}%")

        # Save preserving original format quality
        pil_out = Image.fromarray(edited)
        if ext in ('.jpg', '.jpeg'):
            pil_out.save(dst_path, 'JPEG', quality=95, subsampling=0)
        elif ext == '.png':
            pil_out.save(dst_path, 'PNG')
        else:
            pil_out.save(dst_path)

    print(f"  ✅ Edited images → {output_dir}")
    return output_dir


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Image-space object editor for 3DGS pipelines (v1)")
    ap.add_argument("--images_dir",  required=True,
                    help="Source images directory")
    ap.add_argument("--output_dir",  required=True,
                    help="Destination for edited images")
    ap.add_argument("--prompt",      required=True,
                    help='Edit prompt.  Format: "label1: edit1; label2: edit2"  '
                         'e.g. "bottle: make it metallic gold; table: wood texture"  '
                         'or plain "make it red" to edit all pixels')
    ap.add_argument("--device",      default="cpu",
                    choices=["cpu", "cuda", "mps"])
    ap.add_argument("--mask_cache",  default=None,
                    help="Directory to cache per-image segmentation masks")
    ap.add_argument("--no_backup",   action="store_true",
                    help="Skip backing up originals for in-place edits")
    args = ap.parse_args()

    label_prompts = parse_multi_label_prompt(args.prompt)
    print(f"📝 Parsed edit plan: {label_prompts}")

    edit_images(
        images_dir    = args.images_dir,
        output_dir    = args.output_dir,
        label_prompts = label_prompts,
        device        = args.device,
        mask_cache_dir= args.mask_cache,
        backup        = not args.no_backup,
    )
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()