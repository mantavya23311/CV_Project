"""
segment_object_pointcloud.py  (v7)
═══════════════════════════════════════════════════════════════════════════════

BUGS FIXED vs v6
─────────────────
BUG 1 — read_images_txt only loaded 11/22 images (CRITICAL)
  COLMAP images.txt has TWO lines per image:
    Line 1: pose data
    Line 2: 2D point-track observations (often empty)
  The parser filtered blank lines first, then did i+=2.
  Our mast3r_reconstruct.py writes "\n\n" (line1 + blank line).
  After stripping blanks only 22 pose lines remain, and i+=2 reads
  lines 0,2,4,... = 11 images.  GrabCut only ran on 11/22.
  FIX: read_images_txt now reads the raw file correctly, skipping
  comment lines but keeping the structure so i+=2 works properly.

BUG 2 — Hollow black region on the label
  The label area (white/beige) has low HSV saturation, so GrabCut
  tends to classify it as background in some views.  That inconsistency
  produces a bimodal mask vote: sometimes foreground, sometimes not.
  The label points therefore score ~0.4-0.5 and fall near the threshold.
  FIX: Multi-cue scoring — combine mask vote + depth prior + HSV
  foreground colour prior.  The HSV prior gives label-coloured points
  a boost even when the mask is inconsistent.

NEW FEATURES
─────────────
• HSV COLOUR PRIOR: For each point, sample the real image colour at
  its projected location.  Compute HSV.  If the sampled hue matches
  the object's dominant hue (learned from high-scoring mask points),
  boost the score.  This directly fills in the hollow label region.

• TIGHTER DBSCAN: eps now uses 2× mean-NN (was 3×) to separate the
  object cluster from stray background points more aggressively.

• KEEP MORE POINTS: TARGET raised to 120k (was 80k) and DBSCAN guard
  raised to 150k.  47k points was too sparse for a detailed bottle.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═════════════════════════════════════════════════════════════════════════════
# COLMAP I/O
# ═════════════════════════════════════════════════════════════════════════════

def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            p = line.split()
            cameras[int(p[0])] = dict(model=p[1], w=int(p[2]), h=int(p[3]),
                                      params=list(map(float, p[4:])))
    return cameras


def read_images_txt(path):
    """
    Correctly parse COLMAP images.txt.
    Each image entry = 2 lines:
      Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      Line 2: 2D point observations (may be empty / whitespace-only)
    We skip comment lines (#) and blank lines, then step by 2.
    FIX vs v6: we now read raw lines and strip only comment/blank lines
    so that the i+=2 step correctly skips line 2 of each entry.
    """
    images = {}
    with open(path) as f:
        # Keep ALL non-comment lines (including the empty track line)
        raw = [l.rstrip('\n') for l in f
               if not l.startswith('#')]

    i = 0
    while i < len(raw):
        line = raw[i].strip()
        if not line:
            i += 1
            continue
        p = line.split()
        # Expect at least 10 tokens: id qw qx qy qz tx ty tz cam_id name
        if len(p) < 10:
            i += 1
            continue
        try:
            images[int(p[0])] = dict(
                qvec=np.array(list(map(float, p[1:5]))),
                tvec=np.array(list(map(float, p[5:8]))),
                camera_id=int(p[8]),
                name=p[9])
        except (ValueError, IndexError):
            i += 1
            continue
        i += 2   # skip the 2D point-track line
    return images


def read_points3D_txt(path):
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            p = line.split()
            points.append(dict(
                id=int(p[0]),
                xyz=np.array([float(p[1]), float(p[2]), float(p[3])]),
                rgb=np.array([int(p[4]),   int(p[5]),   int(p[6])]),
                error=float(p[7]),
                rest=' '.join(p[8:])))
    return points


def write_points3D_txt(path, points):
    with open(path, 'w') as f:
        for p in points:
            f.write(f"{p['id']} {p['xyz'][0]:.6f} {p['xyz'][1]:.6f} "
                    f"{p['xyz'][2]:.6f} {p['rgb'][0]} {p['rgb'][1]} "
                    f"{p['rgb'][2]} {p['error']:.6f} {p.get('rest','1.0')}\n")


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY UTILS
# ═════════════════════════════════════════════════════════════════════════════

def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w],
        [  2*x*y+2*z*w, 1-2*x*x-2*z*z,   2*y*z-2*x*w],
        [  2*x*z-2*y*w,   2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def project_points(xyz, R, t, cam):
    xyz_c    = (R @ xyz.T).T + t
    in_front = xyz_c[:, 2] > 0.01
    p  = cam['params']
    fx = p[0];  fy = p[1] if len(p) > 1 else p[0]
    cx = p[2] if len(p) > 2 else cam['w'] / 2
    cy = p[3] if len(p) > 3 else cam['h'] / 2
    z  = xyz_c[:, 2]
    u  = fx * xyz_c[:, 0] / (z + 1e-8) + cx
    v  = fy * xyz_c[:, 1] / (z + 1e-8) + cy
    return np.stack([u, v], 1), in_front, xyz_c[:, 2]


# ═════════════════════════════════════════════════════════════════════════════
# SANITISE / SOR / VOXEL
# ═════════════════════════════════════════════════════════════════════════════

def sanitise_points(points, pct=99.5):
    if not points: return points
    xyz   = np.array([p['xyz'] for p in points])
    valid = np.isfinite(xyz).all(axis=1)
    if valid.sum() > 10:
        c = xyz[valid].mean(0)
        d = np.linalg.norm(xyz - c, axis=1)
        valid &= d <= np.percentile(d[valid], pct)
    clean = []
    for p, v in zip(points, valid):
        if not v: continue
        p = dict(p); p['rgb'] = np.clip(p['rgb'], 0, 255).astype(int)
        clean.append(p)
    removed = len(points) - len(clean)
    if removed: print(f"  🧹 Sanitised {removed} pts")
    return clean


def statistical_outlier_removal(points, k=20, std_ratio=2.0):
    if len(points) < k + 1: return points
    try:
        from scipy.spatial import cKDTree
        xyz = np.array([p['xyz'] for p in points])
        d, _ = cKDTree(xyz).query(xyz, k=k+1)
        md   = d[:, 1:].mean(axis=1)
        thr  = md.mean() + std_ratio * md.std()
        kept = [p for p, ok in zip(points, md <= thr) if ok]
        print(f"  🔬 SOR: {len(points)} → {len(kept)}")
        return kept
    except ImportError:
        return points


def voxel_downsample(points, voxel_size=0.005):
    if not points: return points
    xyz = np.array([p['xyz'] for p in points])
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    seen, order = {}, []
    for i, v in enumerate(map(tuple, vox)):
        if v not in seen: seen[v] = i; order.append(i)
    kept = [points[i] for i in order]
    print(f"  🔲 Voxel ({voxel_size:.5f}): {len(points)} → {len(kept)}")
    return kept


# ═════════════════════════════════════════════════════════════════════════════
# DEPTH PRIOR
# ═════════════════════════════════════════════════════════════════════════════

def compute_depth_prior(points, images, cameras):
    """Gaussian score centred on the modal depth (= object surface depth)."""
    N   = len(points)
    xyz = np.array([p['xyz'] for p in points])
    d_sum = np.zeros(N, np.float64)
    d_cnt = np.zeros(N, np.int32)

    for img_data in images.values():
        R, t = qvec_to_rotmat(img_data['qvec']), img_data['tvec']
        cam  = cameras[img_data['camera_id']]
        uv, inf, depths = project_points(xyz, R, t, cam)
        u_i = uv[:, 0].astype(int); v_i = uv[:, 1].astype(int)
        ib  = inf & (u_i >= 0) & (u_i < cam['w']) & (v_i >= 0) & (v_i < cam['h'])
        idx = np.where(ib)[0]
        d_sum[idx] += depths[idx]; d_cnt[idx] += 1

    valid  = d_cnt > 0
    md     = np.where(valid, d_sum / np.maximum(d_cnt, 1), 0.0)
    d_vals = md[valid & (md > 0)]
    if len(d_vals) < 10: return np.ones(N, np.float32)

    hist, edges = np.histogram(d_vals, bins=50)
    pk     = np.argmax(hist)
    d_mode = (edges[pk] + edges[pk+1]) / 2.0
    q25, q75 = np.percentile(d_vals, [25, 75])
    sigma  = max((q75 - q25) / 2.0, d_mode * 0.05)
    prior  = np.exp(-0.5 * ((md - d_mode) / sigma) ** 2)
    prior[~valid] = 0.0
    print(f"  📏 Depth prior: mode={d_mode:.3f}  σ={sigma:.3f}")
    return prior.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# HSV COLOUR PRIOR  (NEW — fixes hollow label region)
# ═════════════════════════════════════════════════════════════════════════════

def _rgb_to_hsv_batch(rgb_uint8: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV.  Input: [N,3] uint8.  Output: [N,3] float [0,1]."""
    r, g, b = rgb_uint8[:, 0]/255.0, rgb_uint8[:, 1]/255.0, rgb_uint8[:, 2]/255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-8

    h = np.zeros(len(r))
    s = np.where(cmax > 0, delta / cmax, 0.0)
    v = cmax

    m = (cmax == r) & (delta > 1e-8)
    h[m] = ((g[m] - b[m]) / delta[m]) % 6
    m = (cmax == g) & (delta > 1e-8)
    h[m] = (b[m] - r[m]) / delta[m] + 2
    m = (cmax == b) & (delta > 1e-8)
    h[m] = (r[m] - g[m]) / delta[m] + 4
    h = h / 6.0

    return np.stack([h, s, v], axis=1)


def compute_colour_prior(points, images, cameras, images_dir,
                         masks_dict, mask_qualities,
                         n_hue_bins=36):
    """
    Sample the real image colour at each point's projected location.
    Build a hue histogram from high-confidence mask-foreground points.
    Score each point by P(hue | object_hue_distribution).

    This directly fixes the hollow label region: the label (white/beige cap
    and text band) projects onto real image pixels that have object-consistent
    hue, even if the GrabCut mask occasionally classifies it as background.

    Steps:
      1. For each image, load the real photo (not the MASt3R tensor).
      2. Project all points into each camera.
      3. For points that land on mask=1 and have high mask vote, sample
         their hue → build object hue histogram H_obj.
      4. For every point, sample hue from each camera and compare to H_obj.
         colour_prior[i] = mean(H_obj[hue_bin]) over visible cameras.
    """
    print("  🎨 Computing HSV colour prior …")
    N   = len(points)
    xyz = np.array([p['xyz'] for p in points])

    # Accumulate hue samples for object histogram
    obj_hue_counts = np.zeros(n_hue_bins, np.float64)
    # Per-point hue score
    hue_score_sum = np.zeros(N, np.float64)
    hue_vis_cnt   = np.zeros(N, np.int32)

    # Pre-load images at camera resolution
    loaded = {}
    for img_id, img_data in images.items():
        name = img_data['name']
        ip   = os.path.join(images_dir, name)
        if not os.path.exists(ip): continue
        cam  = cameras[img_data['camera_id']]
        W, H = cam['w'], cam['h']
        try:
            img_pil = Image.open(ip).convert('RGB')
            if img_pil.size != (W, H):
                img_pil = img_pil.resize((W, H), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.uint8)
        except Exception:
            continue

        # Mask for this camera
        if name in masks_dict:
            mask_raw = masks_dict[name].astype(np.float32)
            if mask_raw.shape != (H, W):
                mask_raw = np.array(Image.fromarray(
                    (mask_raw*255).astype(np.uint8)).resize((W,H), Image.BILINEAR)
                ).astype(np.float32) / 255.0
        else:
            mask_raw = np.ones((H, W), np.float32)

        q = float(mask_qualities.get(name, 0.5))
        loaded[img_id] = (img_np, mask_raw, img_data, cam, q)

    # ── Pass 1: Build object hue histogram from high-confidence foreground ────
    for img_id, (img_np, mask_f, img_data, cam, q) in loaded.items():
        H, W = img_np.shape[:2]
        Rmat, t = qvec_to_rotmat(img_data['qvec']), img_data['tvec']
        uv, inf, _ = project_points(xyz, Rmat, t, cam)
        u_i = uv[:, 0].astype(int); v_i = uv[:, 1].astype(int)
        ib  = inf & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
        idx = np.where(ib)[0]
        if not len(idx): continue

        # High-confidence foreground: mask > 0.6
        mask_val = mask_f[v_i[idx], u_i[idx]]
        fg_idx   = idx[mask_val > 0.6]
        if not len(fg_idx): continue

        sampled_rgb = img_np[v_i[fg_idx], u_i[fg_idx]]  # [K, 3]
        hsv         = _rgb_to_hsv_batch(sampled_rgb)
        hue_bins    = (hsv[:, 0] * n_hue_bins).astype(int) % n_hue_bins
        # Weight by saturation (achromatic pixels shouldn't dominate hue hist)
        sat_w = hsv[:, 1]
        np.add.at(obj_hue_counts, hue_bins, sat_w * q)

    # Normalise → probability
    if obj_hue_counts.sum() > 0:
        obj_hue_prob = obj_hue_counts / obj_hue_counts.sum()
    else:
        obj_hue_prob = np.ones(n_hue_bins) / n_hue_bins
    print(f"  🎨 Object hue distribution built ({obj_hue_counts.sum():.0f} samples)")

    # ── Pass 2: Score each point by hue match ─────────────────────────────────
    for img_id, (img_np, mask_f, img_data, cam, q) in loaded.items():
        H, W = img_np.shape[:2]
        Rmat, t = qvec_to_rotmat(img_data['qvec']), img_data['tvec']
        uv, inf, _ = project_points(xyz, Rmat, t, cam)
        u_i = uv[:, 0].astype(int); v_i = uv[:, 1].astype(int)
        ib  = inf & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
        idx = np.where(ib)[0]
        if not len(idx): continue

        sampled_rgb = img_np[v_i[idx], u_i[idx]]
        hsv         = _rgb_to_hsv_batch(sampled_rgb)
        hue_bins    = (hsv[:, 0] * n_hue_bins).astype(int) % n_hue_bins
        hue_p       = obj_hue_prob[hue_bins]         # P(hue | object)

        # Saturation weight: highly saturated pixels → more reliable hue
        sat_w = hsv[:, 1]
        # For achromatic pixels (label white/cap white), use value instead
        # White pixels on the label belong to the object but have low sat.
        # We give them a bonus if their value is high (bright = label/cap).
        brightness_bonus = np.where(sat_w < 0.15, hsv[:, 2] * 0.5, 0.0)
        final_hue_score  = np.clip(hue_p * sat_w + brightness_bonus, 0, 1)

        hue_score_sum[idx] += final_hue_score * q
        hue_vis_cnt[idx]   += 1

    colour_prior = np.where(
        hue_vis_cnt > 0,
        hue_score_sum / np.maximum(hue_vis_cnt, 1),
        0.0).astype(np.float32)

    # Normalise to [0,1]
    cp_max = colour_prior.max()
    if cp_max > 0: colour_prior /= cp_max

    print(f"  🎨 Colour prior: mean={colour_prior.mean():.3f}  "
          f"max={colour_prior.max():.3f}")
    return colour_prior


# ═════════════════════════════════════════════════════════════════════════════
# MASK QUALITY CHECKER
# ═════════════════════════════════════════════════════════════════════════════

def _check_mask_quality(mask: np.ndarray) -> float:
    if mask.sum() == 0: return 0.0
    h, w = mask.shape
    cov  = float(mask.mean())
    if not (0.02 <= cov <= 0.75):
        return max(0.1, 1.0 - abs(cov - 0.30) / 0.30)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2: return 0.1
    bb_h = rows[-1] - rows[0] + 1; bb_w = cols[-1] - cols[0] + 1
    fill = float(mask[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1].mean())
    ar   = min(bb_h, bb_w) / max(bb_h, bb_w)
    return float(np.clip(0.4*min(cov/0.30, 1.0) + 0.3*ar + 0.3*fill, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-SCALE GRABCUT + HSV K-MEANS FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

def grabcut_multiscale(img_path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(img_path).convert('RGB')),
                           cv2.COLOR_RGB2BGR)
    h, w   = img.shape[:2]
    result = np.zeros((h, w), dtype=np.uint8)

    for frac in [0.35, 0.55, 0.70]:
        rw = int(w * frac); rh = int(h * frac)
        rx = (w - rw) // 2; ry = (h - rh) // 2
        bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
        msk = np.zeros((h, w), np.uint8)
        try:
            cv2.grabCut(img, msk, (rx, ry, rw, rh), bgd, fgd, 5,
                        cv2.GC_INIT_WITH_RECT)
            fg = np.where((msk == cv2.GC_FGD) | (msk == cv2.GC_PR_FGD),
                          1, 0).astype(np.uint8)
            result = np.maximum(result, fg)
        except cv2.error:
            pass

    if result.sum() == 0:
        cy2, cx2 = h // 2, w // 2
        cv2.ellipse(result, (cx2, cy2), (int(w*0.25), int(h*0.35)), 0, 0, 360, 1, -1)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k, iterations=3)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  k, iterations=1)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.erode(result, ek, iterations=1)
    return result.astype(bool)


def _hsv_kmeans_mask(img_path: str) -> np.ndarray:
    try:
        import cv2
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: raise ValueError
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, w    = img_hsv.shape[:2]
        pixels  = img_hsv.reshape(-1, 3)
        crit    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(pixels, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        labels  = labels.flatten()
        bw      = max(5, min(h, w) // 12)
        border  = np.concatenate([np.arange(w), np.arange((h-1)*w, h*w),
                                   np.arange(0, h*w, w), np.arange(w-1, h*w, w)])
        bg_lbl  = int(np.bincount(labels[border]).argmax())
        mask    = (labels == (1-bg_lbl)).reshape(h, w).astype(np.uint8)
        k2      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=3)
        return mask.astype(bool)
    except Exception:
        img = Image.open(img_path)
        return np.ones((img.height, img.width), dtype=bool)


def generate_masks_auto(image_paths, output_mask_dir):
    print("  ✂️  Multi-scale GrabCut + HSV fallback …")
    os.makedirs(output_mask_dir, exist_ok=True)
    masks, qualities = {}, {}

    for img_path in image_paths:
        name = Path(img_path).name
        try:
            m = grabcut_multiscale(img_path)
            q = _check_mask_quality(m)
            if q < 0.35:
                print(f"    ↺ {name}: GrabCut q={q:.2f} → HSV")
                m = _hsv_kmeans_mask(img_path)
                q = _check_mask_quality(m)
            masks[name]     = m
            qualities[name] = q
            print(f"    ✓ {name}  cov={100*m.mean():.1f}%  q={q:.2f}")
        except Exception as e:
            print(f"    ⚠️  {name}: {e}")
            img = Image.open(img_path)
            masks[name]     = np.ones((img.height, img.width), dtype=bool)
            qualities[name] = 0.2
        Image.fromarray((masks[name].astype(np.uint8)) * 255).save(
            os.path.join(output_mask_dir, name))

    covs    = [float(m.mean()) for m in masks.values()]
    med_cov = float(np.median(covs))
    names   = list(masks.keys())
    print(f"  📊 Median coverage: {100*med_cov:.1f}%  ({len(masks)} masks)")

    # Cross-view consistency rerun
    for i, name in enumerate(names):
        cov = covs[i]
        if abs(cov - med_cov) > 0.30 and qualities[name] < 0.5:
            img_path = next((p for p in image_paths if Path(p).name == name), None)
            if not img_path: continue
            print(f"    ⚠️  {name}: outlier cov={100*cov:.1f}% → rerun")
            try:
                import cv2
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: raise ValueError
                h2, w2 = img_bgr.shape[:2]
                tf  = min(np.sqrt(med_cov*1.5)+0.10, 0.85)
                rw2 = int(w2*tf); rh2 = int(h2*tf)
                rx2 = (w2-rw2)//2; ry2 = (h2-rh2)//2
                bgd2 = np.zeros((1,65),np.float64); fgd2 = np.zeros((1,65),np.float64)
                msk2 = np.zeros((h2,w2),np.uint8)
                cv2.grabCut(img_bgr, msk2, (rx2,ry2,rw2,rh2), bgd2, fgd2, 5,
                            cv2.GC_INIT_WITH_RECT)
                fg2  = np.where((msk2==cv2.GC_FGD)|(msk2==cv2.GC_PR_FGD), 1, 0
                                ).astype(np.uint8)
                k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
                fg2  = cv2.morphologyEx(fg2, cv2.MORPH_CLOSE, k3, iterations=2)
                nq   = _check_mask_quality(fg2.astype(bool))
                if nq > qualities[name]:
                    masks[name] = fg2.astype(bool); qualities[name] = nq
                    Image.fromarray(fg2*255).save(
                        os.path.join(output_mask_dir, name))
                    print(f"      ✓ q={nq:.2f}")
            except Exception as e2:
                print(f"      ⚠️  {e2}")

    return masks, qualities


# ═════════════════════════════════════════════════════════════════════════════
# SAM2
# ═════════════════════════════════════════════════════════════════════════════

def _select_best_sam_mask(raw_masks, shape):
    H, W = shape[:2]; cx, cy = W/2, H/2
    best_s, best = -1, None
    for m in raw_masks:
        seg = m['segmentation']; area = m['area']
        ys, xs = np.where(seg)
        if not len(xs): continue
        dist_n = np.sqrt((xs.mean()-cx)**2+(ys.mean()-cy)**2)/(np.sqrt(cx**2+cy**2)+1e-8)
        ar     = area/(H*W)
        score  = (1-dist_n)*0.6 + ar*0.4
        if score > best_s and 0.02 < ar < 0.60:
            best_s, best = score, seg
    return best


def generate_masks_sam2(image_paths, output_mask_dir, device='cpu'):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    ckpt = os.environ.get("SAM2_CHECKPOINT",
                          os.path.expanduser("~/.cache/sam2/sam2.1_hiera_small.pt"))
    cfg  = os.environ.get("SAM2_CONFIG", "sam2.1/sam2.1_hiera_small.yaml")
    sam2 = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
    gen  = SAM2AutomaticMaskGenerator(
        sam2, points_per_side=32, pred_iou_thresh=0.70,
        stability_score_thresh=0.85, min_mask_region_area=200)
    os.makedirs(output_mask_dir, exist_ok=True)
    masks, qualities = {}, {}
    for img_path in image_paths:
        name   = Path(img_path).name
        img_np = np.array(Image.open(img_path).convert('RGB'))
        raw    = gen.generate(img_np)
        best   = _select_best_sam_mask(raw, img_np.shape) if raw else None
        if best is None:
            best = grabcut_multiscale(img_path)
        masks[name]     = best
        qualities[name] = _check_mask_quality(best)
        Image.fromarray((best.astype(np.uint8))*255).save(
            os.path.join(output_mask_dir, name))
    return masks, qualities


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-CUE PROJECTION VOTE  (mask + depth + colour)
# ═════════════════════════════════════════════════════════════════════════════

def multi_cue_vote(points, images, cameras, masks_dict, mask_qualities,
                   vote_thresh=0.50, min_votes=2,
                   depth_prior=None,    depth_weight=0.25,
                   colour_prior=None,   colour_weight=0.1):
    """
    Combined score = mask_weight*mask_score
                   + depth_weight*depth_prior
                   + colour_weight*colour_prior

    mask_weight is derived so weights sum to 1.
    All priors are in [0,1].

    The colour prior specifically fixes the hollow label area:
    the label projects onto pixels whose hue matches the object's
    dominant hue distribution, so label points score well even when
    GrabCut misclassifies them as background in some views.
    """
    mask_weight = 1.0 - depth_weight - colour_weight
    print(f"  🗳️  Multi-cue vote: {len(points)} pts  thresh={vote_thresh:.2f}  "
          f"weights: mask={mask_weight:.2f} depth={depth_weight:.2f} "
          f"colour={colour_weight:.2f}")

    N   = len(points)
    xyz = np.array([p['xyz'] for p in points])
    wfg  = np.zeros(N, np.float32)
    wtot = np.zeros(N, np.float32)
    vis  = np.zeros(N, np.int32)

    # Pre-load and resize masks
    loaded = {}
    for img_id, img_data in images.items():
        name = img_data['name']
        if name not in masks_dict: continue
        cam  = cameras[img_data['camera_id']]
        W, H = cam['w'], cam['h']
        raw  = masks_dict[name].astype(np.float32)
        if raw.shape != (H, W):
            raw = np.array(Image.fromarray((raw*255).astype(np.uint8)
                           ).resize((W, H), Image.BILINEAR)).astype(np.float32)/255.0
        q = float(mask_qualities.get(name, 0.5))
        loaded[img_id] = (raw, img_data, cam, q)

    for img_id, (mf, img_data, cam, q) in loaded.items():
        H, W = mf.shape[:2]
        R, t = qvec_to_rotmat(img_data['qvec']), img_data['tvec']
        uv, inf, _ = project_points(xyz, R, t, cam)
        u_f, v_f = uv[:, 0], uv[:, 1]
        u_i, v_i = u_f.astype(np.int32), v_f.astype(np.int32)
        ib  = inf & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
        idx = np.where(ib)[0]
        if not len(idx): continue

        vis[idx] += 1
        u0 = np.clip(u_i[idx], 0, W-2); v0 = np.clip(v_i[idx], 0, H-2)
        du = np.clip(u_f[idx]-u0, 0, 1); dv = np.clip(v_f[idx]-v0, 0, 1)
        val = ((1-du)*(1-dv)*mf[v0,u0] + du*(1-dv)*mf[v0,u0+1] +
               (1-du)*dv*mf[v0+1,u0]   + du*dv*mf[v0+1,u0+1])
        wfg[idx]  += val * q
        wtot[idx] += q

    enough     = vis >= min_votes
    mask_score = np.where(wtot > 0, wfg / np.maximum(wtot, 1e-8), 0.0)

    score = mask_weight * mask_score
    if depth_prior  is not None: score += depth_weight  * depth_prior
    if colour_prior is not None: score += colour_weight * colour_prior

    kept = [p for p, k in zip(points, enough & (score >= vote_thresh)) if k]

    sv = score[enough]
    if len(sv):
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(sv, bins=bins)
        print("  📊 Score distribution:")
        for i in range(len(hist)):
            bar = '█' * int(28 * hist[i] / max(hist.max(), 1))
            print(f"     [{bins[i]:.1f}–{bins[i+1]:.1f}]  {bar}  {hist[i]}")
    print(f"  ✅ Kept {len(kept)}/{N}")
    return kept


# ═════════════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════

def adaptive_threshold(points, images, cameras, masks_dict, mask_qualities,
                       depth_prior, colour_prior,
                       target_min=5000, target_max=120000,
                       thresh_start=0.55, thresh_floor=0.35):
    thresh    = thresh_start
    step      = 0.05
    min_votes = max(2, int(len(images) * 0.20))
    best_kept = None; best_n = 0

    while thresh >= thresh_floor:
        kept = multi_cue_vote(
            points, images, cameras, masks_dict, mask_qualities,
            vote_thresh=thresh, min_votes=min_votes,
            depth_prior=depth_prior, depth_weight=0.25,
            colour_prior=colour_prior, colour_weight=0.1)
        n = len(kept)
        print(f"  🔄 thresh={thresh:.2f} → {n} pts")
        if target_min <= n <= target_max: return kept, thresh
        if n >= target_min and n > best_n: best_n, best_kept = n, kept
        if n > target_max: thresh += step; break
        thresh -= step

    final = max(thresh, thresh_floor)
    kept  = multi_cue_vote(
        points, images, cameras, masks_dict, mask_qualities,
        vote_thresh=final, min_votes=min_votes,
        depth_prior=depth_prior, depth_weight=0.25,
        colour_prior=colour_prior, colour_weight=0.1)
    if len(kept) < target_min and best_kept is not None:
        return best_kept, thresh
    return kept, final


# ═════════════════════════════════════════════════════════════════════════════
# DEPTH-AWARE DBSCAN
# ═════════════════════════════════════════════════════════════════════════════

def depth_aware_dbscan(points, images, cameras):
    if len(points) < 50: return points
    N   = len(points)
    xyz = np.array([p['xyz'] for p in points])

    d_sum = np.zeros(N, np.float64); d_cnt = np.zeros(N, np.int32)
    for img_data in images.values():
        R, t = qvec_to_rotmat(img_data['qvec']), img_data['tvec']
        cam  = cameras[img_data['camera_id']]
        uv, inf, depths = project_points(xyz, R, t, cam)
        u_i = uv[:,0].astype(int); v_i = uv[:,1].astype(int)
        ib  = inf & (u_i>=0)&(u_i<cam['w'])&(v_i>=0)&(v_i<cam['h'])
        idx = np.where(ib)[0]
        d_sum[idx]+=depths[idx]; d_cnt[idx]+=1

    valid  = d_cnt > 0
    mean_d = np.where(valid, d_sum/np.maximum(d_cnt,1), np.nan)
    d_vals = mean_d[valid & ~np.isnan(mean_d)]
    pts2   = points
    if len(d_vals) >= 10:
        q25, q75 = np.percentile(d_vals,[25,75])
        iqr=q75-q25; dm=np.median(d_vals)
        lo,hi = dm-2.5*iqr, dm+2.5*iqr
        depth_ok = (mean_d>=lo)&(mean_d<=hi)|~valid
        pts2 = [p for p,ok in zip(points,depth_ok) if ok]
        removed = N-len(pts2)
        if removed: print(f"  📏 Depth IQR: removed {removed} pts")
        if len(pts2) < 50: pts2 = points

    if len(pts2) < 1000: return pts2

    # Downsample for DBSCAN only (don't lose detail after)
    DBSCAN_MAX = 150_000
    run = pts2
    if len(run) > DBSCAN_MAX:
        run = voxel_downsample(run, 0.005)   # finer voxel to keep more detail

    xyz2 = np.array([p['xyz'] for p in run])
    try:
        from scipy.spatial import cKDTree
        samp = xyz2[:min(3000, len(xyz2))]
        nd, _ = cKDTree(samp).query(samp, k=2)
        mnn   = float(nd[:,1].mean())
    except ImportError:
        mnn = float(np.median(xyz2.ptp(axis=0)))*0.02

    # eps = 2× mean-NN (tighter than v6's 3× to better separate object/bg)
    eps = max(0.002, min(0.08, mnn * 1.2))
    print(f"  🔥 DBSCAN: eps={eps:.4f}  n={len(run)}")
    try:
        from sklearn.cluster import DBSCAN
        lbl = DBSCAN(eps=eps, min_samples=5,
                     algorithm='ball_tree', n_jobs=1).fit_predict(xyz2)
    except Exception as e:
        print(f"  ⚠️  DBSCAN failed: {e}"); return pts2

    valid_lbl = lbl[lbl>=0]
    if not len(valid_lbl): return pts2
    cnts = np.bincount(valid_lbl)
    sig  = np.where(cnts >= max(cnts.max()*0.20, len(run)*0.03))[0]
    kept = [p for p,k in zip(run, np.isin(lbl, sig)) if k]
    if len(kept) < 500 and len(pts2) > 1000:
        print("  ⚠️  DBSCAN too aggressive — reverting"); return pts2
    print(f"  🔥 DBSCAN: {len(kept)} pts  {int(np.sum(lbl==-1))} noise removed")
    return kept


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

def geometry_based_filter(points, images, cameras, top_fraction=0.30):
    print(f"  📐 Geometry fallback (top {100*top_fraction:.0f}%)")
    N=len(points); xyz=np.array([p['xyz'] for p in points])
    vis=np.zeros(N,np.int32); ds=np.zeros(N,np.float64); ds2=np.zeros(N,np.float64)
    for img_data in images.values():
        R,t=qvec_to_rotmat(img_data['qvec']),img_data['tvec']
        cam=cameras[img_data['camera_id']]
        uv,inf,depths=project_points(xyz,R,t,cam)
        u_i=uv[:,0].astype(int); v_i=uv[:,1].astype(int)
        ib=inf&(u_i>=0)&(u_i<cam['w'])&(v_i>=0)&(v_i<cam['h'])
        idx=np.where(ib)[0]; z=depths[idx]
        vis[idx]+=1; ds[idx]+=z; ds2[idx]+=z**2
    mu=np.where(vis>0,ds/np.maximum(vis,1),0)
    var=np.where(vis>1,ds2/vis-mu**2,1e6)
    std=np.sqrt(np.maximum(var,0)); valid=vis>=2
    if not valid.sum(): return points
    p95=np.percentile(std[valid],95); std_c=np.clip(std,0,p95)
    vis_n=vis/(vis.max()+1e-8); std_n=1.0-std_c/(std_c.max()+1e-8)
    sc=np.where(valid,0.6*vis_n+0.4*std_n,0.0)
    thr=np.percentile(sc[valid],(1-top_fraction)*100)
    kept=[p for p,k in zip(points,sc>=thr) if k]
    print(f"  ✅ Geometry: {len(kept)}/{N}"); return kept


# ═════════════════════════════════════════════════════════════════════════════
# MASKED IMAGE WRITER
# ═════════════════════════════════════════════════════════════════════════════

def write_masked_images(images_dir, masks_dict, output_dir):
    import cv2
    os.makedirs(output_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(images_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]

    print(f"📸 Writing {len(all_imgs)} masked images ...")

    for name in all_imgs:
        src = os.path.join(images_dir, name)
        dst = os.path.join(output_dir, name)

        if name in masks_dict:

            img = np.array(Image.open(src).convert("RGB"))
            mask = masks_dict[name]

            # Resize mask if needed
            if mask.shape[:2] != img.shape[:2]:
                mask = np.array(
                    Image.fromarray(
                        mask.astype(np.uint8)*255
                    ).resize(
                        (img.shape[1], img.shape[0]),
                        Image.NEAREST
                    )
                ) > 0

            # 🔥 shrink mask slightly to remove halo edges
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            # pure black background
            img[~mask] = [0,0,0]

            Image.fromarray(img).save(dst)

        else:
            orig = Image.open(src)
            Image.new("RGB", orig.size, (0,0,0)).save(dst)

    print(f"✅ Saved masked images → {output_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# .bin sync
# ═════════════════════════════════════════════════════════════════════════════

def try_write_bin(colmap_dir):
    rw = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "../../pipeline/gaussian-splatting/utils")
    if rw not in sys.path: sys.path.append(rw)
    try:
        import read_write_model as rwm
        c,i,p = rwm.read_model(path=colmap_dir, ext=".txt")
        rwm.write_model(c,i,p, path=colmap_dir, ext=".bin")
        print("  ✅ .bin updated")
    except Exception as e:
        print(f"  ⚠️  .bin skipped: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_segmentation(colmap_dir, images_dir, output_dir=None,
                     vote_thresh=0.50, mask_images=False,
                     device='cpu', mask_dir=None, method='auto'):

    if output_dir is None: output_dir = colmap_dir
    os.makedirs(output_dir, exist_ok=True)

    for fn in ["cameras.txt","images.txt","points3D.txt"]:
        fp = os.path.join(colmap_dir, fn)
        if not os.path.exists(fp): raise FileNotFoundError(f"Missing: {fp}")

    print("\n📂 Reading COLMAP model …")
    cameras = read_cameras_txt(os.path.join(colmap_dir,"cameras.txt"))
    images  = read_images_txt (os.path.join(colmap_dir,"images.txt"))
    points  = read_points3D_txt(os.path.join(colmap_dir,"points3D.txt"))
    print(f"   {len(cameras)} cams | {len(images)} imgs | {len(points)} pts")

    if len(images) < len(cameras):
        print(f"  ⚠️  WARNING: images ({len(images)}) < cameras ({len(cameras)}). "
              f"Check images.txt format — expected {len(cameras)}.")

    print("\n🧹 Sanitising …")
    points = sanitise_points(points)

    image_paths = sorted({os.path.join(images_dir, d['name'])
                          for d in images.values()
                          if os.path.exists(os.path.join(images_dir, d['name']))})
    print(f"   Found {len(image_paths)} image files on disk")

    mask_cache     = os.path.join(output_dir, "object_masks_v7")
    masks_dict     = None
    mask_qualities = {}

    # ── Step 1: Masks ─────────────────────────────────────────────────────────
    if mask_dir and os.path.isdir(mask_dir):
        print(f"\n🗂️  Pre-computed masks from {mask_dir} …")
        masks_dict = {}
        for d in images.values():
            mp = os.path.join(mask_dir, d['name'])
            if os.path.exists(mp):
                m = np.array(Image.open(mp).convert('L')) > 127
                masks_dict[d['name']]     = m
                mask_qualities[d['name']] = _check_mask_quality(m)
        print(f"   Loaded {len(masks_dict)} masks")

    if masks_dict is None and method in ('auto','sam2'):
        try:
            print("\n🤖 Trying SAM2 …")
            masks_dict, mask_qualities = generate_masks_sam2(
                image_paths, mask_cache, device)
            print("  ✅ SAM2 succeeded")
        except Exception as e:
            print(f"  ℹ️  SAM2 unavailable: {e}")
            if method=='sam2': method='auto'

    if masks_dict is None and method in ('auto','grabcut','colour'):
        masks_dict, mask_qualities = generate_masks_auto(image_paths, mask_cache)

    # ── Step 2: Priors ────────────────────────────────────────────────────────
    print("\n📏 Computing depth prior …")
    depth_prior = compute_depth_prior(points, images, cameras)

    print("\n🎨 Computing colour prior …")
    colour_prior = compute_colour_prior(
        points, images, cameras, images_dir, masks_dict, mask_qualities)

    # ── Step 3: Multi-cue vote ────────────────────────────────────────────────
    print("\n🗳️  Multi-cue vote (mask + depth + colour) …")

    if method == 'geometry' or not masks_dict:
        print("  ⚠️  No masks — geometry fallback")
        filtered = geometry_based_filter(points, images, cameras, 0.25)
    else:
        use_adaptive = os.getenv("VOK_ADAPTIVE_THRESH","1")=="1"
        if use_adaptive:
            filtered, ft = adaptive_threshold(
                points, images, cameras, masks_dict, mask_qualities,
                depth_prior, colour_prior,
                target_min=5000, target_max=120000,
                thresh_start=vote_thresh, thresh_floor=0.35)
            print(f"  ✅ Final threshold: {ft:.2f}")
        else:
            min_v = max(2, int(len(images)*0.20))
            filtered = multi_cue_vote(
                points, images, cameras, masks_dict, mask_qualities,
                vote_thresh=vote_thresh, min_votes=min_v,
                depth_prior=depth_prior,   depth_weight=0.25,
                colour_prior=colour_prior, colour_weight=0.1)

        if len(filtered) < 500:
            print("  ⚠️  Too few — geometry fallback")
            filtered = geometry_based_filter(points, images, cameras, 0.25)

    # ── Step 4: Depth-aware DBSCAN ────────────────────────────────────────────
    print("\n🔥 Depth-aware DBSCAN …")
    filtered = depth_aware_dbscan(filtered, images, cameras)

    # ── Step 5: SOR ──────────────────────────────────────────────────────────
    print("\n🔬 SOR …")
    filtered = statistical_outlier_removal(filtered, k=20, std_ratio=2.0)

    # ── Step 6: Downsample only if over TARGET ────────────────────────────────
    TARGET    = int(os.environ.get("VOK_TARGET_POINTS", "120000"))
    FLOOR_PTS = int(os.environ.get("VOK_MIN_POINTS",      "5000"))
    if len(filtered) > TARGET:
        xa = np.array([p['xyz'] for p in filtered])
        sn = min(2000,len(filtered))
        s  = xa[np.random.default_rng(42).choice(len(filtered),sn,replace=False)]
        try:
            from scipy.spatial import cKDTree
            nd,_ = cKDTree(s).query(s,k=2); mnn=float(nd[:,1].mean())
        except ImportError:
            mnn = float(np.prod(np.maximum(xa.ptp(0),1e-6))/len(filtered))**(1/3)
        vox = max(mnn*0.5, mnn*(len(filtered)/TARGET)**(1/3))
        filtered = voxel_downsample(filtered, vox)
        if len(filtered) > int(TARGET*1.3):
            filtered = voxel_downsample(filtered,
                vox*(len(filtered)/TARGET)**(1/3))
    else:
        print(f"  ℹ️  {len(filtered)} pts ≤ TARGET {TARGET} — keeping all")

    if len(filtered) < FLOOR_PTS:
        print(f"  ⚠️  Only {len(filtered)} pts — below floor {FLOOR_PTS}.")

    # ── Step 7: Final sanitise ────────────────────────────────────────────────
    filtered = sanitise_points(filtered, pct=99.0)

    # ── Step 8: Write ─────────────────────────────────────────────────────────
    out_pts = os.path.join(output_dir,"points3D.txt")
    write_points3D_txt(out_pts, filtered)
    print(f"\n💾 Final: {len(filtered)} clean object points → {out_pts}")
    try_write_bin(output_dir)

    if mask_images and masks_dict:
        mdir = os.path.join(os.path.dirname(images_dir),"images_masked")
        write_masked_images(images_dir, masks_dict, mdir)
        return mdir

    return images_dir


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Point-cloud isolation v7")
    ap.add_argument("--colmap_dir",  required=True)
    ap.add_argument("--images_dir",  required=True)
    ap.add_argument("--output_dir",  default=None)
    ap.add_argument("--vote_thresh", type=float, default=0.50)
    ap.add_argument("--mask_dir",    default=None)
    ap.add_argument("--mask_images", action="store_true")
    ap.add_argument("--device",      default="cpu",
                    choices=["cpu","cuda","mps"])
    ap.add_argument("--method",      default="auto",
                    choices=["auto","sam2","grabcut","colour","geometry"])
    args = ap.parse_args()
    run_segmentation(**vars(args))
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()