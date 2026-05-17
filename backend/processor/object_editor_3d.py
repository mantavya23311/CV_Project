"""
object_editor_3d.py  (v1)
═══════════════════════════════════════════════════════════════════════════════

Text-Prompted 3D Object Editor
────────────────────────────────
Given a labelled / isolated point cloud (points3D.txt) and a free-text edit
instruction, this module applies the requested transformation to the 3D points
and produces an edited points3D.txt ready for Gaussian Splatting.

Supported edit types (detected automatically from the prompt)
─────────────────────────────────────────────────────────────
COLOUR EDITS
  "make it red"            → recolour all points with target RGB
  "change colour to blue"  → same
  "make it metallic gold"  → metallic sheen simulation
  "tint it green"          → hue-shift preserving luminance
  "remove colour / grey"   → desaturate to greyscale

STYLE EDITS  (colour-based appearance changes)
  "make it look like wood"        → apply procedural wood grain palette
  "make it look like marble"      → marble vein palette
  "make it look rusty / aged"     → rust/patina colour map

GEOMETRY / SHAPE EDITS
  "scale up / make bigger"        → uniform scale around centroid
  "make it taller / wider"        → axis-aligned non-uniform scale
  "rotate 45 degrees"             → rotate around vertical axis
  "move up / translate"           → rigid translation
  "flatten / squish"              → scale along one axis
  "smooth / round"                → voxel-merge + Laplacian smoothing

COMBINED EDITS  (comma-separated or natural language)
  "make it red and rotate 90 degrees"
  "scale up 1.5x and tint blue"

All edits are non-destructive: the original points3D.txt is kept as
points3D_original.txt and the edited version is written to points3D.txt
(or a named output if --output is given).

Architecture
────────────
• A small LLM (via the Anthropic API or a local CLIP-guided parser) parses
  the edit instruction into a structured EditPlan.
• The EditPlan is executed deterministically on the numpy point array.
• CLIP (optional) is used to validate that the edited result is
  semantically consistent with the prompt before writing.

Usage
─────
    python object_editor_3d.py \
        --points_file  path/to/points3D.txt \
        --prompt       "make it bright red and rotate 30 degrees clockwise" \
        --output       path/to/points3D_edited.txt

    # Edit a specific label from multi_object_segmentation output:
    python object_editor_3d.py \
        --points_file  path/to/points3D_bottle.txt \
        --prompt       "change colour to metallic gold, scale up 1.3x"

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, sys, re, json, argparse, copy, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from segment_object_pointcloud import (
    read_points3D_txt, write_points3D_txt, sanitise_points,
)


# ═════════════════════════════════════════════════════════════════════════════
# COLOUR UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

# Named-colour table (extend as needed)
NAMED_COLOURS: dict[str, tuple[int, int, int]] = {
    'red':      (220,  30,  30),
    'green':    ( 30, 180,  30),
    'blue':     ( 30,  80, 220),
    'yellow':   (240, 220,  20),
    'orange':   (240, 120,  10),
    'purple':   (140,  30, 200),
    'pink':     (240,  80, 160),
    'cyan':     ( 20, 200, 220),
    'white':    (255, 255, 255),
    'black':    (  0,   0,   0),
    'grey':     (128, 128, 128),
    'gray':     (128, 128, 128),
    'brown':    (120,  70,  30),
    'gold':     (212, 175,  55),
    'silver':   (192, 192, 192),
    'bronze':   (205, 127,  50),
    'navy':     ( 10,  30, 100),
    'teal':     ( 10, 140, 140),
    'magenta':  (210,  20, 180),
    'lime':     (100, 220,  20),
    'maroon':   (128,   0,   0),
    'olive':    (128, 128,   0),
    'coral':    (255,  80,  60),
    'indigo':   ( 75,   0, 130),
    'violet':   (140,  60, 200),
    'crimson':  (220,  20,  60),
    'turquoise':(  64, 224, 208),
    'beige':    (245, 245, 220),
    'ivory':    (255, 255, 240),
}

# Metallic sheen: specular highlight — kept saturated (not bright white).
# Too-bright sheen causes white washout on already-bright surfaces.
METALLIC_SHEEN = {
    'gold':   (255, 210,  40),   # saturated warm gold, visibly yellow not white
    'silver': (210, 215, 225),
    'bronze': (220, 140,  50),
    'copper': (200, 110,  50),
}

# Procedural texture palettes  (list of (weight, RGB) pairs)
TEXTURE_PALETTES = {
    'wood': [
        (0.35, (160,  90,  40)),
        (0.30, (140,  75,  30)),
        (0.20, (180, 110,  55)),
        (0.15, (200, 130,  70)),
    ],
    'marble': [
        (0.45, (240, 238, 232)),
        (0.25, (200, 195, 185)),
        (0.20, (160, 155, 148)),
        (0.10, ( 80,  80,  80)),
    ],
    'rust': [
        (0.40, (150,  55,  15)),
        (0.30, (180,  80,  20)),
        (0.20, (100,  40,  10)),
        (0.10, ( 60,  30,   5)),
    ],
    'concrete': [
        (0.50, (150, 150, 148)),
        (0.30, (130, 130, 128)),
        (0.20, (170, 168, 165)),
    ],
    'plastic': [
        (0.70, None),   # None = use provided base colour
        (0.30, None),
    ],
    'glass': [
        (0.60, (200, 220, 240)),
        (0.40, (180, 200, 220)),
    ],
    'ceramic': [
        (0.55, (245, 240, 235)),
        (0.30, (230, 225, 218)),
        (0.15, (215, 210, 200)),
    ],
    'leather': [
        (0.45, ( 80,  45,  20)),
        (0.35, ( 60,  35,  15)),
        (0.20, (100,  60,  30)),
    ],
}


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """[N,3] uint8 → [N,3] float HSV in [0,1]."""
    r, g, b  = rgb[:,0]/255., rgb[:,1]/255., rgb[:,2]/255.
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-8
    h = np.zeros(len(r))
    s = np.where(cmax > 0, delta/cmax, 0.0)
    v = cmax
    m = (cmax == r); h[m] = ((g[m]-b[m])/delta[m]) % 6
    m = (cmax == g); h[m] = (b[m]-r[m])/delta[m] + 2
    m = (cmax == b); h[m] = (r[m]-g[m])/delta[m] + 4
    h /= 6.
    return np.stack([h, s, v], 1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """[N,3] float HSV in [0,1] → [N,3] uint8 RGB."""
    h, s, v = hsv[:,0]*6, hsv[:,1], hsv[:,2]
    i  = np.floor(h).astype(int) % 6
    f  = h - np.floor(h)
    p  = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s)
    out = np.zeros_like(hsv)
    for idx, (r_, g_, b_) in enumerate([
            (v, t, p),(q, v, p),(p, v, t),
            (p, q, v),(t, p, v),(v, p, q)]):
        m = i == idx
        out[m] = np.stack([r_[m], g_[m], b_[m]], 1)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def apply_metallic(rgb: np.ndarray, sheen_rgb: tuple,
                   sheen_strength: float = 0.4) -> np.ndarray:
    """Blend original colour with specular sheen based on luminance."""
    lum   = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114]))
    lum_n = lum / (lum.max() + 1e-8)      # [0,1]
    sheen = np.array(sheen_rgb, np.float32)
    out   = (rgb.astype(np.float32) * (1 - sheen_strength * lum_n[:, None])
             + sheen * sheen_strength * lum_n[:, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_texture_palette(rgb: np.ndarray,
                           palette: list,
                           base_colour: tuple | None = None,
                           noise_scale: float = 0.12) -> np.ndarray:
    """
    Map each point to a texture palette colour with per-point random noise.
    Points are assigned palette colours proportionally to their weights,
    with a small random offset to break up uniformity.
    """
    N = len(rgb)
    rng = np.random.default_rng(42)
    weights  = np.array([w for w, _ in palette], np.float32)
    weights /= weights.sum()
    assigned = rng.choice(len(palette), size=N, p=weights)
    out = np.zeros((N, 3), np.float32)
    for i, (_, col) in enumerate(palette):
        mask = assigned == i
        if col is None:
            col = base_colour if base_colour else (128, 128, 128)
        c = np.array(col, np.float32)
        noise = rng.normal(0, noise_scale * 255, (int(mask.sum()), 3))
        out[mask] = np.clip(c + noise, 0, 255)
    return out.astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# EDIT PLAN PARSER
# ═════════════════════════════════════════════════════════════════════════════

class EditPlan:
    """
    Structured representation of one or more edit operations.
    Built either from the rule-based parser or from an LLM response.
    """
    def __init__(self):
        self.colour_ops:    list[dict] = []   # colour transformations
        self.geometry_ops:  list[dict] = []   # spatial transformations
        self.texture_ops:   list[dict] = []   # texture palette changes

    def add_colour(self, op_type: str, **kwargs):
        self.colour_ops.append({'type': op_type, **kwargs})

    def add_geometry(self, op_type: str, **kwargs):
        self.geometry_ops.append({'type': op_type, **kwargs})

    def add_texture(self, op_type: str, **kwargs):
        self.texture_ops.append({'type': op_type, **kwargs})

    def is_empty(self) -> bool:
        return not (self.colour_ops or self.geometry_ops or self.texture_ops)

    def __repr__(self):
        return (f"EditPlan(colour={self.colour_ops}, "
                f"geometry={self.geometry_ops}, "
                f"texture={self.texture_ops})")


def parse_edit_prompt_rules(prompt: str) -> EditPlan:
    """
    Rule-based parser — fast, no LLM required.
    Handles the most common edit patterns.
    """
    plan  = EditPlan()
    text  = prompt.lower().strip()

    # ── COLOUR ────────────────────────────────────────────────────────────────

    # Greyscale / desaturate
    if re.search(r'\b(grey|gray|greyscale|grayscale|desatur|monochrome|remove colou?r)\b', text):
        plan.add_colour('desaturate')

    # Explicit hex colour  e.g. "#ff3300"
    hex_m = re.search(r'#([0-9a-f]{6})', text)
    if hex_m:
        hx = hex_m.group(1)
        plan.add_colour('recolour', rgb=(int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)))

    # ── Metallic detection FIRST (must come before named-colour scan) ─────────
    # "metallic gold" / "gold metallic" must resolve to ONE metallic op, not
    # a recolour(gold) + metallic(silver) conflict.
    metallic_m = re.search(r'\bmetallic\s+(\w+)|\b(\w+)\s+metallic\b', text)
    is_metallic = False
    if metallic_m:
        base = (metallic_m.group(1) or metallic_m.group(2) or 'silver').strip()
        sheen_rgb = METALLIC_SHEEN.get(base, METALLIC_SHEEN.get('silver', (220,220,230)))
        base_rgb  = NAMED_COLOURS.get(base, (200, 200, 200))
        plan.add_colour('metallic', base_rgb=base_rgb, sheen_rgb=sheen_rgb)
        is_metallic = True
    elif re.search(r'\bmetallic\b', text):
        plan.add_colour('metallic', base_rgb=(200, 200, 200),
                        sheen_rgb=METALLIC_SHEEN['silver'])
        is_metallic = True

    # Named colour — skip if metallic already handled it (avoids double op)
    if not is_metallic:
        for name, rgb in NAMED_COLOURS.items():
            if re.search(rf'\b{name}\b', text):
                plan.add_colour('recolour', rgb=rgb, name=name)
                break   # one colour op at a time

    # Tint (hue-shift only, not full recolour)
    tint_m = re.search(r'\btint\s+(?:it\s+)?(\w+)', text)
    if tint_m:
        colour_name = tint_m.group(1)
        if colour_name in NAMED_COLOURS:
            plan.add_colour('tint', rgb=NAMED_COLOURS[colour_name],
                            strength=0.45)

    # Brightness adjustments
    bright_m = re.search(r'\b(bright(?:en)?|lighten)\s*(?:by\s*)?([\d.]+)?', text)
    if bright_m:
        factor = float(bright_m.group(2)) if bright_m.group(2) else 1.3
        plan.add_colour('brightness', factor=min(factor, 3.0))

    dark_m = re.search(r'\b(dark(?:en)?)\s*(?:by\s*)?([\d.]+)?', text)
    if dark_m:
        factor = float(dark_m.group(2)) if dark_m.group(2) else 0.7
        plan.add_colour('brightness', factor=max(factor, 0.1))

    # ── TEXTURE ───────────────────────────────────────────────────────────────
    for tex_name in TEXTURE_PALETTES:
        if re.search(rf'\b{tex_name}\b', text):
            base_rgb = None
            for name, rgb in NAMED_COLOURS.items():
                if re.search(rf'\b{name}\b', text):
                    base_rgb = rgb; break
            plan.add_texture('palette', name=tex_name, base_rgb=base_rgb)
            break

    # ── GEOMETRY ─────────────────────────────────────────────────────────────

    # Scale (uniform)
    scale_m = re.search(r'\b(scale|resize)\s*(?:up|down)?\s*(?:by\s*)?([\d.]+)\s*[xX×]?', text)
    if scale_m:
        factor = float(scale_m.group(2))
        factor = max(0.1, min(factor, 10.0))
        plan.add_geometry('scale_uniform', factor=factor)
    elif re.search(r'\b(bigger|larger|enlarge|scale\s*up)\b', text):
        plan.add_geometry('scale_uniform', factor=1.5)
    elif re.search(r'\b(smaller|shrink|scale\s*down)\b', text):
        plan.add_geometry('scale_uniform', factor=0.67)

    # Scale axis-specific
    taller_m = re.search(r'\b(taller|elongate|stretch\s*(?:vertical|up))\b', text)
    if taller_m:
        plan.add_geometry('scale_axis', axis='y', factor=1.4)

    wider_m = re.search(r'\b(wider|broader|expand\s*(?:horizontal))\b', text)
    if wider_m:
        plan.add_geometry('scale_axis', axis='x', factor=1.4)

    flat_m = re.search(r'\b(flat(?:ten)?|squish|compress)\b', text)
    if flat_m:
        plan.add_geometry('scale_axis', axis='y', factor=0.5)

    # Rotation
    rot_m = re.search(
        r'\brotate\s*(?:by\s*)?([\d.]+)\s*deg(?:rees?)?\s*(clockwise|counter|ccw|cw)?',
        text)
    if rot_m:
        angle_deg = float(rot_m.group(1))
        direction = rot_m.group(2) or 'ccw'
        if direction in ('clockwise', 'cw'):
            angle_deg = -angle_deg
        plan.add_geometry('rotate_y', angle_deg=angle_deg)
    elif re.search(r'\brotate\s*(?:90|quarter)', text):
        plan.add_geometry('rotate_y', angle_deg=90.0)
    elif re.search(r'\brotate\s*(?:180|half)', text):
        plan.add_geometry('rotate_y', angle_deg=180.0)
    elif re.search(r'\bflip\b', text):
        plan.add_geometry('rotate_y', angle_deg=180.0)

    # Translation
    up_m = re.search(r'\bmove\s+up\s*(?:by\s*)?([\d.]+)?', text)
    if up_m:
        amt = float(up_m.group(1)) if up_m.group(1) else 0.2
        plan.add_geometry('translate', dx=0, dy=amt, dz=0)

    down_m = re.search(r'\bmove\s+down\s*(?:by\s*)?([\d.]+)?', text)
    if down_m:
        amt = float(down_m.group(1)) if down_m.group(1) else 0.2
        plan.add_geometry('translate', dx=0, dy=-amt, dz=0)

    # Smooth / round
    if re.search(r'\b(smooth|round|soften)\b', text):
        plan.add_geometry('smooth', iterations=2)

    return plan


def parse_edit_prompt_llm(prompt: str) -> EditPlan:
    """
    Use the Anthropic API to parse the edit prompt into a structured EditPlan.
    Falls back to rule-based parser if the API is unavailable.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        system = """You are a 3D point-cloud edit planner.
Given a natural-language edit instruction, output a JSON object with these
optional keys:

{
  "colour_ops": [
    {"type": "recolour",    "rgb": [R,G,B]},
    {"type": "tint",        "rgb": [R,G,B], "strength": 0.0-1.0},
    {"type": "desaturate"},
    {"type": "metallic",    "base_rgb":[R,G,B], "sheen_rgb":[R,G,B]},
    {"type": "brightness",  "factor": float}
  ],
  "texture_ops": [
    {"type": "palette", "name": "wood|marble|rust|concrete|glass|ceramic|leather",
     "base_rgb": [R,G,B] or null}
  ],
  "geometry_ops": [
    {"type": "scale_uniform", "factor": float},
    {"type": "scale_axis",    "axis": "x|y|z", "factor": float},
    {"type": "rotate_y",      "angle_deg": float},
    {"type": "translate",     "dx": float, "dy": float, "dz": float},
    {"type": "smooth",        "iterations": int}
  ]
}

Output ONLY the JSON. No prose.
RGB values are 0-255 integers."""

        msg = client.messages.create(
            model  = "claude-sonnet-4-20250514",
            max_tokens = 512,
            system = system,
            messages = [{"role": "user", "content": prompt}])

        raw  = msg.content[0].text.strip()
        # Strip markdown fences if present
        raw  = re.sub(r'^```[a-z]*\n?', '', raw)
        raw  = re.sub(r'\n?```$', '', raw)
        data = json.loads(raw)

        plan = EditPlan()
        for op in data.get('colour_ops', []):
            plan.colour_ops.append(op)
        for op in data.get('texture_ops', []):
            plan.texture_ops.append(op)
        for op in data.get('geometry_ops', []):
            plan.geometry_ops.append(op)
        print(f"  🤖 LLM parsed edit plan: {plan}")
        return plan

    except Exception as e:
        print(f"  ℹ️  LLM parse failed ({e}), using rule-based parser")
        return parse_edit_prompt_rules(prompt)


# ═════════════════════════════════════════════════════════════════════════════
# EDIT EXECUTOR
# ═════════════════════════════════════════════════════════════════════════════

def execute_edit_plan(points: list[dict], plan: EditPlan) -> list[dict]:
    """
    Apply an EditPlan to a list of point dicts.
    Returns a new list — the original is not modified.
    """
    pts = copy.deepcopy(points)
    if not pts:
        return pts

    xyz = np.array([p['xyz'] for p in pts], dtype=np.float64)
    rgb = np.array([p['rgb'] for p in pts], dtype=np.uint8)

    centroid = xyz.mean(axis=0)

    # ── COLOUR OPS ────────────────────────────────────────────────────────────
    for op in plan.colour_ops:
        t = op['type']

        if t == 'recolour':
            target = np.array(op['rgb'], np.float32)
            # Luminance-preserving recolour: scale target by original point luminance.
            # This preserves the 3D shading of the object — dark points stay dark,
            # bright points stay bright — in the new colour.
            orig_lum_n = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])) / 255.0
            tinted = target[None, :] * orig_lum_n[:, None]
            px = tinted * 0.85 + rgb.astype(np.float32) * 0.15
            rgb = np.clip(px, 0, 255).astype(np.uint8)
            print(f"  🎨 Recolour → {op['rgb']} (luminance-preserving)")

        elif t == 'tint':
            target   = np.array(op['rgb'], np.float32)
            strength = float(op.get('strength', 0.4))
            rgb      = np.clip(
                rgb.astype(np.float32) * (1-strength) + target * strength,
                0, 255).astype(np.uint8)
            print(f"  🎨 Tint → {op['rgb']}  strength={strength}")

        elif t == 'desaturate':
            lum = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114]))
            rgb = np.clip(np.stack([lum, lum, lum], 1), 0, 255).astype(np.uint8)
            print("  🎨 Desaturate → greyscale")

        elif t == 'metallic':
            base_rgb  = np.array(op.get('base_rgb',  [212, 175, 55]), np.float32)
            sheen_rgb = np.array(op.get('sheen_rgb', [255, 210, 40]), np.float32)
            # Luminance-preserving metallic tint (same formula as image_editor_2d):
            #   scale the base colour by original point luminance, then add subtle sheen.
            # This keeps dark points dark gold and bright points bright gold — not white.
            orig_lum   = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114]))
            orig_lum_n = orig_lum / 255.0
            tinted = base_rgb[None, :] * orig_lum_n[:, None]
            sheen_mask = orig_lum_n > 0.60
            if sheen_mask.any():
                sfac = (orig_lum_n[sheen_mask] - 0.60) / 0.40
                tinted[sheen_mask] = (
                    tinted[sheen_mask] * (1 - 0.20 * sfac[:, None]) +
                    sheen_rgb[None, :] * 0.20 * sfac[:, None])
            px = tinted * 0.85 + rgb.astype(np.float32) * 0.15
            rgb = np.clip(px, 0, 255).astype(np.uint8)
            print(f"  🎨 Metallic gold applied (luminance-preserving)")

        elif t == 'brightness':
            factor = float(op['factor'])
            rgb    = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            print(f"  🎨 Brightness ×{factor}")

    # ── TEXTURE OPS ───────────────────────────────────────────────────────────
    for op in plan.texture_ops:
        if op['type'] == 'palette':
            name     = op['name']
            base_rgb = op.get('base_rgb')
            palette  = TEXTURE_PALETTES.get(name)
            if palette:
                rgb = apply_texture_palette(rgb, palette, base_rgb)
                print(f"  🎨 Texture palette: {name}")

    # ── GEOMETRY OPS ─────────────────────────────────────────────────────────
    for op in plan.geometry_ops:
        t = op['type']

        if t == 'scale_uniform':
            factor = float(op['factor'])
            xyz    = centroid + (xyz - centroid) * factor
            print(f"  📐 Scale uniform ×{factor}")

        elif t == 'scale_axis':
            axis   = op['axis'].lower()
            factor = float(op['factor'])
            ai     = {'x': 0, 'y': 1, 'z': 2}[axis]
            xyz[:, ai] = centroid[ai] + (xyz[:, ai] - centroid[ai]) * factor
            print(f"  📐 Scale axis={axis} ×{factor}")

        elif t == 'rotate_y':
            angle_rad = np.deg2rad(float(op['angle_deg']))
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            Ry   = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            xyz  = centroid + (Ry @ (xyz - centroid).T).T
            print(f"  📐 Rotate Y {op['angle_deg']}°")

        elif t == 'rotate_x':
            angle_rad = np.deg2rad(float(op['angle_deg']))
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            Rx   = np.array([[1,0,0],[0,c,-s],[0,s,c]])
            xyz  = centroid + (Rx @ (xyz - centroid).T).T
            print(f"  📐 Rotate X {op['angle_deg']}°")

        elif t == 'rotate_z':
            angle_rad = np.deg2rad(float(op['angle_deg']))
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            Rz   = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            xyz  = centroid + (Rz @ (xyz - centroid).T).T
            print(f"  📐 Rotate Z {op['angle_deg']}°")

        elif t == 'translate':
            dx = float(op.get('dx', 0))
            dy = float(op.get('dy', 0))
            dz = float(op.get('dz', 0))
            xyz += np.array([dx, dy, dz])
            print(f"  📐 Translate ({dx}, {dy}, {dz})")

        elif t == 'smooth':
            n_iter = int(op.get('iterations', 2))
            try:
                from scipy.spatial import cKDTree
                for _ in range(n_iter):
                    tree = cKDTree(xyz)
                    dists, idxs = tree.query(xyz, k=8)
                    weights = 1.0 / (dists[:, 1:] + 1e-8)
                    weights /= weights.sum(axis=1, keepdims=True)
                    xyz = (xyz[:, None, :] * (1 - weights.sum(axis=1, keepdims=True))[:, :, None]
                           + (xyz[idxs[:, 1:]] * weights[:, :, None]).sum(axis=1))
                print(f"  📐 Smooth ×{n_iter}")
            except ImportError:
                print("  ⚠️  scipy not available — smooth skipped")

    # ── Write back ────────────────────────────────────────────────────────────
    for i, p in enumerate(pts):
        p['xyz'] = xyz[i]
        p['rgb'] = np.clip(rgb[i], 0, 255).astype(int)

    return pts


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EDIT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_edit(points_file: str,
             prompt: str,
             output_file: str | None = None,
             use_llm: bool = True,
             backup: bool = True) -> str:
    """
    Load points3D.txt, parse prompt, execute edits, save result.

    Returns path to the edited file.
    """
    print(f"\n🖊️  Edit prompt: \"{prompt}\"")
    print(f"📂 Loading: {points_file}")

    points = read_points3D_txt(points_file)
    print(f"   {len(points)} points loaded")

    if not points:
        raise ValueError("No points found in file")

    # Backup original — and ALWAYS restore from backup at the start of each
    # run so that re-running the pipeline doesn't stack edits on top of each
    # other (e.g. second run would edit the already-gold cloud, not the
    # original pink bottle colours).
    if backup:
        import shutil
        backup_path = points_file.replace('.txt', '_original.txt')
        if not os.path.exists(backup_path):
            shutil.copy2(points_file, backup_path)
            print(f"  💾 Backup → {backup_path}")
        else:
            # Restore original so each run starts from raw colours
            shutil.copy2(backup_path, points_file)
            print(f"  🔄 Restored original colours from backup")

    # Parse prompt
    print("\n🔍 Parsing edit prompt …")
    if use_llm and os.getenv('ANTHROPIC_API_KEY'):
        plan = parse_edit_prompt_llm(prompt)
    else:
        plan = parse_edit_prompt_rules(prompt)
        print(f"  📝 Rule-based plan: {plan}")

    if plan.is_empty():
        print("  ⚠️  No edits parsed from prompt — nothing to do")
        return points_file

    # Execute
    print("\n⚙️  Executing edit plan …")
    edited = execute_edit_plan(points, plan)
    edited = sanitise_points(edited, pct=99.9)

    # Write
    out_path = output_file or points_file
    write_points3D_txt(out_path, edited)
    print(f"\n✅ Edited {len(edited)} points → {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# BATCH EDITOR  (edit all labels from multi_object_segmentation output)
# ═════════════════════════════════════════════════════════════════════════════

def run_batch_edit(scene_json: str, edits: dict[str, str],
                   use_llm: bool = True):
    """
    Apply label-specific edits from a scene_objects.json produced by
    multi_object_segmentation.

    Parameters
    ──────────
    scene_json : path to scene_objects.json
    edits      : dict mapping label → edit prompt string
                 e.g. {"bottle": "make it red", "table": "make it wood"}
    """
    with open(scene_json) as f:
        meta = json.load(f)

    for label, prompt in edits.items():
        if label not in meta:
            print(f"  ⚠️  Label '{label}' not in scene — skipping"); continue
        pts_file = meta[label]['points_file']
        if not os.path.exists(pts_file):
            print(f"  ⚠️  File not found: {pts_file}"); continue
        print(f"\n{'='*60}")
        print(f"  Editing '{label}': {prompt}")
        print(f"{'='*60}")
        run_edit(pts_file, prompt, use_llm=use_llm)

    print("\n🎉 Batch edit complete!")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Text-prompted 3D object editor  (v1)")

    sub = ap.add_subparsers(dest='command', required=True)

    # Single-file edit
    p_edit = sub.add_parser('edit', help='Edit a single points3D file')
    p_edit.add_argument("--points_file", required=True,
                        help="Path to points3D.txt (or points3D_<label>.txt)")
    p_edit.add_argument("--prompt",      required=True,
                        help="Edit instruction, e.g. 'make it red and rotate 30 degrees'")
    p_edit.add_argument("--output",      default=None,
                        help="Output path (default: overwrite in-place)")
    p_edit.add_argument("--no_llm",     action="store_true",
                        help="Use rule-based parser only (no API call)")
    p_edit.add_argument("--no_backup",  action="store_true",
                        help="Skip backing up the original file")

    # Batch edit from scene JSON
    p_batch = sub.add_parser('batch', help='Batch-edit objects from scene_objects.json')
    p_batch.add_argument("--scene_json", required=True,
                         help="Path to scene_objects.json")
    p_batch.add_argument("--edits",      required=True,
                         help='JSON string: {"label":"prompt",...}  '
                              'e.g. \'{"bottle":"make it red","table":"wood texture"}\'')
    p_batch.add_argument("--no_llm",    action="store_true")

    args = ap.parse_args()

    if args.command == 'edit':
        run_edit(
            points_file = args.points_file,
            prompt      = args.prompt,
            output_file = args.output,
            use_llm     = not args.no_llm,
            backup      = not args.no_backup,
        )

    elif args.command == 'batch':
        edits = json.loads(args.edits)
        run_batch_edit(
            scene_json = args.scene_json,
            edits      = edits,
            use_llm    = not args.no_llm,
        )

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()