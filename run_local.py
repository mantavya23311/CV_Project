"""
VokVision Local Pipeline Bypass
================================
Run this from the ROOT of your cloned vok-vision-main repo:

    cd vok-vision-main
    python run_local.py --images path/to/your/images

What this does:
  1. Copies your images into  storage/uploads/<project_id>/
  2. Builds the exact job payload that the Node.js API / BullMQ
     would normally send to the Python worker
  3. Calls backend/processor/main.py directly, bypassing Redis,
     BullMQ, Flutter, and all cloud services
  4. Output .ply / .splat lands in  storage/outputs/<project_id>/

Usage examples:
    python run_local.py --images ./my_photos/ --iterations 2500 --skip-vlm

    # Auto-discover all objects in the scene:
    python run_local.py --images ./photos/ --multi-seg

    # Segment specific objects by name:
    python run_local.py --images ./photos/ --multi-seg-prompt "bottle, table"

    # Edit the entire point cloud (single-object mode):
    python run_local.py --images ./photos/ --edit-prompt "make it red"

    # Segment then edit one label:
    python run_local.py --images ./photos/ --multi-seg-prompt "bottle, table" \\
        --edit-target bottle --edit-prompt "make it metallic gold"

    # Batch-edit all discovered objects:
    python run_local.py --images ./photos/ --multi-seg \\
        --edit-target batch \\
        --edit-prompt '{"object_0":"make it red","object_1":"wood texture"}'
"""

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

# ── Resolve repo root (this script lives at repo root) ───────────────────────
REPO_ROOT     = Path(__file__).parent.resolve()
PROCESSOR_DIR = REPO_ROOT / "backend" / "processor"
PIPELINE_DIR  = REPO_ROOT / "pipeline"
STORAGE_DIR   = REPO_ROOT / "storage"

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_project_id(name: str) -> str:
    if name:
        return name.replace(" ", "_").lower()
    return "proj_" + str(uuid.uuid4())[:8]


def copy_images(src: Path, dest: Path) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    files = sorted([f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if not files:
        print(f"[ERROR] No images found in {src}")
        sys.exit(1)
    for i, f in enumerate(files, 1):
        dst = dest / f"{i:03d}_{f.name}"
        shutil.copy2(f, dst)
        copied.append(str(dst))
        print(f"  [+] {f.name} → {dst.name}")
    return copied


def build_job_payload(project_id: str, image_paths: list,
                      skip_vlm: bool, iterations: int,
                      output_dir: str) -> dict:
    return {
        "projectId":   project_id,
        "imagePaths":  image_paths,
        "outputDir":   output_dir,
        "skipVlm":     skip_vlm,
        "iterations":  iterations,
        "localRun":    True,
    }


def check_venv():
    venv_python = PROCESSOR_DIR / "venv" / "bin" / "python"
    if not venv_python.exists():
        print("\n[WARN] Processor venv not found at backend/processor/venv/")
        print("       Run setup first:")
        print("         cd backend/processor && python3.10 -m venv venv")
        print("         venv/bin/pip install -r requirements.txt\n")
    current = Path(sys.executable)
    if "venv" not in str(current):
        print(f"[WARN] Running with: {current}")
        print(f"       Recommended: {venv_python}")
        print(f"       Activate: source backend/processor/venv/bin/activate\n")


def patch_sys_path():
    for p in [str(PROCESSOR_DIR), str(PIPELINE_DIR), str(REPO_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def run(args):
    project_id = make_project_id(args.project)
    upload_dir  = STORAGE_DIR / "uploads"  / project_id

    # ── Output dir: MUST match OUTPUT_DIR in backend/processor/config.py ──────
    # The processor config.py sets OUTPUT_DIR = .../storage/outputs  (with 's').
    # run_local.py previously used "output" (no 's'), so the final check was
    # always looking in the wrong place.  Fixed here to always use "outputs".
    output_dir  = STORAGE_DIR / "outputs"  / project_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Gather images ──────────────────────────────────────────────────────
    if args.images:
        src = Path(args.images).resolve()
        if not src.exists():
            print(f"[ERROR] Image path does not exist: {src}")
            sys.exit(1)
        if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
            tmp = STORAGE_DIR / "uploads" / (project_id + "_tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp / src.name)
            src = tmp
        print(f"\n[1/4] Copying images from {src} → {upload_dir}")
        image_paths = copy_images(src, upload_dir)
    else:
        if upload_dir.exists():
            image_paths = sorted([
                str(f) for f in upload_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            ])
            if image_paths:
                print(f"\n[1/4] Using existing images in {upload_dir} "
                      f"({len(image_paths)} files)")
            else:
                print(f"[ERROR] No images in {upload_dir}. "
                      f"Use --images to specify a source.")
                sys.exit(1)
        else:
            print(f"[ERROR] No --images provided and {upload_dir} doesn't exist.")
            print(f"        Usage: python run_local.py --images /path/to/photos/")
            sys.exit(1)

    print(f"         Total: {len(image_paths)} images")

    # ── 2. Build job payload ──────────────────────────────────────────────────
    job = build_job_payload(project_id, image_paths, args.skip_vlm,
                            args.iterations, str(output_dir))
    job_file = STORAGE_DIR / "uploads" / project_id / "_job.json"
    with open(job_file, "w") as f:
        json.dump(job, f, indent=2)

    print(f"\n[2/4] Job payload written → {job_file}")
    print(f"       Project ID  : {project_id}")
    print(f"       Images      : {len(image_paths)}")
    print(f"       Iterations  : {job['iterations']}")
    print(f"       Skip VLM    : {job['skipVlm']}")
    print(f"       Output dir  : {job['outputDir']}")

    # ── 3. Set env vars ───────────────────────────────────────────────────────
    check_venv()
    patch_sys_path()

    main_py = PROCESSOR_DIR / "main.py"
    if not main_py.exists():
        print(f"\n[ERROR] {main_py} not found.")
        print("        Run from repo root: cd vok-vision-main && python run_local.py")
        sys.exit(1)

    print(f"\n[3/4] Invoking {main_py} with local job payload...")
    print("─" * 60)

    # Core
    os.environ["VOK_LOCAL_RUN"]      = "1"
    os.environ["VOK_PROJECT_ID"]     = project_id
    os.environ["VOK_IMAGE_PATHS"]    = json.dumps(image_paths)
    os.environ["VOK_OUTPUT_DIR"]     = str(output_dir)
    os.environ["VOK_SKIP_VLM"]       = "1" if args.skip_vlm else "0"
    os.environ["VOK_ITERATIONS"]     = str(args.iterations)
    os.environ["VOK_JOB_FILE"]       = str(job_file)

    # Multi-object segmentation
    if args.multi_seg_prompt:
        os.environ["VOK_MULTI_SEG_PROMPT"]   = args.multi_seg_prompt
        os.environ["VOK_MULTI_SEG_AUTO"]     = "0"
        os.environ["VOK_MULTI_SEG"]          = "1"
        print(f"       Multi-seg   : ENABLED (prompt='{args.multi_seg_prompt}')")
    elif args.multi_seg:
        os.environ["VOK_MULTI_SEG"]          = "1"
        os.environ["VOK_MULTI_SEG_AUTO"]     = "1"
        os.environ["VOK_MULTI_SEG_PROMPT"]   = ""
        os.environ["VOK_MULTI_N_CLUSTERS"]   = str(args.multi_n_clusters)
        print(f"       Multi-seg   : ENABLED (auto, {args.multi_n_clusters} clusters)")
    else:
        os.environ["VOK_MULTI_SEG"]          = "0"
        os.environ["VOK_MULTI_SEG_AUTO"]     = "0"
        os.environ["VOK_MULTI_SEG_PROMPT"]   = ""
        print("       Multi-seg   : DISABLED  "
              "(use --multi-seg or --multi-seg-prompt to enable)")

    if args.multi_vote_thresh is not None:
        os.environ["VOK_MULTI_VOTE_THRESH"] = str(args.multi_vote_thresh)

    # Object editor
    if args.edit_target and args.edit_prompt:
        os.environ["VOK_EDIT_TARGET"]        = args.edit_target
        os.environ["VOK_EDIT_PROMPT"]        = args.edit_prompt
        os.environ["VOK_EDIT_SINGLE_PROMPT"] = ""
        print(f"       Edit target : {args.edit_target}")
        print(f"       Edit prompt : {args.edit_prompt}")
    elif args.edit_prompt:
        os.environ["VOK_EDIT_SINGLE_PROMPT"] = args.edit_prompt
        os.environ["VOK_EDIT_TARGET"]        = ""
        os.environ["VOK_EDIT_PROMPT"]        = ""
        print(f"       Edit prompt : {args.edit_prompt}  (single-object mode)")
    else:
        os.environ["VOK_EDIT_TARGET"]        = ""
        os.environ["VOK_EDIT_PROMPT"]        = ""
        os.environ["VOK_EDIT_SINGLE_PROMPT"] = ""
        print("       Object edit : DISABLED  (use --edit-prompt to enable)")

    # Edit-before-reconstruct mode (cleanest approach)
    if args.edit_before_reconstruct:
        os.environ["VOK_EDIT_BEFORE_RECONSTRUCT"] = "1"
        print("       Edit mode   : BEFORE RECONSTRUCTION (images edited first)")
    else:
        os.environ["VOK_EDIT_BEFORE_RECONSTRUCT"] = "0"

    # Write .env.local for processors that use python-dotenv
    env_override = PROCESSOR_DIR / ".env.local"
    with open(env_override, "w") as f:
        for k, v in os.environ.items():
            if k.startswith("VOK_"):
                f.write(f"{k}={v}\n")

    # ── Run main.py in-process ────────────────────────────────────────────────
    import importlib.util
    spec = importlib.util.spec_from_file_location("processor_main", main_py)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "run_pipeline"):
            print("DEBUG: Calling run_pipeline...")
            mod.run_pipeline(project_id)
        else:
            print("[ERROR] run_pipeline() not found in main.py")
            sys.exit(1)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"\n[ERROR] main.py exited with code {e.code}")
            sys.exit(e.code)

    # ── 4. Report output ──────────────────────────────────────────────────────
    print("─" * 60)
    print(f"\n[4/4] Pipeline finished. Checking output in {output_dir} ...")

    # Walk entire output tree so we catch model.ply wherever OpenSplat wrote it
    all_files = []
    if output_dir.exists():
        for root, _dirs, files in os.walk(output_dir):
            for fname in files:
                fpath = Path(root) / fname
                all_files.append(fpath)

    if all_files:
        print(f"\n  Output files in {output_dir}:")
        for fpath in sorted(all_files)[:40]:          # cap at 40 lines
            rel  = fpath.relative_to(output_dir)
            size = fpath.stat().st_size
            print(f"   {rel}  ({size/1024:.1f} KB)")
        if len(all_files) > 40:
            print(f"   … and {len(all_files)-40} more files")

        splat_files = [f for f in all_files
                       if f.suffix in (".splat", ".ply", ".glb")]
        if splat_files:
            print(f"\n  [SUCCESS] 3D output: {splat_files[0]}")
        else:
            print("\n  [NOTE] No .splat / .ply yet — "
                  "pipeline may still be running or failed mid-way.")
    else:
        print(f"\n  [NOTE] Output directory is empty: {output_dir}")
        print("         The pipeline may have failed before writing output.")
        print("         Check the logs above carefully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VokVision local pipeline bypass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # Basic run (MASt3R + segmentation + OpenSplat):
  python run_local.py --images ./photos/ --iterations 2500 --skip-vlm

  # Auto-discover all objects:
  python run_local.py --images ./photos/ --multi-seg

  # Segment specific objects by name:
  python run_local.py --images ./photos/ --multi-seg-prompt "bottle, table"

  # Segment then edit one label (edits merged into splat automatically):
  python run_local.py --images ./photos/ --multi-seg-prompt "bottle, table" \\
      --edit-target bottle --edit-prompt "make it metallic gold"

  # Batch-edit all labels:
  python run_local.py --images ./photos/ --multi-seg \\
      --edit-target batch \\
      --edit-prompt '{"object_0":"make it red","object_1":"wood texture"}'
        """,
    )

    parser.add_argument("--images", "-i", default=None,
                        help="Folder of input images (or single image).")
    parser.add_argument("--project", "-p", default="test_project_001",
                        help="Project name / ID (default: test_project_001)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip the Gemini VLM audit (no API key needed)")
    parser.add_argument("--iterations", "-n", type=int, default=2500,
                        help="OpenSplat iterations (default: 2500)")

    seg_grp   = parser.add_argument_group("Multi-object segmentation")
    seg_mutex = seg_grp.add_mutually_exclusive_group()
    seg_mutex.add_argument("--multi-seg", action="store_true",
                           help="Enable multi-object segmentation (auto-discover mode).")
    seg_mutex.add_argument("--multi-seg-prompt", metavar="LABELS", default=None,
                           help='Comma-separated labels, e.g. "bottle, table"')
    seg_grp.add_argument("--multi-n-clusters", type=int, default=6,
                         dest="multi_n_clusters",
                         help="Clusters for auto-discover mode (default: 6)")
    seg_grp.add_argument("--multi-vote-thresh", type=float, default=None,
                         dest="multi_vote_thresh",
                         help="Min vote fraction for label assignment (default: 0.40)")

    edit_grp = parser.add_argument_group("3D object editing")
    edit_grp.add_argument("--edit-prompt", metavar="PROMPT", default=None,
                          help='Edit instruction, e.g. "make it red". '
                               'Without --edit-target this edits the whole cloud.')
    edit_grp.add_argument("--edit-target", metavar="TARGET", default=None,
                          help='"single", "batch", or a label name from '
                               '--multi-seg-prompt.')
    edit_grp.add_argument("--edit-before-reconstruct", action="store_true",
                          dest="edit_before_reconstruct",
                          help="Edit images BEFORE MASt3R reconstruction — cleanest "
                               "approach for colour changes.  Requires --edit-prompt.")

    args = parser.parse_args()
    run(args)