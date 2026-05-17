#!/bin/bash
# ============================================================
#  VokVision — one-shot local pipeline runner
#  Place this at the ROOT of vok-vision-main/
#  Usage:
#    chmod +x run_pipeline.sh
#    ./run_pipeline.sh                        # use images already in storage/
#    ./run_pipeline.sh --images ./my_photos/  # specify image folder
#    ./run_pipeline.sh --skip-vlm             # no Gemini API key needed
#    ./run_pipeline.sh --iterations 3000      # fast test (not full quality)
# ============================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/backend/processor/venv"
PYTHON="$VENV/bin/python3"

# ── Check venv exists ──────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo ""
  echo "[ERROR] Python venv not found at: $VENV"
  echo ""
  echo "  Set it up first:"
  echo "    cd backend/processor"
  echo "    python3.10 -m venv venv"
  echo "    venv/bin/pip install --upgrade pip"
  echo "    venv/bin/pip install -r requirements.txt"
  echo ""
  exit 1
fi

echo "  Using Python: $PYTHON"
echo "  Repo root:    $REPO_ROOT"
echo ""

# ── Activate venv and run ──────────────────────────────────
source "$VENV/bin/activate"
cd "$REPO_ROOT"

python run_local.py "$@"
