#!/usr/bin/env bash
# rebuild_opensplat.sh
# ═══════════════════════════════════════════════════════════════════════════════
# Rebuild OpenSplat against the PyTorch version installed in the processor venv.
#
# Run from the vok-vision-main repo root:
#   bash rebuild_opensplat.sh
#
# The error  "undefined symbol: _ZNK3c105Error4whatEv"  means OpenSplat was
# compiled against a different libtorch ABI than your current venv provides.
# This script detects the correct cmake prefix from your venv and rebuilds.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENSPLAT_DIR="$REPO_ROOT/pipeline/opensplat"
VENV_PYTHON="$REPO_ROOT/backend/processor/venv/bin/python"

echo "═══════════════════════════════════════════════════════════"
echo " OpenSplat Rebuild Script"
echo " Repo root   : $REPO_ROOT"
echo " OpenSplat   : $OPENSPLAT_DIR"
echo " Venv Python : $VENV_PYTHON"
echo "═══════════════════════════════════════════════════════════"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -d "$OPENSPLAT_DIR" ]; then
    echo "❌ OpenSplat directory not found: $OPENSPLAT_DIR"
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Venv Python not found: $VENV_PYTHON"
    echo "   Activate your venv or edit VENV_PYTHON at the top of this script."
    exit 1
fi

# ── Get PyTorch cmake prefix path ─────────────────────────────────────────────
echo ""
echo "🔍 Detecting PyTorch cmake prefix …"
TORCH_CMAKE=$("$VENV_PYTHON" -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")

if [ -z "$TORCH_CMAKE" ]; then
    echo "❌ Could not detect torch.utils.cmake_prefix_path."
    echo "   Make sure PyTorch is installed in the venv:"
    echo "   $VENV_PYTHON -m pip install torch"
    exit 1
fi

TORCH_VER=$("$VENV_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$("$VENV_PYTHON" -c "import torch; print(torch.version.cuda or 'cpu-only')" 2>/dev/null)

echo "   PyTorch version : $TORCH_VER"
echo "   CUDA version    : $CUDA_VER"
echo "   cmake prefix    : $TORCH_CMAKE"

# ── Check CUDA / CPU build ────────────────────────────────────────────────────
if [ "$CUDA_VER" = "cpu-only" ] || [ "$CUDA_VER" = "None" ]; then
    CUDA_FLAG="-DOPENSPLAT_BUILD_WITH_TORCH=ON -DCMAKE_CUDA_COMPILER=OFF"
    echo "   Build mode      : CPU"
else
    # Check nvcc is available
    if command -v nvcc &>/dev/null; then
        NVCC_VER=$(nvcc --version | grep "release" | awk '{print $NF}' | tr -d ',')
        echo "   nvcc version    : $NVCC_VER"
        CUDA_FLAG=""
    else
        echo "   ⚠️  nvcc not found — building CPU version"
        CUDA_FLAG="-DOPENSPLAT_BUILD_WITH_TORCH=ON -DCMAKE_CUDA_COMPILER=OFF"
    fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────
BUILD_DIR="$OPENSPLAT_DIR/build"
echo ""
echo "🔨 Rebuilding OpenSplat …"
echo "   Build dir: $BUILD_DIR"

# Wipe old build to avoid stale cmake cache
if [ -d "$BUILD_DIR" ]; then
    echo "   Removing old build dir …"
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

echo ""
echo "⚙️  Running cmake …"
cmake .. \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    -DCMAKE_BUILD_TYPE=Release \
    $CUDA_FLAG

echo ""
echo "🔧 Building (this takes a few minutes) …"
cmake --build . --config Release -j2

# ── Verify ────────────────────────────────────────────────────────────────────
OPENSPLAT_BIN="$BUILD_DIR/opensplat"
if [ ! -f "$OPENSPLAT_BIN" ]; then
    echo "❌ Build failed — opensplat binary not found at $OPENSPLAT_BIN"
    exit 1
fi

echo ""
echo "✅ Build succeeded: $OPENSPLAT_BIN"

# Quick sanity check
echo ""
echo "🧪 Running opensplat --version …"
"$OPENSPLAT_BIN" --version 2>&1 | head -5 || true

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " ✅ OpenSplat rebuilt successfully!"
echo ""
echo " Now re-run your pipeline:"
echo "   python run_local.py --edit-before-reconstruct \\"
echo "       --edit-target bottle --edit-prompt 'make it metallic gold'"
echo "═══════════════════════════════════════════════════════════"