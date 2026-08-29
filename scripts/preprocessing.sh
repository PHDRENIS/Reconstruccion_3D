#!/usr/bin/env bash
# Preprocessing SUN RGB-D: PNG(mm)->NPY(m), resize, binarizar
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "[1/4] png_to_npy"
python -m src.preprocessing.png_to_npy
echo "[2/4] Resizing depth/masks"
python -m src.preprocessing.Resizing_depth --input "$ROOT/data/SUNRGBD/Train/depth" --output "$ROOT/data/SUNRGBD/Train/depth_input" 2>&1 | head
python -m src.preprocessing.Resizing_masks --input "$ROOT/data/SUNRGBD/Train/mask" --output "$ROOT/data/SUNRGBD/Train/masks_resized"
echo "[3/4] quitar _abs"
python -m src.preprocessing.quitar_abs --root "$ROOT/data/SUNRGBD"
echo "[4/4] to_binary"
python -m src.preprocessing.to_binary --depth-dir "$ROOT/data/SUNRGBD/Train/depth_input" --mask-dir "$ROOT/data/SUNRGBD/Train/masks"
echo "Done."
