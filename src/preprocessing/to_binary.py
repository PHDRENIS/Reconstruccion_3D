# Regenera mascaras binarias (255 donde depth>0) a partir de .npy.
# Uso: python -m src.preprocessing.to_binary [--depth-dir DIR] [--mask-dir DIR]
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPTH = REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth_input"
DEFAULT_MASK = REPO_ROOT / "data" / "SUNRGBD" / "Train" / "masks"

# Legacy (no usar): C:\Users\victo\OneDrive\Documents\TT\SUNRBG_IMAGES\...


def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def regenerar_mascaras(depth_dir=DEFAULT_DEPTH, mask_dir=DEFAULT_MASK):
    print("Iniciando regeneración de máscaras basada en Depth Input.")
    depth_dir = Path(depth_dir)
    mask_dir = Path(mask_dir)

    if not depth_dir.exists():
        print(f"[ERROR] No existe la carpeta fuente: {depth_dir}")
        return

    ensure_dir(mask_dir)

    files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])

    print(f"\nProcesando carpeta: {depth_dir}")
    print(f"Generando {len(files)} máscaras.")

    for filename in tqdm(files):
        depth_path = depth_dir / filename
        depth = np.load(str(depth_path))

        mask = (depth > 0).astype(np.uint8) * 255

        name_no_ext = os.path.splitext(filename)[0]
        mask_filename = name_no_ext + ".png"
        mask_path = mask_dir / mask_filename

        cv2.imwrite(str(mask_path), mask)

    print("\nProceso terminado")
    print("Ahora tus máscaras tienen exactamente los mismos nombres que tus archivos de profundidad.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerar mascaras binarias desde depth .npy")
    parser.add_argument("--depth-dir", type=str, default=str(DEFAULT_DEPTH))
    parser.add_argument("--mask-dir", type=str, default=str(DEFAULT_MASK))
    args = parser.parse_args()
    regenerar_mascaras(args.depth_dir, args.mask_dir)
