# Convierte PNG de profundidad (mm) -> NPY (m).
# Uso: python -m src.preprocessing.png_to_npy [--input DIR] [--output DIR]
# Por defecto opera sobre data/SUNRGBD/Train/depth_input -> depth_input_npy y depth_gt -> depth_gt_npy.
# Rutas legacy absolutas (C:\Users\victo\...) se conservan como comentario de referencia.
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# Raiz del repo (TT Limpio/)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TASKS = [
    {
        "input": REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth_input",
        "output": REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth_input_npy",
    },
    {
        "input": REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth_gt",
        "output": REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth_gt_npy",
    },
]

# Legacy (no usar): C:\Users\victo\OneDrive\Documents\TT\SUNRBG_IMAGES\...


def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def convertir_png_a_npy(tareas=None):
    print("Iniciando conversión de PNG (milímetros) a NPY (metros).")
    tareas = tareas or DEFAULT_TASKS
    for tarea in tareas:
        input_dir = Path(tarea["input"])
        output_dir = Path(tarea["output"])

        if not input_dir.exists():
            print(f"[ERROR] No se encontró la carpeta: {input_dir}")
            continue

        ensure_dir(output_dir)

        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
        print(f"\nProcesando carpeta: {input_dir}")
        print(f"Archivos encontrados: {len(files)}")

        for filename in tqdm(files):
            path_in = input_dir / filename
            depth_mm = cv2.imread(str(path_in), cv2.IMREAD_UNCHANGED)

            if depth_mm is None:
                print(f"Error leyendo: {filename}")
                continue

            depth_meters = depth_mm.astype(np.float32) / 1000.0

            name_no_ext = os.path.splitext(filename)[0]
            path_out = output_dir / (name_no_ext + ".npy")

            np.save(str(path_out), depth_meters)

    print("\nConversión finalizada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PNG (mm) -> NPY (m)")
    parser.add_argument("--input", type=str, default=None, help="Carpeta con PNGs de entrada")
    parser.add_argument("--output", type=str, default=None, help="Carpeta de salida NPY")
    args = parser.parse_args()

    if args.input and args.output:
        convertir_png_a_npy([{"input": Path(args.input), "output": Path(args.output)}])
    else:
        convertir_png_a_npy()
