import argparse
import os
import cv2
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "SUNRGBD" / "Train" / "depth_input"
TARGET_SIZE = (640, 480)

# Legacy (no usar):
# INPUT_DIR = "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth\\"
# OUTPUT_DIR = "C:\\Users\\victo\\OneDrive\\Documents\\TT\\Preprocessing_scripts\\data\\dataset_final\\depth_input\\"


def main(input_dir=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] No existe {input_dir}. Usa --input para apuntar a tu carpeta.")
        return

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    print(f"Procesando {len(files)} Inputs de Profundidad...")

    for filename in tqdm(files):
        depth_img = cv2.imread(str(input_dir / filename), cv2.IMREAD_UNCHANGED)

        if depth_img is not None:
            depth_resized = cv2.resize(depth_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_dir / filename), depth_resized)

    print("Depth Input Terminado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    main(args.input, args.output)
