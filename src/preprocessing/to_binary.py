#USed to make makes of some lost in previous steps.
import os
import cv2
import numpy as np
from tqdm import tqdm


TASKS = [
    {
        "depth_input": "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth_input\\",  # Fuente .npy correcta
        "mask_output": "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\masks\\"         # Destino .png (se sobrescribirá)
    }
]
# =================================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def regenerar_mascaras():
    print("Iniciando regeneración de máscaras basada en Depth Input.")

    for task in TASKS:
        depth_dir = task["depth_input"]
        mask_dir = task["mask_output"]
        
        if not os.path.exists(depth_dir):
            print(f"[ERROR] No existe la carpeta fuente: {depth_dir}")
            continue

        ensure_dir(mask_dir) 

        files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        
        print(f"\nProcesando carpeta: {depth_dir}")
        print(f"Generando {len(files)} máscaras.")

        for filename in tqdm(files):
            depth_path = os.path.join(depth_dir, filename)
            depth = np.load(depth_path)
            
        
            mask = (depth > 0).astype(np.uint8) * 255

            name_no_ext = os.path.splitext(filename)[0]
            mask_filename = name_no_ext + ".png"
            mask_path = os.path.join(mask_dir, mask_filename)
            
            cv2.imwrite(mask_path, mask)

    print("\nProceso terminado")
    print("Ahora tus máscaras tienen exactamente los mismos nombres que tus archivos de profundidad.")

if __name__ == "__main__":
    regenerar_mascaras()