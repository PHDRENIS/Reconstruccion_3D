#USed to convert depth png images in millimeters to npy files in meters, some lost in previous steps.
import os
import cv2
import numpy as np
from tqdm import tqdm


RUTAS = [
    {
        "input": "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth_input\\", 
        "output": "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth_input_npy\\" 
    },
    {
        "input": "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth_gt\\",    
        "output": "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth_gt_npy\\"    
    }
]


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convertir_png_a_npy():
    print("Iniciando conversión de PNG (milímetros) a NPY (metros).")

    for tarea in RUTAS:
        input_dir = tarea["input"]
        output_dir = tarea["output"]
        
        # Verificar que la carpeta de entrada exista
        if not os.path.exists(input_dir):
            print(f"ERROR No se encontró la carpeta: {input_dir}")
            continue

        ensure_dir(output_dir)
        
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
        print(f"\nProcesando carpeta: {input_dir}")
        print(f"Archivos encontrados: {len(files)}")

        for filename in tqdm(files):
            
            path_in = os.path.join(input_dir, filename)
            depth_mm = cv2.imread(path_in, cv2.IMREAD_UNCHANGED)

            if depth_mm is None:
                print(f"Error leyendo: {filename}")
                continue

            
            depth_meters = depth_mm.astype(np.float32) / 1000.0

            name_no_ext = os.path.splitext(filename)[0]
            path_out = os.path.join(output_dir, name_no_ext + ".npy")
            
            np.save(path_out, depth_meters)

    print("\nConversión de entrenamiento finalizada")

if __name__ == "__main__":
    convertir_png_a_npy()