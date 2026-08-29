import os
import cv2
from tqdm import tqdm

INPUT_DIR = "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth\\"
OUTPUT_DIR = "C:\\Users\\victo\\OneDrive\\Documents\\TT\\Preprocessing_scripts\\data\\dataset_final\\depth_input\\"
TARGET_SIZE = (640, 480)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.png')])
print(f"Procesando {len(files)} Inputs de Profundidad...")

for filename in tqdm(files):
    depth_img = cv2.imread(os.path.join(INPUT_DIR, filename), cv2.IMREAD_UNCHANGED)
    
    if depth_img is not None:
        depth_resized = cv2.resize(depth_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), depth_resized)

print("Depth Input Terminado")