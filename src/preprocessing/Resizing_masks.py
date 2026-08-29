import os
import cv2
from tqdm import tqdm

INPUT_DIR = "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\depth_masks\\"
OUTPUT_DIR = "C:\\Users\\victo\\OneDrive\\Documents\\TT\\Preprocessing_scripts\\data\\dataset_final\\masks\\"
TARGET_SIZE = (640, 480)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.png')])
print(f"Procesando {len(files)} Máscaras...")

for filename in tqdm(files):
    mask = cv2.imread(os.path.join(INPUT_DIR, filename), 0)
    
    if mask is not None:
        mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), mask_resized)

print("Mascaras Terminadas")