from ultralytics import YOLO
import os
from glob import glob

model = YOLO("yolo11s-sunrgbd-seg.pt")

source_dir = "Validation/rgb"
output_dir = "imagenes_inferencia"

os.makedirs(output_dir, exist_ok=True)

imagenes = sorted(glob(os.path.join(source_dir, "*.jpg")))
ya_procesadas = set(
    f.replace(output_dir + "/", "") for f in glob(os.path.join(output_dir, "*.jpg"))
)
faltantes = [img for img in imagenes if os.path.basename(img) not in ya_procesadas]

print(f"Ya procesadas: {len(ya_procesadas)}, faltantes: {len(faltantes)}")

total_objetos = 0
for i, img_path in enumerate(faltantes):
    results = model(img_path, verbose=False)
    for r in results:
        if r.masks is not None:
            total_objetos += len(r.masks)
        r.save(os.path.join(output_dir, os.path.basename(img_path)))

    if (i + 1) % 10 == 0:
        print(f"Procesadas: {i + 1}/{len(faltantes)}")

print(f"Completadas: {len(faltantes)} imagenes restantes")
print(f"Total objetos detectados: {total_objetos}")
print(f"Resultados en: {output_dir}")
