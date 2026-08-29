#!/usr/bin/env python3
"""
Medicion dimensional de la habitacion
Extrae bounding box y dimensiones del modelo 3D.
"""

import numpy as np
import open3d as o3d
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Posible reconstruccion final")
OUTPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Metricas Reconstruccion")

INPUT_FILES = [
    ("room_final.ply", "Malla Limpia (room_final)"),
    ("room_labeled.ply", "Nube Etiquetada (room_labeled)"),
]

def measure_dimensions(filepath, label):
    print(f"\n--- {label} ---")
    
    pcd = o3d.io.read_point_cloud(str(filepath))
    if pcd.is_empty():
        print(f"ERROR: No se pudo cargar {filepath}")
        return None
    
    pts = np.asarray(pcd.points)
    
    # Bounding box
    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    dimensions = bbox_max - bbox_min
    
    # Area del piso (proyeccion XY)
    x_range = dimensions[0]
    y_range = dimensions[1]
    z_range = dimensions[2]
    floor_area = x_range * y_range
    volume = x_range * y_range * z_range
    
    results = {
        'label': label,
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'ancho_x': round(x_range, 3),
        'largo_y': round(y_range, 3),
        'alto_z': round(z_range, 3),
        'area_piso': round(floor_area, 2),
        'volumen': round(volume, 2),
        'n_points': len(pts)
    }
    
    print(f"  Bounding Box: X[{bbox_min[0]:.2f}, {bbox_max[0]:.2f}], Y[{bbox_min[1]:.2f}, {bbox_max[1]:.2f}], Z[{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]")
    print(f"  Dimensiones: {x_range:.2f} m x {y_range:.2f} m x {z_range:.2f} m")
    print(f"  Area piso: {floor_area:.2f} m²")
    print(f"  Volumen: {volume:.2f} m³")
    
    return results

# Ejecutar
all_results = []
for filename, label in INPUT_FILES:
    filepath = INPUT_DIR / filename
    if filepath.exists():
        res = measure_dimensions(filepath, label)
        if res:
            all_results.append(res)

# Guardar
with open(OUTPUT_DIR / "room_dimensions.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("MEDICIONES DIMENSIONALES DEL MODELO 3D\n")
    f.write("NOTA: Para comparacion con mediciones fisicas futuras\n")
    f.write("="*60 + "\n\n")
    
    for res in all_results:
        f.write(f"--- {res['label']} ---\n")
        f.write(f"Puntos: {res['n_points']:,}\n")
        f.write(f"Bounding Box min: {res['bbox_min']}\n")
        f.write(f"Bounding Box max: {res['bbox_max']}\n")
        f.write(f"Ancho (X): {res['ancho_x']} m\n")
        f.write(f"Largo (Y): {res['largo_y']} m\n")
        f.write(f"Alto (Z): {res['alto_z']} m\n")
        f.write(f"Area piso: {res['area_piso']} m²\n")
        f.write(f"Volumen: {res['volumen']} m³\n\n")
    
    f.write("="*60 + "\n")
    f.write("ESPACIO RESERVADO PARA COMPARACION CON MEDICIONES FISICAS:\n")
    f.write("Ancho real: ___ m\n")
    f.write("Largo real: ___ m\n")
    f.write("Alto real:  ___ m\n")
    f.write("="*60 + "\n")

print(f"\nResultados guardados en: {OUTPUT_DIR / 'room_dimensions.txt'}")
