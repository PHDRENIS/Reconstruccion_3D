#!/usr/bin/env python3
"""
Visualizacion de planos detectados para figura del documento
Genera una imagen con los planos RANSAC coloreados.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Posible reconstruccion final")
OUTPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Metricas Reconstruccion")

# Procesar ambas mallas para generar visualizaciones
INPUT_FILES = [
    ("room_final.ply", "Malla Limpia (room_final)"),
    ("room_labeled.ply", "Nube Etiquetada (room_labeled)"),
]

def visualize_detected_planes(pcd_path, label, suffix):
    print(f"\nVisualizando: {label}")
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        print(f"ERROR: No se pudo cargar {pcd_path}")
        return
    
    print(f"Puntos: {len(pcd.points):,}")
    
    # Voxel downsample
    pcd_ds = pcd.voxel_down_sample(0.02)
    print(f"Puntos downsample: {len(pcd_ds.points):,}")
    
    # RANSAC
    remaining = pcd_ds
    planes = []
    colors = [
        [1.0, 0.2, 0.2],   # Rojo - Pared 1
        [0.2, 0.4, 1.0],   # Azul - Pared 2
        [0.2, 1.0, 0.2],   # Verde - Pared 3
        [1.0, 1.0, 0.2],   # Amarillo - Pared 4
        [0.6, 0.3, 0.1],   # Marron - Piso
        [0.5, 0.5, 0.5],   # Gris - Techo
        [1.0, 0.5, 0.0],   # Naranja
        [0.8, 0.2, 0.8],   # Morado
    ]
    
    all_pts = []
    all_colors = []
    
    for i in range(8):
        if len(remaining.points) < 500:
            break
        
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.06, ransac_n=3, num_iterations=3000
        )
        
        a, b, c, d = plane_model
        nz = abs(c)
        inlier_pts = np.asarray(remaining.points)[inliers]
        
        if len(inlier_pts) < 500:
            break
        
        color = colors[i % len(colors)]
        all_pts.append(inlier_pts)
        all_colors.append(np.tile(color, (len(inlier_pts), 1)))
        
        tipo = "horizontal" if nz > 0.85 else "vertical" if nz < 0.3 else "inclinado"
        print(f"  Plano {i}: {tipo}, pts={len(inlier_pts):,}, nz={nz:.3f}")
        
        remaining = remaining.select_by_index(inliers, invert=True)
    
    # Puntos restantes en gris claro
    rest_pts = np.asarray(remaining.points)
    if len(rest_pts) > 0:
        all_pts.append(rest_pts)
        all_colors.append(np.ones((len(rest_pts), 3)) * 0.85)
    
    # Concatenar
    combined_pts = np.vstack(all_pts)
    combined_colors = np.vstack(all_colors)
    
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(combined_pts)
    pcd_colored.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Guardar nube coloreada
    output_pcd = OUTPUT_DIR / f"planes_detected_colored_{suffix}.ply"
    o3d.io.write_point_cloud(str(output_pcd), pcd_colored)
    print(f"  Nube coloreada guardada: {output_pcd}")
    
    # Visualizar (abre ventana, capturar imagen)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd_colored)
    
    # Configurar vista
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_front([0.5, -0.5, 0.5])
    ctr.set_lookat([0, 0, 1])
    ctr.set_up([0, 0, 1])
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capturar imagen
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    # Convertir a imagen y guardar
    img_array = np.asarray(img)
    img_path = OUTPUT_DIR / f"planes_detected_{suffix}.png"
    plt.imsave(str(img_path), img_array)
    print(f"  Imagen guardada: {img_path}")

# Ejecutar para ambas mallas
for filename, label in INPUT_FILES:
    filepath = INPUT_DIR / filename
    if filepath.exists():
        suffix = filename.replace(".ply", "")
        visualize_detected_planes(filepath, label, suffix)
    else:
        print(f"ERROR: No se encontro {filepath}")

print("\nVisualizacion completada.")
