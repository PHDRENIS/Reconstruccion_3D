import open3d as o3d
import numpy as np
import json
from pathlib import Path

# Cargar nube de puntos
cloud_path = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/reconstruction_cloud.ply")
cloud = o3d.io.read_point_cloud(str(cloud_path))

# Convertir a array
pts = np.asarray(cloud.points)

# Recorte Z (0.3m a 2.8m)
z_mask = (pts[:, 2] >= 0.3) & (pts[:, 2] <= 2.8)
pts_zcrop = pts[z_mask]

# Crear nube con recorte Z
cloud_zcrop = o3d.geometry.PointCloud()
cloud_zcrop.points = o3d.utility.Vector3dVector(pts_zcrop)

# Filtro estadístico
cloud_clean, _ = cloud_zcrop.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
pts_clean = np.asarray(cloud_clean.points)

# Centro de masa (mediana)
center_x = np.median(pts_clean[:, 0])
center_y = np.median(pts_clean[:, 1])

# Mejor rango: ±2.0m desde centro
r = 2.0
x_min = center_x - r
x_max = center_x + r
y_min = center_y - r
y_max = center_y + r

xy_mask = (pts_clean[:, 0] >= x_min) & (pts_clean[:, 0] <= x_max) & \
          (pts_clean[:, 1] >= y_min) & (pts_clean[:, 1] <= y_max)

pts_final = pts_clean[xy_mask]

cloud_final = o3d.geometry.PointCloud()
cloud_final.points = o3d.utility.Vector3dVector(pts_final)

# Guardar nube limpia
output_cloud = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/reconstruction_cloud_cleaned.ply")
o3d.io.write_point_cloud(str(output_cloud), cloud_final)

# Recomputar bounding box
bbox = cloud_final.get_axis_aligned_bounding_box()
dims = bbox.max_bound - bbox.min_bound

real_width = 3.810
real_length = 4.978
real_height = 2.527

err_w = ((dims[0] - real_width) / real_width) * 100
err_l = ((dims[1] - real_length) / real_length) * 100
err_h = ((dims[2] - real_height) / real_height) * 100

results = {
    "method": "Z_crop [0.3,2.8] + statistical_outlier(std=1.0) + symmetric_crop(±2.0m from median center)",
    "original_points": len(cloud.points),
    "final_points": len(pts_final),
    "center": [center_x, center_y],
    "dimensions": {
        "width_x": {"real": real_width, "model": float(dims[0]), "error_percent": float(err_w)},
        "length_y": {"real": real_length, "model": float(dims[1]), "error_percent": float(err_l)},
        "height_z": {"real": real_height, "model": float(dims[2]), "error_percent": float(err_h)}
    }
}

output_json = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/dimensions_final.json")
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"Nube limpia guardada: {output_cloud}")
print(f"Resultados guardados: {output_json}")
print(f"\nDimensiones finales:")
print(f"  X (ancho): {dims[0]:.3f}m (error: {err_w:+.1f}%)")
print(f"  Y (largo): {dims[1]:.3f}m (error: {err_l:+.1f}%)")
print(f"  Z (alto):  {dims[2]:.3f}m (error: {err_h:+.1f}%)")
