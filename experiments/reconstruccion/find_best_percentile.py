import open3d as o3d
import numpy as np
import json
from pathlib import Path

# Cargar nube de puntos
cloud_path = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/reconstruction_cloud.ply")
cloud = o3d.io.read_point_cloud(str(cloud_path))

print(f"Nube original: {len(cloud.points)} puntos")

# Convertir a array
pts = np.asarray(cloud.points)

# Recorte Z (0.3m a 2.8m)
z_mask = (pts[:, 2] >= 0.3) & (pts[:, 2] <= 2.8)
pts_zcrop = pts[z_mask]
print(f"Tras recorte Z [0.3, 2.8]: {len(pts_zcrop)} puntos")

# Crear nube con recorte Z
cloud_zcrop = o3d.geometry.PointCloud()
cloud_zcrop.points = o3d.utility.Vector3dVector(pts_zcrop)

# Filtro estadístico más agresivo
cloud_clean, _ = cloud_zcrop.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
print(f"Tras filtro estadístico (std=1.0): {len(cloud_clean.points)} puntos")

# Aplicar percentiles más restrictivos en X e Y
pts_clean = np.asarray(cloud_clean.points)

# Probar diferentes percentiles para encontrar el mejor balance
percentiles_to_test = [1, 2, 5, 10]
real_width = 3.810
real_length = 4.978
real_height = 2.527

best_result = None
best_score = float('inf')

for p in percentiles_to_test:
    x_min, x_max = np.percentile(pts_clean[:, 0], [p, 100-p])
    y_min, y_max = np.percentile(pts_clean[:, 1], [p, 100-p])
    
    xy_mask = (pts_clean[:, 0] >= x_min) & (pts_clean[:, 0] <= x_max) & \
              (pts_clean[:, 1] >= y_min) & (pts_clean[:, 1] <= y_max)
    
    pts_final = pts_clean[xy_mask]
    
    if len(pts_final) < 1000:
        continue
    
    cloud_final = o3d.geometry.PointCloud()
    cloud_final.points = o3d.utility.Vector3dVector(pts_final)
    
    bbox = cloud_final.get_axis_aligned_bounding_box()
    dims = bbox.max_bound - bbox.min_bound
    
    err_w = abs(((dims[0] - real_width) / real_width) * 100)
    err_l = abs(((dims[1] - real_length) / real_length) * 100)
    err_h = abs(((dims[2] - real_height) / real_height) * 100)
    
    score = err_w + err_l + err_h  # Suma de errores absolutos
    
    print(f"\nPercentil {p}-{100-p}%:")
    print(f"  Puntos: {len(pts_final)}")
    print(f"  Dimensiones: {dims}")
    print(f"  Errores: X={err_w:.1f}%, Y={err_l:.1f}%, Z={err_h:.1f}%")
    print(f"  Score: {score:.1f}")
    
    if score < best_score:
        best_score = score
        best_result = {
            "percentile": p,
            "points": len(pts_final),
            "dimensions": dims.tolist(),
            "errors": {"width": err_w, "length": err_l, "height": err_h},
            "bounds": {
                "x": [x_min, x_max],
                "y": [y_min, y_max],
                "z": [0.3, 2.8]
            }
        }

print(f"\n{'='*50}")
print(f"MEJOR RESULTADO: Percentil {best_result['percentile']}-{100-best_result['percentile']}%")
print(f"  Dimensiones: {best_result['dimensions']}")
print(f"  Errores: X={best_result['errors']['width']:.1f}%, Y={best_result['errors']['length']:.1f}%, Z={best_result['errors']['height']:.1f}%")
print(f"{'='*50}")

# Guardar resultados del mejor
output_path = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/dimensions_best.json")
with open(output_path, "w") as f:
    json.dump(best_result, f, indent=2)

print(f"\nMejor resultado guardado en: {output_path}")
