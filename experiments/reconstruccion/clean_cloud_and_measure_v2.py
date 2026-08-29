import open3d as o3d
import numpy as np
import json
from pathlib import Path

# Cargar nube de puntos
cloud_path = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/reconstruction_cloud.ply")
cloud = o3d.io.read_point_cloud(str(cloud_path))

print(f"Nube original: {len(cloud.points)} puntos")

# Convertir a array para manipulación
pts = np.asarray(cloud.points)

# Recorte Z (0.3m a 2.8m)
z_mask = (pts[:, 2] >= 0.3) & (pts[:, 2] <= 2.8)
pts_zcrop = pts[z_mask]
print(f"Tras recorte Z [0.3, 2.8]: {len(pts_zcrop)} puntos")

# Crear nueva nube con recorte Z
cloud_zcrop = o3d.geometry.PointCloud()
cloud_zcrop.points = o3d.utility.Vector3dVector(pts_zcrop)

# Filtro estadístico más agresivo (std_ratio=1.0)
cloud_clean, inlier_indices = cloud_zcrop.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
print(f"Tras filtro estadístico (std=1.0): {len(cloud_clean.points)} puntos")

# Recomputar bounding box
bbox = cloud_clean.get_axis_aligned_bounding_box()
min_bound = bbox.min_bound
max_bound = bbox.max_bound

dimensions = max_bound - min_bound
print(f"\nBounding Box:")
print(f"  Min: {min_bound}")
print(f"  Max: {max_bound}")
print(f"  Dimensiones: {dimensions}")

# Medidas reales (metros)
real_width = 3.810
real_length = 4.978
real_height = 2.527

model_width = dimensions[0]
model_length = dimensions[1]
model_height = dimensions[2]

error_width = ((model_width - real_width) / real_width) * 100
error_length = ((model_length - real_length) / real_length) * 100
error_height = ((model_height - real_height) / real_height) * 100

print(f"\nComparación:")
print(f"  Ancho (X): Real={real_width:.3f}m, Modelo={model_width:.3f}m, Error={error_width:+.1f}%")
print(f"  Largo (Y): Real={real_length:.3f}m, Modelo={model_length:.3f}m, Error={error_length:+.1f}%")
print(f"  Alto (Z):  Real={real_height:.3f}m, Modelo={model_height:.3f}m, Error={error_height:+.1f}%")

# Si sigue mal, intentar recorte por percentiles en X/Y
if model_width > 6.0 or model_length > 7.0:
    print("\n--- Aplicando recorte por percentiles en X/Y ---")
    pts_clean = np.asarray(cloud_clean.points)
    
    # Percentiles 1% y 99% para X e Y
    x_min, x_max = np.percentile(pts_clean[:, 0], [1, 99])
    y_min, y_max = np.percentile(pts_clean[:, 1], [1, 99])
    
    xy_mask = (pts_clean[:, 0] >= x_min) & (pts_clean[:, 0] <= x_max) & \
              (pts_clean[:, 1] >= y_min) & (pts_clean[:, 1] <= y_max)
    
    pts_final = pts_clean[xy_mask]
    
    cloud_final = o3d.geometry.PointCloud()
    cloud_final.points = o3d.utility.Vector3dVector(pts_final)
    
    bbox_final = cloud_final.get_axis_aligned_bounding_box()
    dimensions_final = bbox_final.max_bound - bbox_final.min_bound
    
    print(f"Tras percentiles 1-99%: {len(cloud_final.points)} puntos")
    print(f"  X range: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  Y range: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  Dimensiones: {dimensions_final}")
    
    error_width_final = ((dimensions_final[0] - real_width) / real_width) * 100
    error_length_final = ((dimensions_final[1] - real_length) / real_length) * 100
    error_height_final = ((dimensions_final[2] - real_height) / real_height) * 100
    
    print(f"\nComparación final:")
    print(f"  Ancho (X): Real={real_width:.3f}m, Modelo={dimensions_final[0]:.3f}m, Error={error_width_final:+.1f}%")
    print(f"  Largo (Y): Real={real_length:.3f}m, Modelo={dimensions_final[1]:.3f}m, Error={error_length_final:+.1f}%")
    print(f"  Alto (Z):  Real={real_height:.3f}m, Modelo={dimensions_final[2]:.3f}m, Error={error_height_final:+.1f}%")
    
    # Guardar resultados finales
    results = {
        "method": "Z_crop + statistical_outlier (std=1.0) + percentile_1_99_XY",
        "original_points": len(cloud.points),
        "after_zcrop": len(pts_zcrop),
        "after_statistical_filter": len(cloud_clean.points),
        "after_percentile": len(cloud_final.points),
        "bounding_box": {
            "min": bbox_final.min_bound.tolist(),
            "max": bbox_final.max_bound.tolist(),
            "dimensions": dimensions_final.tolist()
        },
        "comparison": {
            "width": {"real": real_width, "model": dimensions_final[0], "error_percent": error_width_final},
            "length": {"real": real_length, "model": dimensions_final[1], "error_percent": error_length_final},
            "height": {"real": real_height, "model": dimensions_final[2], "error_percent": error_height_final}
        }
    }
    
    # Guardar nube final
    output_cloud_final = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/reconstruction_cloud_final_cleaned.ply")
    o3d.io.write_point_cloud(str(output_cloud_final), cloud_final)
    print(f"\nNube final guardada en: {output_cloud_final}")
else:
    # Guardar resultados intermedios
    results = {
        "method": "Z_crop + statistical_outlier (std=1.0)",
        "original_points": len(cloud.points),
        "after_zcrop": len(pts_zcrop),
        "after_statistical_filter": len(cloud_clean.points),
        "bounding_box": {
            "min": min_bound.tolist(),
            "max": max_bound.tolist(),
            "dimensions": dimensions.tolist()
        },
        "comparison": {
            "width": {"real": real_width, "model": model_width, "error_percent": error_width},
            "length": {"real": real_length, "model": model_length, "error_percent": error_length},
            "height": {"real": real_height, "model": model_height, "error_percent": error_height}
        }
    }

output_path = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/dimensions_cleaned.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResultados guardados en: {output_path}")
