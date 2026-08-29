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

# Recorte Z (0.3m a 2.8m, como en los fotogramas individuales)
z_mask = (pts[:, 2] >= 0.3) & (pts[:, 2] <= 2.8)
pts_zcrop = pts[z_mask]
print(f"Tras recorte Z [0.3, 2.8]: {len(pts_zcrop)} puntos")

# Crear nueva nube con recorte Z
cloud_zcrop = o3d.geometry.PointCloud()
cloud_zcrop.points = o3d.utility.Vector3dVector(pts_zcrop)

# Filtro estadístico de outliers (nb_neighbors=20, std_ratio=2.0)
cloud_clean, inlier_indices = cloud_zcrop.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"Tras filtro estadístico: {len(cloud_clean.points)} puntos")

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
real_width = 3.810   # X
real_length = 4.978  # Y
real_height = 2.527  # Z

model_width = dimensions[0]   # X
model_length = dimensions[1]  # Y
model_height = dimensions[2]  # Z

# Calcular errores relativos
error_width = ((model_width - real_width) / real_width) * 100
error_length = ((model_length - real_length) / real_length) * 100
error_height = ((model_height - real_height) / real_height) * 100

print(f"\nComparación:")
print(f"  Ancho (X): Real={real_width:.3f}m, Modelo={model_width:.3f}m, Error={error_width:+.1f}%")
print(f"  Largo (Y): Real={real_length:.3f}m, Modelo={model_length:.3f}m, Error={error_length:+.1f}%")
print(f"  Alto (Z):  Real={real_height:.3f}m, Modelo={model_height:.3f}m, Error={error_height:+.1f}%")

# Guardar resultados
results = {
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

# Guardar nube limpia para visualización
output_cloud = Path("C:/Users/renea/Desktop/IPN/TT/Nueva reconstruccion/reconstruction_cloud_cleaned.ply")
o3d.io.write_point_cloud(str(output_cloud), cloud_clean)
print(f"Nube limpia guardada en: {output_cloud}")
