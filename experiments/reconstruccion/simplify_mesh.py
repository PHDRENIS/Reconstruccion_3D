#!/usr/bin/env python3
"""Simplifica malla masiva usando vertex clustering (rapido) + quadric."""
import open3d as o3d, os
from pathlib import Path

base = Path(r"C:\Users\renea\Desktop\IPN\TT\Nueva reconstrucción")

print("Cargando malla 18M...")
mesh = o3d.io.read_triangle_mesh(str(base / "reconstruction_mesh.ply"))
ot = len(mesh.triangles)
print(f"  {len(mesh.vertices):,} v, {ot:,} t")

# Step 1: Vertex clustering (fast, lossy but preserves shape)
print("\nStep 1: Vertex clustering -> 1M tris...")
mesh2 = mesh.simplify_vertex_clustering(
    voxel_size=0.02,
    contraction=o3d.geometry.SimplificationContraction.Average,
)
mesh2.remove_degenerate_triangles()
print(f"  -> {len(mesh2.vertices):,} v, {len(mesh2.triangles):,} t")

# Step 2: Quadric decimation for quality
print("\nStep 2: Quadric decimation -> 500K...")
mesh3 = mesh2.simplify_quadric_decimation(target_number_of_triangles=500_000)
mesh3.remove_degenerate_triangles()
mesh3.compute_vertex_normals()
out = str(base / "reconstruction_mesh_lite.ply")
o3d.io.write_triangle_mesh(out, mesh3)
mb = os.path.getsize(out)/1e6
print(f"  Lite: {len(mesh3.vertices):,} v, {len(mesh3.triangles):,} t, {mb:.0f}MB")
print(f"  {out}")

# Ultra-lite
print("\nStep 3: Quadric -> 200K...")
mesh4 = mesh2.simplify_quadric_decimation(target_number_of_triangles=200_000)
mesh4.remove_degenerate_triangles()
mesh4.compute_vertex_normals()
out2 = str(base / "reconstruction_mesh_ultralite.ply")
o3d.io.write_triangle_mesh(out2, mesh4)
mb2 = os.path.getsize(out2)/1e6
print(f"  UltraLite: {len(mesh4.vertices):,} v, {len(mesh4.triangles):,} t, {mb2:.0f}MB")
print(f"  {out2}")
print("Done.")
