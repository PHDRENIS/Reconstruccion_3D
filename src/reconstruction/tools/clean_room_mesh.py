#!/usr/bin/env python3
"""
Limpieza rapida de malla: elimina piso y techo, perfila paredes con RANSAC.
"""

import argparse, sys
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cloud", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--floor-dist", type=float, default=0.08)
    p.add_argument("--voxel", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Cargando nube...")
    pcd = o3d.io.read_point_cloud(args.cloud)
    print(f"  Puntos originales: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(args.voxel)
    print(f"  Puntos downsampled: {len(pcd.points)}")

    # Fase 1: Detectar horizontales (piso/techo)
    floor_plane, ceiling_plane = None, None
    remaining = pcd
    planes_found = []

    for _ in range(3):
        plane, inliers = remaining.segment_plane(
            distance_threshold=0.04, ransac_n=3, num_iterations=2000
        )
        a, b, c, d = plane
        normal = np.array([a, b, c])
        nz = abs(c)

        if nz > 0.85:
            points_arr = np.asarray(remaining.points)[inliers]
            centroid_z = float(points_arr[:, 2].mean())
            planes_found.append((plane, centroid_z, "horizontal"))
            print(f"  Plano horizontal: Z={centroid_z:.2f}, normal_z={nz:.3f}")
        remaining = remaining.select_by_index(inliers, invert=True)

    if planes_found:
        h_planes = [p for p in planes_found if p[2] == "horizontal"]
        if len(h_planes) >= 2:
            h_planes.sort(key=lambda x: x[1])
            floor_plane = h_planes[0][0]
            ceiling_plane = h_planes[-1][0]
        elif len(h_planes) == 1:
            if h_planes[0][1] < 1.0:
                floor_plane = h_planes[0][0]
            else:
                ceiling_plane = h_planes[0][0]

    # Fase 2: Eliminar piso y techo
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    keep = np.ones(len(points), dtype=bool)

    for plane, label in [(floor_plane, "piso"), (ceiling_plane, "techo")]:
        if plane is None:
            continue
        a, b, c, d_val = plane
        dists = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d_val)
        mask = dists < args.floor_dist
        keep[mask] = 0
        print(f"  Eliminado {label}: {int(mask.sum())} puntos")

    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(points[keep])
    if colors is not None:
        pcd_clean.colors = o3d.utility.Vector3dVector(colors[keep])
    print(f"  Puntos restantes: {len(pcd_clean.points)}")

    # Fase 3: Detectar paredes
    remaining = pcd_clean
    walls = []
    for _ in range(4):
        if len(remaining.points) < 100:
            break
        plane_w, inliers = remaining.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=2000
        )
        a, b, c, d_val = plane_w
        nz = abs(c)
        if nz < 0.3:
            walls.append(plane_w)
            print(f"  Pared detectada: nz={nz:.3f}")
        remaining = remaining.select_by_index(inliers, invert=True)

    # Fase 4: Malla
    pcd_clean.estimate_normals()
    pcd_clean.orient_normals_consistent_tangent_plane(30)

    print("  Reconstruyendo malla (Poisson)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=9)
    print(f"  Malla: {len(mesh.vertices)} vert, {len(mesh.triangles)} tris")

    # Eliminar triangulos con baja densidad
    if len(densities) > 0:
        density_thresh = np.quantile(densities, 0.05)
        verts_to_remove = densities < density_thresh
        mesh.remove_vertices_by_mask(verts_to_remove)

    mesh.compute_vertex_normals()

    # Simplificar
    target_tris = min(200000, len(mesh.triangles) // 10)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    print(f"  Malla simplificada: {len(mesh.vertices)} vert, {len(mesh.triangles)} tris")

    # Guardar
    mesh_path = out / "reconstruction_mesh_cleaned.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"  Malla: {mesh_path}")

    cloud_path = out / "reconstruction_cloud_cleaned.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd_clean)
    print(f"  Nube: {cloud_path}")

    # Resumen
    lines = [
        f"original_points: {len(pcd.points)}",
        f"cleaned_points: {len(pcd_clean.points)}",
        f"walls_found: {len(walls)}",
        f"mesh_vertices: {len(mesh.vertices)}",
        f"mesh_triangles: {len(mesh.triangles)}",
        f"mesh_path: {mesh_path}",
        f"cloud_path: {cloud_path}",
    ]
    (out / "cleaning_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
