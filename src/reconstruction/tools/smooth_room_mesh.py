#!/usr/bin/env python3
"""
Pipeline de limpieza de malla de habitacion:
  1) Filtros estadisticos (eliminar picos)
  2) Voxel downsampling (uniformizar)
  3) RANSAC -> separar estructura/objetos
  4) Paredes/piso planos + objetos Poisson suavizado
  5) Merge + Taubin smooth + simplificar
"""

import argparse, sys
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--voxel", type=float, default=0.012)
    p.add_argument("--stat-nn", type=int, default=20)
    p.add_argument("--stat-std", type=float, default=2.0)
    p.add_argument("--radius-nn", type=int, default=12)
    p.add_argument("--radius-r", type=float, default=0.06)
    p.add_argument("--plane-dist", type=float, default=0.06)
    p.add_argument("--ransac-iter", type=int, default=3000)
    p.add_argument("--poisson-depth", type=int, default=9)
    return p.parse_args()


def plane_to_flat_mesh(plane_model, inlier_pts, color):
    a, b, c, d_val = plane_model
    normal = np.array([a, b, c])
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    pts2d = np.column_stack([np.dot(inlier_pts, u), np.dot(inlier_pts, v)])
    try:
        hull = ConvexHull(pts2d)
    except Exception:
        return None
    verts2d = pts2d[hull.vertices]
    p0 = -d_val * normal
    verts3d = np.outer(verts2d[:, 0], u) + np.outer(verts2d[:, 1], v) + p0
    nv = len(verts3d)
    tris = [[0, i, i + 1] for i in range(1, nv - 1)]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts3d)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    cols = np.tile(color, (nv, 1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    mesh.compute_vertex_normals()
    return mesh


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === PASO 1: Cargar ===
    print("=== PASO 1: Cargar y convertir a nube ===")
    mesh_in = o3d.io.read_triangle_mesh(args.mesh)
    print(f"  Verts: {len(mesh_in.vertices):,}  Tris: {len(mesh_in.triangles):,}")
    pcd = mesh_in.sample_points_uniformly(number_of_points=min(5_000_000, len(mesh_in.vertices)))
    print(f"  Sampled: {len(pcd.points):,} points")

    # === PASO 2: Filtros ===
    print(f"\n=== PASO 2: Filtros estadisticos (eliminar picos) ===")
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=args.stat_nn, std_ratio=args.stat_std)
    print(f"  Statistical: {n_before:,} -> {len(pcd.points):,}")
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_radius_outlier(nb_points=args.radius_nn, radius=args.radius_r)
    print(f"  Radius: {n_before:,} -> {len(pcd.points):,}")

    # === PASO 3: Voxel ===
    print(f"\n=== PASO 3: Voxel downsampling ===")
    pcd = pcd.voxel_down_sample(args.voxel)
    print(f"  {len(pcd.points):,} pts @ {args.voxel}m")

    pts = np.asarray(pcd.points)

    # === PASO 4: RANSAC ===
    print(f"\n=== PASO 4: RANSAC - deteccion de planos ===")
    remaining = pcd
    remaining_indices = np.arange(len(pts))
    objects_mask = np.ones(len(pts), dtype=bool)
    wall_meshes = []
    floor_mesh = None
    found_floor = False
    n_planes = 0

    WALL_COLORS = [[0.72, 0.70, 0.68], [0.75, 0.73, 0.71], [0.70, 0.68, 0.66], [0.73, 0.71, 0.69]]

    for iteration in range(10):
        if len(remaining.points) < 200:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=args.plane_dist, ransac_n=3, num_iterations=args.ransac_iter,
        )
        a, b, c, d_val = plane_model
        nz = abs(c)
        global_inliers = remaining_indices[inliers]
        inlier_pts = pts[global_inliers]
        z_mean = float(inlier_pts[:, 2].mean())

        if nz > 0.85:
            if not found_floor and z_mean < 1.2:
                found_floor = True
                mf = plane_to_flat_mesh(plane_model, inlier_pts, [0.35, 0.32, 0.28])
                if mf is not None:
                    floor_mesh = mf
                    print(f" Piso: Z={z_mean:.2f}, {len(inlier_pts):,} pts, {len(mf.triangles)} tris")
            else:
                print(f" Techo/otro HZ descartado: Z={z_mean:.2f}")
            objects_mask[global_inliers] = False
        elif nz < 0.3:
            ci = len(wall_meshes) % len(WALL_COLORS)
            mw = plane_to_flat_mesh(plane_model, inlier_pts, WALL_COLORS[ci])
            if mw is not None and len(mw.triangles) > 4:
                wall_meshes.append(mw)
                print(f" Pared {len(wall_meshes)}: nz={nz:.3f}, {len(inlier_pts):,} pts, {len(mw.triangles)} tris")
            objects_mask[global_inliers] = False

        keep_local = np.ones(len(remaining.points), dtype=bool)
        keep_local[inliers] = False
        remaining_indices = remaining_indices[keep_local]
        remaining = remaining.select_by_index(inliers, invert=True)
        n_planes += 1

    print(f"  Planos detectados: {n_planes}")

    # === PASO 5: Objetos ===
    obj_idx = np.where(objects_mask)[0]
    print(f"\n=== PASO 5: Objetos ({len(obj_idx):,} pts) ===")
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(pts[obj_idx])
    mesh_obj = o3d.geometry.TriangleMesh()

    if len(obj_idx) > 500:
        pcd_obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_obj.orient_normals_consistent_tangent_plane(50)
        print(f"  Poisson depth={args.poisson_depth}...")
        mesh_obj, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_obj, depth=args.poisson_depth
        )
        if len(densities) > 0:
            mesh_obj.remove_vertices_by_mask(densities < np.quantile(densities, 0.03))
        mesh_obj.compute_vertex_normals()
        target = min(120_000, max(5000, len(mesh_obj.triangles) // 4))
        mesh_obj = mesh_obj.simplify_quadric_decimation(target)
        print(f"  Objetos mesh: {len(mesh_obj.vertices):,} v, {len(mesh_obj.triangles):,} t")

    # === PASO 6: Merge ===
    print(f"\n=== PASO 6: Merge final ===")
    mesh_final = o3d.geometry.TriangleMesh()
    if floor_mesh is not None:
        mesh_final += floor_mesh
    for w in wall_meshes:
        mesh_final += w
    if len(mesh_obj.triangles) > 0:
        mesh_final += mesh_obj

    mesh_final.remove_duplicated_vertices()
    mesh_final.remove_degenerate_triangles()
    mesh_final.remove_non_manifold_edges()
    print(f"  Pre-smooth: {len(mesh_final.vertices):,} v, {len(mesh_final.triangles):,} t")

    # === PASO 7: Smooth ===
    print(f"\n=== PASO 7: Taubin smooth + simplificar ===")
    mesh_final = mesh_final.filter_smooth_taubin(number_of_iterations=8)
    mesh_final.compute_vertex_normals()
    target_tris = min(180_000, max(20_000, len(mesh_final.triangles) // 3))
    mesh_final = mesh_final.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    mesh_final.remove_degenerate_triangles()
    mesh_final.compute_vertex_normals()

    mesh_path = out / "room_smooth.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh_final)

    summary = [
        f"walls: {len(wall_meshes)}",
        f"floor: {'yes' if floor_mesh else 'no'}",
        f"objects_points: {len(obj_idx):,}",
        f"mesh_vertices: {len(mesh_final.vertices):,}",
        f"mesh_triangles: {len(mesh_final.triangles):,}",
        f"mesh_path: {mesh_path}",
    ]
    (out / "room_smooth_summary.txt").write_text("\n".join(summary))
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
