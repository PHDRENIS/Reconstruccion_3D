#!/usr/bin/env python3
"""
Reconstruccion de habitacion limpia final:
  1) Recortar nube a dimensiones reales de habitacion
  2) Filtros estadisticos agresivos
  3) RANSAC para paredes/piso/techo
  4) Paredes planas + piso plano + objetos Poisson
  5) Merge + Taubin smooth
"""

import argparse, sys
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cloud", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=3.5)
    p.add_argument("--xy-percentile", type=float, default=2.0)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--stat-std", type=float, default=1.5)
    p.add_argument("--plane-dist", type=float, default=0.04)
    p.add_argument("--poisson-depth", type=int, default=10)
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

    print("=== PASO 1: Cargar y recortar nube ===")
    pcd = o3d.io.read_point_cloud(args.cloud)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"  Original: {len(pts):,} puntos")

    # Crop Z
    z_mask = (pts[:, 2] >= args.z_min) & (pts[:, 2] <= args.z_max)
    # Crop XY
    x_low, x_high = np.percentile(pts[:, 0], [args.xy_percentile, 100 - args.xy_percentile])
    y_low, y_high = np.percentile(pts[:, 1], [args.xy_percentile, 100 - args.xy_percentile])
    xy_mask = (pts[:, 0] >= x_low) & (pts[:, 0] <= x_high) & (pts[:, 1] >= y_low) & (pts[:, 1] <= y_high)
    crop_mask = z_mask & xy_mask
    print(f"  Recorte Z [{args.z_min}, {args.z_max}] XY [{x_low:.1f},{x_high:.1f}] [{y_low:.1f},{y_high:.1f}]")
    print(f"  Puntos recortados: {crop_mask.sum():,} ({crop_mask.sum()/len(pts)*100:.1f}%)")

    pcd.points = o3d.utility.Vector3dVector(pts[crop_mask])
    if cols is not None:
        pcd.colors = o3d.utility.Vector3dVector(cols[crop_mask])

    # === PASO 2: Filtros agresivos ===
    print(f"\n=== PASO 2: Filtros estadisticos ===")
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=args.stat_std)
    print(f"  Statistical: {n_before:,} -> {len(pcd.points):,}")
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_radius_outlier(nb_points=12, radius=0.05)
    print(f"  Radius: {n_before:,} -> {len(pcd.points):,}")

    # === PASO 3: Voxel ===
    print(f"\n=== PASO 3: Voxel downsampling ===")
    pcd = pcd.voxel_down_sample(args.voxel)
    print(f"  {len(pcd.points):,} pts")

    pts = np.asarray(pcd.points)

    # === PASO 4: RANSAC ===
    print(f"\n=== PASO 4: RANSAC - deteccion de planos ===")
    remaining = pcd
    remaining_indices = np.arange(len(pts))
    objects_mask = np.ones(len(pts), dtype=bool)
    wall_meshes = []
    floor_mesh = None
    ceil_mesh = None
    n_planes = 0

    WALL_GRAY = [0.72, 0.72, 0.72]
    FLOOR_CLR = [0.30, 0.25, 0.20]

    for iteration in range(12):
        if len(remaining.points) < 200:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=args.plane_dist, ransac_n=3, num_iterations=3000,
        )
        a, b, c, d_val = plane_model
        nz = abs(c)
        global_inliers = remaining_indices[inliers]
        inlier_pts = pts[global_inliers]
        z_mean = float(inlier_pts[:, 2].mean())

        if nz > 0.8:
            if z_mean < 0.5:
                mf = plane_to_flat_mesh(plane_model, inlier_pts, FLOOR_CLR)
                if mf is not None and floor_mesh is None:
                    floor_mesh = mf
                    print(f" Piso: Z={z_mean:.2f}, {len(inlier_pts):,} pts")
            elif z_mean > 3.0:
                if ceil_mesh is None:
                    print(f" Techo: Z={z_mean:.2f}, descartado")
            objects_mask[global_inliers] = False
        elif nz < 0.25:
            mw = plane_to_flat_mesh(plane_model, inlier_pts, WALL_GRAY)
            if mw is not None and len(mw.triangles) > 4:
                wall_meshes.append(mw)
                print(f" Pared {len(wall_meshes)}: nz={nz:.3f}, {len(inlier_pts):,} pts")
            objects_mask[global_inliers] = False

        keep_local = np.ones(len(remaining.points), dtype=bool)
        keep_local[inliers] = False
        remaining_indices = remaining_indices[keep_local]
        remaining = remaining.select_by_index(inliers, invert=True)
        n_planes += 1

    print(f"  Planos: {n_planes}")

    # === PASO 5: Objetos ===
    obj_idx = np.where(objects_mask)[0]
    print(f"\n=== PASO 5: Objetos Poisson ({len(obj_idx):,} pts) ===")
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(pts[obj_idx])
    mesh_obj = o3d.geometry.TriangleMesh()

    if len(obj_idx) > 500:
        pcd_obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
        pcd_obj.orient_normals_consistent_tangent_plane(60)
        mesh_obj, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_obj, depth=args.poisson_depth
        )
        if len(dens) > 0:
            mesh_obj.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
        mesh_obj.compute_vertex_normals()
        tgt = min(100_000, max(5000, len(mesh_obj.triangles) // 3))
        mesh_obj = mesh_obj.simplify_quadric_decimation(tgt)
        print(f"  Objetos: {len(mesh_obj.vertices):,} v, {len(mesh_obj.triangles):,} t")

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
    print(f"  {len(mesh_final.vertices):,} v, {len(mesh_final.triangles):,} t")

    # === PASO 7: Smooth ===
    print(f"\n=== PASO 7: Taubin smooth + simplificar ===")
    mesh_final = mesh_final.filter_smooth_taubin(number_of_iterations=6)
    mesh_final.compute_vertex_normals()
    mesh_final.remove_degenerate_triangles()

    final_path = out / "room_clean_final.ply"
    o3d.io.write_triangle_mesh(str(final_path), mesh_final)

    s = [
        f"cropped_Z: [{args.z_min}, {args.z_max}]",
        f"cropped_XY: [{x_low:.1f}, {x_high:.1f}] x [{y_low:.1f}, {y_high:.1f}]",
        f"walls: {len(wall_meshes)}",
        f"floor: {'yes' if floor_mesh else 'no'}",
        f"objects_points: {len(obj_idx):,}",
        f"mesh_verts: {len(mesh_final.vertices):,}",
        f"mesh_tris: {len(mesh_final.triangles):,}",
        f"path: {final_path}",
    ]
    (out / "room_clean_summary.txt").write_text("\n".join(s))
    print("\n" + "\n".join(s))


if __name__ == "__main__":
    main()
