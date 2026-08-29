#!/usr/bin/env python3
"""
Reconstruccion final con piso forzado:
  1) Recortar nube a dimensiones de habitacion
  2) Filtros estadisticos
  3) Piso manual en Z minima, paredes via RANSAC
  4) Todo como Poisson con bordes limpios
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
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (nv, 1)))
    mesh.compute_vertex_normals()
    return mesh


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Cargando nube ===")
    pcd = o3d.io.read_point_cloud(args.cloud)
    pts_all = np.asarray(pcd.points)
    print(f"  Original: {len(pts_all):,} pts")

    # === Crop intelligently ===
    z_low, z_high = np.percentile(pts_all[:, 2], [1, 99])
    x_low, x_high = np.percentile(pts_all[:, 0], [1, 99])
    y_low, y_high = np.percentile(pts_all[:, 1], [1, 99])
    print(f"  ROI: X[{x_low:.1f},{x_high:.1f}] Y[{y_low:.1f},{y_high:.1f}] Z[{z_low:.1f},{z_high:.1f}]")

    zm = (pts_all[:, 0] >= x_low) & (pts_all[:, 0] <= x_high) & \
         (pts_all[:, 1] >= y_low) & (pts_all[:, 1] <= y_high) & \
         (pts_all[:, 2] >= z_low) & (pts_all[:, 2] <= z_high)
    pts = pts_all[zm]
    cols_all = np.asarray(pcd.colors) if pcd.has_colors() else None
    cols = cols_all[zm] if cols_all is not None else np.ones((len(pts), 3)) * 0.55
    print(f"  Cropped: {len(pts):,} pts")

    # === Filters ===
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts)
    n_before = len(pcd2.points)
    pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=20, std_ratio=args.stat_std)
    print(f"  Statistical: {n_before:,} -> {len(pcd2.points):,}")
    n_before = len(pcd2.points)
    pcd2, _ = pcd2.remove_radius_outlier(nb_points=12, radius=0.05)
    print(f"  Radius: {n_before:,} -> {len(pcd2.points):,}")
    pcd2 = pcd2.voxel_down_sample(args.voxel)
    print(f"  Voxel({args.voxel}m): {len(pcd2.points):,} pts")

    pts = np.asarray(pcd2.points)
    n_total = len(pts)

    # === Detect planes ===
    print(f"\n=== RANSAC planos ===")
    remaining = pcd2
    remaining_idx = np.arange(n_total)
    obj_mask = np.ones(n_total, dtype=bool)
    wall_meshes = []
    n_planes = 0
    z_min_room = pts[:, 2].min()
    z_max_room = pts[:, 2].max()

    for iteration in range(10):
        if len(remaining.points) < 200:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=args.plane_dist, ransac_n=3, num_iterations=3000
        )
        a, b, c, d_val = plane_model
        nz = abs(c)
        gi = remaining_idx[inliers]
        ipts = pts[gi]

        if nz < 0.25:
            mw = plane_to_flat_mesh(plane_model, ipts, [0.70, 0.70, 0.70])
            if mw is not None and len(mw.triangles) > 4 and len(mw.vertices) > 5:
                wall_meshes.append(mw)
                print(f" Pared {len(wall_meshes)}: nz={nz:.3f}, {len(ipts):,} pts, {len(mw.triangles)}t")
            obj_mask[gi] = False
        # Horizontal planes become floor/ceiling -- remove them
        if nz > 0.8:
            obj_mask[gi] = False

        keep = np.ones(len(remaining.points), dtype=bool)
        keep[inliers] = False
        remaining_idx = remaining_idx[keep]
        remaining = remaining.select_by_index(inliers, invert=True)
        n_planes += 1

    # === Floor: manual from lowest Z ===
    floor_z = pts[:, 2].min() + 0.05
    floor_pts = pts[(pts[:, 2] >= floor_z - 0.06) & (pts[:, 2] <= floor_z + 0.06)]
    xr = [floor_pts[:, 0].min(), floor_pts[:, 0].max()]
    yr = [floor_pts[:, 1].min(), floor_pts[:, 1].max()]

    if len(floor_pts) > 100:
        floor_verts = np.array([
            [xr[0], yr[0], floor_z], [xr[1], yr[0], floor_z],
            [xr[1], yr[1], floor_z], [xr[0], yr[1], floor_z]
        ])
        floor_mesh = o3d.geometry.TriangleMesh()
        floor_mesh.vertices = o3d.utility.Vector3dVector(floor_verts)
        floor_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]], dtype=np.int32))
        floor_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.30, 0.25, 0.20], (4, 1)))
        floor_mesh.compute_vertex_normals()
        print(f"\n Piso manual: Z={floor_z:.2f}, {len(floor_pts):,} pts, area=({xr[1]-xr[0]:.1f}x{yr[1]-yr[0]:.1f})m")
    else:
        floor_mesh = None
        print("  Piso: sin puntos suficientes")

    # === Objects Poisson ===
    obj_idx = np.where(obj_mask)[0]
    print(f"\n=== Objetos Poisson ({len(obj_idx):,} pts) ===")
    pcd_o = o3d.geometry.PointCloud()
    pcd_o.points = o3d.utility.Vector3dVector(pts[obj_idx])
    mesh_obj = o3d.geometry.TriangleMesh()

    if len(obj_idx) > 500:
        pcd_o.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
        pcd_o.orient_normals_consistent_tangent_plane(60)
        mesh_obj, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_o, depth=args.poisson_depth)
        if len(dens) > 0:
            mesh_obj.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
        mesh_obj.compute_vertex_normals()
        tgt = min(100_000, max(5000, len(mesh_obj.triangles) // 3))
        mesh_obj = mesh_obj.simplify_quadric_decimation(tgt)
        print(f"  {len(mesh_obj.vertices):,} v, {len(mesh_obj.triangles):,} t")

    # === Merge ===
    print(f"\n=== Merge ===")
    mesh = o3d.geometry.TriangleMesh()
    if floor_mesh is not None:
        mesh += floor_mesh
    for w in wall_meshes:
        mesh += w
    if len(mesh_obj.triangles) > 0:
        mesh += mesh_obj
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    print(f"  {len(mesh.vertices):,} v, {len(mesh.triangles):,} t")

    print(f"\n=== Smooth ===")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=6)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    p = out / "room_final_v3.ply"
    o3d.io.write_triangle_mesh(str(p), mesh)

    s = f"room: X[{x_low:.1f},{x_high:.1f}] Y[{y_low:.1f},{y_high:.1f}] Z[{z_low:.1f},{z_high:.1f}]\nwalls: {len(wall_meshes)}\nfloor: {'yes' if floor_mesh else 'no'}\nobjects: {len(obj_idx):,} pts\nmesh: {len(mesh.vertices):,}v {len(mesh.triangles):,}t\npath: {p}"
    (out / "room_v3_summary.txt").write_text(s)
    print("\n" + s)


if __name__ == "__main__":
    main()
