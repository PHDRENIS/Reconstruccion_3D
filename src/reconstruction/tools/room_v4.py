#!/usr/bin/env python3
"""
Reconstruccion con crop por densidad Z - recorta al cluster principal.
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
    p.add_argument("--plane-dist", type=float, default=0.04)
    p.add_argument("--poisson-depth", type=int, default=10)
    return p.parse_args()


def plane_to_flat_mesh(plane_model, inlier_pts, color):
    a, b, c, d_val = plane_model
    normal = np.array([a, b, c])
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.9: u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal; u /= np.linalg.norm(u)
    v = np.cross(normal, u); v /= np.linalg.norm(v)
    pts2d = np.column_stack([np.dot(inlier_pts, u), np.dot(inlier_pts, v)])
    try: hull = ConvexHull(pts2d)
    except Exception: return None
    verts2d = pts2d[hull.vertices]
    p0 = -d_val * normal
    verts3d = np.outer(verts2d[:, 0], u) + np.outer(verts2d[:, 1], v) + p0
    nv = len(verts3d)
    tris = np.array([[0, i, i + 1] for i in range(1, nv - 1)], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts3d)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
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

    # === Density-based Z crop ===
    z_hist, z_edges = np.histogram(pts_all[:, 2], bins=200)
    peak_bin = np.argmax(z_hist)
    z_center = (z_edges[peak_bin] + z_edges[peak_bin + 1]) / 2
    z_low = z_center - 1.8
    z_high = z_center + 1.8
    print(f"  Z peak: {z_center:.1f}, crop: [{z_low:.1f}, {z_high:.1f}]")

    # XY: 2nd/98th percentile of the Z-cropped points
    zm = (pts_all[:, 2] >= z_low) & (pts_all[:, 2] <= z_high)
    pts_crop = pts_all[zm]
    x_low, x_high = np.percentile(pts_crop[:, 0], [2, 98])
    y_low, y_high = np.percentile(pts_crop[:, 1], [2, 98])
    print(f"  XY crop: [{x_low:.1f},{x_high:.1f}] [{y_low:.1f},{y_high:.1f}]")

    xym = (pts_crop[:, 0] >= x_low) & (pts_crop[:, 0] <= x_high) & \
          (pts_crop[:, 1] >= y_low) & (pts_crop[:, 1] <= y_high)
    pts = pts_crop[xym]
    print(f"  After crop: {len(pts):,} pts ({len(pts)/len(pts_all)*100:.1f}%)")

    # === Filters ===
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts)
    # Stat filter
    nb = len(pcd2.points)
    pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    print(f"  Stat: {nb:,} -> {len(pcd2.points):,}")
    # Radius filter
    nb = len(pcd2.points)
    pcd2, _ = pcd2.remove_radius_outlier(nb_points=12, radius=0.05)
    print(f"  Radius: {nb:,} -> {len(pcd2.points):,}")

    pcd2 = pcd2.voxel_down_sample(args.voxel)
    pts = np.asarray(pcd2.points)
    print(f"  Voxel({args.voxel}m): {len(pts):,} pts")

    # === RANSAC ===
    print(f"\n=== RANSAC ===")
    remaining = pcd2
    rem_idx = np.arange(len(pts))
    obj_mask = np.ones(len(pts), dtype=bool)
    walls = []

    for _ in range(12):
        if len(remaining.points) < 200: break
        pm, il = remaining.segment_plane(distance_threshold=args.plane_dist, ransac_n=3, num_iterations=3000)
        a, b, c, _ = pm; nz = abs(c)
        gi = rem_idx[il]; ipts = pts[gi]
        if nz < 0.25:
            mw = plane_to_flat_mesh(pm, ipts, [0.70, 0.70, 0.70])
            if mw and len(mw.triangles) > 4 and len(mw.vertices) > 5:
                walls.append(mw)
                print(f" Pared {len(walls)}: nz={nz:.3f}, {len(ipts):,} pts")
        if nz > 0.25:
            obj_mask[gi] = False
        kl = np.ones(len(remaining.points), dtype=bool); kl[il] = False
        rem_idx = rem_idx[kl]; remaining = remaining.select_by_index(il, invert=True)

    # === Floor from lowest Z ===
    floor_z = pts[:, 2].min() + 0.08
    fpts = pts[(pts[:, 2] >= floor_z - 0.08) & (pts[:, 2] <= floor_z + 0.08)]
    if len(fpts) > 500:
        # Find floor area from the cropped bounds
        xr = [pts[:, 0].min() + 0.1, pts[:, 0].max() - 0.1]
        yr = [pts[:, 1].min() + 0.1, pts[:, 1].max() - 0.1]
        fv = np.array([[xr[0], yr[0], floor_z], [xr[1], yr[0], floor_z], [xr[1], yr[1], floor_z], [xr[0], yr[1], floor_z]])
        fm = o3d.geometry.TriangleMesh()
        fm.vertices = o3d.utility.Vector3dVector(fv)
        fm.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]], dtype=np.int32))
        fm.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.30, 0.25, 0.20], (4, 1)))
        fm.compute_vertex_normals()
        print(f"\n Piso: Z={floor_z:.2f}, {len(fpts):,} pts")
    else:
        fm = None

    # === Objects ===
    oi = np.where(obj_mask)[0]
    print(f"\n=== Objetos ({len(oi):,} pts) ===")
    po = o3d.geometry.PointCloud(); po.points = o3d.utility.Vector3dVector(pts[oi])
    mo = o3d.geometry.TriangleMesh()
    if len(oi) > 500:
        po.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
        po.orient_normals_consistent_tangent_plane(60)
        mo, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(po, depth=args.poisson_depth)
        if len(dens) > 0: mo.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
        mo.compute_vertex_normals()
        mo = mo.simplify_quadric_decimation(min(100_000, max(5000, len(mo.triangles) // 3)))
        print(f"  {len(mo.vertices):,}v {len(mo.triangles):,}t")

    # === Merge ===
    print(f"\n=== Merge ===")
    mesh = o3d.geometry.TriangleMesh()
    if fm: mesh += fm
    for w in walls: mesh += w
    if len(mo.triangles) > 0: mesh += mo
    mesh.remove_duplicated_vertices(); mesh.remove_degenerate_triangles(); mesh.remove_non_manifold_edges()
    print(f"  {len(mesh.vertices):,}v {len(mesh.triangles):,}t")

    mesh = mesh.filter_smooth_taubin(number_of_iterations=6)
    mesh.compute_vertex_normals(); mesh.remove_degenerate_triangles()

    p = out / "room_final_v4.ply"
    o3d.io.write_triangle_mesh(str(p), mesh)
    s = f"room: {x_high-x_low:.1f}x{y_high-y_low:.1f}x{z_high-z_low:.1f}m\nwalls: {len(walls)}\nfloor: {'yes' if fm else 'no'}\nmesh: {len(mesh.vertices):,}v {len(mesh.triangles):,}t\npath: {p}"
    (out / "room_v4_summary.txt").write_text(s)
    print("\n" + s)


if __name__ == "__main__":
    main()
