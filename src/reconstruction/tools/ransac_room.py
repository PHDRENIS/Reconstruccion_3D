#!/usr/bin/env python3
"""
Post-procesamiento RANSAC: piso + paredes planas.
Dos pasadas: deteccion en nube ligera, proyeccion desde nube densa.
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
    return p.parse_args()


def plane_to_flat_mesh(plane_model, points, color):
    """Proyecta puntos a un plano y crea malla plana con bordes recortados."""
    a, b, c, d_val = plane_model
    normal = np.array([a, b, c])

    # Base vectors for 2D projection
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    pts2d = np.column_stack([np.dot(points, u), np.dot(points, v)])
    try:
        hull = ConvexHull(pts2d)
    except Exception:
        # Fallback: bounding rect
        xmin, ymin = pts2d.min(axis=0)
        xmax, ymax = pts2d.max(axis=0)
        pts2d_rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        verts2d = pts2d_rect
    else:
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

    # === LOAD & SAMPLE ===
    print("Cargando malla...")
    mesh_in = o3d.io.read_triangle_mesh(args.mesh)
    print(f"  Original: {len(mesh_in.vertices):,} v, {len(mesh_in.triangles):,} t")
    pcd = mesh_in.sample_points_uniformly(number_of_points=2_500_000)
    print(f"  Sampled: {len(pcd.points):,} pts")

    # === FILTERS ===
    print("\nFiltrado...")
    for lbl, nn, sr in [("Stat1", 20, 2.0), ("Stat2", 20, 1.5), ("Radius", 12, 0.08)]:
        nb = len(pcd.points)
        if "Stat" in lbl:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nn, std_ratio=sr)
        else:
            pcd, _ = pcd.remove_radius_outlier(nb_points=nn, radius=sr)
        print(f"  {lbl}: {nb:,} -> {len(pcd.points):,}")

    # Dense cloud (after filtering) for projection
    pcd_dense = pcd
    pts_dense = np.asarray(pcd_dense.points)
    print(f"  Puntos densos: {len(pts_dense):,}")

    # Light cloud for RANSAC
    pcd_light = pcd.voxel_down_sample(0.02)
    pts_light = np.asarray(pcd_light.points)
    print(f"  Puntos RANSAC: {len(pts_light):,}")

    # === RANSAC PLANE DETECTION ===
    print("\nRANSAC (max 6 planos)...")
    remaining = pcd_light
    horizontal_planes = []
    vertical_planes = []
    used_indices = set()

    for _ in range(6):
        if len(remaining.points) < 200:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.06, ransac_n=3, num_iterations=3000
        )
        a, b, c, d_val = plane_model
        nz = abs(c)
        inlier_pts = np.asarray(remaining.points)[inliers]
        if len(inlier_pts) < 500:
            continue

        z_mean = float(inlier_pts[:, 2].mean())
        area_2d = (inlier_pts[:, 0].max() - inlier_pts[:, 0].min()) * (inlier_pts[:, 1].max() - inlier_pts[:, 1].min())

        if nz > 0.85:
            horizontal_planes.append((plane_model, inlier_pts, z_mean, area_2d))
            print(f"  HZ plane: Z={z_mean:.2f}, area={area_2d:.1f}m², {len(inlier_pts):,} pts")
        elif nz < 0.25:
            vertical_planes.append((plane_model, inlier_pts, area_2d))
            print(f"  VT plane: nz={nz:.3f}, area={area_2d:.1f}m², {len(inlier_pts):,} pts")

        remaining = remaining.select_by_index(inliers, invert=True)

    # === CLASSIFY HORIZONTALS ===
    z_global_median = float(np.median(pts_dense[:, 2]))
    print(f"\nMediana Z global: {z_global_median:.2f}")

    floor_plane, floor_pts = None, None
    for pm, ipts, zmean, area in sorted(horizontal_planes, key=lambda x: x[2]):
        if zmean < z_global_median:
            floor_plane, floor_pts = pm, ipts
            print(f"  PISO: Z={zmean:.2f}, {len(ipts):,} pts")
            break

    # === CLASSIFY VERTICALS (keep best 4) ===
    vertical_planes.sort(key=lambda x: x[2], reverse=True)
    vertical_planes = vertical_planes[:4]

    # === BUILD FLAT MESHES FROM DENSE POINTS ===
    WALL_COLORS = [
        [0.68, 0.67, 0.66],
        [0.71, 0.70, 0.69],
        [0.65, 0.64, 0.63],
        [0.73, 0.72, 0.71],
    ]

    meshes = []
    total_v, total_t = 0, 0

    # Floor
    if floor_plane is not None:
        a, b, c, d_val = floor_plane
        dists = np.abs(a * pts_dense[:, 0] + b * pts_dense[:, 1] + c * pts_dense[:, 2] + d_val)
        floor_mask = dists < 0.15
        if floor_mask.sum() > 500:
            fm = plane_to_flat_mesh(floor_plane, pts_dense[floor_mask], [0.28, 0.22, 0.18])
            meshes.append(("Piso", fm))
            total_v += len(fm.vertices)
            total_t += len(fm.triangles)
            print(f"\n  Piso mesh: {len(fm.vertices)} v, {len(fm.triangles)} t")

    # Walls
    for wi, (pm, ipts, area) in enumerate(vertical_planes):
        a, b, c, d_val = pm
        dists = np.abs(a * pts_dense[:, 0] + b * pts_dense[:, 1] + c * pts_dense[:, 2] + d_val)
        wall_mask = dists < 0.15
        if wall_mask.sum() < 500:
            continue
        wm = plane_to_flat_mesh(pm, pts_dense[wall_mask], WALL_COLORS[wi % len(WALL_COLORS)])
        meshes.append((f"Pared {wi+1}", wm))
        total_v += len(wm.vertices)
        total_t += len(wm.triangles)
        print(f"  Pared {wi+1} mesh: {len(wm.vertices)} v, {len(wm.triangles)} t")

    # === MERGE ===
    print(f"\nMerge ({len(meshes)} componentes)...")
    mesh_final = o3d.geometry.TriangleMesh()
    for label, m in meshes:
        mesh_final += m

    mesh_final.remove_duplicated_vertices()
    mesh_final.remove_degenerate_triangles()
    mesh_final.remove_non_manifold_edges()

    print(f"  Pre-smooth: {len(mesh_final.vertices):,} v, {len(mesh_final.triangles):,} t")

    # === SMOOTH ===
    print("Suavizando...")
    mesh_final = mesh_final.filter_smooth_taubin(number_of_iterations=6)
    mesh_final.compute_vertex_normals()
    mesh_final.remove_degenerate_triangles()

    # === EXPORT ===
    out_path = out / "room_paredes_piso.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh_final)

    summary_lines = [
        f"planes_detected: {len(horizontal_planes) + len(vertical_planes)}",
        f"walls_used: {len([m for l,m in meshes if 'Pared' in l])}",
        f"floor_used: {'yes' if floor_plane else 'no'}",
        f"mesh_vertices: {len(mesh_final.vertices):,}",
        f"mesh_triangles: {len(mesh_final.triangles):,}",
        f"path: {out_path}",
    ]
    (out / "room_paredes_piso_summary.txt").write_text("\n".join(summary_lines))
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
