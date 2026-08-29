#!/usr/bin/env python3
"""
RANSAC room: piso + paredes planas (mallas rectangulares limpias).
"""

import argparse, sys
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def rectangular_mesh_from_points(pts_3d, normal, color):
    """Crea un rectangulo alineado con el plano en las dimensiones de los puntos."""
    # Project to 2D
    u_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u_vec, normal)) > 0.9:
        u_vec = np.array([0.0, 1.0, 0.0])
    u_vec = u_vec - np.dot(u_vec, normal) * normal
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    v_vec /= np.linalg.norm(v_vec)

    coords_u = np.dot(pts_3d, u_vec)
    coords_v = np.dot(pts_3d, v_vec)

    umin, umax = coords_u.min(), coords_u.max()
    vmin, vmax = coords_v.min(), coords_v.max()

    # Slightly expand
    margin = 0.05
    umin -= margin; umax += margin
    vmin -= margin; vmax += margin

    # Project centroid to plane
    centroid = pts_3d.mean(axis=0)
    # Calculate plane offset
    d_val = -np.dot(normal, centroid)

    p0 = -d_val * normal
    rect_verts = np.array([
        p0 + umin * u_vec + vmin * v_vec,
        p0 + umax * u_vec + vmin * v_vec,
        p0 + umax * u_vec + vmax * v_vec,
        p0 + umin * u_vec + vmax * v_vec,
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(rect_verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]], dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (4, 1)))
    mesh.compute_vertex_normals()
    return mesh


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Cargando malla...")
    mesh_in = o3d.io.read_triangle_mesh(args.mesh)
    print(f"  Original: {len(mesh_in.vertices):,} v, {len(mesh_in.triangles):,} t")
    pcd = mesh_in.sample_points_uniformly(number_of_points=2_500_000)
    print(f"  Sampled: {len(pcd.points):,} pts")

    # Filters
    print("\nFiltrado...")
    for lbl, nn, sr in [("Stat1", 20, 2.0), ("Stat2", 20, 1.5), ("Radius", 12, 0.08)]:
        nb = len(pcd.points)
        if "Stat" in lbl:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nn, std_ratio=sr)
        else:
            pcd, _ = pcd.remove_radius_outlier(nb_points=nn, radius=sr)
        print(f"  {lbl}: {nb:,} -> {len(pcd.points):,}")

    pts_dense = np.asarray(pcd.points)
    print(f"  Puntos densos: {len(pts_dense):,}")

    # Light cloud for RANSAC
    pcd_light = pcd.voxel_down_sample(0.02)
    pts_light = np.asarray(pcd_light.points)
    print(f"  Puntos RANSAC: {len(pts_light):,}")

    # RANSAC
    print("\nRANSAC (max 6 planos)...")
    remaining = pcd_light
    horizontal_planes = []
    vertical_planes = []

    for iteration in range(10):
        if len(remaining.points) < 200:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.06, ransac_n=3, num_iterations=5000
        )
        a, b, c, d_val = plane_model
        nz = abs(c)
        inlier_pts = np.asarray(remaining.points)[inliers]
        if len(inlier_pts) < 500:
            continue

        z_mean = float(inlier_pts[:, 2].mean())
        area = (inlier_pts[:, 0].max() - inlier_pts[:, 0].min()) * \
               (inlier_pts[:, 1].max() - inlier_pts[:, 1].min())

        if nz > 0.85:
            horizontal_planes.append((plane_model, z_mean, area, len(inlier_pts)))
            print(f"  HZ: Z={z_mean:.2f}, area={area:.1f}m², {len(inlier_pts):,} pts")
        elif nz < 0.25:
            vertical_planes.append((plane_model, area, len(inlier_pts)))
            print(f"  VT: nz={nz:.3f}, area={area:.1f}m², {len(inlier_pts):,} pts")

        remaining = remaining.select_by_index(inliers, invert=True)

    # Classify
    z_global_median = float(np.median(pts_dense[:, 2]))
    print(f"\nMediana Z global: {z_global_median:.2f}")

    floor_plane_model = None
    for pm, zmean, area, n_pts in sorted(horizontal_planes, key=lambda x: x[1]):
        if zmean < z_global_median:
            floor_plane_model = pm
            print(f"  PISO: Z={zmean:.2f}, {n_pts} pts, area={area:.1f}m²")
            break

    # Keep best 4 walls
    vertical_planes.sort(key=lambda x: x[1], reverse=True)
    vertical_planes = vertical_planes[:4]

    # Build meshes from DENSE points
    WALL_COLORS = [[0.68, 0.67, 0.66], [0.71, 0.70, 0.69], [0.65, 0.64, 0.63], [0.73, 0.72, 0.71]]
    meshes = []

    # Floor
    if floor_plane_model is not None:
        a, b, c, d_val = floor_plane_model
        normal = np.array([a, b, c])
        dists = np.abs(a * pts_dense[:, 0] + b * pts_dense[:, 1] + c * pts_dense[:, 2] + d_val)
        floor_mask = dists < 0.40
        floor_pts = pts_dense[floor_mask]
        if len(floor_pts) > 500:
            fm = rectangular_mesh_from_points(floor_pts, normal, [0.28, 0.22, 0.18])
            meshes.append(("Piso", fm))
            print(f"\n  Piso mesh: {len(fm.triangles)} tris, area={(floor_pts[:,0].max()-floor_pts[:,0].min()):.1f}x{(floor_pts[:,1].max()-floor_pts[:,1].min()):.1f}m")

    # Walls
    for wi, (pm, area, n_pts) in enumerate(vertical_planes):
        a, b, c, d_val = pm
        normal = np.array([a, b, c])
        dists = np.abs(a * pts_dense[:, 0] + b * pts_dense[:, 1] + c * pts_dense[:, 2] + d_val)
        wall_mask = dists < 0.40
        wall_pts = pts_dense[wall_mask]
        if len(wall_pts) < 500:
            continue
        wm = rectangular_mesh_from_points(wall_pts, normal, WALL_COLORS[wi % len(WALL_COLORS)])
        meshes.append((f"Pared {wi+1}", wm))
        print(f"  Pared {wi+1} mesh: {len(wm.triangles)} tris")

    # Merge
    print(f"\nMerge ({len(meshes)} componentes)...")
    mesh_final = o3d.geometry.TriangleMesh()
    for label, m in meshes:
        mesh_final += m

    mesh_final.remove_duplicated_vertices()
    mesh_final.remove_degenerate_triangles()
    mesh_final.remove_non_manifold_edges()
    mesh_final.compute_vertex_normals()

    print(f"  Final: {len(mesh_final.vertices)} v, {len(mesh_final.triangles)} t")

    out_path = out / "room_paredes_piso.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh_final)

    s = [
        f"walls: {len([m for l,m in meshes if 'Pared' in l])}",
        f"floor: {'yes' if floor_plane_model is not None else 'no'}",
        f"mesh_vertices: {len(mesh_final.vertices)}",
        f"mesh_triangles: {len(mesh_final.triangles)}",
        f"path: {out_path}",
    ]
    (out / "room_paredes_piso_summary.txt").write_text("\n".join(s))
    print("\n" + "\n".join(s))


if __name__ == "__main__":
    main()
