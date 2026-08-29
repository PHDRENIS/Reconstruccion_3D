#!/usr/bin/env python3
"""
Reconstruccion avanzada de habitacion: paredes planas + objetos organicos.
- RANSAC para detectar piso, techo y paredes (planos planos)
- Proyeccion de paredes a planos perfectos con bordes recortados
- Poisson solo para objetos no-planares
- Merge final para malla tipo "habitacion real"
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
    p.add_argument("--voxel", type=float, default=0.008)
    p.add_argument("--plane-dist", type=float, default=0.06)
    p.add_argument("--ransac-iter", type=int, default=2500)
    p.add_argument("--poisson-depth", type=int, default=9)
    return p.parse_args()


def plane_to_mesh(plane, inlier_points, color=None):
    """Proyecta puntos a un plano y crea una malla plana con bordes recortados."""
    a, b, c, d = plane
    normal = np.array([a, b, c])
    points_2d = project_to_plane_2d(inlier_points, normal)

    if len(points_2d) < 10:
        return None

    try:
        hull = ConvexHull(points_2d)
    except Exception:
        return None

    verts_2d = points_2d[hull.vertices]
    verts_3d = unproject_from_plane_2d(verts_2d, normal, d)

    tri_mesh = o3d.geometry.TriangleMesh()
    tri_mesh.vertices = o3d.utility.Vector3dVector(verts_3d)

    # Triangulate convex hull polygon (fan triangulation from vertex 0)
    nv = len(verts_3d)
    tris_list = []
    for i in range(1, nv - 1):
        tris_list.append([0, i, i + 1])
    tri_mesh.triangles = o3d.utility.Vector3iVector(np.array(tris_list, dtype=np.int32))
    tri_mesh.compute_vertex_normals()

    if color is not None:
        nv = len(verts_3d)
        cols = np.tile(color, (nv, 1))
        tri_mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

    return tri_mesh


def project_to_plane_2d(points_3d, normal):
    """Proyecta puntos 3D a coordenadas 2D en el plano."""
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return np.column_stack([np.dot(points_3d, u), np.dot(points_3d, v)])


def unproject_from_plane_2d(points_2d, normal, dist):
    """Convierte puntos 2D del plano de vuelta a 3D."""
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    p0 = -dist * normal
    return np.outer(points_2d[:, 0], u) + np.outer(points_2d[:, 1], v) + p0


WALL_COLORS = [
    [0.7, 0.7, 0.7],  # grey
    [0.75, 0.75, 0.75],
    [0.8, 0.8, 0.8],
    [0.65, 0.65, 0.65],
]


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Cargando nube...")
    pcd = o3d.io.read_point_cloud(args.cloud)
    n_orig = len(pcd.points)
    print(f"  Original: {n_orig} puntos")
    pcd = pcd.voxel_down_sample(args.voxel)
    print(f"  Downsampled: {len(pcd.points)} puntos")

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3)) * 0.5
    all_idx = np.arange(len(pts))

    remaining = pcd
    floor_pts, ceil_pts = None, None
    wall_meshes = []
    objects_mask = np.ones(len(pts), dtype=bool)
    plane_count = 0

    for iteration in range(8):
        if len(remaining.points) < 200:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=args.plane_dist,
            ransac_n=3,
            num_iterations=args.ransac_iter,
        )
        a, b, c, d_val = plane_model
        nz = abs(c)
        inlier_indices = all_idx[:len(remaining.points)][inliers]

        # Obtener puntos del plano en coordenadas globales
        inlier_pts = pts[inlier_indices]
        centroid_z = float(inlier_pts[:, 2].mean())

        if nz > 0.85:
            # Horizontal: piso o techo
            if centroid_z < 1.0:
                floor_pts = inlier_pts
                print(f"  Piso: Z={centroid_z:.2f}, {len(inlier_pts)} pts")
            else:
                ceil_pts = inlier_pts
                print(f"  Techo: Z={centroid_z:.2f}, {len(inlier_pts)} pts")
            objects_mask[inlier_indices] = False
        elif nz < 0.3:
            # Vertical: pared
            color_wall = WALL_COLORS[len(wall_meshes) % len(WALL_COLORS)]
            mesh_wall = plane_to_mesh(plane_model, inlier_pts, color_wall)
            if mesh_wall is not None and len(mesh_wall.triangles) > 4:
                wall_meshes.append(mesh_wall)
                print(f"  Pared {len(wall_meshes)}: {len(inlier_pts)} pts, nz={nz:.3f}, {len(mesh_wall.triangles)} tris")
            objects_mask[inlier_indices] = False
        else:
            # Inclinado — dejar como objeto (escaleras, etc.)
            pass

        remaining = remaining.select_by_index(inliers, invert=True)
        plane_count += 1

    # Malla de objetos (Poisson)
    obj_pts = pts[objects_mask]
    obj_cols = cols[objects_mask]
    print(f"  Objetos: {len(obj_pts)} puntos")

    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(obj_pts)

    mesh_objects = o3d.geometry.TriangleMesh()
    if len(obj_pts) > 500:
        pcd_objects.estimate_normals()
        pcd_objects.orient_normals_consistent_tangent_plane(50)
        print(f"  Reconstruyendo objetos (Poisson depth={args.poisson_depth})...")
        mesh_objects, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_objects, depth=args.poisson_depth
        )
        if len(densities) > 0:
            thresh = np.quantile(densities, 0.02)
            verts_to_remove = densities < thresh
            mesh_objects.remove_vertices_by_mask(verts_to_remove)
        mesh_objects.compute_vertex_normals()
        # Simplificar objetos
        target_t = min(150000, len(mesh_objects.triangles) // 5)
        mesh_objects = mesh_objects.simplify_quadric_decimation(target_t)
    print(f"  Objetos mesh: {len(mesh_objects.vertices)} vert, {len(mesh_objects.triangles)} tris")

    # Merge: paredes + objetos
    print("Merge final...")
    mesh_final = o3d.geometry.TriangleMesh()
    if len(mesh_objects.triangles) > 0:
        mesh_final += mesh_objects
    for wm in wall_meshes:
        mesh_final += wm

    mesh_final.remove_duplicated_vertices()
    mesh_final.remove_degenerate_triangles()
    mesh_final.compute_vertex_normals()

    mesh_path = out / "room_final.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh_final)

    # Guardar nube etiquetada
    labels = np.zeros(len(pts), dtype=np.int32)
    labels[~objects_mask] = 1
    pcd_labeled = o3d.geometry.PointCloud()
    pcd_labeled.points = pcd.points
    wall_cols = np.array(WALL_COLORS)
    out_cols = np.zeros((len(pts), 3))
    out_cols[objects_mask] = obj_cols[:len(out_cols[objects_mask])] if len(obj_cols) > 0 else [0.5, 0.5, 0.5]
    out_cols[~objects_mask] = [0.3, 0.3, 0.8]
    pcd_labeled.colors = o3d.utility.Vector3dVector(out_cols)
    cloud_path = out / "room_labeled.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd_labeled)

    summary = [
        f"original_points: {n_orig}",
        f"downsampled: {len(pcd.points)}",
        f"planes_detected: {plane_count}",
        f"walls: {len(wall_meshes)}",
        f"objects_points: {len(obj_pts)}",
        f"mesh_vertices: {len(mesh_final.vertices)}",
        f"mesh_triangles: {len(mesh_final.triangles)}",
        f"mesh_path: {mesh_path}",
        f"cloud_path: {cloud_path}",
    ]
    (out / "room_summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
