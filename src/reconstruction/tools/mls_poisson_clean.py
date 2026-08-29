#!/usr/bin/env python3
"""
MLS Smoothing + Screened Poisson para suavizar la superficie
sin alterar la geometria de la reconstruccion original.
"""

import argparse, sys
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cloud", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--voxel", type=float, default=0.012)
    p.add_argument("--mls-radius", type=float, default=0.08)
    p.add_argument("--poisson-depth", type=int, default=10)
    p.add_argument("--target-tris", type=int, default=200000)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === LOAD ===
    print("Cargando nube original...")
    pcd = o3d.io.read_point_cloud(args.cloud)
    n_orig = len(pcd.points)
    print(f"  Original: {n_orig:,} puntos")

    # === STATISTICAL OUTLIER x2 ===
    print("\nFiltros de outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)
    print(f"  Stat1 (std=1.8): {n_orig:,} -> {len(pcd.points):,}")

    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    print(f"  Stat2 (std=1.2): {n_before:,} -> {len(pcd.points):,}")

    # === VOXEL ===
    print(f"\nVoxel downsampling ({args.voxel}m)...")
    pcd = pcd.voxel_down_sample(args.voxel)
    print(f"  {len(pcd.points):,} puntos")

    # === NORMALS + SMOOTHING (MLS not available in 0.19) ===
    print("\nEstimando normales (radio grande = suavizado)...")
    # Large radius for smoother normals = smoother surface
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=args.mls_radius * 2, max_nn=40)
    pcd.estimate_normals(search_param)
    pcd.orient_normals_consistent_tangent_plane(80)

    # === Bilateral-like: weight points by distance to local surface ===
    # We can't use MLS, so we do: Poisson with good normals + Taubin post
    pcd_smooth = pcd

    # Downsample for Poisson if too many points
    n_pts = len(pcd_smooth.points)
    target = 800_000
    if n_pts > target:
        pcd_poisson = pcd_smooth.random_down_sample(target / n_pts)
        print(f"  Downsampled para Poisson: {len(pcd_poisson.points):,} pts")
    else:
        pcd_poisson = pcd_smooth

    # === SCREENED POISSON ===
    print(f"\nScreened Poisson (depth={args.poisson_depth})...")
    mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_poisson, depth=args.poisson_depth
    )
    print(f"  Raw mesh: {len(mesh_raw.vertices):,} v, {len(mesh_raw.triangles):,} t")

    # Filter by density
    if len(densities) > 0:
        q = np.quantile(densities, 0.02)
        mesh_raw.remove_vertices_by_mask(densities < q)
        print(f"  After density filter: {len(mesh_raw.vertices):,} v, {len(mesh_raw.triangles):,} t")

    mesh_raw.compute_vertex_normals()
    mesh_raw.remove_degenerate_triangles()

    # === TAUBIN SMOOTH ===
    print("\nTaubin smooth (8 iter)...")
    mesh = mesh_raw.filter_smooth_taubin(number_of_iterations=8)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    # === SIMPLIFY ===
    nt = len(mesh.triangles)
    if nt > args.target_tris:
        print(f"\nSimplificando ({nt:,} -> {args.target_tris:,} tris)...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()

    print(f"\n  Final: {len(mesh.vertices):,} v, {len(mesh.triangles):,} t")

    # === EXPORT ===
    mesh_path = out / "room_mls.ply"
    cloud_path = out / "room_mls_cloud.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(cloud_path), pcd_smooth)

    s = [
        f"original_points: {n_orig:,}",
        f"after_filters: {len(pcd.points):,}",
        f"mesh_vertices: {len(mesh.vertices):,}",
        f"mesh_triangles: {len(mesh.triangles):,}",
        f"normal_radius: {args.mls_radius * 2}",
        f"poisson_depth: {args.poisson_depth}",
        f"mesh_path: {mesh_path}",
        f"cloud_path: {cloud_path}",
    ]
    (out / "room_mls_summary.txt").write_text("\n".join(s))
    print("\n" + "\n".join(s))


if __name__ == "__main__":
    main()
