#!/usr/bin/env python3
"""
Crop por densidad + DBSCAN + Screened Poisson.
Elimina ruido lejano y clusters aislados, conservando solo la habitacion.
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
    p.add_argument("--normal-radius", type=float, default=0.10)
    p.add_argument("--poisson-depth", type=int, default=10)
    p.add_argument("--target-tris", type=int, default=200000)
    p.add_argument("--dbscan-eps", type=float, default=0.15)
    p.add_argument("--dbscan-min", type=int, default=30)
    return p.parse_args()


def density_bbox(pts, cell_size=0.15, mass_pct=98):
    """XY crop by density: finds the main cluster in 2D histogram."""
    x, y = pts[:, 0], pts[:, 1]
    x_bins = np.arange(x.min() - cell_size, x.max() + cell_size, cell_size)
    y_bins = np.arange(y.min() - cell_size, y.max() + cell_size, cell_size)
    hist, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])
    hist_sorted = np.sort(hist.ravel())[::-1]
    cumsum = np.cumsum(hist_sorted)
    total = cumsum[-1]
    threshold_idx = np.searchsorted(cumsum, total * mass_pct / 100)
    threshold_val = hist_sorted[max(0, threshold_idx - 1)] if threshold_idx > 0 else 1
    mask = hist >= threshold_val
    yi, xi = np.where(mask)
    if len(xi) == 0:
        return pts
    # xe has len = len(x_bins)+1 or len(x_bins) depending on numpy version
    x_min = xe[xi.min()]
    x_max = xe[min(len(xe)-1, xi.max()+1)]
    y_min = ye[yi.min()]
    y_max = ye[min(len(ye)-1, yi.max()+1)]
    # Add margin
    margin = cell_size
    return pts[(x >= x_min - margin) & (x <= x_max + margin) & 
               (y >= y_min - margin) & (y <= y_max + margin)]


def density_z_crop(pts, height=1.8):
    """Z crop around the densest Z bin."""
    z = pts[:, 2]
    hist, edges = np.histogram(z, bins=100)
    peak_idx = np.argmax(hist)
    z_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    z_min = z_center - height
    z_max = z_center + height
    mask = (z >= z_min) & (z <= z_max)
    return pts[mask], z_center


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === LOAD ===
    print("Cargando nube original...")
    pcd = o3d.io.read_point_cloud(args.cloud)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3)) * 0.55
    n_orig = len(pts)
    print(f"  Original: {n_orig:,} puntos")
    print(f"  BBox: X[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] Y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] Z[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")

    # === OUTLIER FILTERS ===
    print("\nFiltros de outliers...")
    pcd.points = o3d.utility.Vector3dVector(pts)
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)
    print(f"  Stat1 (std=1.8): {n_before:,} -> {len(pcd.points):,}")

    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    print(f"  Stat2 (std=1.2): {n_before:,} -> {len(pcd.points):,}")

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    # === XY DENSITY CROP ===
    print("\nCrop XY por densidad...")
    n_before = len(pts)
    pts = density_bbox(pts, cell_size=0.12, mass_pct=98)
    print(f"  {n_before:,} -> {len(pts):,} pts")

    # === Z DENSITY CROP ===
    print("\nCrop Z por histograma...")
    n_before = len(pts)
    pts, z_center = density_z_crop(pts, height=1.8)
    print(f"  {n_before:,} -> {len(pts):,} (centro Z={z_center:.1f}m)")

    if len(pts) < 1000:
        print("ERROR: Muy pocos puntos tras crop")
        return

    # === VOXEL before DBSCAN ===
    print(f"\nVoxel ({args.voxel}m) before DBSCAN...")
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts)
    pcd2 = pcd2.voxel_down_sample(args.voxel)
    pts_down = np.asarray(pcd2.points)
    print(f"  {len(pts_down):,} pts")

    # === DBSCAN on downsampled cloud ===
    print(f"\nDBSCAN (eps={args.dbscan_eps}, min={args.dbscan_min})...")
    labels = np.array(pcd2.cluster_dbscan(eps=args.dbscan_eps, min_points=args.dbscan_min))
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:6]:
        lbl = "Noise" if u == -1 else f"Cluster {u}"
        print(f"  {lbl}: {c:,} pts")

    non_noise = [(u, c) for u, c in zip(unique, counts) if u != -1]
    if non_noise:
        best_label = max(non_noise, key=lambda x: x[1])[0]
        main_mask = labels == best_label
        pts_down = pts_down[main_mask]
        print(f"  Conservando cluster {best_label}: {len(pts_down):,} pts")
    print(f"  Final: {len(pts_down):,} pts")

    # Rebuild pcd2 with filtered points
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts_down)

    # Target for Poisson
    n_pts = len(pcd2.points)
    target = 800_000
    if n_pts > target:
        pcd_pois = pcd2.random_down_sample(target / n_pts)
    else:
        pcd_pois = pcd2
    print(f"  Poisson input: {len(pcd_pois.points):,} pts")

    print(f"\nNormales (radius={args.normal_radius}m)...")
    pcd_pois.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius * 2, max_nn=40))
    pcd_pois.orient_normals_consistent_tangent_plane(80)

    # === POISSON ===
    print(f"\nScreened Poisson (depth={args.poisson_depth})...")
    mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_pois, depth=args.poisson_depth
    )
    print(f"  Raw: {len(mesh_raw.vertices):,}v, {len(mesh_raw.triangles):,}t")

    if len(densities) > 0:
        q = np.quantile(densities, 0.02)
        mesh_raw.remove_vertices_by_mask(densities < q)
    mesh_raw.compute_vertex_normals()
    mesh_raw.remove_degenerate_triangles()

    # === TAUBIN ===
    print("\nTaubin smooth (8 iter)...")
    mesh = mesh_raw.filter_smooth_taubin(number_of_iterations=8)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    # === SIMPLIFY ===
    nt = len(mesh.triangles)
    if nt > args.target_tris:
        print(f"Simplificando {nt:,} -> {args.target_tris:,}...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()

    # === EXPORT ===
    mesh_path = out / "room_cropped.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    s = [
        f"original: {n_orig:,} pts",
        f"after_crop_dbscan: {len(pts):,} pts",
        f"mesh_vertices: {len(mesh.vertices):,}",
        f"mesh_triangles: {len(mesh.triangles):,}",
        f"z_center: {z_center:.1f}",
        f"poisson_depth: {args.poisson_depth}",
        f"path: {mesh_path}",
    ]
    (out / "room_cropped_summary.txt").write_text("\n".join(s))
    print("\n" + "\n".join(s))


if __name__ == "__main__":
    main()
