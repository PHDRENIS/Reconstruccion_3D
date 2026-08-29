#!/usr/bin/env python3
"""
Reconstruccion 3D v2 - parametros optimizados:
  - Filtro bilateral + clip depth por frame
  - Statistical outlier removal por frame  
  - ICP estricto (distance=0.04, iter=80, fitness>0.4)
  - TSDF fino (voxel=0.005, sdf_trunc=0.03)
  - Z crop [0.3, 2.8] sin piso ni techo
  - Taubin smooth final
"""

import argparse, json, sys
from pathlib import Path
import cv2, numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ir-dir", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--yolo-model", default="")
    p.add_argument("--depth-scale", type=float, default=0.001)
    p.add_argument("--depth-trunc", type=float, default=5.0)
    p.add_argument("--z-min", type=float, default=0.3)
    p.add_argument("--z-max", type=float, default=2.8)
    p.add_argument("--icp-dist", type=float, default=0.04)
    p.add_argument("--icp-iter", type=int, default=80)
    p.add_argument("--min-fitness", type=float, default=0.4)
    p.add_argument("--voxel-length", type=float, default=0.005)
    p.add_argument("--sdf-trunc", type=float, default=0.03)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--yolo-conf", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    # Load pairs
    ir_dir = Path(args.ir_dir)
    depth_dir = Path(args.depth_dir)
    ir_files = sorted(ir_dir.glob("yolo_data_*.jpg")) + sorted(ir_dir.glob("yolo_data_*.png"))
    depth_map = {p.stem: p for p in depth_dir.glob("yolo_data_*.png")}
    pairs = [(ir, depth_map[ir.stem]) for ir in ir_files if ir.stem in depth_map]
    pairs = pairs[::args.frame_step]
    print(f"Frames: {len(pairs)}")

    # YOLO
    yolo = None
    if args.yolo_model and Path(args.yolo_model).exists():
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model)
        print("YOLO IR cargado")

    # Sample first to get dims
    d0 = cv2.imread(str(pairs[0][1]), cv2.IMREAD_UNCHANGED)
    d0 = np.squeeze(d0)
    h, w = d0.shape[:2]

    # Intrinsics
    fx = fy = float(w) * 1.05
    cx = w / 2.0
    cy = h / 2.0
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    prev_pcd = None
    prev_pose = np.eye(4)
    prev_rgbd = None
    pose = np.eye(4)
    poses = [pose.copy()]
    integrated = 0
    skipped = 0
    fitness_vals = []
    rmse_vals = []
    valid_ratios = []
    det_counts = []

    for idx, (ir_path, dp_path) in enumerate(pairs):
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(dp_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth_raw is None:
            skipped += 1; continue

        depth_raw = np.squeeze(depth_raw)
        if ir.shape[:2] != (h, w):
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_AREA)

        # --- Pre-process depth ---
        depth_m = depth_raw.astype(np.float32) * args.depth_scale
        valid = (depth_m > args.z_min) & (depth_m < args.z_max)
        if valid.sum() < 500:
            skipped += 1; continue
        valid_ratios.append(float(valid.sum()) / valid.size)

        # Bilateral filter on valid depth (float32)
        depth_vis = depth_m.copy(); depth_vis[~valid] = 0
        depth_vis_f32 = depth_vis.astype(np.float32)
        depth_fused = cv2.bilateralFilter(depth_vis_f32, d=5, sigmaColor=0.5, sigmaSpace=5)
        depth_fused = np.clip(depth_fused, 0, args.depth_trunc)

        # YOLO mask
        n_det = 0
        if yolo is not None:
            ir3 = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            res = yolo(ir3, conf=args.yolo_conf, verbose=False)
            if res and res[0].masks is not None and len(res[0].masks) > 0:
                for m in res[0].masks.data.cpu().numpy():
                    if (m > 0.5).sum() >= 200:
                        n_det += 1
        det_counts.append(n_det)

        # Create RGBD
        color = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        depth_img = o3d.geometry.Image(depth_fused.astype(np.float32))
        color_img = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd = pcd.voxel_down_sample(voxel_size=0.015)

        # Statistical outlier removal per frame
        if len(pcd.points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)

        if len(pcd.points) < 100:
            skipped += 1; continue

        # Z crop
        pts_arr = np.asarray(pcd.points)
        z_mask = (pts_arr[:, 2] >= args.z_min) & (pts_arr[:, 2] <= args.z_max)
        if z_mask.sum() < 100:
            skipped += 1; continue
        pcd = pcd.select_by_index(np.where(z_mask)[0])

        pcd.estimate_normals()

        # ICP
        if prev_pcd is None:
            pose = np.eye(4)
        else:
            init_guess = np.eye(4)
            if prev_rgbd is not None:
                try:
                    odo_ok, odo, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                        rgbd, prev_rgbd, intrinsics, np.eye(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    )
                    if odo_ok:
                        init_guess = odo
                except Exception:
                    pass

            reg = o3d.pipelines.registration.registration_icp(
                pcd, prev_pcd, max_correspondence_distance=args.icp_dist,
                init=init_guess,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.icp_iter),
            )

            if reg.fitness < args.min_fitness:
                skipped += 1; continue

            fitness_vals.append(reg.fitness)
            rmse_vals.append(reg.inlier_rmse)
            pose = prev_pose @ np.linalg.inv(reg.transformation)

        volume.integrate(rgbd, intrinsics, pose)
        poses.append(pose.copy())
        prev_pcd = pcd
        prev_pose = pose
        prev_rgbd = rgbd
        integrated += 1

        if integrated % 30 == 0:
            print(f"  Frame {integrated}/{len(pairs)} integrated, {skipped} skipped")

    print(f"\nIntegrated: {integrated}/{len(pairs)}, Skipped: {skipped}")

    if integrated < 5:
        print("ERROR: Muy pocos frames integrados")
        return

    # Extract mesh
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Smooth
    print("Taubin smoothing...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    # Simplify
    n_tris = len(mesh.triangles)
    if n_tris > 500_000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=300_000)
    elif n_tris > 200_000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=min(200_000, n_tris // 2))
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    mesh_path = out / "room_v2.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Mesh: {mesh_path}")

    cloud_path = out / "room_v2_cloud.ply"
    pcd_out = volume.extract_point_cloud()
    # Crop point cloud by Z too
    pts_out = np.asarray(pcd_out.points)
    zm = (pts_out[:, 2] >= args.z_min) & (pts_out[:, 2] <= args.z_max)
    pcd_out = pcd_out.select_by_index(np.where(zm)[0])
    o3d.io.write_point_cloud(str(cloud_path), pcd_out)

    summary = {
        "frames_total": len(pairs),
        "frames_integrated": integrated,
        "frames_skipped": skipped,
        "icp_fitness_mean": round(float(np.mean(fitness_vals)), 4) if fitness_vals else 0,
        "icp_rmse_mean": round(float(np.mean(rmse_vals)), 4) if rmse_vals else 0,
        "avg_detections": round(float(np.mean(det_counts)), 2) if det_counts else 0,
        "avg_valid_depth": round(float(np.mean(valid_ratios)), 4) if valid_ratios else 0,
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "z_crop": [args.z_min, args.z_max],
        "voxel_length": args.voxel_length,
        "mesh_path": str(mesh_path),
        "cloud_path": str(cloud_path),
    }

    with open(out / "room_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
