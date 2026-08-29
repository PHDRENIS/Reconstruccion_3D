#!/usr/bin/env python3
"""
Reconstruccion 3D final: YOLO IR + Depth Completion + ICP + TSDF
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--voxel-length", type=float, default=0.008)
    parser.add_argument("--sdf-trunc", type=float, default=0.04)
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_dir = Path(args.ir_dir)
    depth_dir = Path(args.depth_dir)
    pairs = sorted(ir_dir.glob("yolo_data_*.jpg")) + sorted(ir_dir.glob("yolo_data_*.png"))
    depth_files = {p.stem: p for p in depth_dir.glob("yolo_data_*.png")}
    paired = [(ir, depth_files[ir.stem]) for ir in pairs if ir.stem in depth_files]
    if args.max_frames > 0:
        paired = paired[:args.max_frames]

    print(f"Frames: {len(paired)}")

    import open3d as o3d
    from ultralytics import YOLO

    yolo = YOLO(args.yolo_model)

    sample_ir = cv2.imread(str(paired[0][0]), cv2.IMREAD_GRAYSCALE)
    sample_depth = cv2.imread(str(paired[0][1]), cv2.IMREAD_UNCHANGED)
    h, w = sample_depth.shape[:2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsics.intrinsic_matrix[0, 0],
                                                    intrinsics.intrinsic_matrix[1, 1],
                                                    intrinsics.intrinsic_matrix[0, 2],
                                                    intrinsics.intrinsic_matrix[1, 2])

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    total_dets, mask_ratios, rmse_vals, fitness_vals, valid_ratios = [], [], [], [], []
    prev_pcd, prev_pose, prev_rgbd = None, np.eye(4), None
    pose = np.eye(4)

    for idx, (ir_path, dp_path) in enumerate(paired):
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(dp_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth_raw is None:
            continue

        depth_raw = np.squeeze(depth_raw)
        if ir.shape[:2] != (h, w):
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_AREA)

        depth_m = depth_raw.astype(np.float32) * args.depth_scale
        valid = (depth_m > 0) & (depth_m < args.max_depth)
        if valid.sum() == 0:
            continue
        valid_ratios.append(float(valid.sum()) / valid.size)

        # YOLO mask
        ir3 = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        res = yolo(ir3, conf=args.yolo_conf, verbose=False)
        yolo_mask = np.zeros((h, w), dtype=np.uint8)
        det_count = 0
        if res and res[0].masks is not None and len(res[0].masks) > 0:
            for m_data in res[0].masks.data.cpu().numpy():
                mb = (m_data > 0.5).astype(np.uint8) * 255
                if mb.sum() >= 200:
                    yolo_mask = cv2.bitwise_or(yolo_mask, mb)
                    det_count += 1
        total_dets.append(det_count)
        mask_ratios.append(float((yolo_mask > 0).sum()) / yolo_mask.size)

        depth_fused = depth_m.copy()
        depth_fused = np.clip(depth_fused, 0, args.max_depth)

        color = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        depth_arr = depth_fused.astype(np.float32).copy()
        depth_img = o3d.geometry.Image(depth_arr)
        color_img = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd.estimate_normals()

        if prev_pcd is not None:
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
                pcd, prev_pcd, max_correspondence_distance=0.08,
                init=init_guess,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
            )
            fitness_vals.append(reg.fitness)
            rmse_vals.append(reg.inlier_rmse)
            pose = prev_pose @ np.linalg.inv(reg.transformation)

        volume.integrate(rgbd, intrinsics, pose)
        prev_pcd, prev_pose, prev_rgbd = pcd, pose, rgbd

        if (idx + 1) % 50 == 0:
            print(f"  Frame {idx+1}/{len(paired)}")

    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = out_dir / "reconstruction_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Mesh: {mesh_path}")

    pcd_out = volume.extract_point_cloud()
    pcd_path = out_dir / "reconstruction_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_out)
    print(f"Cloud: {pcd_path}")

    summary = {
        "frames": len(paired),
        "avg_yolo_dets": round(np.mean(total_dets), 2) if total_dets else 0,
        "avg_mask_ratio": round(np.mean(mask_ratios), 4) if mask_ratios else 0,
        "avg_valid_depth": round(np.mean(valid_ratios), 4) if valid_ratios else 0,
        "icp_fitness": round(np.mean(fitness_vals), 4) if fitness_vals else 0,
        "icp_rmse": round(np.mean(rmse_vals), 4) if rmse_vals else 0,
        "mesh_vertices": int(np.asarray(mesh.vertices).shape[0]) if mesh.has_vertices() else 0,
        "mesh_triangles": int(np.asarray(mesh.triangles).shape[0]) if mesh.has_triangles() else 0,
        "mesh_path": str(mesh_path),
        "cloud_path": str(pcd_path),
    }

    with open(out_dir / "evaluation.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n=== RESUMEN ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
