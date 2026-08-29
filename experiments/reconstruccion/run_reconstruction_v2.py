#!/usr/bin/env python3
"""
Reconstruccion V1 adaptada para nueva secuencia (338 frames + metadata reales).
Usa intrinsecos reales de la camara desde metadata CSV.
"""

import argparse, json, sys, os
from pathlib import Path
import cv2, numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ir-dir", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--yolo-model", required=True)
    p.add_argument("--depth-scale", type=float, default=0.001)
    p.add_argument("--max-depth", type=float, default=8.0)
    p.add_argument("--voxel-length", type=float, default=0.008)
    p.add_argument("--sdf-trunc", type=float, default=0.03)
    p.add_argument("--yolo-conf", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def imread_unicode(path):
    """cv2.imread con soporte unicode en Windows."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def read_intrinsics_from_metadata(metadata_dir: Path):
    """Lee intrinsecos del primer CSV de metadata."""
    for f in sorted(metadata_dir.glob("yolo_data_0000_Depth_metadata.csv")):
        text = f.read_text(encoding="utf-8")
        fx = fy = cx = cy = None
        for line in text.splitlines():
            if line.startswith("Fx,"):
                fx = float(line.split(",")[1].strip())
            elif line.startswith("Fy,"):
                fy = float(line.split(",")[1].strip())
            elif line.startswith("PPx,"):
                cx = float(line.split(",")[1].strip())
            elif line.startswith("PPy,"):
                cy = float(line.split(",")[1].strip())
        if fx and fy and cx and cy:
            return fx, fy, cx, cy
    return None


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

    # Read intrinsics from metadata
    metadata_dir = Path(args.metadata_dir)
    intrinsics_data = read_intrinsics_from_metadata(metadata_dir)
    if intrinsics_data:
        fx, fy, cx, cy = intrinsics_data
        print(f"Intrinsecos reales: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    else:
        w, h = 640, 480
        fx = fy = float(w) * 1.05
        cx = w / 2
        cy = h / 2
        print(f"Intrinsecos estimados (sin metadata): fx={fx:.2f}")

    sample_ir = imread_unicode(paired[0][0])
    sample_depth = imread_unicode(paired[0][1])
    sample_depth = np.squeeze(sample_depth)
    h, w = sample_depth.shape[:2]

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    total_dets, mask_ratios, rmse_vals, fitness_vals, valid_ratios = [], [], [], [], []
    prev_pcd, prev_pose, prev_rgbd = None, np.eye(4), None
    pose = np.eye(4)

    for idx, (ir_path, dp_path) in enumerate(paired):
        ir = imread_unicode(ir_path)
        depth_raw = imread_unicode(dp_path)
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
        det_count = 0
        yolo_mask = np.zeros((h, w), dtype=np.uint8)
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
        depth_img = o3d.geometry.Image(depth_fused.astype(np.float32))
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
    cloud_path = out_dir / "reconstruction_cloud.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd_out)
    print(f"Cloud: {cloud_path}")

    summary = {
        "frames": len(paired),
        "intrinsics": f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}",
        "avg_yolo_dets": round(np.mean(total_dets), 2) if total_dets else 0,
        "avg_mask_ratio": round(np.mean(mask_ratios), 4) if mask_ratios else 0,
        "avg_valid_depth": round(np.mean(valid_ratios), 4) if valid_ratios else 0,
        "icp_fitness": round(np.mean(fitness_vals), 4) if fitness_vals else 0,
        "icp_rmse": round(np.mean(rmse_vals), 4) if rmse_vals else 0,
        "mesh_vertices": int(np.asarray(mesh.vertices).shape[0]) if mesh.has_vertices() else 0,
        "mesh_triangles": int(np.asarray(mesh.triangles).shape[0]) if mesh.has_triangles() else 0,
        "mesh_path": str(mesh_path),
        "cloud_path": str(cloud_path),
    }

    with open(out_dir / "evaluation.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n=== RESUMEN ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
