#!/usr/bin/env python3
"""
IR + Depth reconstruction pipeline (Route A).
Depth completion with existing model (optional), ICP odometry, TSDF fusion.
Outputs mesh and evaluation summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IR + Depth reconstruction pipeline")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--use-depth-model", action="store_true")
    parser.add_argument("--depth-checkpoint", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--voxel-length", type=float, default=0.01)
    parser.add_argument("--sdf-trunc", type=float, default=0.04)
    parser.add_argument("--icp-voxel", type=float, default=0.02)
    parser.add_argument("--icp-distance", type=float, default=0.06)
    parser.add_argument("--icp-iter", type=int, default=40)
    parser.add_argument("--use-odometry-init", action="store_true")
    return parser.parse_args()


class EncoderWrapper(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv2d(5, 3, kernel_size=3, padding=1, bias=True)

        from torchvision.models import efficientnet_b0

        base = efficientnet_b0(weights=None)
        self.encoder = EncoderWrapper(base.features)

        self.dec1 = _conv_block(1280 + 112, 256)
        self.dec2 = _conv_block(256 + 40, 128)
        self.dec3 = _conv_block(128 + 24, 64)
        self.dec4 = _conv_block(64 + 16, 32)
        self.dec5 = _conv_block(32, 16)
        self.final_conv = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)

        feats = []
        out = x
        for idx, layer in enumerate(self.encoder.features):
            out = layer(out)
            if idx in (1, 2, 3, 5):
                feats.append(out)

        skip16, skip24, skip40, skip112 = feats
        x = out

        x = nn.functional.interpolate(x, size=skip112.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip112], dim=1)
        x = self.dec1(x)

        x = nn.functional.interpolate(x, size=skip40.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip40], dim=1)
        x = self.dec2(x)

        x = nn.functional.interpolate(x, size=skip24.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip24], dim=1)
        x = self.dec3(x)

        x = nn.functional.interpolate(x, size=skip16.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip16], dim=1)
        x = self.dec4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec5(x)
        x = self.final_conv(x)
        x = torch.relu(x)
        return x


def _load_pairs(ir_dir: Path, depth_dir: Path) -> List[Tuple[Path, Path]]:
    ir_files = sorted(ir_dir.glob("yolo_data_*.jpg")) + sorted(ir_dir.glob("yolo_data_*.png"))
    depth_files = {p.stem: p for p in depth_dir.glob("yolo_data_*.png")}
    pairs = []
    for ir in ir_files:
        depth = depth_files.get(ir.stem)
        if depth is not None:
            pairs.append((ir, depth))
    return pairs


def _make_intrinsics(width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    import open3d as o3d

    if fx > 0 and fy > 0 and cx > 0 and cy > 0:
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )


def _depth_completion(
    model: DepthCompletionModel,
    device: str,
    ir_gray: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rgb = np.repeat(ir_gray[..., None], 3, axis=2)
    input_tensor = np.concatenate(
        [rgb, depth_m[..., None], mask.astype(np.float32)[..., None]], axis=2
    )
    input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
    return pred


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_dir = Path(args.ir_dir)
    depth_dir = Path(args.depth_dir)

    pairs = _load_pairs(ir_dir, depth_dir)
    if not pairs:
        raise SystemExit("No se encontraron pares IR+Depth")

    if args.frame_step > 1:
        pairs = pairs[:: args.frame_step]

    ir_sample = cv2.imread(str(pairs[0][0]), cv2.IMREAD_GRAYSCALE)
    depth_sample = cv2.imread(str(pairs[0][1]), cv2.IMREAD_UNCHANGED)
    if ir_sample is None or depth_sample is None:
        raise SystemExit("No se pudo leer IR/Depth de muestra")

    height, width = depth_sample.shape[:2]

    try:
        import open3d as o3d
    except Exception as exc:
        raise SystemExit(
            "Open3D no está disponible en este Python. "
            "Instala Open3D en un entorno compatible (Python 3.10-3.12) o usa otro equipo. "
            f"Detalle: {exc}"
        )

    intrinsics = _make_intrinsics(width, height, args.fx, args.fy, args.cx, args.cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    model = None
    if args.use_depth_model:
        if not args.depth_checkpoint:
            raise SystemExit("--depth-checkpoint es requerido si --use-depth-model")
        model = DepthCompletionModel().to(args.device)
        state = torch.load(args.depth_checkpoint, map_location=args.device)
        model.load_state_dict(state, strict=True)
        model.eval()

    poses = []
    pose = np.eye(4)
    prev_pcd = None
    prev_pose = np.eye(4)
    prev_rgbd = None
    fitness_scores = []
    rmse_scores = []
    valid_ratios = []

    for idx, (ir_path, depth_path) in enumerate(pairs):
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth_raw is None:
            continue

        if ir.shape[:2] != (height, width):
            ir = cv2.resize(ir, (width, height), interpolation=cv2.INTER_AREA)

        ir = cv2.bilateralFilter(ir, d=5, sigmaColor=50, sigmaSpace=50)
        ir_norm = ir.astype(np.float32) / 255.0

        if depth_raw.dtype != np.uint16:
            raise SystemExit(f"Depth no es uint16: {depth_path}")

        depth_m = depth_raw.astype(np.float32) * args.depth_scale
        mask = (depth_m > 0) & (depth_m < args.max_depth)
        valid_ratios.append(float(mask.sum()) / float(mask.size))

        if model is not None:
            pred = _depth_completion(model, args.device, ir_norm, depth_m, mask)
            depth_fused = depth_m.copy()
            depth_fused[~mask] = pred[~mask]
        else:
            depth_fused = depth_m

        depth_fused = np.clip(depth_fused, 0, args.max_depth)

        color = (np.repeat(ir_norm[..., None], 3, axis=2) * 255.0).astype(np.uint8)
        depth_image = o3d.geometry.Image(depth_fused.astype(np.float32))
        color_image = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd = pcd.voxel_down_sample(voxel_size=args.icp_voxel)
        pcd.estimate_normals()

        if prev_pcd is None:
            pose = np.eye(4)
        else:
            init_guess = np.eye(4)
            if args.use_odometry_init and prev_rgbd is not None:
                odo_ok, odo, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd,
                    prev_rgbd,
                    intrinsics,
                    np.eye(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                )
                if odo_ok:
                    init_guess = odo

            reg = o3d.pipelines.registration.registration_icp(
                pcd,
                prev_pcd,
                max_correspondence_distance=args.icp_distance,
                init=init_guess,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=args.icp_iter
                ),
            )
            fitness_scores.append(float(reg.fitness))
            rmse_scores.append(float(reg.inlier_rmse))
            pose = prev_pose @ np.linalg.inv(reg.transformation)

        volume.integrate(rgbd, intrinsics, pose)
        poses.append(pose)
        prev_pcd = pcd
        prev_pose = pose
        prev_rgbd = rgbd

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = out_dir / "reconstruction_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    pcd_out = volume.extract_point_cloud()
    pcd_path = out_dir / "reconstruction_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_out)

    summary = {
        "frames": len(pairs),
        "valid_depth_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "valid_depth_ratio_min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
        "valid_depth_ratio_max": float(np.max(valid_ratios)) if valid_ratios else 0.0,
        "icp_fitness_mean": float(np.mean(fitness_scores)) if fitness_scores else 0.0,
        "icp_rmse_mean": float(np.mean(rmse_scores)) if rmse_scores else 0.0,
        "mesh_vertices": int(np.asarray(mesh.vertices).shape[0]),
        "mesh_triangles": int(np.asarray(mesh.triangles).shape[0]),
        "mesh_path": str(mesh_path),
        "point_cloud_path": str(pcd_path),
    }

    summary_path = out_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Mesh:", mesh_path)
    print("Point cloud:", pcd_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
