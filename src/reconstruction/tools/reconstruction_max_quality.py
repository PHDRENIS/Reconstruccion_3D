#!/usr/bin/env python3
"""
Reconstruccion 3D de maxima calidad (solo geometria):
  Depth Completion → ICP Multi-escala → Pose Graph → TSDF → Poisson → Post-proc
"""

import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np
import torch
import torch.nn as nn


# ===================================================================
# Depth Completion Model (Efficient-UNet)
# ===================================================================
def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class EncoderFeats(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
    def forward(self, x):
        return self.features(x)

class DepthCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv2d(5, 3, 3, padding=1, bias=True)
        from torchvision.models import efficientnet_b0
        base = efficientnet_b0(weights=None)
        self.encoder = EncoderFeats(base.features)
        self.dec1 = _conv_block(1280 + 112, 256)
        self.dec2 = _conv_block(256 + 40, 128)
        self.dec3 = _conv_block(128 + 24, 64)
        self.dec4 = _conv_block(64 + 16, 32)
        self.dec5 = _conv_block(32, 16)
        self.final_conv = nn.Sequential(nn.Conv2d(16, 1, 1, bias=True))

    def forward(self, x):
        x = self.adapter(x)
        feats = []; out = x
        for idx, layer in enumerate(self.encoder.features):
            out = layer(out)
            if idx in (1, 2, 3, 5): feats.append(out)
        s16, s24, s40, s112 = feats; x = out
        x = nn.functional.interpolate(x, size=s112.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s112], 1); x = self.dec1(x)
        x = nn.functional.interpolate(x, size=s40.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s40], 1); x = self.dec2(x)
        x = nn.functional.interpolate(x, size=s24.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s24], 1); x = self.dec3(x)
        x = nn.functional.interpolate(x, size=s16.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s16], 1); x = self.dec4(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dec5(x); x = self.final_conv(x)
        return torch.relu(x)


# ===================================================================
# Main Pipeline
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Reconstruccion 3D Maxima Calidad")
    p.add_argument("--ir-dir", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--depth-model", required=True)
    p.add_argument("--depth-scale", type=float, default=0.001)
    p.add_argument("--depth-trunc", type=float, default=5.0)
    p.add_argument("--z-min", type=float, default=0.3)
    p.add_argument("--z-max", type=float, default=2.8)
    p.add_argument("--tsdf-voxel", type=float, default=0.006)
    p.add_argument("--sdf-trunc", type=float, default=0.03)
    p.add_argument("--poisson-depth", type=int, default=10)
    p.add_argument("--target-tris", type=int, default=300000)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def depth_completion(model, device, ir, depth_m, mask, max_d):
    ir_n = ir.astype(np.float32) / 255.0
    tensor = np.concatenate([
        np.repeat(ir_n[..., None], 3, axis=2),
        depth_m[..., None],
        mask.astype(np.float32)[..., None]
    ], axis=2)
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()
    out = depth_m.copy()
    out[~mask] = pred[~mask]
    return np.clip(out, 0, max_d)


def multi_scale_icp(source, target, init_guess=None, voxel_sizes=None, distances=None, iters=None):
    """ICP multi-escala: coarse -> medium -> fine"""
    if init_guess is None:
        init_guess = np.eye(4)
    if voxel_sizes is None:
        voxel_sizes = [0.04, 0.02, 0.01]
    if distances is None:
        distances = [0.10, 0.06, 0.03]
    if iters is None:
        iters = [60, 80, 100]

    import open3d as o3d

    T = init_guess.copy()
    for vi, di, it in zip(voxel_sizes, distances, iters):
        src = source.voxel_down_sample(vi) if vi > 0 else source
        tgt = target.voxel_down_sample(vi) if vi > 0 else target
        if len(src.points) < 50 or len(tgt.points) < 50:
            continue
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vi*2, max_nn=20))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vi*2, max_nn=20))
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_correspondence_distance=di, init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it),
        )
        T = reg.transformation
    return T, reg.fitness, reg.inlier_rmse


def detect_loop_closures(pcds, poses, min_frames_apart=20, max_distance=0.5, min_fitness=0.35):
    """Detecta loop closures entre frames no consecutivos."""
    import open3d as o3d
    loops = []
    step = max(1, min_frames_apart // 2)
    for i in range(0, len(pcds) - min_frames_apart, step):
        p_i = pcds[i]
        center_i = np.asarray(p_i.points).mean(axis=0)
        for j in range(i + min_frames_apart, len(pcds), step):
            center_j = np.asarray(pcds[j].points).mean(axis=0)
            if np.linalg.norm(center_i - center_j) > max_distance:
                continue
            T_cl, fitness, rmse = multi_scale_icp(
                pcds[i], pcds[j],
                init_guess=poses[j] @ np.linalg.inv(poses[i]),
                voxel_sizes=[0.03, 0.015], distances=[0.08, 0.04], iters=[60, 80]
            )
            if fitness > min_fitness:
                loops.append((i, j, T_cl, fitness))
                if len(loops) % 5 == 0:
                    print(f"  Loop closures: {len(loops)}")
    return loops


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    import open3d as o3d

    # === Load data ===
    ir_dir = Path(args.ir_dir)
    depth_dir = Path(args.depth_dir)
    ir_files = sorted(ir_dir.glob("yolo_data_*.jpg")) + sorted(ir_dir.glob("yolo_data_*.png"))
    d_map = {p.stem: p for p in depth_dir.glob("yolo_data_*.png")}
    pairs = [(ir, d_map[ir.stem]) for ir in ir_files if ir.stem in d_map]
    pairs = pairs[::args.frame_step]
    print(f"Frames: {len(pairs)}")

    # Sample dims
    d0 = cv2.imread(str(pairs[0][1]), cv2.IMREAD_UNCHANGED)
    d0 = np.squeeze(d0)
    h, w = d0.shape[:2]
    fx = fy = float(w) * 1.05; cx = w / 2; cy = h / 2
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # === Load depth model ===
    print("\n=== ETAPA 1: Depth Completion ===")
    t0 = time.time()
    dcm = DepthCompletionModel().to(device)
    state = torch.load(args.depth_model, map_location=device)
    dcm.load_state_dict(state, strict=True)
    dcm.eval()
    print(f"  Modelo cargado: {args.depth_model}")

    # === Process frames ===
    rgbds = []
    pcds_icp = []
    valid_indices = []
    print(f"  Procesando {len(pairs)} frames con depth completion...")

    for idx, (ir_p, dp_p) in enumerate(pairs):
        ir = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE)
        dr = cv2.imread(str(dp_p), cv2.IMREAD_UNCHANGED)
        if ir is None or dr is None: continue
        dr = np.squeeze(dr)
        if ir.shape[:2] != (h, w):
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_AREA)

        depth_m = dr.astype(np.float32) * args.depth_scale
        mask = (depth_m > 0) & (depth_m < args.depth_trunc)
        if mask.sum() < 500: continue

        # Depth completion
        depth_c = depth_completion(dcm, device, ir, depth_m, mask, args.depth_trunc)
        depth_c[mask] = depth_m[mask]  # preserve valid pixels

        # Preprocess
        depth_c = cv2.bilateralFilter(depth_c.astype(np.float32), 5, 0.3, 5)
        depth_c = np.clip(depth_c, 0, args.depth_trunc)

        # RGBD
        color = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth_c.astype(np.float32)),
            depth_scale=1.0, depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Point cloud for ICP (light)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        z_arr = np.asarray(pcd.points)[:, 2]
        valid_z = (z_arr >= args.z_min) & (z_arr <= args.z_max)
        if valid_z.sum() < 200: continue
        pcd = pcd.select_by_index(np.where(valid_z)[0])
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

        rgbds.append(rgbd)
        pcds_icp.append(pcd)
        valid_indices.append(idx)

        if len(valid_indices) % 50 == 0:
            print(f"    {len(valid_indices)} frames validos")

    print(f"  Frames validos: {len(pcds_icp)}/{len(pairs)}")
    print(f"  Tiempo depth completion: {time.time()-t0:.1f}s")

    # === ETAPA 2: Multi-scale ICP ===
    print("\n=== ETAPA 2: ICP Multi-escala ===")
    t0 = time.time()
    poses = [np.eye(4)]
    pcds_aligned = [pcds_icp[0]]
    fitness_vals, rmse_vals = [], []
    skipped = 0

    for i in range(1, len(pcds_icp)):
        init = np.eye(4)
        try:
            odo_ok, odo_trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbds[i], rgbds[i-1], intrinsics, np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            )
            if odo_ok: init = odo_trans
        except Exception: pass

        T_icp, fit, rmse = multi_scale_icp(pcds_icp[i], pcds_aligned[-1], init)
        if fit < 0.3: skipped += 1; continue

        fitness_vals.append(fit); rmse_vals.append(rmse)
        new_pose = poses[-1] @ np.linalg.inv(T_icp)
        poses.append(new_pose)
        pcds_aligned.append(pcds_icp[i])

        if i % 60 == 0:
            print(f"  Frame {i}/{len(pcds_icp)} | fit={fit:.3f} | rmse={rmse:.4f}")

    print(f"  Alineados: {len(poses)}/{len(pcds_icp)}, Skipped: {skipped}")
    print(f"  ICP fitness: {np.mean(fitness_vals):.4f} ± {np.std(fitness_vals):.4f}")
    print(f"  ICP RMSE: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"  Tiempo ICP: {time.time()-t0:.1f}s")

    # === ETAPA 3: Pose Graph Optimization ===
    print("\n=== ETAPA 3: Pose Graph Optimization ===")
    t0 = time.time()
    pg = o3d.pipelines.registration.PoseGraph()
    info_default = np.eye(6)

    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(poses[0]))
    for i in range(1, len(poses)):
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(poses[i]))
        edge = o3d.pipelines.registration.PoseGraphEdge(
            i-1, i,
            np.linalg.inv(poses[i-1]) @ poses[i],
            info_default,
            uncertain=False
        )
        pg.edges.append(edge)

    # Loop closure detection
    print("  Detectando loop closures...")
    loops = detect_loop_closures(pcds_aligned, poses)
    for src, tgt, T_lc, fit in loops:
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(src, tgt, T_lc, info_default, uncertain=True))
    print(f"  Loop closures: {len(loops)}")

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.04, edge_prune_threshold=0.25, reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    poses_opt = [pg.nodes[i].pose for i in range(len(pg.nodes))]
    print(f"  Tiempo pose graph: {time.time()-t0:.1f}s")

    # === ETAPA 4: TSDF Fusion ===
    print("\n=== ETAPA 4: TSDF Fusion ===")
    t0 = time.time()
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(min(len(rgbds), len(poses_opt))):
        vol.integrate(rgbds[i], intrinsics, poses_opt[i])
    print(f"  Tiempo TSDF: {time.time()-t0:.1f}s")

    # === ETAPA 5: Screened Poisson ===
    print("\n=== ETAPA 5: Screened Poisson Surface Reconstruction ===")
    t0 = time.time()
    pcd_acc = vol.extract_point_cloud()
    n_points = len(pcd_acc.points)
    # Downsample for Poisson
    target_poisson = min(n_points, 800_000)
    if n_points > target_poisson:
        ratio = target_poisson / n_points
        pcd_poisson = pcd_acc.random_down_sample(ratio)
    else:
        pcd_poisson = pcd_acc

    print(f"  Nube Poisson: {len(pcd_poisson.points):,} pts")
    pcd_poisson.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    pcd_poisson.orient_normals_consistent_tangent_plane(50)

    mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_poisson, depth=args.poisson_depth
    )
    if len(densities) > 0:
        mesh_raw.remove_vertices_by_mask(densities < np.quantile(densities, 0.03))
    mesh_raw.compute_vertex_normals()
    mesh_raw.remove_degenerate_triangles()
    print(f"  Poisson mesh: {len(mesh_raw.vertices):,} v, {len(mesh_raw.triangles):,} t")
    print(f"  Tiempo Poisson: {time.time()-t0:.1f}s")

    # === ETAPA 6: Post-procesamiento ===
    print("\n=== ETAPA 6: Post-procesamiento ===")
    t0 = time.time()
    mesh = mesh_raw.filter_smooth_taubin(number_of_iterations=8)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    # Simplify
    nt = len(mesh.triangles)
    if nt > args.target_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # Z crop on mesh
    vm = np.asarray(mesh.vertices)
    z_ok = (vm[:, 2] >= args.z_min) & (vm[:, 2] <= args.z_max)
    if z_ok.sum() > 100:
        keep_idx = np.where(z_ok)[0]
        mesh = mesh.select_by_index(keep_idx)
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    print(f"  Final mesh: {len(mesh.vertices):,} v, {len(mesh.triangles):,} t")
    print(f"  Tiempo post-proc: {time.time()-t0:.1f}s")

    # === Export ===
    mesh_path = out_dir / "room_max_quality.ply"
    cloud_path = out_dir / "room_cloud.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(cloud_path), pcd_acc)

    # Z range
    pts_all = np.asarray(pcd_acc.points)
    z_all = pts_all[:, 2]

    summary = {
        "frames_total": len(pairs),
        "frames_valid": len(pcds_icp),
        "frames_aligned": len(poses),
        "frames_skipped": skipped,
        "loop_closures": len(loops),
        "icp_fitness_mean": round(float(np.mean(fitness_vals)), 4) if fitness_vals else 0,
        "icp_rmse_mean": round(float(np.mean(rmse_vals)), 4) if rmse_vals else 0,
        "z_span": [round(float(z_all.min()), 2), round(float(z_all.max()), 2)],
        "z_range": round(float(z_all.max() - z_all.min()), 2),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "mesh_path": str(mesh_path),
        "cloud_path": str(cloud_path),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== RESUMEN ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
