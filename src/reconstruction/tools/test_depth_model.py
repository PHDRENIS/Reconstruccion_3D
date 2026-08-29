#!/usr/bin/env python3
"""
Run depth completion model on RGB + depth + mask and export previews.
This is a functional test, not a GT evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test depth completion model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth-raw", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--out-dir", default="output/tests")
    parser.add_argument("--max-depth", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise SystemExit(f"No se pudo leer RGB: {args.rgb}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0

    depth_raw = np.fromfile(args.depth_raw, dtype=np.uint16)
    expected = args.width * args.height
    if depth_raw.size != expected:
        raise SystemExit(
            f"Depth size mismatch: {depth_raw.size} vs expected {expected}"
        )

    depth_raw = depth_raw.reshape(args.height, args.width)
    depth_m = depth_raw.astype(np.float32) * args.depth_scale
    mask = (depth_m > 0) & (depth_m < args.max_depth)

    depth_norm = depth_m.copy()
    depth_norm[~mask] = 0

    input_tensor = np.concatenate(
        [rgb, depth_norm[..., None], mask.astype(np.float32)[..., None]], axis=2
    )
    input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).unsqueeze(0)

    model = DepthCompletionModel()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        pred = model(input_tensor).squeeze(0).squeeze(0).numpy()

    pred_vis = pred.copy()
    pred_vis = np.clip(pred_vis / args.max_depth, 0, 1) * 255.0
    pred_vis = pred_vis.astype(np.uint8)
    pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

    raw_vis = depth_norm.copy()
    raw_vis = np.clip(raw_vis / args.max_depth, 0, 1) * 255.0
    raw_vis = raw_vis.astype(np.uint8)
    raw_vis = cv2.applyColorMap(raw_vis, cv2.COLORMAP_JET)

    cv2.imwrite(str(out_dir / "depth_raw_colormap.png"), raw_vis)
    cv2.imwrite(str(out_dir / "depth_pred_colormap.png"), pred_vis)

    hole_ratio = float((~mask).sum()) / float(mask.size)
    pred_nonzero = float((pred > 0).sum()) / float(pred.size)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "rgb": str(Path(args.rgb).resolve()),
        "depth_raw": str(Path(args.depth_raw).resolve()),
        "depth_scale": args.depth_scale,
        "hole_ratio": hole_ratio,
        "pred_nonzero_ratio": pred_nonzero,
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "pred_mean": float(pred.mean()),
    }

    with open(out_dir / "depth_model_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Depth raw preview:", out_dir / "depth_raw_colormap.png")
    print("Depth pred preview:", out_dir / "depth_pred_colormap.png")
    print("Summary:", out_dir / "depth_model_summary.json")


if __name__ == "__main__":
    main()
