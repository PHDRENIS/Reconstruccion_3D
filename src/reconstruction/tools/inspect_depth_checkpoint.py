#!/usr/bin/env python3
"""
Inspect a depth model checkpoint and report input/output characteristics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect depth checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"No existe: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        print("Checkpoint type:", type(state))
        return

    keys = list(state.keys())
    print("Top-level keys:", keys[:10])

    # If already a state dict, use it
    state_dict = state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state_dict = state["model_state_dict"]

    sd_keys = list(state_dict.keys())
    print("State dict size:", len(sd_keys))
    print("State dict sample:", sd_keys[:10])

    # First conv weight
    first_conv = None
    for key in sd_keys:
        weight = state_dict[key]
        if hasattr(weight, "ndim") and weight.ndim == 4:
            first_conv = (key, tuple(weight.shape))
            break
    if first_conv:
        print("First conv:", first_conv[0], first_conv[1])

    # Output conv if present
    final_keys = [k for k in sd_keys if "final" in k or "output" in k]
    if final_keys:
        print("Final keys sample:", final_keys[:5])


if __name__ == "__main__":
    main()
