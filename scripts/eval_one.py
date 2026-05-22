"""Run a few candidate predictors on a single train sample and report PSNR vs GT.

Usage:
    python scripts/eval_one.py path/to/dataset/train/<sample_id>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_sample
from src.geometry import psnr, normalized_score
from src.pipelines.lidar_warp import render as render_lidar_warp


def temporal_mean(sample) -> np.ndarray:
    cam = sample.target_camera
    img0 = sample.load_image("t0", cam).astype(np.float32)
    img1 = sample.load_image("t1", cam).astype(np.float32)
    alpha = sample.target_alpha
    out = (1.0 - alpha) * img0 + alpha * img1
    return out.clip(0, 255).astype(np.uint8)


def report(name: str, pred: np.ndarray, gt: np.ndarray, dt: float) -> None:
    p = psnr(pred, gt)
    s = normalized_score(p)
    print(f"  {name:20s}  PSNR={p:6.2f} dB  score={s:5.1f}  time={dt:5.2f}s")


def main(sample_dir: str) -> None:
    sample = load_sample(sample_dir)
    gt = sample.load_target_gt()
    if gt is None:
        print(f"[!] No GT in {sample_dir} — cannot evaluate. Pick a train sample.")
        sys.exit(1)

    print(f"sample: {sample.sample_id}")
    print(f"  target_camera: {sample.target_camera}")
    print(f"  delta_s: {sample.delta_s}   alpha: {sample.target_alpha:.3f}")
    print(f"  gt shape: {gt.shape}")

    out_dir = ROOT / "outputs" / "eval_one" / sample.sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gt).save(out_dir / "gt.jpg", quality=95)

    candidates = {
        "temporal_mean": temporal_mean,
        "lidar_warp_r1": lambda s: render_lidar_warp(s, splat_radius=1),
        "lidar_warp_r2": lambda s: render_lidar_warp(s, splat_radius=2),
    }

    print("\npredictions:")
    for name, fn in candidates.items():
        t = time.time()
        pred = fn(sample)
        dt = time.time() - t
        report(name, pred, gt, dt)
        Image.fromarray(pred).save(out_dir / f"{name}.jpg", quality=95)

    print(f"\nartifacts in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
