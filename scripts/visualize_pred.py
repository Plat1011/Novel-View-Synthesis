"""Render t0, prediction, t1, gt side by side with labels for one sample.

Usage:
    python scripts/visualize_pred.py path/to/train/<sample_id> [--pipeline inverse_warp]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_sample
from src.geometry import psnr
from src.pipelines.lidar_warp import render as render_forward_warp
from src.pipelines.lidar_inverse_warp import render as render_inverse_warp


def temporal_mean(sample):
    cam = sample.target_camera
    img0 = sample.load_image("t0", cam).astype(np.float32)
    img1 = sample.load_image("t1", cam).astype(np.float32)
    a = sample.target_alpha
    return ((1.0 - a) * img0 + a * img1).clip(0, 255).astype(np.uint8)


def _rife(sample):
    from src.pipelines.rife import render as r
    return r(sample)


def _rife_dis(sample):
    from src.pipelines.rife_dis import render as r
    return r(sample)


PIPELINES = {
    "temporal_mean": temporal_mean,
    "forward_warp": lambda s: render_forward_warp(s, splat_radius=1),
    "inverse_warp": lambda s: render_inverse_warp(s, splat_radius=1),
    "rife": _rife,
    "rife_dis": _rife_dis,
}


def label(img: np.ndarray, text: str, bg=(0, 0, 0), fg=(255, 255, 255)) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]
    pad = 8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + 2 * pad), bg, -1)
    cv2.putText(img, text, (pad, th + pad - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--pipeline", default="inverse_warp", choices=list(PIPELINES))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sample = load_sample(args.sample_dir)
    cam = sample.target_camera
    img0 = sample.load_image("t0", cam)
    img1 = sample.load_image("t1", cam)
    gt = sample.load_target_gt()
    pred = PIPELINES[args.pipeline](sample)

    psnr_pred = psnr(pred, gt) if gt is not None else float("nan")
    psnr_mean = psnr(temporal_mean(sample), gt) if gt is not None else float("nan")

    a = sample.target_alpha
    panels = [
        label(img0, f"t0  (camera: {cam})"),
        label(pred, f"prediction [{args.pipeline}]  alpha={a:.2f}  PSNR={psnr_pred:.2f}"),
        label(img1, "t1"),
    ]
    if gt is not None:
        panels.append(label(gt, f"GT (target)  baseline mean PSNR={psnr_mean:.2f}"))

    grid = np.concatenate(panels, axis=0)
    out = args.out or (ROOT / "outputs" / "viz" / f"{sample.sample_id}_{args.pipeline}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out, quality=92)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
