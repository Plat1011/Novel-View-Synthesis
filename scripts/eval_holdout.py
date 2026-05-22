"""Evaluate several pipelines on N random train samples and report mean PSNR / score.

Usage:
    python scripts/eval_holdout.py --train-dir final_dataset_v5_participants_small/train --n 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_sample
from src.geometry import psnr, normalized_score
from src.pipelines.lidar_warp import render as render_forward_warp
from src.pipelines.lidar_inverse_warp import render as render_inverse_warp


def temporal_mean(sample):
    cam = sample.target_camera
    img0 = sample.load_image("t0", cam).astype(np.float32)
    img1 = sample.load_image("t1", cam).astype(np.float32)
    a = sample.target_alpha
    return ((1.0 - a) * img0 + a * img1).clip(0, 255).astype(np.uint8)


def _rife_render(sample):
    from src.pipelines.rife import render as r
    return r(sample)


def _rife_dis_render(sample):
    from src.pipelines.rife_dis import render as r
    return r(sample)


PIPELINES = {
    "temporal_mean": temporal_mean,
    "forward_warp_r1": lambda s: render_forward_warp(s, splat_radius=1),
    "inverse_warp_r1": lambda s: render_inverse_warp(s, splat_radius=1),
    "rife": _rife_render,
    "rife_dis": _rife_dis_render,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these pipeline names")
    args = ap.parse_args()

    samples = sorted(p for p in args.train_dir.iterdir() if p.is_dir())
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    samples = samples[: args.n]

    pipes = PIPELINES
    if args.only:
        pipes = {k: v for k, v in PIPELINES.items() if k in args.only}

    results = {name: {"psnr": [], "time": []} for name in pipes}
    for sd in tqdm(samples, desc="samples"):
        sample = load_sample(sd)
        gt = sample.load_target_gt()
        if gt is None:
            continue
        for name, fn in pipes.items():
            t = time.time()
            pred = fn(sample)
            dt = time.time() - t
            results[name]["psnr"].append(psnr(pred, gt))
            results[name]["time"].append(dt)

    print(f"\nresults over {len(samples)} samples (seed={args.seed})\n")
    print(f"{'pipeline':22s}  {'mean PSNR':>10s}  {'median':>7s}  {'mean score':>10s}  {'time/s':>8s}")
    for name, st in results.items():
        ps = np.asarray(st["psnr"])
        if ps.size == 0:
            continue
        sc = np.asarray([normalized_score(p) for p in ps])
        t = np.asarray(st["time"])
        print(f"{name:22s}  {ps.mean():>10.3f}  {np.median(ps):>7.3f}  {sc.mean():>10.2f}  {t.mean():>8.2f}")


if __name__ == "__main__":
    main()
