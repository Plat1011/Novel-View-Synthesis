"""First real pipeline: blend t0+t1 of the target camera via lidar-warped depth.

Both t0 and t1 frames of the target camera are warped into the target pose using the lidar
point cloud. The two warps are then blended (alpha = position of target time within [t0, t1]),
holes are filled with simple in-painting / temporal mean.
"""
from __future__ import annotations

import numpy as np

from ..data import Sample
from ..geometry import lidar_warp


def render(sample: Sample, splat_radius: int = 1) -> np.ndarray:
    cam = sample.target_camera
    intr_src = sample.intrinsics[cam]
    intr_tgt = sample.intrinsics[cam]  # same camera, same intrinsics
    pose_tgt = sample.pose("target", cam)

    lidar = sample.load_lidar()
    xyz = lidar["xyz"]

    img0 = sample.load_image("t0", cam)
    img1 = sample.load_image("t1", cam)

    w0, m0, _ = lidar_warp(xyz, img0, intr_src, sample.pose("t0", cam), intr_tgt, pose_tgt, splat_radius)
    w1, m1, _ = lidar_warp(xyz, img1, intr_src, sample.pose("t1", cam), intr_tgt, pose_tgt, splat_radius)

    alpha = sample.target_alpha
    blended = np.zeros_like(w0, dtype=np.float32)
    weight = np.zeros(w0.shape[:2], dtype=np.float32)

    # Two-source blend where both available
    both = m0 & m1
    only0 = m0 & ~m1
    only1 = m1 & ~m0

    blended[both] = (1.0 - alpha) * w0[both].astype(np.float32) + alpha * w1[both].astype(np.float32)
    weight[both] = 1.0
    blended[only0] = w0[only0]
    weight[only0] = 1.0
    blended[only1] = w1[only1]
    weight[only1] = 1.0

    out = blended.astype(np.uint8)
    holes = weight == 0
    if holes.any():
        mean = ((img0.astype(np.float32) + img1.astype(np.float32)) * 0.5).astype(np.uint8)
        out = np.where(holes[..., None], mean, out)
    return out
