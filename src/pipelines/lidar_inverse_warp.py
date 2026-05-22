"""Inverse warp: lidar gives target-view depth, each target pixel samples its colour from sources.

For each target pixel:
    1. project the lidar to the target view -> dense depth (z-buffer, optional splat)
    2. unproject target pixel + depth -> 3D world point
    3. project this 3D point into each source view -> sub-pixel coords
    4. bilinear sample source RGB
    5. blend t0/t1 by alpha (the target's position within [t0, t1])

Holes (target pixels without lidar coverage) are filled with the temporal mean of the target
camera's two input frames.
"""
from __future__ import annotations

import numpy as np

from ..data import Sample
from ..geometry import (
    Intrinsics,
    cam_to_world,
    lidar_depth_map,
    project_points,
    world_to_cam,
)


def _unproject_target_depth(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Pinhole only. Returns (H,W,3) camera-frame 3D points for valid depth pixels (NaN elsewhere)."""
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    K_inv = np.linalg.inv(intr.K)
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1)  # H,W,3
    dirs = pix @ K_inv.T
    return dirs * depth[..., None]


def _bilinear_sample(img: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """img HxWx3 uint8, uv (...,2) float coords, valid (...) bool. Returns (rgb..., valid_out)."""
    H, W = img.shape[:2]
    u = uv[..., 0]
    v = uv[..., 1]
    in_bounds = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    ok = valid & in_bounds & np.isfinite(u) & np.isfinite(v)

    out = np.zeros((*uv.shape[:-1], 3), dtype=np.float32)
    if not np.any(ok):
        return out, ok

    uu = u[ok].astype(np.float32)
    vv = v[ok].astype(np.float32)
    u0 = np.floor(uu).astype(np.int32)
    v0 = np.floor(vv).astype(np.int32)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    du = (uu - u0).astype(np.float32)
    dv = (vv - v0).astype(np.float32)

    img_f = img.astype(np.float32)
    c00 = img_f[v0, u0]
    c10 = img_f[v0, u1]
    c01 = img_f[v1, u0]
    c11 = img_f[v1, u1]
    w00 = ((1.0 - du) * (1.0 - dv))[:, None]
    w10 = (du * (1.0 - dv))[:, None]
    w01 = ((1.0 - du) * dv)[:, None]
    w11 = (du * dv)[:, None]
    sampled = w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11

    out[ok] = sampled
    return out, ok


def _sample_from_source(
    pts_world: np.ndarray,  # H,W,3
    valid: np.ndarray,  # H,W
    src_img: np.ndarray,
    src_intr: Intrinsics,
    src_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_pts = pts_world.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    pts_cam = world_to_cam(flat_pts, src_c2w)
    pixels, in_view = project_points(pts_cam, src_intr)
    ok_flat = flat_valid & in_view
    H, W = valid.shape
    uv = pixels.reshape(H, W, 2)
    ok = ok_flat.reshape(H, W)
    rgb, ok = _bilinear_sample(src_img, uv, ok)
    return rgb, ok


def render(sample: Sample, splat_radius: int = 1) -> np.ndarray:
    cam = sample.target_camera
    intr = sample.intrinsics[cam]
    pose_tgt = sample.pose("target", cam)

    lidar = sample.load_lidar()
    xyz = lidar["xyz"]

    depth_tgt = lidar_depth_map(xyz, pose_tgt, intr, splat_radius=splat_radius)
    has_depth = np.isfinite(depth_tgt)

    pts_cam = _unproject_target_depth(np.where(has_depth, depth_tgt, 0.0), intr)
    pts_world = cam_to_world(pts_cam.reshape(-1, 3), pose_tgt).reshape(*pts_cam.shape)

    img0 = sample.load_image("t0", cam)
    img1 = sample.load_image("t1", cam)

    rgb0, ok0 = _sample_from_source(pts_world, has_depth, img0, intr, sample.pose("t0", cam))
    rgb1, ok1 = _sample_from_source(pts_world, has_depth, img1, intr, sample.pose("t1", cam))

    alpha = sample.target_alpha
    out = np.zeros_like(rgb0)
    weight = np.zeros(rgb0.shape[:2], dtype=np.float32)

    both = ok0 & ok1
    only0 = ok0 & ~ok1
    only1 = ok1 & ~ok0

    out[both] = (1.0 - alpha) * rgb0[both] + alpha * rgb1[both]
    weight[both] = 1.0
    out[only0] = rgb0[only0]
    weight[only0] = 1.0
    out[only1] = rgb1[only1]
    weight[only1] = 1.0

    holes = weight == 0
    if holes.any():
        mean = ((img0.astype(np.float32) + img1.astype(np.float32)) * 0.5)
        out = np.where(holes[..., None], mean, out)

    return np.clip(out, 0, 255).astype(np.uint8)
