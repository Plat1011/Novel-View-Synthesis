"""Projections, depth from lidar, depth-based warping."""
from __future__ import annotations

import cv2
import numpy as np

from .data import Intrinsics


def world_to_cam(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """(N,3) world points -> (N,3) in camera frame using OpenCV camera axes (x right, y down, z forward)."""
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    return points_world @ R.T + t


def project_points(points_cam: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) camera-frame points to (N,2) pixels using the camera's distortion model.

    Returns (pixels Nx2 float, valid mask N bool). 'valid' is True when point is in front of camera
    (z>0) and projects inside the image. Distortion is applied per the intrinsics.
    """
    n = points_cam.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    in_front = points_cam[:, 2] > 1e-6
    pts = points_cam[in_front]

    K = intr.K
    D = intr.distortion_coeffs.astype(np.float64).reshape(-1)
    model = intr.distortion_model.lower()

    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)

    if model in ("fisheye", "equidistant", "kannala_brandt", "kannala-brandt"):
        # OpenCV fisheye expects (1,N,3) input
        d_fish = D[:4] if D.size >= 4 else np.pad(D, (0, 4 - D.size))
        uv, _ = cv2.fisheye.projectPoints(pts.reshape(1, -1, 3), rvec, tvec, K, d_fish.reshape(1, 4))
        uv = uv.reshape(-1, 2)
    elif model in ("none", "", "pinhole"):
        uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, K, None)
        uv = uv.reshape(-1, 2)
    else:
        # plumb_bob / rad_tan / opencv / brown — radial+tangential
        uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, K, D)
        uv = uv.reshape(-1, 2)

    pixels = np.full((n, 2), np.nan, dtype=np.float64)
    pixels[in_front] = uv

    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < intr.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < intr.height)
    )
    valid = in_front & in_bounds & np.isfinite(pixels[:, 0]) & np.isfinite(pixels[:, 1])
    return pixels, valid


def lidar_depth_map(
    xyz_world: np.ndarray,
    c2w: np.ndarray,
    intr: Intrinsics,
    splat_radius: int = 0,
) -> np.ndarray:
    """Project a world-frame point cloud into the camera and z-buffer it into a depth map.

    Returns (H, W) float32; pixels with no point are NaN. splat_radius>0 paints a small disc per
    point to densify; in case of overlap keeps the nearest depth.
    """
    pts_cam = world_to_cam(xyz_world, c2w)
    pixels, valid = project_points(pts_cam, intr)
    z = pts_cam[:, 2]

    H, W = intr.height, intr.width
    depth = np.full((H, W), np.inf, dtype=np.float32)

    px = pixels[valid]
    zz = z[valid].astype(np.float32)
    u = np.round(px[:, 0]).astype(np.int32)
    v = np.round(px[:, 1]).astype(np.int32)

    if splat_radius <= 0:
        # Sort by depth desc so nearest wins in the final assignment
        order = np.argsort(-zz)
        u, v, zz = u[order], v[order], zz[order]
        depth[v, u] = np.minimum(depth[v, u], zz)
    else:
        r = splat_radius
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                if du * du + dv * dv > r * r:
                    continue
                uu = u + du
                vv = v + dv
                m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                if not np.any(m):
                    continue
                np.minimum.at(depth, (vv[m], uu[m]), zz[m])

    depth[~np.isfinite(depth)] = np.nan
    return depth


def cam_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    R, t = c2w[:3, :3], c2w[:3, 3]
    return points_cam @ R.T + t


def lidar_warp(
    xyz_world: np.ndarray,
    src_img: np.ndarray,
    src_intr: Intrinsics,
    src_c2w: np.ndarray,
    tgt_intr: Intrinsics,
    tgt_c2w: np.ndarray,
    splat_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp source RGB into target view via lidar correspondences.

    For each lidar point: sample colour at its projection in src view, splat at its projection in
    target view with z-buffer. Returns (rgb HxWx3 uint8, mask HxW bool, depth HxW float32 with NaN
    where mask is False).
    """
    src_pts_cam = world_to_cam(xyz_world, src_c2w)
    src_px, src_valid = project_points(src_pts_cam, src_intr)
    tgt_pts_cam = world_to_cam(xyz_world, tgt_c2w)
    tgt_px, tgt_valid = project_points(tgt_pts_cam, tgt_intr)

    both = src_valid & tgt_valid
    if not np.any(both):
        H, W = tgt_intr.height, tgt_intr.width
        return (
            np.zeros((H, W, 3), dtype=np.uint8),
            np.zeros((H, W), dtype=bool),
            np.full((H, W), np.nan, dtype=np.float32),
        )

    sx = np.clip(np.round(src_px[both, 0]).astype(np.int32), 0, src_intr.width - 1)
    sy = np.clip(np.round(src_px[both, 1]).astype(np.int32), 0, src_intr.height - 1)
    colors = src_img[sy, sx]  # N,3 uint8

    tx = np.round(tgt_px[both, 0]).astype(np.int32)
    ty = np.round(tgt_px[both, 1]).astype(np.int32)
    z = tgt_pts_cam[both, 2].astype(np.float32)

    H, W = tgt_intr.height, tgt_intr.width
    out_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    out_depth = np.full((H, W), np.inf, dtype=np.float32)
    out_mask = np.zeros((H, W), dtype=bool)

    r = max(0, int(splat_radius))
    offsets = [(0, 0)] if r == 0 else [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r]

    # Farthest first so closest writes last and wins in the final assignment.
    order = np.argsort(-z)
    tx, ty, z, colors = tx[order], ty[order], z[order], colors[order]

    for du, dv in offsets:
        uu = tx + du
        vv = ty + dv
        m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        if not np.any(m):
            continue
        uu_m = uu[m]
        vv_m = vv[m]
        z_m = z[m]
        c_m = colors[m]
        closer = z_m < out_depth[vv_m, uu_m]
        if not np.any(closer):
            continue
        uu_m = uu_m[closer]
        vv_m = vv_m[closer]
        z_m = z_m[closer]
        c_m = c_m[closer]
        out_depth[vv_m, uu_m] = z_m
        out_rgb[vv_m, uu_m] = c_m
        out_mask[vv_m, uu_m] = True

    out_depth[~out_mask] = np.nan
    return out_rgb, out_mask, out_depth


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def normalized_score(psnr_db: float) -> float:
    return max(0.0, min(100.0, (max(10.0, min(30.0, psnr_db)) - 10.0) / 20.0 * 100.0))
