"""Organizer baseline reproduction: RIFE + DIS optical flow blend.

For the target camera, compute DIS flow between t0 and t1, build a flow-warped interpolant,
blend with the temporal mean, then average with RIFE's output. Mirrors baseline_flow.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..data import Sample
from .rife import infer_pair


def _warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(
        img,
        (xs + dx).astype(np.float32),
        (ys + dy).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


_DIS = None


def _dis() -> "cv2.DISOpticalFlow":
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS


def render(sample: Sample, w_dis: float = 0.55, w_mean: float = 0.45, w_rife: float = 0.40) -> np.ndarray:
    cam = sample.target_camera
    alpha = float(sample.target_alpha)
    img0 = sample.load_image("t0", cam)
    img1 = sample.load_image("t1", cam)

    mean = (img0.astype(np.float32) + img1.astype(np.float32)) * 0.5

    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    flow = _dis().calc(gray0, gray1, None)
    dx, dy = flow[..., 0], flow[..., 1]
    w0 = _warp(img0, -alpha * dx, -alpha * dy)
    w1 = _warp(img1, (1.0 - alpha) * dx, (1.0 - alpha) * dy)
    dis_warped = (1.0 - alpha) * w0.astype(np.float32) + alpha * w1.astype(np.float32)
    dis_blend = (w_dis * dis_warped + w_mean * mean).clip(0, 255)

    rife = infer_pair(img0, img1, timestep=alpha).astype(np.float32)

    return ((1.0 - w_rife) * dis_blend + w_rife * rife).clip(0, 255).astype(np.uint8)
