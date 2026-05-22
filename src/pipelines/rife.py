"""RIFE-based frame interpolation as a pipeline.

Wraps the upstream ECCV2022-RIFE inference: given the target camera's t0 and t1 frames,
produce a frame at `target_alpha` between them.

Expected layout (repo paths):
    baseline/baseline_files/baseline_ensemble/
        train_log/{IFNet_HDv3.py, RIFE_HDv3.py, refine.py, flownet.pkl}
        ECCV2022-RIFE/  (cloned from https://github.com/hzwer/ECCV2022-RIFE)

The first call loads the model (cached in this module).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..data import Sample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "baseline" / "baseline_files" / "baseline_ensemble"
_RIFE_REPO = _BASELINE_DIR / "ECCV2022-RIFE"

_MODEL = None
_DEVICE = None


def _ensure_paths_on_sys_path() -> None:
    for p in (str(_RIFE_REPO), str(_BASELINE_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL

    if not _RIFE_REPO.exists():
        raise FileNotFoundError(
            f"Upstream RIFE not found at {_RIFE_REPO}. Clone it first:\n"
            f"  cd {_BASELINE_DIR}\n"
            f"  git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git"
        )

    _ensure_paths_on_sys_path()

    import torch
    from train_log.RIFE_HDv3 import Model  # type: ignore

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Model()
    m.load_model(str(_BASELINE_DIR / "train_log"), -1)
    m.eval()
    _MODEL = m
    return _MODEL


def _to_tensor(img: np.ndarray):
    import torch
    arr = np.ascontiguousarray(img)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(_DEVICE)


def _pad_to_multiple(t, multiple: int = 64):
    import torch.nn.functional as F
    h, w = t.shape[-2:]
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    return F.pad(t, (0, pw - w, 0, ph - h)), (h, w)


def infer_pair(img0: np.ndarray, img1: np.ndarray, timestep: float = 0.5) -> np.ndarray:
    """Interpolate a frame between img0 and img1 at the given timestep (in [0, 1])."""
    model = _load_model()
    import torch

    t0 = _to_tensor(img0)
    t1 = _to_tensor(img1)
    t0p, (h, w) = _pad_to_multiple(t0)
    t1p, _ = _pad_to_multiple(t1)
    with torch.no_grad():
        out = model.inference(t0p, t1p, timestep=timestep)
    arr = out[0, :, :h, :w].permute(1, 2, 0).cpu().numpy() * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def render(sample: Sample) -> np.ndarray:
    cam = sample.target_camera
    img0 = sample.load_image("t0", cam)
    img1 = sample.load_image("t1", cam)
    alpha = float(sample.target_alpha)
    return infer_pair(img0, img1, timestep=alpha)
