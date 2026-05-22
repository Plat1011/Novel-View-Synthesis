"""Sample loading: meta.json, images, lidar."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

CAMERA_NAMES = ("front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear")
TIMESTEPS = ("t0", "t1")


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str
    distortion_coeffs: np.ndarray

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["width"]),
            height=int(d["height"]),
            distortion_model=str(d.get("distortion_model", "none")),
            distortion_coeffs=np.asarray(d.get("distortion_coeffs", []), dtype=np.float64),
        )

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class Sample:
    sample_dir: Path
    meta: dict
    intrinsics: dict[str, Intrinsics]
    poses_c2w: dict[str, dict[str, np.ndarray]]  # {"t0"|"t1"|"target": {cam: 4x4}}

    @property
    def sample_id(self) -> str:
        return self.meta["sample_id"]

    @property
    def target_camera(self) -> str:
        return self.meta["target_camera"]

    @property
    def delta_s(self) -> float:
        return float(self.meta["delta_s"])

    @property
    def target_alpha(self) -> float:
        """Position of target time within [t0, t1], in [0, 1]."""
        ts = self.meta.get("timestamps_ns")
        if ts is None or "target" not in ts:
            return 0.5
        t0, t1, tt = float(ts["t0"]), float(ts["t1"]), float(ts["target"])
        if t1 == t0:
            return 0.5
        return (tt - t0) / (t1 - t0)

    def image_path(self, step: str, camera: str) -> Path:
        return self.sample_dir / "input" / step / f"{camera}.jpg"

    def load_image(self, step: str, camera: str) -> np.ndarray:
        with Image.open(self.image_path(step, camera)) as im:
            return np.asarray(im.convert("RGB"))

    def load_target_gt(self) -> np.ndarray | None:
        p = self.sample_dir / "target" / f"{self.target_camera}.jpg"
        if not p.exists():
            return None
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"))

    def load_lidar(self) -> dict[str, np.ndarray]:
        with np.load(self.sample_dir / "input" / "lidar.npz") as z:
            out = {k: z[k] for k in z.files}
        return out

    def pose(self, step: str, camera: str | None = None) -> np.ndarray:
        cam = camera or self.target_camera
        return np.asarray(self.poses_c2w[step][cam], dtype=np.float64)


def load_sample(sample_dir: str | Path) -> Sample:
    sample_dir = Path(sample_dir)
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    intrinsics = {name: Intrinsics.from_dict(meta["intrinsics"][name]) for name in CAMERA_NAMES}
    poses = {
        step: {cam: np.asarray(mat, dtype=np.float64) for cam, mat in step_poses.items()}
        for step, step_poses in meta["poses_c2w"].items()
    }
    return Sample(sample_dir=sample_dir, meta=meta, intrinsics=intrinsics, poses_c2w=poses)


def iter_samples(root: str | Path):
    root = Path(root)
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield load_sample(p)
