"""Build a submission from a test split using a chosen render function.

Layout produced:
    <out_dir>/<sample_id>/pred.jpg

Usage as a library:
    from src.submit import build_submission
    from src.pipelines.lidar_warp import render
    build_submission("dataset/test", "submission", render)

CLI:
    python -m src.submit --test-dir dataset/test --out submission --pipeline lidar_warp
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .data import Sample, load_sample


RenderFn = Callable[[Sample], np.ndarray]


def _save_jpg(arr: np.ndarray, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=quality, subsampling=1)


def build_submission(
    test_dir: str | Path,
    out_dir: str | Path,
    render: RenderFn,
    quality: int = 95,
    skip_existing: bool = True,
) -> dict:
    """Run `render` on every sample in `test_dir`, write pred.jpg, return per-sample timings."""
    test_dir = Path(test_dir)
    out_dir = Path(out_dir)
    samples = sorted(p for p in test_dir.iterdir() if p.is_dir())
    stats = {"n": len(samples), "rendered": 0, "skipped": 0, "failed": [], "total_s": 0.0}

    t0 = time.time()
    for sample_dir in tqdm(samples, desc="render"):
        pred_path = out_dir / sample_dir.name / "pred.jpg"
        if skip_existing and pred_path.exists():
            stats["skipped"] += 1
            continue
        try:
            sample = load_sample(sample_dir)
            pred = render(sample)
            _save_jpg(pred, pred_path, quality=quality)
            stats["rendered"] += 1
        except Exception as e:
            stats["failed"].append((sample_dir.name, repr(e)))

    stats["total_s"] = time.time() - t0
    return stats


def zip_submission(out_dir: str | Path, zip_path: str | Path) -> Path:
    out_dir = Path(out_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(out_dir.rglob("*.jpg")):
            zf.write(f, arcname=f.relative_to(out_dir).as_posix())
    return zip_path


def _resolve_pipeline(name: str) -> RenderFn:
    """Resolve a pipeline name like 'lidar_warp' or 'pkg.module:func'."""
    if ":" in name:
        module_name, func_name = name.split(":", 1)
    else:
        module_name, func_name = f"src.pipelines.{name}", "render"
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--out", default=Path("submission"), type=Path)
    ap.add_argument("--zip", default=Path("submission.zip"), type=Path)
    ap.add_argument("--pipeline", default="lidar_warp", help="pipeline module name in src.pipelines, or 'pkg.mod:func'")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--no-skip", action="store_true", help="re-render even if pred.jpg exists")
    ap.add_argument("--clean", action="store_true", help="wipe out dir before rendering")
    args = ap.parse_args()

    render = _resolve_pipeline(args.pipeline)

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    stats = build_submission(args.test_dir, args.out, render, quality=args.quality, skip_existing=not args.no_skip)
    print(f"rendered={stats['rendered']}  skipped={stats['skipped']}  failed={len(stats['failed'])}  total={stats['total_s']:.1f}s")
    for name, err in stats["failed"][:10]:
        print(f"  FAIL {name}: {err}")
    if stats["failed"]:
        print(f"  ... ({len(stats['failed'])} total)")

    zip_path = zip_submission(args.out, args.zip)
    print(f"submission archive: {zip_path}  ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
