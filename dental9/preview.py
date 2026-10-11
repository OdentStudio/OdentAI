"""A quick bone surface without the network — something to place the crop box on.

The point is speed, not accuracy: a few seconds after reading, on any machine,
the user sees where the jaws are inside a large field of view and draws a box
around the region of interest. Only that box then goes to the network.

The threshold is found per scan (two-threshold Otsu: air / soft tissue /
bone), not fixed in HU: CBCT is not calibrated, and one clinic's bone is
another's soft tissue.
"""
import json
import os
import time
from typing import Callable

import numpy as np
import SimpleITK as sitk

from . import io as _io
from . import mesh as _mesh
from . import winpath as _winpath

# Coarse on purpose: at 1 mm a 200 mm field is 200³ voxels — a second of
# work and a mesh light enough for the viewport. Jaws are unmistakable at
# this size; nobody is going to plan on this surface.
PREVIEW_SPACING_MM = 1.0
# The smoothed 1 mm grid is kept next to the preview so that a new density
# rebuilds the surface in a fraction of a second, without reading the scan.
VOLUME_FILE = "preview_volume.mha"
# Density is what the user moves: 50 is the automatic threshold, 0 the
# air / soft tissue boundary (skin), 100 the densest bone and enamel.
DEFAULT_DENSITY = 50.0


def _inside(vol: np.ndarray) -> np.ndarray:
    """Voxels inside the field of view: the corners outside the reconstruction
    cylinder are padding at -1000 (or the scanner's own fill value) and would
    drag the histogram."""
    inside = vol[vol > vol.min() + 1]
    if inside.size < 1000:
        inside = vol.ravel()
    # A subsample is plenty for a histogram and keeps it quick on large scans.
    if inside.size > 2_000_000:
        inside = inside[:: inside.size // 2_000_000]
    return inside


def thresholds(vol: np.ndarray) -> dict:
    """Two Otsu thresholds (air / soft tissue / bone) and the top of the range.

    The upper Otsu threshold is the automatic bone threshold. On some scans it
    lands in soft tissue and the preview shows skin instead of the jaws
    (reported 2026-10-11 from outside testing) — the density slider is the way
    out, and these three values are its scale.
    """
    inside = _inside(vol)
    img = sitk.GetImageFromArray(inside.astype(np.float32).reshape(1, 1, -1))
    f = sitk.OtsuMultipleThresholdsImageFilter()
    f.SetNumberOfThresholds(2)
    f.SetNumberOfHistogramBins(256)
    f.Execute(img)
    low, auto = (float(v) for v in f.GetThresholds())
    high = float(np.percentile(inside, 99.5))
    return {"low": low, "auto": auto, "high": max(high, auto + 1.0)}


def bone_threshold(vol: np.ndarray) -> float:
    """The automatic bone threshold: the upper of the two Otsu thresholds."""
    return thresholds(vol)["auto"]


def density_to_threshold(density: float, t: dict) -> float:
    """0..100 onto the scan's own scale: 0 = low, 50 = auto, 100 = high.

    Relative rather than in HU: CBCT is not calibrated, and the same number
    would mean bone on one scanner and soft tissue on another.
    """
    d = min(max(float(density), 0.0), 100.0)
    if d <= 50.0:
        return t["low"] + (t["auto"] - t["low"]) * d / 50.0
    return t["auto"] + (t["high"] - t["auto"]) * (d - 50.0) / 50.0


def _surface(vol: np.ndarray, grid: sitk.Image, thr: float, outdir: str) -> int:
    pd = _mesh.build_surface(vol, grid, thr, taubin=10, pass_band=0.1, decimate=0.5)
    _pts, tris = _mesh.write_stl(pd, os.path.join(outdir, "preview.stl"))
    return int(tris)


def rebuild_preview(outdir: str, density: float,
                    log: Callable[[str], None] = print) -> dict:
    """A new surface at another density from the kept 1 mm grid: no scan read."""
    t0 = time.time()
    # The json is read and written by Python, which takes any path; only the
    # native reader and writer go through winpath.
    with open(os.path.join(outdir, "preview.json"), encoding="utf-8") as f:
        info = json.load(f)
    vol_path, cleanup_in = _winpath.readable(os.path.join(outdir, VOLUME_FILE))
    try:
        grid = sitk.ReadImage(vol_path)
    finally:
        cleanup_in()
    vol = sitk.GetArrayFromImage(grid).astype(np.float32)
    thr = density_to_threshold(density, info["thresholds"])
    stage, finish_out = _winpath.writable(outdir)
    try:
        tris = _surface(vol, grid, thr, stage)
    finally:
        finish_out()
    info.update(density=float(density), threshold=thr, triangles=tris,
                seconds=round(time.time() - t0, 2))
    with open(os.path.join(outdir, "preview.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log(f"  preview at density {density:.0f} (threshold {thr:.0f}): "
        f"{tris} triangles in {info['seconds']} s -> {outdir}")
    return info


def make_preview(path: str, outdir: str, log: Callable[[str], None] = print,
                 density: float = DEFAULT_DENSITY) -> dict:
    real_path = path
    path, cleanup_in = _winpath.readable(path)
    outdir_real = outdir
    outdir, finish_out = _winpath.writable(outdir)
    try:
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        log(f"reading: {real_path}")
        src = _io.read_volume(path)
        sp = src.GetSpacing()
        fov = [round(n * v, 1) for n, v in zip(src.GetSize(), sp)]
        log(f"  volume {src.GetSize()}, spacing {tuple(round(s, 3) for s in sp)} mm, "
            f"field of view {fov[0]}x{fov[1]}x{fov[2]} mm")
        lo, hi = _io.physical_bounds(src)

        # Smooth before downsampling so the coarse grid does not alias noise
        # into specks — CBCT is noisy, and the specks would hide the jaw.
        sm = sitk.SmoothingRecursiveGaussian(sitk.Cast(src, sitk.sitkFloat32),
                                             PREVIEW_SPACING_MM * 0.6)
        grid = _io.to_training_grid(sm, PREVIEW_SPACING_MM)
        vol = sitk.GetArrayFromImage(grid).astype(np.float32)
        t = thresholds(vol)
        thr = density_to_threshold(density, t)
        log(f"  bone threshold {thr:.0f} at density {density:.0f} (Otsu {t['auto']:.0f}, "
            f"scale {t['low']:.0f}..{t['high']:.0f}), preview grid {vol.shape}")
        # int16 is plenty for a 1 mm preview and halves the file.
        sitk.WriteImage(sitk.Cast(sitk.Round(grid), sitk.sitkInt16),
                        os.path.join(outdir, VOLUME_FILE))

        tris = _surface(vol, grid, thr, outdir)
        info = {"input": real_path, "size": list(src.GetSize()), "spacing": list(sp),
                "field_of_view_mm": fov, "threshold": thr, "density": float(density),
                "thresholds": t, "triangles": int(tris),
                "bounds_min": [float(v) for v in lo], "bounds_max": [float(v) for v in hi],
                "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(outdir, "preview.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        log(f"  preview: {tris} triangles in {info['seconds']} s -> {outdir_real}")
        return info
    finally:
        finish_out()
        cleanup_in()
