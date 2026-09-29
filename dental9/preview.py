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


def bone_threshold(vol: np.ndarray) -> float:
    """The upper of the two Otsu thresholds, computed only inside the field of
    view: the corners outside the reconstruction cylinder are padding at -1000
    (or the scanner's own fill value) and would drag the histogram."""
    inside = vol[vol > vol.min() + 1]
    if inside.size < 1000:
        inside = vol.ravel()
    # A subsample is plenty for a histogram and keeps it quick on large scans.
    if inside.size > 2_000_000:
        inside = inside[:: inside.size // 2_000_000]
    img = sitk.GetImageFromArray(inside.astype(np.float32).reshape(1, 1, -1))
    f = sitk.OtsuMultipleThresholdsImageFilter()
    f.SetNumberOfThresholds(2)
    f.SetNumberOfHistogramBins(256)
    f.Execute(img)
    return float(f.GetThresholds()[-1])


def make_preview(path: str, outdir: str, log: Callable[[str], None] = print) -> dict:
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
        thr = bone_threshold(vol)
        log(f"  bone threshold {thr:.0f} (Otsu), preview grid {vol.shape}")

        pd = _mesh.build_surface(vol, grid, thr, taubin=10, pass_band=0.1,
                                 decimate=0.5)
        stl = os.path.join(outdir, "preview.stl")
        _pts, tris = _mesh.write_stl(pd, stl)
        info = {"input": real_path, "size": list(src.GetSize()), "spacing": list(sp),
                "field_of_view_mm": fov, "threshold": thr, "triangles": int(tris),
                "bounds_min": [float(v) for v in lo], "bounds_max": [float(v) for v in hi],
                "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(outdir, "preview.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        log(f"  preview: {tris} triangles in {info['seconds']} s -> {outdir_real}")
        return info
    finally:
        finish_out()
        cleanup_in()
