#!/usr/bin/env python3
"""The same scan on two devices of a BUILT worker: time and label agreement.

Written for the macOS build (2026-10-23): does CoreML run this network at all,
how much faster than the CPU, and does it give the same labels. The runner has
no patient data, so the default scan is a synthetic phantom — agreement on it
says the two paths compute the same thing, not that the result is good. On a
real Mac pass a real scan instead.

    python scripts/bench_devices.py dist/dental9/dental9 models/dental9.onnx \\
        [--scan <scan>] [--devices cpu coreml]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import SimpleITK as sitk


def phantom(path: str) -> None:
    """A jaw-sized block at 0.3 mm: one working tile (160x320x320) of soft
    tissue with a bone arch, teeth-like posts and a canal-like tube."""
    z, y, x = np.mgrid[0:160, 0:320, 0:320].astype(np.float32)
    a = np.full((160, 320, 320), -1000, np.float32)
    r = np.hypot(y - 200, x - 160)
    a[(r < 140) & (z > 20) & (z < 140)] = 40                     # soft tissue
    arch = (np.abs(r - 100) < 18) & (z > 40) & (z < 110)
    a[arch] = 1100                                               # bone
    posts = arch & (z > 90) & ((np.arctan2(y - 200, x - 160) * 12).astype(int) % 2 == 0)
    a[posts] = 2200                                              # teeth
    a[(np.abs(r - 100) < 3) & (np.abs(z - 60) < 3)] = 100        # canal
    rng = np.random.default_rng(0)
    a += rng.normal(0, 40, a.shape).astype(np.float32)
    img = sitk.GetImageFromArray(a.astype(np.int16))
    img.SetSpacing((0.3, 0.3, 0.3))
    sitk.WriteImage(img, path)


def run(exe, model, scan, out, device):
    t = time.time()
    r = subprocess.run([exe, scan, "-o", out, "-m", model, "--device", device,
                        "--no-stl"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t
    log = r.stdout + r.stderr
    if r.returncode != 0:
        print(log[-3000:])
        return None, dt, log
    rep = json.load(open(os.path.join(out, "report.json"), encoding="utf-8"))
    return rep, dt, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("model")
    ap.add_argument("--scan", default="")
    ap.add_argument("--devices", nargs="+", default=["cpu", "coreml"])
    a = ap.parse_args()
    exe, model = os.path.abspath(a.exe), os.path.abspath(a.model)
    tmp = tempfile.mkdtemp(prefix="odentai_bench_")
    scan = a.scan or os.path.join(tmp, "phantom.nii.gz")
    if not a.scan:
        phantom(scan)

    labels = {}
    for dev in a.devices:
        out = os.path.join(tmp, dev)
        rep, dt, log = run(exe, model, scan, out, dev)
        if rep is None:
            print(f"{dev:8} FAILED after {dt:.0f} s")
            continue
        print(f"{dev:8} {rep['device']:22} total {dt:6.1f} s, inference "
              f"{rep['predict_seconds']:6.1f} s, tile {rep['tile']}")
        labels[dev] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out, "labels.nii.gz")))

    if len(labels) == 2:
        (d1, l1), (d2, l2) = labels.items()
        print(f"voxels with the same label, {d1} vs {d2}: {float((l1 == l2).mean()) * 100:.3f}%")
        for k in sorted(set(np.unique(l1)) | set(np.unique(l2))):
            if k == 0:
                continue
            g, o = l1 == k, l2 == k
            s = g.sum() + o.sum()
            print(f"  class {k}: Dice {2 * (g & o).sum() / s:.4f}, voxels {g.sum()} vs {o.sum()}")
    elif len(labels) < len(a.devices):
        sys.exit(1)


if __name__ == "__main__":
    main()
