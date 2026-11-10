#!/usr/bin/env python3
"""Smoke test of a BUILT worker: does the frozen executable actually run?

The release workflow builds on machines without a graphics card, so this
runs on the CPU and on a synthetic volume (no patient data). It does not
judge segmentation quality — that is eval_vs_manual.py — only that the
frozen build has everything it needs: onnxruntime with its provider,
SimpleITK, VTK, the weights next to the binary, and every command the add-on
calls (--providers, --preview, a segmentation, --crop).

    python scripts/smoke_test.py dist/dental9/dental9[.exe] models/dental9.onnx
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import SimpleITK as sitk


def make_scan(path: str) -> None:
    """Air, soft tissue and a bone block, 0.4 mm voxels. Sized so the 0.3 mm
    grid is exactly one smallest tile (64x128x128): a single CPU tile."""
    a = np.full((48, 96, 96), -1000, np.int16)
    a[8:40, 16:80, 16:80] = 60
    a[16:32, 30:66, 30:66] = 1400
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((0.4, 0.4, 0.4))
    sitk.WriteImage(img, path)


def run(cmd, label):
    print(f"--- {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    print(r.stdout.replace("\r", "\n")[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {label} (exit code {r.returncode})")
    return r


def main() -> None:
    # Absolute: Windows' CreateProcess does not resolve a relative path
    # written with forward slashes.
    exe, model = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    tmp = tempfile.mkdtemp(prefix="odentai_smoke_")
    scan = os.path.join(tmp, "scan.nii.gz")
    make_scan(scan)

    r = run([exe, "--providers"], "providers")
    if "CPUExecutionProvider" not in r.stdout:
        raise SystemExit("FAILED: onnxruntime lists no CPU provider")

    out = os.path.join(tmp, "preview")
    run([exe, scan, "-o", out, "--preview"], "bone preview")
    for f in ("preview.stl", "preview.json"):
        if not os.path.isfile(os.path.join(out, f)):
            raise SystemExit(f"FAILED: --preview wrote no {f}")

    out = os.path.join(tmp, "seg")
    run([exe, scan, "-o", out, "-m", model, "--device", "cpu", "-c", "mandible"],
        "segmentation on the CPU")
    with open(os.path.join(out, "report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("device") != "CPU":
        raise SystemExit(f"FAILED: report says device {rep.get('device')!r}")

    out = os.path.join(tmp, "crop")
    run([exe, scan, "-o", out, "-m", model, "--device", "cpu", "-c", "mandible",
         "--crop", "2", "2", "2", "30", "30", "15"], "segmentation inside a box")
    with open(os.path.join(out, "report.json"), encoding="utf-8") as f:
        if "crop" not in json.load(f):
            raise SystemExit("FAILED: --crop left no trace in report.json")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
