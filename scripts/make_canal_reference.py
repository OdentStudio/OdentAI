#!/usr/bin/env python3
"""The reference intensity curve for the second canal pass.

Why. The mandibular canal is held up by the contrast of its cortical wall.
On scanners with a compressed intensity range (measured: p99.5 = 1630 against
2400 for the other sources) that contrast drops from 350 HU to 52 and the
network stops asserting the canal: on the problem scan 0.22 cm3 in six
fragments instead of 0.6 in two. Yet the information IS in the scan — map
the intensities onto the domain the network is confident in and the canal is
found along 88% of its course.

The curve is built from MANDIBLE CROPS, not whole scans: a crop has no skull
and no neck, and the histogram proportions are entirely different. Measured
on the same scan: a whole-scan reference gives 83% coverage, a crop one 88%.

Taken from the sources where the network is confident (DentVoxel and
ToothFairy3) and averaged by the median — a single reference scan would make
the result hostage to its quirks.

    python3 deploy/scripts/make_canal_reference.py --out deploy/models/dental9.json
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import SimpleITK as sitk

MARGIN_VOX = 20            # 6 mm around the mandible bbox
MANDIBLE = 1


def crop_to_mandible(img: np.ndarray, lab: np.ndarray) -> np.ndarray:
    idx = np.argwhere(lab == MANDIBLE)
    lo = np.maximum(idx.min(0) - MARGIN_VOX, 0)
    hi = np.minimum(idx.max(0) + MARGIN_VOX + 1, img.shape)
    return img[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="nnUNet_raw/Dataset003_Dental9")
    ap.add_argument("--out", required=True, help="the dental9.json next to the weights")
    ap.add_argument("--prefixes", nargs="+", default=["dv_", "tf3_"],
                    help="sources the network is confident on")
    ap.add_argument("--per-prefix", type=int, default=10)
    ap.add_argument("--levels", type=int, default=256)
    a = ap.parse_args()

    random.seed(31)
    files = []
    for p in a.prefixes:
        f = sorted(glob.glob(os.path.join(a.raw, "labelsTr", f"{p}*.nii.gz")))
        files += random.sample(f, min(a.per_prefix, len(f)))

    q = np.linspace(0, 100, a.levels + 1)
    curves = []
    for lf in files:
        im = lf.replace("labelsTr", "imagesTr").replace(".nii.gz", "_0000.nii.gz")
        lab = sitk.GetArrayFromImage(sitk.ReadImage(lf))
        if (lab == MANDIBLE).sum() < 1000:
            continue
        img = sitk.GetArrayFromImage(sitk.ReadImage(im)).astype(np.float32)
        curves.append(np.percentile(crop_to_mandible(img, lab), q))
    if not curves:
        raise SystemExit("not a single scan with a labelled mandible found")

    ref = np.median(np.array(curves), axis=0)
    cfg = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    cfg["canal_refine"] = {
        "_comment": "Second canal pass: the intensities of the mandible crop are "
                    "mapped onto this curve. Built by make_canal_reference.py.",
        "built_from": len(curves),
        "quantiles": [round(float(v), 4) for v in q],
        "reference_hu": [round(float(v), 1) for v in ref],
        "margin_voxels": MARGIN_VOX,
        # Signs of a broken canal, calibrated on 30 labelled cases: each one
        # alone trips on at most 3% of healthy canals, while a broken canal
        # trips all five. Computed without ground truth.
        "triggers": {
            "volume_below_cm3": 0.45,
            "side_volume_below_cm3": 0.15,
            "asymmetry_below": 0.45,
            "span_fraction_below": 0.30,
            "components_per_side_above": 2,
        },
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"curve from {len(curves)} crops -> {a.out}")
    i995 = int(0.995 * (len(ref) - 1))
    print(f"  median {ref[len(ref)//2]:.0f} HU, p99.5 {ref[i995]:.0f} HU, max {ref[-1]:.0f}")


if __name__ == "__main__":
    main()
