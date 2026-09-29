#!/usr/bin/env python3
"""An honest evaluation of the production pipeline: our ONNX path against manual labels.

Compared not with nnU-Net's prediction but with the annotator's hand — nnU-Net
is the third column here, to see how much is lost in the port to production
as opposed to lost by the model itself.

    python3 deploy/scripts/eval_vs_manual.py --gt <labelsTs> --ours <dir> --nnunet <dir>
"""
import argparse
import glob
import json
import os

import numpy as np
import SimpleITK as sitk

NAMES = {1: "Mandible", 2: "Upper Skull", 3: "Upper Teeth", 4: "Lower Teeth",
         5: "Mandibular Canal", 6: "Maxillary Sinus", 7: "Nasal Cavity",
         8: "Pharynx", 9: "Soft Palate"}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def load(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ours", required=True, help="folder with <case>/labels.nii.gz")
    ap.add_argument("--nnunet", default="", help="folder with <case>.nii.gz")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cases = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(a.ours, "*", "labels.nii.gz")))
    print(f"cases: {len(cases)}")

    res = {k: {"ours": [], "nnunet": [], "pair": []} for k in NAMES}
    for c in cases:
        gt = load(os.path.join(a.gt, f"{c}.nii.gz"))
        ours = load(os.path.join(a.ours, c, "labels.nii.gz"))
        nn = load(os.path.join(a.nnunet, f"{c}.nii.gz")) if a.nnunet else None
        for k in NAMES:
            g, o = gt == k, ours == k
            # A class may be absent from the frame: a slab does not always reach
            # the sinuses. Counting Dice = 0 there is unfair, so skip.
            if not g.any() and not o.any():
                continue
            if g.any():
                res[k]["ours"].append(dice(g, o))
                if nn is not None:
                    res[k]["nnunet"].append(dice(g, nn == k))
            if nn is not None:
                res[k]["pair"].append(dice(o, nn == k))

    print(f"\n{'class':<18}{'n':>4}{'our ONNX':>10}{'nnU-Net':>10}{'diff':>9}"
          f"{'  path agreement':>20}")
    rows = {}
    for k, n in NAMES.items():
        o = res[k]["ours"]
        if not o:
            continue
        mo = float(np.mean(o))
        mn = float(np.mean(res[k]["nnunet"])) if res[k]["nnunet"] else float("nan")
        mp = float(np.mean(res[k]["pair"])) if res[k]["pair"] else float("nan")
        rows[n] = {"n": len(o), "ours": round(mo, 4), "nnunet": round(mn, 4),
                   "agreement": round(mp, 4)}
        print(f"{n:<18}{len(o):>4}{mo:>10.4f}{mn:>10.4f}{mo-mn:>+9.4f}{mp:>20.4f}")

    valid = [r for r in rows.values() if r["ours"] == r["ours"]]
    print(f"\n{'mean':<18}{'':>4}{np.mean([r['ours'] for r in valid]):>10.4f}"
          f"{np.mean([r['nnunet'] for r in valid]):>10.4f}"
          f"{np.mean([r['ours'] for r in valid])-np.mean([r['nnunet'] for r in valid]):>+9.4f}"
          f"{np.mean([r['agreement'] for r in valid]):>20.4f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"cases": cases, "classes": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
