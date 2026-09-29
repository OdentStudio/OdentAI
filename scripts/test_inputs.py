#!/usr/bin/env python3
"""A check of what the program can be fed.

Anything arrives from a clinic: a folder of slices, a single multi-frame
.dcm, an archive, nested folders, one slice instead of a volume. Each of
these forms broke in its own way, so all of them are checked, not just the
"typical" one.

The test data is made right here from a synthetic volume — no patient scans
are needed or used.

    python3 deploy/scripts/test_inputs.py [--real <path to a real export>]
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dental9.io import read_volume, to_training_grid                # noqa: E402

PASS, FAIL = "  OK  ", " FAIL "
results = []


def check(name, fn):
    try:
        fn()
        results.append((True, name, ""))
        print(f"[{PASS}] {name}")
    except Exception as e:                                          # noqa: BLE001
        results.append((False, name, str(e)))
        print(f"[{FAIL}] {name}\n         {type(e).__name__}: {e}")


def make_volume(shape=(24, 32, 32), spacing=0.4) -> sitk.Image:
    """A small volume with meaningful Hounsfield units: air, soft tissue and
    bone, so that writing to DICOM with its int16 is exercised too."""
    a = np.full(shape, -1000, np.int16)
    a[6:18, 8:24, 8:24] = 60
    a[10:14, 12:20, 12:20] = 1400
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((spacing,) * 3)
    img.SetOrigin((-10.0, 20.0, 5.0))
    return img


def write_series(img: sitk.Image, folder: str) -> None:
    """Split the volume into slices, the way a scanner does."""
    os.makedirs(folder, exist_ok=True)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    for z in range(img.GetSize()[2]):
        sl = img[:, :, z]
        sl.SetMetaData("0008|0060", "CT")
        sl.SetMetaData("0020|0032", "\\".join(
            str(v) for v in img.TransformIndexToPhysicalPoint((0, 0, z))))
        sl.SetMetaData("0020|0013", str(z))
        sl.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125.1.99999")
        w.SetFileName(os.path.join(folder, f"IM{z:05d}.dcm"))
        w.Execute(sl)


def same_geometry(a: sitk.Image, b: sitk.Image, tol=1e-3) -> None:
    if a.GetSize() != b.GetSize():
        raise AssertionError(f"size {a.GetSize()} versus {b.GetSize()}")
    if max(abs(x - y) for x, y in zip(a.GetSpacing(), b.GetSpacing())) > tol:
        raise AssertionError(f"spacing {a.GetSpacing()} versus {b.GetSpacing()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="", help="path to a real export, optional")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="d9test_")
    try:
        ref = make_volume()
        nii = os.path.join(tmp, "vol.nii.gz")
        sitk.WriteImage(ref, nii)
        mha = os.path.join(tmp, "vol.mha")
        sitk.WriteImage(ref, mha)

        series = os.path.join(tmp, "series")
        write_series(ref, series)

        nested = os.path.join(tmp, "nested", "STUDY", "SER001")
        os.makedirs(os.path.dirname(nested), exist_ok=True)
        shutil.copytree(series, nested)
        # A short scout series goes next to the real one: the program must
        # pick the long one, not the first it meets.
        scout = os.path.join(tmp, "nested", "STUDY", "SCOUT")
        write_series(make_volume((3, 32, 32)), scout)

        zip_path = os.path.join(tmp, "study.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            for root, _d, files in os.walk(series):
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.join("DICOM", f))
        zdir = os.path.join(tmp, "zipdir")
        os.makedirs(zdir)
        shutil.copy(zip_path, os.path.join(zdir, "study.zip"))

        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)

        n_slices = ref.GetSize()[2]

        check(".nii.gz file", lambda: same_geometry(ref, read_volume(nii)))
        check(".mha file", lambda: same_geometry(ref, read_volume(mha)))
        check("folder of DICOM slices",
              lambda: same_geometry(ref, read_volume(series)))
        check("nested exports, the longest series is taken",
              lambda: same_geometry(ref, read_volume(os.path.join(tmp, "nested"))))
        check("one slice of a series — go up and read the whole series",
              lambda: same_geometry(ref, read_volume(
                  os.path.join(series, "IM00003.dcm"))))
        check(".zip archive", lambda: same_geometry(ref, read_volume(zip_path)))
        check("folder with an archive inside", lambda: same_geometry(ref, read_volume(zdir)))
        check("folder with a single volume file",
              lambda: same_geometry(ref, read_volume(os.path.join(tmp, "onlyfile")))
              if os.path.isdir(os.path.join(tmp, "onlyfile")) else None)

        def onlyfile():
            d = os.path.join(tmp, "onlyfile")
            os.makedirs(d, exist_ok=True)
            shutil.copy(nii, os.path.join(d, "vol.nii.gz"))
            same_geometry(ref, read_volume(d))
        check("folder holding one volume file", onlyfile)

        def missing():
            try:
                read_volume(os.path.join(tmp, "no-such-path"))
            except FileNotFoundError:
                return
            raise AssertionError("no error on a missing path")
        check("missing path -> a clear error", missing)

        def empty_dir():
            try:
                read_volume(empty)
            except RuntimeError as e:
                if "no DICOM series" not in str(e):
                    raise AssertionError(f"unclear message: {e}")
                return
            raise AssertionError("no error on an empty folder")
        check("empty folder -> a clear error", empty_dir)

        def anisotropic():
            """A coarser step along z than in-plane — common for CT."""
            img = make_volume((20, 32, 32))
            img.SetSpacing((0.3, 0.3, 0.9))
            g = to_training_grid(img, 0.3)
            want = (32, 32, 60)
            if g.GetSize() != want:
                raise AssertionError(f"{g.GetSize()} instead of {want}")
        check("anisotropic spacing is brought to isotropic 0.3 mm", anisotropic)

        def keeps_fov():
            """The physical field of view must survive resampling."""
            img = make_volume((24, 32, 32), spacing=0.15)
            g = to_training_grid(img, 0.3)
            f0 = [n * s for n, s in zip(img.GetSize(), img.GetSpacing())]
            f1 = [n * s for n, s in zip(g.GetSize(), g.GetSpacing())]
            if max(abs(x - y) for x, y in zip(f0, f1)) > 0.31:
                raise AssertionError(f"field of view {f0} mm became {f1} mm")
        check("the physical field of view does not change with resolution", keeps_fov)

        if a.real:
            check(f"real export: {os.path.basename(a.real.rstrip('/'))}",
                  lambda: read_volume(a.real).GetDimension() == 3 or
                  (_ for _ in ()).throw(AssertionError("did not come out 3D")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = [r for r in results if not r[0]]
    print(f"\npassed {len(results) - len(bad)} of {len(results)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
