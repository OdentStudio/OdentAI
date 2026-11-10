#!/usr/bin/env python3
"""Pack the add-on zip: add-on code + the built binary. No weights.

The weights are trained on ToothFairy data (CC BY-NC-SA 4.0) and may be
distributed only as a separate asset under that licence, so they never go
into this zip: the add-on downloads them after the user accepts the licence.
Their archive is made by scripts/pack_weights.py.

Usage:
    python scripts/pack_addon.py --dist dist/dental9 --out dist/OdentAI.zip
"""
import argparse
import os
import sys
import zipfile

# The add-on is packed per platform: it carries a native binary. The add-on
# code is the same everywhere.
PLATFORM = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(
    sys.platform, sys.platform)


def add_tree(z: zipfile.ZipFile, src: str, prefix: str) -> int:
    """File permissions are carried into the archive as they are.

    Otherwise the binary arrives without its executable bit: Blender unpacks
    the zip with the standard module, which does not restore permissions. The
    add-on fixes this itself on first launch, but with a read-only install
    folder there would be nothing to fix it with.
    """
    n = 0
    for root, _dirs, files in os.walk(src):
        for f in files:
            if f.endswith(".onnx"):
                raise SystemExit(f"weights found in {root}: {f} — they must not "
                                 "be packed into the add-on zip")
            full = os.path.join(root, f)
            rel = os.path.relpath(full, src)
            info = zipfile.ZipInfo.from_file(full, os.path.join(prefix, rel))
            info.external_attr = (os.stat(full).st_mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(full, "rb") as fh:
                z.writestr(info, fh.read(), compresslevel=6)
            n += 1
    return n


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--addon", default=os.path.join(here, "addon", "OdentAI"))
    ap.add_argument("--dist", default=os.path.join(here, "dist", "dental9"),
                    help="the PyInstaller build folder (onedir)")
    ap.add_argument("--out", default=os.path.join(
        here, "dist", f"OdentAI_{PLATFORM}.zip"))
    a = ap.parse_args()

    for p in (a.addon, a.dist):
        if not os.path.isdir(p):
            raise SystemExit(f"folder missing: {p}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # The folder inside the zip is the add-on's module name in Blender
        # (and its key in the preferences), so it is taken from the source
        # folder rather than spelled out twice.
        root = os.path.basename(os.path.normpath(a.addon))
        n_code = add_tree(z, a.addon, root)
        n_bin = add_tree(z, a.dist, os.path.join(root, "bin", "dental9"))

    size = os.path.getsize(a.out) / 1e6
    print(f"{a.out}")
    print(f"  add-on code: {n_code} files")
    print(f"  binary:      {n_bin} files")
    print("  weights:     not included (separate archive, scripts/pack_weights.py)")
    print(f"  size:        {size:.0f} MB")


if __name__ == "__main__":
    main()
