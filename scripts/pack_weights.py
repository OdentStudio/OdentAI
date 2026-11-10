#!/usr/bin/env python3
"""Pack the model weights as their own archive: OdentAI-weights-1.zip.

The weights are trained on ToothFairy data (CC BY-NC-SA 4.0) and may be
distributed only as a separate asset under that licence — never inside the
add-on zip. This archive is what the add-on downloads after the user accepts
the licence, and what "Install from file" takes. It carries the four model
files and LICENSE.md (licence, data source, papers to cite).

The add-on verifies every file against the checksums in its WEIGHTS_FILES, so
new weights mean a new archive name, a new URL and new checksums there.

Usage:
    python scripts/pack_weights.py [--models models] [--out dist/OdentAI-weights-1.zip]
"""
import argparse
import hashlib
import os
import zipfile

FILES = ("dental9.onnx", "dental9.json", "teeth_fdi.onnx", "teeth_fdi.json")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(here, "models"))
    ap.add_argument("--out", default=os.path.join(here, "dist", "OdentAI-weights-1.zip"))
    a = ap.parse_args()

    # The checksums the release was built against: the archive must hold
    # exactly the files the add-on expects.
    want = {}
    with open(os.path.join(here, "SHA256.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith("models/"):
                want[os.path.basename(parts[1])] = parts[0]
    lic = os.path.join(here, "weights", "LICENSE.md")
    if not os.path.isfile(lic):
        raise SystemExit(f"file missing: {lic}")
    for n in FILES:
        p = os.path.join(a.models, n)
        if not os.path.isfile(p):
            raise SystemExit(f"file missing: {p}")
        if sha256(p) != want.get(n):
            raise SystemExit(f"{n} does not match SHA256.txt: not the released weights")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # Stored, not deflated: onnx does not compress.
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        z.write(lic, "LICENSE.md", compress_type=zipfile.ZIP_DEFLATED)
        for n in FILES:
            z.write(os.path.join(a.models, n), n)
    print(a.out)
    print(f"  size:   {os.path.getsize(a.out) / 1e6:.0f} MB")
    print(f"  sha256: {sha256(a.out)}")


if __name__ == "__main__":
    main()
