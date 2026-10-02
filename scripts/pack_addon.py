#!/usr/bin/env python3
"""Pack the add-on zip: add-on code + the built binary + the weights.

The weights go as separate files inside bin/dental9/, not into the executable:
replacing them later means replacing two files (dental9.onnx and dental9.json)
with no rebuild and no reinstall of the add-on. By the same token the model
can live anywhere at all, with its path set in the add-on preferences.

Usage:
    python3 deploy/scripts/pack_addon.py \
        --dist deploy/dist/dental9 \
        --model deploy/models/dental9.onnx
"""
import argparse
import os
import sys
import zipfile

# The add-on is packed per platform: it carries a native binary. The add-on
# code and the weights are the same everywhere.
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
    ap.add_argument("--model", default=os.path.join(here, "models", "dental9.onnx"))
    ap.add_argument("--out", default=os.path.join(
        here, "dist", f"OdentAI_{PLATFORM}.zip"))
    ap.add_argument("--teeth-model", default=os.path.join(here, "models", "teeth_fdi.onnx"),
                    help="weights of the second pass (FDI tooth numbering); go next to "
                         "the main ones as teeth_fdi.onnx + teeth_fdi.json")
    ap.add_argument("--no-teeth-model", action="store_true",
                    help="pack without the second pass: the Separate teeth option will not work")
    ap.add_argument("--no-model", action="store_true",
                    help="pack without weights: they will be delivered separately")
    a = ap.parse_args()

    for p in (a.addon, a.dist):
        if not os.path.isdir(p):
            raise SystemExit(f"folder missing: {p}")
    cfg = os.path.splitext(a.model)[0] + ".json"
    if not a.no_model:
        for p in (a.model, cfg):
            if not os.path.isfile(p):
                raise SystemExit(f"file missing: {p}\n"
                                 "the weights and their .json are made by deploy/scripts/export_onnx.py")

    tcfg = os.path.splitext(a.teeth_model)[0] + ".json"
    with_teeth = not a.no_model and not a.no_teeth_model
    if with_teeth:
        for p in (a.teeth_model, tcfg):
            if not os.path.isfile(p):
                raise SystemExit(f"file missing: {p}\n"
                                 "the second-pass weights are made by export_onnx.py in ~/precise_teeth "
                                 "(or use --no-teeth-model)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # ZIP_STORED for the weights: onnx is already compressed inside, and
    # spending minutes on deflate for one percent is pointless. Code is compressed.
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # The folder inside the zip is the add-on's module name in Blender
        # (and its key in the preferences), so it is taken from the source
        # folder rather than spelled out twice.
        root = os.path.basename(os.path.normpath(a.addon))
        n_code = add_tree(z, a.addon, root)
        n_bin = add_tree(z, a.dist, os.path.join(root, "bin", "dental9"))
        n_model = 0
        if not a.no_model:
            # Inside the bundle the weights are always called dental9.onnx,
            # whatever the file is called outside: the add-on looks them up by
            # that name, and replacing the model later means dropping in a file
            # with the same name.
            pairs = [(a.model, "dental9.onnx"), (cfg, "dental9.json")]
            if with_teeth:
                pairs += [(a.teeth_model, "teeth_fdi.onnx"), (tcfg, "teeth_fdi.json")]
            for p, name in pairs:
                z.write(p, os.path.join(root, "bin", "dental9", name),
                        compress_type=zipfile.ZIP_STORED)
                n_model += 1

    size = os.path.getsize(a.out) / 1e6
    print(f"{a.out}")
    print(f"  add-on code: {n_code} files")
    print(f"  binary:      {n_bin} files")
    print(f"  weights:     {n_model} files" if n_model else "  weights:     not included")
    print(f"  size:        {size:.0f} MB")


if __name__ == "__main__":
    main()
