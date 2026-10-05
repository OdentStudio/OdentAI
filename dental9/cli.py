"""The command line. This is what the Blender add-on calls — as a separate
process, so that an onnxruntime crash or running out of video memory cannot
bring Blender down with it."""
import argparse
import json
import os
import sys
import time

from .classes import CLASSES, resolve
from .pipeline import Options, segment


def _default_model() -> str:
    """The model is looked for next to the executable. It ships as a separate
    file, not inside the binary: hundreds of megabytes that must be
    replaceable without rebuilding everything."""
    base = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    for name in ("dental9.onnx", os.path.join("models", "dental9.onnx")):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return os.path.abspath(os.path.join(base, "dental9.onnx"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dental9",
        description="CBCT (DICOM or NIfTI) to STL surfaces for nine anatomical classes")
    p.add_argument("input", nargs="?", help="folder with a DICOM series, or a volume file")
    p.add_argument("-o", "--out", help="where to write the STL files")
    p.add_argument("-m", "--model", default=None, help="the .onnx weights file")
    p.add_argument("--config", default="", help="dental9.json, if not next to the model")
    p.add_argument("-c", "--classes", nargs="+", default=None,
                   metavar="CLASS", help="which classes to compute; all by default. "
                   + "Available: " + ", ".join(k.key for k in CLASSES))
    p.add_argument("--device", choices=("auto", "dml", "cuda", "cpu"), default="auto",
                   help="auto = the GPU if onnxruntime can see one")
    p.add_argument("--tile", type=int, nargs=3, default=None,
                   metavar=("Z", "Y", "X"), help="tile size; the training patch by default")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--taubin", type=int, default=20,
                   help="mesh smoothing iterations, 0 disables it")
    p.add_argument("--pass-band", type=float, default=0.1)
    p.add_argument("--decimate", type=float, default=0.0,
                   help="fraction of triangles to drop, 0.5 halves the mesh")
    p.add_argument("--no-sub-voxel", action="store_true",
                   help="build from the binary mask (one-voxel staircase edges)")
    p.add_argument("--no-canal-refine", action="store_true",
                   help="skip the second canal pass on scans where the canal "
                        "comes out broken")
    p.add_argument("--canal-refine-always", action="store_true",
                   help="run the second canal pass on every scan, not only when "
                        "the first pass looks broken (for evaluation)")
    p.add_argument("--canal-refine-mode", choices=("match", "fine"), default="match",
                   help="second canal pass: match = intensities onto the training "
                        "curve; fine = the mandible again on a 0.25 mm grid")
    p.add_argument("--canal-enhance", type=int, nargs="?", const=8, default=0,
                   metavar="RADIUS",
                   help="experimental: bone window + 3D adaptive histogram "
                        "equalisation before the second canal pass only; RADIUS "
                        "in voxels of the 0.3 mm grid (8 when omitted)")
    p.add_argument("--canal-enhance-window", choices=("fixed", "auto"), default="fixed",
                   help="fixed = -100..1700 HU; auto = from the scan (median of "
                        "non-air voxels .. their 99th percentile)")
    p.add_argument("--labels", action="store_true", help="also write labels.nii.gz")
    p.add_argument("--no-stl", action="store_true",
                   help="labels only, no surfaces — for accuracy evaluation")
    p.add_argument("--separate-teeth", action="store_true",
                   help="second pass: individual teeth with FDI numbers, implants, "
                        "implant crowns and bridges as separate STL files in teeth/")
    p.add_argument("--teeth-model", default=None,
                   help="the .onnx weights of the teeth-numbering model; by default "
                        "teeth_fdi.onnx next to the main model")
    p.add_argument("--crop", type=float, nargs=6, default=None,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="compute only inside this box: two opposite corners in mm, "
                        "patient coordinates (the frame the STL files are in)")
    p.add_argument("--preview", action="store_true",
                   help="no network: a quick bone surface by threshold "
                        "(preview.stl + preview.json in -o), to draw a crop box on")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--providers", action="store_true",
                   help="list the onnxruntime providers and exit")
    p.add_argument("--diagnose", action="store_true",
                   help="check the hardware: whether the graphics card is really "
                        "usable and how long one scan will take")
    p.add_argument("--json", action="store_true",
                   help="machine-readable check output, used by the add-on")
    return p


def _utf8_streams() -> None:
    """Print in UTF-8 whatever the console locale is.

    On Windows a piped stdout takes the locale code page (cp1252 and the
    like), and the first path with a letter outside it — a Polish "ł" in a
    folder name was the real case — kills the run with
    "'charmap' codec can't encode character". The add-on reads the pipe as
    UTF-8; a real console just shows odd bytes for the rare letter.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                   # noqa: BLE001
            pass


def main(argv=None) -> int:
    _utf8_streams()
    a = build_parser().parse_args(argv)

    if a.providers:
        from .infer import available_providers
        print("\n".join(available_providers()))
        return 0
    if a.diagnose:
        from .diagnose import collect, format_text
        rep = collect(a.model or _default_model())
        print(json.dumps(rep, ensure_ascii=False) if a.json else format_text(rep))
        # The exit code lets the caller tell "all fine" from "works, but on
        # the CPU" without parsing the text.
        return 0 if not any(x["level"] == "bad" for x in rep["advice"]) else 3

    # checked by hand rather than required=True: otherwise --providers, which
    # needs neither a scan nor an output folder, would demand both
    if not a.input or not a.out:
        print("an input scan and -o are required", file=sys.stderr)
        return 2

    try:
        from .winpath import sweep_stale
        sweep_stale()
    except Exception:                                   # noqa: BLE001
        pass

    if a.preview:
        from .preview import make_preview
        try:
            make_preview(a.input, a.out)
        except Exception as e:
            _report_error(e)
            return 1
        return 0

    last = [0.0]

    def progress(done, total):
        now = time.time()
        if done == total or now - last[0] > 1.0:
            last[0] = now
            # \r and flush: the add-on reads the output line by line, but in
            # a terminal this should look like one line updating in place
            end = "\n" if done == total else "\r"
            print(f"tile {done}/{total} ({100*done//total}%)", end=end, flush=True)

    opts = Options(model=a.model or _default_model(), config=a.config,
                   tile=a.tile, overlap=a.overlap, device=a.device,
                   classes=resolve(a.classes), taubin=a.taubin,
                   pass_band=a.pass_band, decimate=a.decimate,
                   sub_voxel=not a.no_sub_voxel, save_labels=a.labels or a.no_stl,
                   canal_refine=not a.no_canal_refine,
                   canal_refine_force=a.canal_refine_always,
                   canal_refine_mode=a.canal_refine_mode,
                   canal_enhance=a.canal_enhance,
                   canal_enhance_window=a.canal_enhance_window,
                   make_stl=not a.no_stl,
                   threads=a.threads,
                   separate_teeth=a.separate_teeth,
                   teeth_model=a.teeth_model or "",
                   crop=a.crop)
    try:
        segment(a.input, a.out, opts, progress=progress)
    except Exception as e:
        _report_error(e)
        return 1
    return 0


def _report_error(e: BaseException) -> None:
    """The error in readable form, then the traceback and the memory state.

    The ERROR line comes first and alone: the add-on shows it to the user.
    The rest is for whoever reads the log afterwards.
    """
    import traceback

    from .infer import readable_error
    from .sysinfo import snapshot
    sys.stdout.flush()
    print(f"ERROR: {readable_error(e)}", file=sys.stderr)
    print("---- details for the developer ----", file=sys.stderr)
    print(f"type: {type(e).__name__}"
          + (f", caused by {type(e.__cause__).__name__}" if e.__cause__ else ""),
          file=sys.stderr)
    print(f"memory: {snapshot()}", file=sys.stderr)
    print("".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip(),
          file=sys.stderr)
    print("---- end of details ----", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
