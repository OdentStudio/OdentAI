"""The whole path: CT -> labels -> one STL per class. One function per scan.

Kept apart from cli.py because the Blender add-on drives the same code: it
calls `segment()` in a separate process rather than the command line.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

from . import infer as _infer
from . import io as _io
from . import mesh as _mesh
from . import sysinfo as _sysinfo
from . import teeth as _teeth
from . import winpath as _winpath
from .classes import CLASSES, Klass

# A sane voxel spacing for dental CBCT. Outside this range the scan metadata is
# broken, and refusing at once beats computing for half an hour on garbage:
# the whole pipeline works in physical millimetres, so a wrong spacing shifts
# both the intensity window and the mesh scale. Caught on a Sirona export that
# claimed 1.0 mm instead of 0.15 — the field of view came out as "616 mm".
SANE_SPACING_MM = (0.04, 1.2)
# Above this the frame cannot be a dental CBCT (the largest are ~260 mm): a
# hint that the spacing in the file is wrong — 1.0 mm passes the range above.
MAX_DENTAL_FOV_MM = 300
# The logit buffer is ten classes over the whole volume. Above this we warn
# that a lot of RAM is needed; above a share of the available RAM we refuse,
# otherwise the machine swaps and freezes instead of failing honestly.
WARN_BUFFER_GB = 4.0
# Resident memory of an open onnxruntime session with its weights, measured.
SESSION_RAM_GB = 2.0

# Signs of a broken canal. Thresholds calibrated on 30 labelled cases: each
# one trips on at most 3% of healthy canals. Also stored in dental9.json — the
# config next to the weights takes precedence.
DEFAULT_TRIGGERS = {
    "volume_below_cm3": 0.45,
    "side_volume_below_cm3": 0.15,
    "asymmetry_below": 0.45,
    "span_fraction_below": 0.30,
    "components_per_side_above": 2,
}
# The fine canal pass is skipped above this many voxels: a normal mandible
# with its margin is 60-120 M at 0.25 mm, which takes 25-75 s on a GPU.
FINE_MAX_VOXELS = 150_000_000


@dataclass
class Options:
    model: str
    config: str = ""
    tile: Optional[Sequence[int]] = None
    overlap: float = 0.5
    device: str = "auto"
    classes: Sequence[Klass] = dc_field(default_factory=lambda: list(CLASSES))
    taubin: int = 20            # mesh smoothing; 0 disables it
    pass_band: float = 0.1
    decimate: float = 0.0       # 0.5 = half the triangles
    sub_voxel: bool = True      # isosurface from the signed field, not the binary mask
    save_labels: bool = False   # also write a NIfTI label map next to the STLs
    canal_refine: bool = True   # second canal pass when the first comes out broken
    canal_refine_force: bool = False   # always, even when the first pass looks sound
    canal_refine_mode: str = "fine"    # "fine" = the mandible again on a finer grid; "match" = intensities onto the training curve
    canal_fine_spacing: float = 0.25   # grid of the "fine" second pass, mm
    canal_fine_margin_mm: float = 15.0  # context around the mandible for it
    make_stl: bool = True       # off for bulk accuracy evaluation
    threads: int = 0
    separate_teeth: bool = False   # second pass: individual teeth with FDI numbers
    teeth_model: str = ""          # weights for that pass; default teeth_fdi.onnx next to the main model
    crop: Optional[Sequence[float]] = None   # x0 y0 z0 x1 y1 z1, mm in patient coordinates (the STL frame)


def _free_ram_gb() -> Optional[float]:
    """How much memory can actually be taken. Available, not total: on a
    working machine Blender and its scene already hold half of it."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 2 ** 30
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:                                   # noqa: BLE001
        pass
    return None


def load_config(opts: Options) -> dict:
    """Preprocessing parameters: next to the model, else from the package folder.

    A separate file rather than constants in code: retraining changes the
    intensity window, the spacing and the patch size, forgetting one of them
    is easy, and the result would look plausible while being wrong.
    """
    cand = [opts.config] if opts.config else []
    cand += [os.path.splitext(opts.model)[0] + ".json",
             os.path.join(os.path.dirname(os.path.abspath(opts.model)), "dental9.json")]
    # In the frozen build the fallback config sits inside the bundle, next to
    # the unpacked libraries, not in the source tree.
    base = getattr(sys, "_MEIPASS", None)
    if base:
        cand.append(os.path.join(base, "dental9.json"))
    cand.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "configs", "dental9.json"))
    for c in cand:
        if c and os.path.isfile(c):
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("dental9.json with the preprocessing parameters not found")


def _match_to_reference(crop: np.ndarray, ref: dict) -> np.ndarray:
    """Map the crop's intensities onto the reference curve of the training domain.

    The mapping is monotone and defined by quantiles, so the ordering of
    intensities is preserved; the curve lives in the json next to the weights
    and changes together with the model.
    """
    q = np.asarray(ref["quantiles"], dtype=np.float64)
    dst = np.asarray(ref["reference_hu"], dtype=np.float64)
    src = np.percentile(crop, q)
    # np.interp needs strictly increasing knots: on compressed scans adjacent
    # quantiles often coincide.
    src, keep = np.unique(src, return_index=True)
    return np.interp(crop, src, dst[keep]).astype(np.float32)


def _canal_health(labels: np.ndarray, spacing, thr: dict) -> dict:
    """Break the canal down into features and decide whether it came out broken.

    Thresholds calibrated on 30 labelled training cases (specks removed, sides
    without a mandible not counted): each one alone trips on at most 3% of
    healthy canals, while a broken canal trips all five at once.

    Everything is computed without ground truth, so it works in production.
    """
    vox = float(np.prod(spacing)) / 1000.0
    canal = _mesh.drop_small(labels == 5, spacing, 0.02)   # specks do not count
    mand = labels == 1
    out = {"volume_cm3": float(canal.sum()) * vox, "reasons": []}
    if not mand.any():
        out["reasons"].append("no mandible")
        return out

    mi = np.argwhere(mand)
    mid = (mi[:, 2].min() + mi[:, 2].max()) // 2
    span_mand = float(mi[:, 1].max() - mi[:, 1].min() + 1) * spacing[1]

    sides = []
    for sl in (slice(0, mid), slice(mid, None)):
        half_m = mand[:, :, sl]
        # A side where the frame cuts the mandible off cannot be held to account.
        if half_m.sum() < 0.2 * mand.sum():
            continue
        half = np.zeros_like(canal)
        half[:, :, sl] = canal[:, :, sl]
        span = 0.0
        if half.any():
            hi = np.argwhere(half)
            span = float(hi[:, 1].max() - hi[:, 1].min() + 1) * spacing[1]
        sides.append({"vol": float(half.sum()) * vox, "span": span,
                      "comps": _mesh.count_components(half)})
    if not sides:
        return out

    vols = [x["vol"] for x in sides]
    out.update(sides=len(sides), vol_min=min(vols),
               asym=min(vols) / max(vols) if max(vols) > 0 else 0.0,
               span_frac=min(x["span"] for x in sides) / span_mand if span_mand else 0.0,
               comps_max=max(x["comps"] for x in sides))

    if out["volume_cm3"] < thr["volume_below_cm3"]:
        out["reasons"].append(f"volume {out['volume_cm3']:.2f} cm3")
    if out["vol_min"] < thr["side_volume_below_cm3"]:
        out["reasons"].append(f"weak side {out['vol_min']:.2f} cm3")
    if out["asym"] < thr["asymmetry_below"]:
        out["reasons"].append(f"asymmetry {out['asym']:.2f}")
    if out["span_frac"] < thr["span_fraction_below"]:
        out["reasons"].append(f"course covered {out['span_frac']:.2f}")
    if out["comps_max"] > thr["components_per_side_above"]:
        out["reasons"].append(f"{out['comps_max']} pieces on one side")
    return out


def _fine_canal(src: sitk.Image, grid: sitk.Image, labels: np.ndarray, sess,
                cfg: dict, opts: "Options", log) -> np.ndarray:
    """The canal predicted on a finer grid around the mandible, as a mask on `grid`.

    2026-10-05, a 0.15 mm scan: at 0.3 mm the left canal came out as two short
    fragments (105 mm3, half its course) — rotations, tile shifts, mirroring,
    denoising and intensity matching all failed to bring the front half back,
    while 0.25 mm found it whole (300 mm3, the full course) and 0.35 mm lost
    it almost entirely. Thin canal walls drop below what the network resolves
    at 0.3 mm. 0.25 is within the scale range nnU-Net trains with.

    Context matters: cropping tightly to the mandible lost two thirds of that
    canal, hence the margin.
    """
    # The box comes from the largest piece of mandible only: stray specks
    # labelled mandible blew one box up to 123x171x147 mm, 197 M voxels at
    # 0.25 mm and 324 s (2026-10-06, a 750-slice scan).
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(
        sitk.GetImageFromArray((labels == 1).astype(np.uint8))), sortByObjectSize=True)
    idx = np.argwhere(sitk.GetArrayViewFromImage(cc) == 1)[:, ::-1]   # z, y, x -> x, y, z
    pts = np.array([grid.TransformIndexToPhysicalPoint([int(v) for v in c])
                    for c in (idx.min(0), idx.max(0))])
    m = float(opts.canal_fine_margin_mm)
    sub = _io.crop_to_box(src, pts.min(0) - m, pts.max(0) + m)
    n = int(np.prod([round(s * v / opts.canal_fine_spacing)
                     for s, v in zip(sub.GetSize(), sub.GetSpacing())]))
    if n > FINE_MAX_VOXELS:
        log(f"  fine pass skipped: the mandible box is {n / 1e6:.0f} M voxels "
            f"at {opts.canal_fine_spacing} mm")
        return None
    fine = _io.to_training_grid(sub, opts.canal_fine_spacing)
    v = _io.normalize_ct(sitk.GetArrayFromImage(fine).astype(np.float32),
                         _norm_props(cfg["normalization"]))
    log(f"  fine pass: {opts.canal_fine_spacing} mm, {v.shape}")
    logits, _ = _infer.predict_logits_auto(sess, v, opts.tile or cfg["tile"], opts.overlap,
                                           None, np.float16 if v.size > 150e6 else np.float32, log)
    c = sitk.GetImageFromArray((logits.argmax(0) == 5).astype(np.uint8))
    del logits
    c.CopyInformation(fine)
    return sitk.GetArrayFromImage(sitk.Resample(c, grid, sitk.Transform(),
                                                sitk.sitkNearestNeighbor, 0)).astype(bool)


def _refine_canal(vol: np.ndarray, labels: np.ndarray, sess, cfg: dict,
                  opts: "Options", spacing, log, src=None, grid=None) -> np.ndarray:
    """A second pass over the canal only, and only around the mandible.

    Runs when `_canal_health` says the canal came out broken; ONLY the canal is
    taken from it, the other eight classes are not touched at all.

    Two modes. "fine" (the default since 2026-10-07) predicts the mandible
    again on a 0.25 mm grid — thin canal walls fall below what the network
    resolves at 0.3 mm. "match" is the older pass: intensities inside the
    mandible mapped onto the training curve and predicted again at 0.3 mm.

    Why fine is the default: on 22 clinic scans the match pass helped on none
    and on one (0.15 mm, the left canal in two fragments) made it worse, while
    the fine pass restored that canal whole, 105 -> 305 mm3. On the 70 labelled
    held-out cases neither pass ever runs — the trigger does not fire there —
    so the change costs nothing measurable: the two are byte-identical on that
    set (docs/canal_fine_eval/REPORT_dice_ru.md).

    Neither pass may be forced on every scan. Measured against manual labels on
    those 70: forcing fine costs 0.0075 canal Dice, forcing match 0.0039.
    """
    ref = cfg.get("canal_refine")
    if not ref:
        return labels
    thr = ref.get("triggers", DEFAULT_TRIGGERS)
    health = _canal_health(labels, spacing, thr)
    if not health["reasons"] and not opts.canal_refine_force:
        return labels
    mand = labels == 1
    if not mand.any():
        log("  canal looks broken but there is no mandible — second pass skipped")
        return labels

    log("  canal looks broken (" + "; ".join(health["reasons"]) + ") — second pass"
        if health["reasons"] else "  canal second pass (forced)")
    if opts.canal_refine_mode == "fine" and src is not None:
        canal2 = _fine_canal(src, grid, labels, sess, cfg, opts, log)
        if canal2 is None:
            return labels
        return _merge_canal(labels, canal2, health, spacing, thr, log)
    idx = np.argwhere(mand)
    pad = int(ref.get("margin_voxels", 20))
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.array(vol.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    # The second pass works from RAW Hounsfield units, so normalisation is
    # redone: vol arrives already normalised, while the curve is given in HU.
    crop_hu = _denormalize(vol[sl], cfg["normalization"])
    matched = _match_to_reference(crop_hu, ref)
    vol2 = _io.normalize_ct(matched, _norm_props(cfg["normalization"]))
    logits2, _ = _infer.predict_logits_auto(sess, vol2, opts.tile or cfg["tile"],
                                            opts.overlap, None, np.float32, log)
    canal2 = np.zeros(vol.shape, bool)
    canal2[sl] = logits2.argmax(0) == 5
    del logits2
    return _merge_canal(labels, canal2, health, spacing, thr, log)


def _merge_canal(labels, canal2, health, spacing, thr, log) -> np.ndarray:
    out = labels.copy()
    # Take the second pass's canal only where the first pass found nothing
    # more definite: teeth and the upper skull are not given away.
    take = canal2 & ((labels == 0) | (labels == 5) | (labels == 1))
    out[out == 5] = 0
    out[take] = 5
    after = _canal_health(out, spacing, thr)
    log(f"  after the second pass: {after['volume_cm3']:.2f} cm3"
        + (" — still " + "; ".join(after["reasons"]) if after["reasons"] else ", looks sound"))
    # If the second pass did worse, roll back: no point risking what worked to
    # rescue what did not.
    if after["volume_cm3"] < health["volume_cm3"]:
        log("  the second pass found less than the first — keeping the first")
        return labels
    return out


def _norm_props(norm: dict) -> dict:
    return {"percentile_00_5": norm["clip"][0], "percentile_99_5": norm["clip"][1],
            "mean": norm["mean"], "std": norm["std"]}


def _denormalize(v: np.ndarray, norm: dict) -> np.ndarray:
    """Back to Hounsfield units. The clipping done at normalisation cannot be
    undone, but it only touches values outside [-1000, 3492], and the canal is
    far inside."""
    return v * float(norm["std"]) + float(norm["mean"])


def _signed_fields(logits: np.ndarray, ids: Sequence[int]) -> Dict[int, np.ndarray]:
    """Per class: its logit minus the best of the others.

    The zero of that field is exactly the argmax boundary, but placed by the
    values rather than rounded to a voxel. The best and second-best are found
    once for all classes: the winner's rival is the runner-up, everyone else's
    rival is the winner. Done in slabs along z, otherwise a large volume would
    need four copies of itself side by side.
    """
    C, Z = logits.shape[0], logits.shape[1]
    out = {i: np.empty(logits.shape[1:], np.float32) for i in ids}
    chunk = max(1, Z // 8)
    for z0 in range(0, Z, chunk):
        z1 = min(z0 + chunk, Z)
        part = logits[:, z0:z1].astype(np.float32)
        order = np.argpartition(part, C - 2, axis=0)
        best_i = order[C - 1]
        top = np.take_along_axis(part, order[C - 2:], axis=0)
        best_v, second_v = top.max(0), top.min(0)
        for i in ids:
            rival = np.where(best_i == i, second_v, best_v)
            out[i][z0:z1] = part[i] - rival
    return out


def _peak_ram_estimate_gb(n: int, num_classes: int, n_out: int,
                          teeth: bool, fields: bool) -> float:
    """The most RAM the run will hold at once, from the grid size alone.

    Two candidates for the peak: during inference (volume, logit buffer,
    weight sum) and while the signed fields are built (volume, logit buffer,
    labels and one float32 field per output class). The HU copy for the teeth
    pass lives through both. On top: the onnxruntime session itself, measured
    at about 2 GB resident with DirectML (2026-09-29, 367³ grid: estimate
    within 0.1 GB of the measured 6.5 GB peak).
    """
    acc = num_classes * (2 if n > 150_000_000 else 4)
    during = 4 + acc + 4
    after = 4 + acc + 1 + (4 * n_out if fields else 0)
    return SESSION_RAM_GB + n * (max(during, after) + (2 if teeth else 0)) / 2 ** 30


def _log_system(log) -> None:
    """What the machine is, at the top of every log: the first thing to ask
    when a run fails on someone else's computer."""
    import platform
    total, avail = _sysinfo.ram_gb()
    log(f"system: {platform.system()} {platform.release()}, {_sysinfo.cpu_name()}"
        + (f", RAM {avail:.1f} of {total:.1f} GB free" if total else ""))
    for g in _sysinfo.gpus():
        log(f"  graphics: {_sysinfo.describe_gpu(g)}")
    try:
        import onnxruntime as ort
        log(f"  onnxruntime {ort.__version__}: {', '.join(ort.get_available_providers())}")
    except Exception:                                   # noqa: BLE001
        pass


def segment(path: str, outdir: str, opts: Options,
            log: Callable[[str], None] = print,
            progress: Optional[Callable[[int, int], None]] = None) -> dict:
    # Non-ASCII paths on Windows are handled by staging (see winpath.py). The
    # user-facing report keeps the real paths; only the native libraries see
    # the staged ones.
    real_path, real_outdir = path, outdir
    path, cleanup_in = _winpath.readable(path)
    outdir, finish_out = _winpath.writable(outdir)
    try:
        return _segment(path, outdir, real_path, real_outdir, opts, log, progress)
    finally:
        # Nested: a failed move of the output must not leave the staged copy
        # of the scan behind.
        try:
            finish_out()
        finally:
            cleanup_in()


def _segment(path: str, outdir: str, real_path: str, real_outdir: str,
             opts: Options, log, progress) -> dict:
    cfg = load_config(opts)
    os.makedirs(outdir, exist_ok=True)
    t_all = time.time()

    _log_system(log)
    log(f"reading: {real_path}")
    src = _io.read_volume(path)
    log(f"  volume {src.GetSize()}, spacing {tuple(round(s, 3) for s in src.GetSpacing())} mm")

    # Labels go back onto the scan's own grid, the whole of it, even when only
    # a box of it was computed: outside the box they are simply zero.
    src_full = src
    if opts.crop:
        c = [float(v) for v in opts.crop]
        lo_b, hi_b = np.minimum(c[:3], c[3:]), np.maximum(c[:3], c[3:])
        src = _io.crop_to_box(src, lo_b, hi_b)
        log(f"  cropped to the box {tuple(round(float(v), 1) for v in lo_b)} - "
            f"{tuple(round(float(v), 1) for v in hi_b)} mm: {src.GetSize()} voxels")

    sp = src.GetSpacing()
    fov = [round(n * v, 1) for n, v in zip(src.GetSize(), sp)]
    lo, hi = SANE_SPACING_MM
    if not all(lo <= v <= hi for v in sp):
        raise RuntimeError(
            f"voxel spacing {tuple(round(v, 3) for v in sp)} mm is outside the "
            f"sane range for dental CBCT ({lo}-{hi} mm). The scan metadata is "
            f"most likely broken: at this spacing the field of view comes out "
            f"as {fov} mm. Everything would be computed at the wrong scale.")
    log(f"  field of view {fov[0]}x{fov[1]}x{fov[2]} mm")

    # The logit buffer scales with the scan, not the tile, and on a large
    # volume it outweighs everything else put together. Checked BEFORE
    # resampling, from the grid size alone: the resampled volume itself takes
    # gigabytes, and on 2026-10-02 a scan with broken metadata (1 mm voxels,
    # a 616 mm frame) was resampled to 2053^3 — ~30 GB gone — and only then
    # refused. On a 16 GB laptop that is a frozen machine, not a message.
    n_grid = int(np.prod([round(n * v / cfg["spacing"]) for n, v in zip(src.GetSize(), sp)]))
    buf_gb = n_grid * cfg["num_classes"] * (2 if n_grid > 150_000_000 else 4) / 2**30
    if buf_gb > WARN_BUFFER_GB:
        log(f"  large volume: about {buf_gb:.1f} GB of RAM needed for the "
            "label buffer")
    ram = _free_ram_gb()
    if ram and buf_gb > 0.7 * ram:
        hint = ""
        if max(fov) > MAX_DENTAL_FOV_MM:
            hint = (f" The field of view, {max(fov):.0f} mm, is larger than any "
                    "dental CBCT: the voxel size written in the file is most "
                    "likely wrong.")
        raise RuntimeError(
            f"this scan needs about {buf_gb:.1f} GB of RAM for the label buffer, "
            f"and only {ram:.1f} GB is available. Crop the scan to the area of "
            "interest, or run it on a machine with more memory." + hint)

    grid = _io.to_training_grid(src, cfg["spacing"])
    vol = sitk.GetArrayFromImage(grid).astype(np.float32)
    log(f"  resampled to {cfg['orientation']}, {cfg['spacing']} mm: {vol.shape}")
    # The label buffer is not the peak: the signed fields for the meshes
    # (one float32 volume per class) are built while it is still alive. Logged
    # so that a run that dies in swap can be told from one that dies on the GPU.
    src_gb = (src_full.GetNumberOfPixels() * src_full.GetSizeOfPixelComponent()
              * src_full.GetNumberOfComponentsPerPixel()) / 2 ** 30
    peak_gb = src_gb + _peak_ram_estimate_gb(vol.size, cfg["num_classes"],
                                             len(opts.classes), opts.separate_teeth,
                                             opts.sub_voxel and opts.make_stl)
    log(f"  estimated peak RAM about {peak_gb:.1f} GB"
        + (f", {ram:.1f} GB free now" if ram else ""))
    if ram and peak_gb > ram:
        log("  WARNING: more than the free memory — Windows will page to disk, "
            "which is slow and may fail. Crop the scan to the area of interest.")

    # Raw HU are needed by the teeth pass: the numbering network has its own
    # intensity window. int16 costs half of float32 and HU fit in it.
    vol_hu = vol.astype(np.int16) if opts.separate_teeth else None

    norm = cfg["normalization"]
    vol = _io.normalize_ct(vol, {"percentile_00_5": norm["clip"][0],
                                 "percentile_99_5": norm["clip"][1],
                                 "mean": norm["mean"], "std": norm["std"]})

    t0 = time.time()
    # A reference, not a bare session: if the card is reset mid-run the
    # session is recreated, and the canal pass below must use the new one.
    ref = _infer.SessionRef(opts.model, opts.device, opts.threads)
    sess = ref.sess
    dev = _infer.session_device(sess)
    log(f"computing on: {dev}")
    # A silent fallback to the CPU is the worst thing that can happen here:
    # the same result, but six minutes instead of fifteen seconds, and nobody
    # understands why. On Linux it usually means the CUDA provider failed to load.
    if opts.device != "cpu" and "GPU" not in dev:
        log("  WARNING: the graphics card is not being used, computing on the CPU "
            "(about 15 times slower)")
        log(f"  available providers: {', '.join(_infer.available_providers())}")
    tile = _infer.fit_tile(opts.tile or cfg["tile"], vol.shape)
    # What was asked for is the tile fitted to the volume: a small (cropped)
    # volume shrinks it for geometry, which is not a memory problem and must
    # not be reported as one.
    fitted = tile
    log(f"  tile {tile[0]}x{tile[1]}x{tile[2]}, overlap {opts.overlap}")

    # On large volumes a float32 logit buffer runs to gigabytes, and float16
    # is precise enough for argmax by a wide margin: the gap between competing
    # classes is orders of magnitude larger than the half-precision step.
    acc_dtype = np.float16 if vol.size > 150_000_000 else np.float32
    logits, tile = _infer.predict_logits_auto(ref, vol, tile, opts.overlap,
                                              progress, acc_dtype, log)
    sess = ref.sess
    labels = logits.argmax(0).astype(np.uint8)
    t_pred = time.time() - t0
    log(f"  tile in use: {tile[0]}x{tile[1]}x{tile[2]}")
    log(f"  inference took {t_pred:.1f} s; {_sysinfo.snapshot()}")

    if opts.canal_refine and any(k.id == 5 for k in opts.classes):
        new = _refine_canal(vol, labels, ref, cfg, opts, grid.GetSpacing(), log, src, grid)
        sess = ref.sess
        if new is not labels:
            labels = new
            # The signed fields came from the first pass and no longer match
            # the canal — its surface is built from the mask instead.
            refined_canal = True
        else:
            refined_canal = False
    else:
        refined_canal = False

    ids = [k.id for k in opts.classes]
    fields = _signed_fields(logits, ids) if (opts.sub_voxel and opts.make_stl) else {}
    del logits
    log(f"  surfaces prepared; {_sysinfo.snapshot()}")

    sp = grid.GetSpacing()
    vox_cm3 = float(np.prod(sp)) / 1000.0
    report = {"input": real_path, "grid": list(vol.shape), "tile": list(tile),
              "requested_tile": list(fitted),
              "predict_seconds": round(t_pred, 1),
              "device": _infer.session_device(sess), "classes": {}}

    for k in opts.classes:
        mask = labels == k.id
        raw_n = _mesh.count_components(mask)
        mask = _mesh.drop_small(mask, sp, k.min_cm3)
        vol_cm3 = float(mask.sum()) * vox_cm3
        if not mask.any():
            log(f"  {k.name}: not found")
            report["classes"][k.key] = {"volume_cm3": 0.0, "components": 0, "stl": None}
            continue

        if not opts.make_stl:
            report["classes"][k.key] = {"volume_cm3": round(vol_cm3, 2),
                                        "components": _mesh.count_components(mask),
                                        "stl": None}
            continue

        if opts.sub_voxel and not (refined_canal and k.id == 5):
            f = fields[k.id]
            f[~mask & (f > 0)] = -1.0          # dropped specks are silenced in the field
            surface = f
        else:
            surface = mask.astype(np.float32) - 0.5

        pd = _mesh.build_surface(surface, grid, 0.0, opts.taubin,
                                 opts.pass_band, opts.decimate)
        name = f"{k.id:02d}_{k.key}.stl"
        pts, tris = _mesh.write_stl(pd, os.path.join(outdir, name))
        n = _mesh.count_components(mask)
        log(f"  {k.name}: {vol_cm3:.1f} cm3, {n} object(s)"
            + (f", {raw_n - n} speck(s) dropped" if raw_n > n else "")
            + f", {tris} triangles -> {name}")
        report["classes"][k.key] = {"volume_cm3": round(vol_cm3, 2), "components": n,
                                    "triangles": int(tris), "stl": name}

    if opts.separate_teeth:
        tm = opts.teeth_model or os.path.join(os.path.dirname(os.path.abspath(opts.model)), "teeth_fdi.onnx")
        tcfg_path = os.path.splitext(tm)[0] + ".json"
        if not os.path.isfile(tm):
            raise FileNotFoundError(f"teeth model not found: {tm}")
        if not os.path.isfile(tcfg_path):
            raise FileNotFoundError(f"no {os.path.basename(tcfg_path)} next to the teeth model — "
                                    "the weights and their json travel together")
        with open(tcfg_path, encoding="utf-8") as f:
            tcfg = json.load(f)
        log("separating teeth (FDI numbering)")
        t1 = time.time()
        sess2 = _infer.SessionRef(tm, opts.device, opts.threads)
        numbered, treport = _teeth.separate_teeth(
            vol_hu, labels, grid, sess2, tcfg, None, opts.overlap, fields, outdir,
            opts.taubin, opts.pass_band, opts.decimate, opts.make_stl, log, progress)
        treport["seconds"] = round(time.time() - t1, 1)
        report["teeth"] = treport
        log(f"  teeth pass took {treport['seconds']} s")
        if opts.save_labels:
            seg = _io.resample_labels_to(src_full, numbered, grid)
            sitk.WriteImage(seg, os.path.join(outdir, "teeth_labels.nii.gz"), useCompression=True)
        del sess2, numbered

    if opts.save_labels:
        seg = _io.resample_labels_to(src_full, labels, grid)
        p = os.path.join(outdir, "labels.nii.gz")
        sitk.WriteImage(seg, p, useCompression=True)
        log("  labels on the original grid -> labels.nii.gz")

    report["seconds"] = round(time.time() - t_all, 1)
    report["peak_ram_gb"] = _sysinfo.process_peak_gb()
    if opts.crop:
        report["crop"] = [float(v) for v in opts.crop]
    log(f"  {_sysinfo.snapshot()}")
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"done in {report['seconds']} s -> {real_outdir}")
    return report
