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
from . import teeth as _teeth
from . import winpath as _winpath
from .classes import CLASSES, Klass

# A sane voxel spacing for dental CBCT. Outside this range the scan metadata is
# broken, and refusing at once beats computing for half an hour on garbage:
# the whole pipeline works in physical millimetres, so a wrong spacing shifts
# both the intensity window and the mesh scale. Caught on a Sirona export that
# claimed 1.0 mm instead of 0.15 — the field of view came out as "616 mm".
SANE_SPACING_MM = (0.04, 1.2)
# The logit buffer is ten classes over the whole volume. Above this we warn
# that a lot of RAM is needed; above a share of the available RAM we refuse,
# otherwise the machine swaps and freezes instead of failing honestly.
WARN_BUFFER_GB = 4.0

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
    make_stl: bool = True       # off for bulk accuracy evaluation
    threads: int = 0
    separate_teeth: bool = False   # second pass: individual teeth with FDI numbers
    teeth_model: str = ""          # weights for that pass; default teeth_fdi.onnx next to the main model


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


def _refine_canal(vol: np.ndarray, labels: np.ndarray, sess, cfg: dict,
                  opts: "Options", spacing, log) -> np.ndarray:
    """A second pass over the canal only, and only inside the mandible.

    The user's idea, and exactly right: when the canal comes out broken the
    intensities are to blame, not the network. Map them onto the training
    domain WITHIN THE MANDIBLE and predict again, taking ONLY the canal from
    the second pass. The other eight classes are not touched at all — measured,
    none of them moved by a hundredth.

    Measured on a scan where the canal was lost: 0.22 cm3 in six fragments
    (33% of the course) became 0.61 cm3 in four (88%), reference 0.60.
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
        finish_out()
        cleanup_in()


def _segment(path: str, outdir: str, real_path: str, real_outdir: str,
             opts: Options, log, progress) -> dict:
    cfg = load_config(opts)
    os.makedirs(outdir, exist_ok=True)
    t_all = time.time()

    log(f"reading: {real_path}")
    src = _io.read_volume(path)
    log(f"  volume {src.GetSize()}, spacing {tuple(round(s, 3) for s in src.GetSpacing())} mm")

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

    grid = _io.to_training_grid(src, cfg["spacing"])
    vol = sitk.GetArrayFromImage(grid).astype(np.float32)
    log(f"  resampled to {cfg['orientation']}, {cfg['spacing']} mm: {vol.shape}")

    # The logit buffer scales with the scan, not the tile, and on a large
    # volume it outweighs everything else put together.
    buf_gb = vol.size * cfg["num_classes"] * (2 if vol.size > 150_000_000 else 4) / 2**30
    if buf_gb > WARN_BUFFER_GB:
        log(f"  large volume: about {buf_gb:.1f} GB of RAM needed for the "
            "label buffer")
    ram = _free_ram_gb()
    if ram and buf_gb > 0.7 * ram:
        raise RuntimeError(
            f"this scan needs about {buf_gb:.1f} GB of RAM for the label buffer, "
            f"and only {ram:.1f} GB is available. Crop the scan to the area of "
            "interest, or run it on a machine with more memory.")

    # Raw HU are needed by the teeth pass: the numbering network has its own
    # intensity window. int16 costs half of float32 and HU fit in it.
    vol_hu = vol.astype(np.int16) if opts.separate_teeth else None

    norm = cfg["normalization"]
    vol = _io.normalize_ct(vol, {"percentile_00_5": norm["clip"][0],
                                 "percentile_99_5": norm["clip"][1],
                                 "mean": norm["mean"], "std": norm["std"]})

    t0 = time.time()
    sess = _infer.make_session(opts.model, opts.device, opts.threads)
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
    log(f"  tile {tile[0]}x{tile[1]}x{tile[2]}, overlap {opts.overlap}")

    # On large volumes a float32 logit buffer runs to gigabytes, and float16
    # is precise enough for argmax by a wide margin: the gap between competing
    # classes is orders of magnitude larger than the half-precision step.
    acc_dtype = np.float16 if vol.size > 150_000_000 else np.float32
    logits, tile = _infer.predict_logits_auto(sess, vol, tile, opts.overlap,
                                              progress, acc_dtype, log)
    labels = logits.argmax(0).astype(np.uint8)
    t_pred = time.time() - t0
    log(f"  tile in use: {tile[0]}x{tile[1]}x{tile[2]}")
    log(f"  inference took {t_pred:.1f} s")

    if opts.canal_refine and any(k.id == 5 for k in opts.classes):
        new = _refine_canal(vol, labels, sess, cfg, opts, grid.GetSpacing(), log)
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

    sp = grid.GetSpacing()
    vox_cm3 = float(np.prod(sp)) / 1000.0
    report = {"input": real_path, "grid": list(vol.shape), "tile": list(tile),
              "requested_tile": list(opts.tile or cfg["tile"]),
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
        sess2 = _infer.make_session(tm, opts.device, opts.threads)
        numbered, treport = _teeth.separate_teeth(
            vol_hu, labels, grid, sess2, tcfg, None, opts.overlap, fields, outdir,
            opts.taubin, opts.pass_band, opts.decimate, opts.make_stl, log, progress)
        treport["seconds"] = round(time.time() - t1, 1)
        report["teeth"] = treport
        log(f"  teeth pass took {treport['seconds']} s")
        if opts.save_labels:
            seg = _io.resample_labels_to(src, numbered, grid)
            sitk.WriteImage(seg, os.path.join(outdir, "teeth_labels.nii.gz"), useCompression=True)
        del sess2, numbered

    if opts.save_labels:
        seg = _io.resample_labels_to(src, labels, grid)
        p = os.path.join(outdir, "labels.nii.gz")
        sitk.WriteImage(seg, p, useCompression=True)
        log("  labels on the original grid -> labels.nii.gz")

    report["seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"done in {report['seconds']} s -> {real_outdir}")
    return report
