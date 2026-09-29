"""Individual teeth with FDI numbers — a second pass on top of dental9.

How it works (project ~/precise_teeth): dental9 yields the Upper/Lower Teeth
masks, each is cropped to its bbox + 5 mm, and in that crop the numbering
network sees a whole arch in one tile. Network outputs: 1..32 — FDI teeth
(11..18, 21..28, 31..38, 41..48), 33 — implant screw, 34 — implant crown,
35 — bridge.

Division of labour: the GEOMETRY of the teeth stays with dental9 (its
boundaries are verified against manual labels); the numbering network decides
WHERE THE BOUNDARY BETWEEN NEIGHBOURS RUNS and WHICH NUMBER each piece gets.
Implants, implant crowns and bridges come entirely from the second pass:
dental9 counts them as teeth.

Measured on 20 held-out DentVoxel cases (a scanner the network never saw):
Dice per tooth 0.952, number correct for 99.1%, nothing missed, nothing extra.
"""
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from . import infer as _infer
from . import io as _io
from . import mesh as _mesh

UPPER_TEETH, LOWER_TEETH = 3, 4          # dental9 classes
IMPLANT, IMPLANT_CROWN, BRIDGE = 33, 34, 35

# File and object names. A dentist reads an FDI number without a legend.
TOOTH_NAMES = {
    1: "Central Incisor", 2: "Lateral Incisor", 3: "Canine", 4: "First Premolar",
    5: "Second Premolar", 6: "First Molar", 7: "Second Molar", 8: "Third Molar",
}
PROSTHESIS = {IMPLANT: "implant", IMPLANT_CROWN: "implant_crown", BRIDGE: "bridge"}

# Anything smaller than this is a speck. The smallest real tooth (a lower
# incisor) is about 0.3 cm3; a root fragment at the frame edge is 0.05.
MIN_TOOTH_CM3 = 0.03
# Beyond this distance from the nearest numbered voxel the dental9 mask gets
# no number: it is either a dental9 speck or a tooth the second pass did not
# see — and silently handing it to a neighbour is worse than dropping it.
ASSIGN_MM = 2.0


def _components(mask: np.ndarray):
    """Connected components (26-connectivity), labels 1..k by decreasing size."""
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8))))
    arr = sitk.GetArrayFromImage(cc)
    return arr, int(arr.max())


def _bbox(arr: np.ndarray, value: int):
    idx = np.where(arr == value)
    return np.array([i.min() for i in idx]), np.array([i.max() + 1 for i in idx])


def _nearest(lab: np.ndarray, spacing_xyz) -> Tuple[np.ndarray, np.ndarray]:
    """Per voxel: distance (mm) to the nearest non-zero voxel and its label.
    Danielsson's Voronoi map — no scipy, which the bundle does not carry."""
    img = sitk.GetImageFromArray(lab.astype(np.uint8))
    img.SetSpacing(tuple(float(v) for v in spacing_xyz))
    f = sitk.DanielssonDistanceMapImageFilter()
    f.UseImageSpacingOn()
    f.SquaredDistanceOff()
    dist = sitk.GetArrayFromImage(f.Execute(img))
    near = sitk.GetArrayFromImage(f.GetVoronoiMap()).astype(np.uint8)
    return dist, near


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    img = sitk.BinaryDilate(sitk.GetImageFromArray(mask.astype(np.uint8)), [r, r, r])
    return sitk.GetArrayFromImage(img) > 0


def fdi(tooth_id: int) -> int:
    """id 1..32 -> FDI number 11..48."""
    q, t = divmod(tooth_id - 1, 8)
    return (q + 1) * 10 + t + 1


def tooth_label(tooth_id: int) -> str:
    f = fdi(tooth_id)
    side = {1: "Upper Right", 2: "Upper Left", 3: "Lower Left", 4: "Lower Right"}[f // 10]
    return f"Tooth {f} ({side} {TOOTH_NAMES[f % 10]})"


def clean_arch_mask(mask: np.ndarray, spacing, min_cm3: float = 0.135,
                    gap_mm: float = 25.0) -> np.ndarray:
    """The arch mask before the bbox: large pieces (>= 0.135 cm3) always stay —
    in scans with no front teeth the upper arch is two posterior groups 26+ mm
    apart, and the rule "no further than N mm from the largest" lost half the
    arch. Small pieces join only if they lie within gap_mm of what is already
    taken; lone voxels and false hits 100 mm from the arch stay out of the bbox.
    """
    cc, k = _components(mask)
    if k <= 1:
        return mask
    vox_cm3 = float(np.prod(spacing)) / 1000.0
    sizes = np.bincount(cc.ravel(), minlength=k + 1)[1:].astype(float)   # voxels in component i+1
    cm3 = sizes * vox_cm3
    boxes = {i: _bbox(cc, i + 1) for i in range(k) if sizes[i] >= 300}
    keep = {i for i in boxes if cm3[i] >= min_cm3}
    if not keep:
        keep = {int(np.argmax(sizes))}
        boxes.setdefault(int(np.argmax(sizes)), _bbox(cc, int(np.argmax(sizes)) + 1))
    lo = np.min([boxes[i][0] for i in keep], 0)
    hi = np.max([boxes[i][1] for i in keep], 0)
    gap = np.array([gap_mm / s for s in spacing[::-1]])       # in voxels along (z, y, x)
    small = [i for i in boxes if i not in keep]
    changed = True
    while changed:
        changed = False
        for i in small:
            if i in keep:
                continue
            a, b = boxes[i]
            if (np.maximum(0, np.maximum(a - hi, lo - b)) <= gap).all():
                keep.add(i); lo, hi = np.minimum(lo, a), np.maximum(hi, b); changed = True
    return np.isin(cc, [i + 1 for i in keep])


def _crop_box(mask: np.ndarray, margin_vox: int, shape) -> Tuple[slice, ...]:
    idx = np.where(mask)
    return tuple(slice(max(0, int(i.min()) - margin_vox), min(int(n), int(i.max()) + margin_vox + 1))
                 for i, n in zip(idx, shape))


def _one_instance_per_number(lab: np.ndarray, spacing) -> np.ndarray:
    """One tooth, one connected region. The largest piece of each number stays;
    the other pieces with the same number go to the nearest neighbour: two
    pieces sharing a number are either a speck or a numbering slip at the edge,
    and in both cases the neighbour is right more often than the duplicate."""
    out = lab.copy()
    for t in np.unique(lab):
        if t == 0 or t in PROSTHESIS:
            continue
        m = lab == t
        cc, k = _components(m)
        if k <= 1:
            continue
        out[m & (cc != 1)] = 0                      # component 1 is the largest
    # orphaned pieces -> the nearest surviving number (within ASSIGN_MM)
    orphan = (lab > 0) & (out == 0)
    if orphan.any() and out.any():
        dist, near = _nearest(out, spacing)
        take = orphan & (dist <= ASSIGN_MM)
        out[take] = near[take]
    return out


def _contacts(lab: np.ndarray):
    """Per label: how many boundary voxels in total and how many of them face
    each other label. One pass of shifts along the three axes instead of a
    dilation per tooth — on an arch crop that is a fraction of a second."""
    shell: Dict[int, int] = {}
    touch: Dict[int, Dict[int, int]] = {}
    for ax in range(3):
        a = np.moveaxis(lab, ax, 0)
        x, y = a[:-1], a[1:]
        diff = x != y
        for p, q in ((x[diff], y[diff]), (y[diff], x[diff])):
            ids, cnt = np.unique(p, return_counts=True)
            for i, c in zip(ids, cnt):
                if i:
                    shell[int(i)] = shell.get(int(i), 0) + int(c)
            both = (p > 0) & (q > 0)
            pairs = p[both].astype(np.int32) * 256 + q[both].astype(np.int32)
            ids, cnt = np.unique(pairs, return_counts=True)
            for k, c in zip(ids, cnt):
                i, j = int(k) // 256, int(k) % 256
                touch.setdefault(i, {})[j] = touch.setdefault(i, {}).get(j, 0) + int(c)
    return shell, touch


def qc_teeth(lab: np.ndarray, spacing, cfg: dict, own: set, log) -> Tuple[np.ndarray, list]:
    """Check on top that every number is a tooth, not a piece of its neighbour.

    What is caught, and where the thresholds come from (9 clinic test cases
    and 529 TF3 labelled cases):

    1. TWO NUMBERS ON ONE TOOTH — the network drew a boundary through a tooth
       and called the halves 24 and 25. Real neighbours touch over a contact
       facet — under 25% of the shell in every one of the nine healthy cases.
       A fragment of the same tooth abuts its "neighbour" over a whole cross
       section: 30-65%. A 25% threshold separates the two without a single
       overlap. Merge into one tooth; the larger part keeps the number.
    2. A NUMBERED SPECK — an isolated piece under 10% of the median volume for
       that tooth type (medians over 529 cases live in teeth_fdi.json). Drop
       it. Between 10 and 35% — keep, but flag in the report: it may be a tooth
       cut by the frame edge.
    """
    q = cfg.get("qc", {})
    merge_frac = float(q.get("merge_contact_fraction", 0.25))
    speck_frac = float(q.get("speck_below_median_fraction", 0.10))
    small_frac = float(q.get("small_below_median_fraction", 0.35))
    typical = {int(k): float(v) for k, v in cfg.get("typical_volume_cm3", {}).items()}
    vox_cm3 = float(np.prod(spacing)) / 1000.0
    out = lab.copy()
    notes = []

    for _round in range(8):                     # merges can expose new candidates
        ids = [int(t) for t in np.unique(out) if t and t in own]
        if not ids:
            break
        vol = {t: float((out == t).sum()) * vox_cm3 for t in ids}
        shell, touch = _contacts(out)
        changed = False
        for t in sorted(ids, key=lambda t: vol[t]):          # smallest first
            if t not in vol:
                continue
            med = typical.get(fdi(t) % 10, 0.5)
            nb = {j: c for j, c in touch.get(t, {}).items() if j in own and j != t}
            best, best_c = (max(nb.items(), key=lambda x: x[1]) if nb else (None, 0))
            frac = best_c / shell[t] if shell.get(t) else 0.0
            if best is not None and frac >= merge_frac:
                keep, drop = (best, t) if vol[best] >= vol[t] else (t, best)
                out[out == drop] = keep
                notes.append(f"tooth {fdi(drop)} merged into {fdi(keep)}: "
                             f"{vol[drop]:.2f} cm3 touching it over {frac*100:.0f}% of its surface")
                vol[keep] = vol[keep] + vol.pop(drop)
                changed = True
                break                                       # contacts are stale now
            if best is None and vol[t] < speck_frac * med:
                out[out == t] = 0
                notes.append(f"tooth {fdi(t)} dropped: {vol[t]:.2f} cm3 is "
                             f"{vol[t]/med*100:.0f}% of a typical one and touches nothing")
                vol.pop(t)
                changed = True
                break
        if not changed:
            break

    ids = [int(t) for t in np.unique(out) if t and t in own]
    for t in ids:
        med = typical.get(fdi(t) % 10, 0.5)
        v = float((out == t).sum()) * vox_cm3
        if v < small_frac * med:
            notes.append(f"tooth {fdi(t)} is small: {v:.2f} cm3, {v/med*100:.0f}% of typical "
                         "— cut by the frame edge, or not a whole tooth")
    for n in notes:
        log("  " + n)
    return out, notes


def separate_teeth(vol_hu: np.ndarray, labels: np.ndarray, grid: sitk.Image,
                   sess, cfg: dict, tile: Optional[Sequence[int]], overlap: float,
                   fields: Dict[int, np.ndarray], outdir: str, taubin: int,
                   pass_band: float, decimate: float, make_stl: bool,
                   log: Callable[[str], None], progress=None) -> Tuple[np.ndarray, dict]:
    """The second pass over both arches. Returns the number map (uint8, 0..35)
    on the grid and a report for report.json.

    vol_hu  — the volume in Hounsfield units (dental9 normalises its own way;
              the numbering network has its own window, stored in its json);
    labels  — the dental9 label map;
    fields  — dental9's signed fields for the teeth classes, if available: the
              outer tooth surface is built from them without voxel steps.
    """
    spacing = grid.GetSpacing()                       # (x, y, z) mm
    vox_cm3 = float(np.prod(spacing)) / 1000.0
    margin = int(round(cfg.get("crop_margin_mm", 5.0) / spacing[0]))
    divisor = cfg.get("tile_divisor", [32, 32, 32])
    norm = cfg["normalization"]
    props = {"percentile_00_5": norm["clip"][0], "percentile_99_5": norm["clip"][1],
             "mean": norm["mean"], "std": norm["std"]}
    upper_ids = set(cfg.get("upper_ids", range(1, 17)))
    lower_ids = set(cfg.get("lower_ids", range(17, 33)))

    numbered = np.zeros(labels.shape, np.uint8)
    report = {"arches": {}, "teeth": {}, "prosthesis": {}}
    for arch, d9_id, own in (("upper", UPPER_TEETH, upper_ids), ("lower", LOWER_TEETH, lower_ids)):
        mask = labels == d9_id
        if mask.sum() * vox_cm3 < 0.5:
            log(f"  {arch} arch: too little tooth mask, skipped")
            report["arches"][arch] = {"skipped": "no teeth"}
            continue
        box_mask = clean_arch_mask(mask, spacing)
        sl = _crop_box(box_mask, margin, labels.shape)
        crop = _io.normalize_ct(vol_hu[sl].astype(np.float32), props)
        t = _infer.fit_tile(tile or cfg["tile"], crop.shape, divisor)
        logits, used = _infer.predict_logits_auto(sess, crop, t, overlap, progress,
                                                  np.float32, log, divisor)
        pred = logits.argmax(0).astype(np.uint8)
        del logits
        # only this arch and prostheses; teeth of the other arch inside the crop are not ours
        allowed = np.isin(pred, list(own) + [IMPLANT, IMPLANT_CROWN, BRIDGE])
        pred[~allowed] = 0
        # prostheses only inside the (dilated) dental9 mask: that is where it counted them as teeth
        near_mask = _dilate(mask[sl], 3)
        pred[np.isin(pred, list(PROSTHESIS)) & ~near_mask] = 0

        # Geometry is the dental9 mask; the number comes from the nearest numbered voxel.
        sub = np.zeros(pred.shape, np.uint8)
        pros = np.isin(pred, list(PROSTHESIS))
        sub[pros] = pred[pros]
        teeth_here = mask[sl] & ~pros
        seed = np.where(np.isin(pred, list(own)), pred, 0).astype(np.uint8)
        if seed.any():
            dist, near = _nearest(seed, spacing)
            take = teeth_here & (dist <= ASSIGN_MM)
            sub[take] = near[take]
        lost = float((teeth_here & (sub == 0)).sum()) * vox_cm3
        sub = _one_instance_per_number(sub, spacing)
        sub, qc_notes = qc_teeth(sub, spacing, cfg, own, log)
        region = numbered[sl]
        region[(region == 0) & (sub > 0)] = sub[(region == 0) & (sub > 0)]
        n_teeth = len([v for v in np.unique(sub) if v in own])
        log(f"  {arch} arch: crop {crop.shape[0]}x{crop.shape[1]}x{crop.shape[2]}, "
            f"tile {used[0]}x{used[1]}x{used[2]}, {n_teeth} teeth numbered"
            + (f", {lost:.2f} cm3 of tooth mask left unnumbered" if lost > 0.05 else ""))
        report["arches"][arch] = {"crop": list(crop.shape), "tile": list(used),
                                  "teeth": n_teeth, "unnumbered_cm3": round(lost, 2),
                                  "qc": qc_notes}

    # --- one STL per tooth ---
    tdir = os.path.join(outdir, "teeth")
    if make_stl:
        os.makedirs(tdir, exist_ok=True)
    for t in sorted(int(v) for v in np.unique(numbered)):
        if t == 0:
            continue
        m = numbered == t
        m = _mesh.drop_small(m, spacing, MIN_TOOTH_CM3)
        if not m.any():
            continue
        vol = float(m.sum()) * vox_cm3
        if t in PROSTHESIS:
            key = PROSTHESIS[t]
            entry = {"volume_cm3": round(vol, 2), "components": _mesh.count_components(m), "stl": None}
        else:
            key = str(fdi(t))
            entry = {"fdi": fdi(t), "name": tooth_label(t), "volume_cm3": round(vol, 2), "stl": None}
        if make_stl:
            arch_field = fields.get(UPPER_TEETH if t in upper_ids else LOWER_TEETH) if t not in PROSTHESIS else None
            if arch_field is not None:
                # outer surface from dental9's signed field, the boundary with
                # the neighbour from the numbers; the field outside this tooth is silenced
                f = arch_field.copy()
                f[~m & (f > 0)] = -1.0
                surface = f
            else:
                surface = m.astype(np.float32) - 0.5
            pd = _mesh.build_surface(surface, grid, 0.0, taubin, pass_band, decimate)
            name = f"tooth_{fdi(t)}.stl" if t not in PROSTHESIS else f"{key}.stl"
            _pts, tris = _mesh.write_stl(pd, os.path.join(tdir, name))
            entry["stl"] = os.path.join("teeth", name)
            entry["triangles"] = int(tris)
        (report["prosthesis"] if t in PROSTHESIS else report["teeth"])[key] = entry
    n = len(report["teeth"])
    log(f"  teeth separated: {n}"
        + (", " + ", ".join(f"{k} {v['components']}" for k, v in report["prosthesis"].items()) if report["prosthesis"] else ""))
    return numbered, report
