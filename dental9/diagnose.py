"""Hardware check: what is there, what is missing and what to do about it.

Written for the person in the clinic, not for a developer. The main case it
exists for: the graphics card was not picked up, segmentation takes six
minutes instead of fifteen seconds, and there is no error message — because
there was no error, onnxruntime simply took the CPU.

Checked by doing, not by listing: open a session on the chosen provider and
push a small tile through it. The list of available providers lies —
DirectML and CUDA are in the build but may fail to come up (no driver, no
cuDNN, card busy).
"""
import json
import os
import platform
import sys
import time
from typing import Optional

import numpy as np

# A small tile for the probe. A multiple of the (32, 64, 64) divisor and small
# enough for the check to take seconds even on the CPU.
PROBE_TILE = (64, 128, 128)
# A standard scan: 100 mm frame (334³), tile 160×320×320 with overlap 0.5 —
# that is 16 tiles.
CASE_TILES = 16
CASE_TILE_VOXELS = 160 * 320 * 320
NUM_CLASSES = 10

OK, WARN, BAD = "ok", "warn", "bad"


def _ram_gb() -> Optional[float]:
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
            return m.ullTotalPhys / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:                                   # noqa: BLE001
        return None


def _gpu_names() -> list:
    """Graphics card names — so the user sees the check found their card.
    Ask the system, not onnxruntime: it does not show names."""
    try:
        import subprocess
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=20).stdout
        else:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:                                   # noqa: BLE001
        return []


def _accum_seconds_per_voxel() -> float:
    """What it costs to accumulate one tile into the logit buffer.

    Measured separately, and for a reason: on a graphics card this is half of
    the total time. The network does a tile in a fraction of a second, then
    the CPU adds ten classes over the whole volume — an estimate "from the
    network alone" was off by a factor of two.
    """
    tile = PROBE_TILE
    out = np.random.randn(NUM_CLASSES, *tile).astype(np.float32)
    g = np.random.rand(*tile).astype(np.float32)
    acc = np.zeros((NUM_CLASSES, *tile), np.float32)
    w = np.zeros(tile, np.float32)
    t0 = time.time()
    for _ in range(3):
        acc += out * g
        w += g
    return (time.time() - t0) / 3.0 / float(np.prod(tile))


def _probe(model: str, device: str, full_tile=None) -> dict:
    """Open a session and actually run one tile.

    On a GPU, then also the working tile the segmentation uses. The small
    probe only proves the card comes up; whether the real tile fits into its
    memory, and how close it runs to the driver timeout, is a separate
    question — and the one that fails on laptops (2026-09-29, RTX 2070 Max-Q:
    the check was green, the run died on the second tile).
    """
    from . import infer as _infer
    r = {"device_asked": device}
    try:
        t0 = time.time()
        sess = _infer.make_session(model, device)
        r["load_seconds"] = round(time.time() - t0, 1)
        r["provider"] = sess.get_providers()[0]
        r["device_used"] = _infer.session_device(sess)
        x = np.random.randn(1, 1, *PROBE_TILE).astype(np.float32)
        name = sess.get_inputs()[0].name
        sess.run(None, {name: x})                       # warm-up, the first run
        t0 = time.time()                                # is always the slowest
        sess.run(None, {name: x})
        dt = time.time() - t0
        r["probe_seconds"] = round(dt, 2)
        per_vox = dt / float(np.prod(PROBE_TILE)) + _accum_seconds_per_voxel()
        r["case_seconds_estimate"] = int(round(
            CASE_TILES * CASE_TILE_VOXELS * per_vox))
        r["status"] = OK
    except Exception as e:                              # noqa: BLE001
        r["status"] = BAD
        r["error"] = (_infer.readable_error(e).splitlines() or [""])[0][:300]
        return r
    if full_tile and "GPU" in r.get("device_used", ""):
        t = tuple(int(v) for v in full_tile)
        r["full_tile"] = list(t)
        try:
            x = np.random.randn(1, 1, *t).astype(np.float32)
            sess.run(None, {name: x})                   # warm-up for this shape
            t0 = time.time()
            sess.run(None, {name: x})
            r["full_tile_seconds"] = round(time.time() - t0, 2)
            from . import sysinfo
            r["vram_after_full_tile"] = sysinfo.vram_used()
        except Exception as e:                          # noqa: BLE001
            r["full_tile_error"] = _infer.readable_error(e)[:400]
    return r


def _advice(rep: dict) -> list:
    """Advice. Each line is about what to do, not about what happened."""
    out = []
    gpu = rep.get("gpu", {})
    used = gpu.get("device_used", "")
    win = rep["platform"]["system"] == "Windows"

    if gpu.get("status") == OK and "GPU" in used:
        est = gpu.get("case_seconds_estimate")
        out.append((OK, f"Graphics card is working: {used}. About {est} s per "
                        "scan, plus a few seconds for reading and meshing."))
        t = gpu.get("full_tile")
        ts = "x".join(str(v) for v in t) if t else ""
        if gpu.get("full_tile_error"):
            out.append((WARN, f"The working tile {ts} does NOT fit this graphics "
                              "card. Segmentation will retry with smaller tiles; if "
                              "it still fails, crop the scan or compute on the CPU. "
                              f"onnxruntime says: {gpu['full_tile_error'][:200]}"))
        elif gpu.get("full_tile_seconds", 0) > 1.5:
            out.append((WARN, f"The working tile {ts} takes "
                              f"{gpu['full_tile_seconds']} s on this card — close to "
                              "the 2 s Windows driver timeout, which resets the card. "
                              "On a laptop, plug in the charger and pick maximum "
                              "performance."))
    else:
        est = (rep.get("cpu") or gpu or {}).get("case_seconds_estimate")
        how_long = (f", roughly {est // 60} min per scan instead of ten seconds."
                    if est else ".")
        out.append((BAD, "The graphics card is NOT being used — this will run on "
                         "the CPU" + how_long))
        # A card present in the system that the provider could not bring up
        # is a different illness with a different cure, and the user needs to
        # know which one they have.
        if rep.get("gpu_names"):
            out.append((WARN, "A graphics card is present: "
                              f"{', '.join(rep['gpu_names'])}. So the problem is "
                              "the driver or this build, not the hardware."))
        else:
            out.append((WARN, "No graphics card could be detected. There may "
                              "simply be none — then everything works correctly, "
                              "just slowly."))
        if win:
            out.append((WARN, "Windows: DirectML works on any graphics card, "
                              "including integrated and AMD ones. The usual cause "
                              "is an outdated display driver — update it and "
                              "check again."))
            if not rep["providers"].get("has_dml"):
                out.append((BAD, "This build has no DirectML. A build with "
                                 "onnxruntime-directml is needed — tell the "
                                 "developer."))
        else:
            out.append((WARN, "Linux: CUDA and cuDNN are bundled, so the only thing "
                              "needed from the system is an NVIDIA driver that "
                              "supports CUDA 12 (version 525 or newer). Check with "
                              "'nvidia-smi' in a terminal: if it is not found, the "
                              "driver is not installed."))
        if gpu.get("error"):
            out.append((WARN, f"onnxruntime says: {gpu['error']}"))

    ram = rep["platform"].get("ram_gb")
    if ram and ram < 8:
        out.append((WARN, f"Only {ram:.0f} GB of RAM. The label buffer takes "
                          "several gigabytes on a large volume; 8 GB or less may "
                          "not be enough."))
    m = rep.get("model", {})
    if not m.get("exists"):
        out.append((BAD, "Model file not found. Set the path to dental9.onnx in "
                         "the add-on preferences."))
    elif not m.get("config"):
        out.append((BAD, "There is no dental9.json next to the model — without it "
                         "the intensity window and the grid spacing are unknown. "
                         "These two files always change together."))
    return out


def collect(model: str) -> dict:
    import onnxruntime as ort

    from . import infer as _infer
    from . import sysinfo

    have = _infer.available_providers()
    rep = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "frozen": bool(getattr(sys, "frozen", False)),
            "ram_gb": _ram_gb(),
            "cpu_count": os.cpu_count(),
        },
        "onnxruntime": {"version": ort.__version__, "providers": have},
        "providers": {"has_dml": "DmlExecutionProvider" in have,
                      "has_cuda": "CUDAExecutionProvider" in have},
        "gpu_names": _gpu_names(),
        "gpus": sysinfo.gpus(),
        "model": {"path": model, "exists": os.path.isfile(model)},
    }
    rep["platform"]["cpu"] = sysinfo.cpu_name()
    if rep["model"]["exists"]:
        rep["model"]["size_mb"] = round(os.path.getsize(model) / 1e6)
        cfg = None
        for c in (os.path.splitext(model)[0] + ".json",
                  os.path.join(os.path.dirname(model), "dental9.json")):
            if os.path.isfile(c):
                try:
                    with open(c, encoding="utf-8") as f:
                        cfg = json.load(f)
                    break
                except Exception:                       # noqa: BLE001
                    pass
        rep["model"]["config"] = bool(cfg)
        if cfg:
            rep["model"].update({"epoch": cfg.get("epoch"),
                                 "precision": cfg.get("precision", "fp32"),
                                 "dataset": cfg.get("dataset"),
                                 "spacing": cfg.get("spacing")})
        # One probe only: if the card did not come up, onnxruntime already
        # fell back to the CPU and the measurement is a CPU one — no point
        # measuring it twice.
        rep["gpu"] = _probe(model, "auto", (cfg or {}).get("tile"))
    rep["advice"] = [{"level": lvl, "text": txt} for lvl, txt in _advice(rep)]
    return rep


def format_text(rep: dict) -> str:
    from . import sysinfo
    p = rep["platform"]
    lines = [
        "hardware check",
        "-" * 60,
        f"system:        {p['system']} {p['release']} ({p['machine']})",
        f"cpu:           {p['cpu_count']} cores"
        + (f", {p['ram_gb']:.0f} GB RAM" if p.get("ram_gb") else ""),
        f"processor:     {p.get('cpu', '')}",
        f"graphics:      {'; '.join(sysinfo.describe_gpu(g) for g in rep.get('gpus', [])) or ', '.join(rep['gpu_names']) or 'not detected'}",
        f"onnxruntime:   {rep['onnxruntime']['version']}",
        f"providers:     {', '.join(rep['onnxruntime']['providers'])}",
    ]
    m = rep["model"]
    if m["exists"]:
        lines.append(f"model:         {m['size_mb']} MB"
                     + (f", {m.get('precision')}, epoch {m.get('epoch')}"
                        if m.get("config") else ", dental9.json missing"))
    else:
        lines.append(f"model:         NOT FOUND ({m['path']})")
    for key in ("gpu", "cpu"):
        r = rep.get(key)
        if not r:
            continue
        where = "GPU" if "GPU" in r.get("device_used", "") else "CPU"
        if r["status"] == OK:
            lines.append(f"probe on {where}:  {r['device_used']}, "
                         f"{r['probe_seconds']} s for the probe tile "
                         f"-> ~{r['case_seconds_estimate']} s per scan")
        else:
            lines.append(f"probe on {where}:  failed — {r.get('error', '')}")
        if r.get("full_tile"):
            ts = "x".join(str(v) for v in r["full_tile"])
            lines.append(f"working tile:  {ts}: "
                         + (f"FAILED — {r['full_tile_error']}" if r.get("full_tile_error")
                            else f"{r.get('full_tile_seconds')} s"
                            + (f", VRAM used {r['vram_after_full_tile']}"
                               if r.get("vram_after_full_tile") else "")))
    lines.append("-" * 60)
    mark = {OK: "[ ok ]", WARN: "[ !  ]", BAD: "[ !! ]"}
    for a in rep["advice"]:
        lines.append(f"{mark[a['level']]} {a['text']}")
    return "\n".join(lines)
