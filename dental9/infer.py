"""Sliding-window inference on onnxruntime. Torch is not on the production path.

Reproduces nnU-Net's prediction rather than inventing its own: a step of half
a tile, Gaussian weights, LOGITS accumulated (before argmax), edge padding.
Agreement with nnUNetv2_predict is checked by deploy/scripts/eval_vs_manual.py.

Compute on the GPU: DirectML on Windows, CUDA on Linux; the CPU is only the
fallback path, an order of magnitude slower.
"""
import os
import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# The tile divisor is the product of the network strides: five downsamplings
# on the first axis, six on the others. Not negotiable: otherwise the skip
# connections do not add up with the decoder.
DIVISOR = (32, 64, 64)


def available_providers() -> List[str]:
    import onnxruntime as ort
    return list(ort.get_available_providers())


def pick_providers(device: str = "auto") -> List[str]:
    """Provider order. The first available one in the list is the one that runs.

    DirectML goes before CUDA on purpose: on Windows it exists on every
    graphics card (integrated and AMD included) and needs neither the CUDA
    Toolkit nor a matching NVIDIA driver — and nobody in a clinic will install
    those.
    """
    have = available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    order = {"auto": ["DmlExecutionProvider", "CUDAExecutionProvider"],
             "dml": ["DmlExecutionProvider"],
             "cuda": ["CUDAExecutionProvider"]}[device]
    picked = [p for p in order if p in have]
    if not picked and device != "auto":
        # Each build carries one GPU provider (DirectML on Windows, CUDA on
        # Linux): say what to choose instead of listing provider names.
        name = {"dml": "DirectML", "cuda": "CUDA"}[device]
        raise RuntimeError(
            f"{name} is not part of this build; set Compute on to Automatic "
            f"(available here: {have})")
    return picked + ["CPUExecutionProvider"]


def make_session(path: str, device: str = "auto", threads: int = 0):
    import onnxruntime as ort
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model file not found: {path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
    # DirectML arenas grow on inputs of varying size, and our tile is constant
    # only within one scan — at the edges it gets cropped.
    opts.enable_mem_pattern = False
    # The weights live inside the add-on, i.e. under the user's profile, and
    # a user called 王伟 has a non-ASCII path there. Same defence as for the
    # scan; the session holds the graph in memory, so the staged copy (if a
    # copy was needed at all) can go right away.
    from . import winpath
    safe, cleanup = winpath.readable(path)
    try:
        sess = ort.InferenceSession(safe, opts, providers=pick_providers(device))
    finally:
        cleanup()
    return sess


def session_device(sess) -> str:
    p = sess.get_providers()[0]
    return {"DmlExecutionProvider": "DirectML (GPU)",
            "CUDAExecutionProvider": "CUDA (GPU)",
            "CPUExecutionProvider": "CPU"}.get(p, p)


def _gaussian(patch: Sequence[int], sigma_scale: float = 0.125) -> np.ndarray:
    """Tile weights as in nnU-Net: a Gaussian with sigma = 1/8 of the side.

    The point is to discount the tile edges. A voxel at the edge lacks half
    of its context and the prediction there is systematically worse; in the
    overlap it is outvoted by the centre of the neighbouring tile. Without
    this the seams show as steps, most visibly on the thin mandibular canal.
    """
    coords = [np.arange(n, dtype=np.float32) - (n - 1) / 2.0 for n in patch]
    g = np.ones(tuple(patch), np.float32)
    for ax, (c, n) in enumerate(zip(coords, patch)):
        s = max(n * sigma_scale, 1e-6)
        line = np.exp(-(c ** 2) / (2.0 * s * s)).astype(np.float32)
        g *= line.reshape([-1 if i == ax else 1 for i in range(3)])
    g /= g.max()
    # A zero in the corners would become a hole where there is only one tile (frame edge).
    g = np.maximum(g, g.max() * 1e-3)
    return g


def fit_tile(tile: Sequence[int], shape: Sequence[int],
             divisor: Sequence[int] = DIVISOR) -> Tuple[int, int, int]:
    """A tile no larger than the volume, a multiple of the divisor, but at least
    two divisors.

    Two, not one: with one, a single voxel per axis is left at the bottom of
    the network and InstanceNorm fails — there is nothing to take a variance
    over.

    The divisor is a parameter, not a module constant: the tooth-numbering
    model has (32, 32, 32), dental9 has (32, 64, 64). It lives in the json
    next to the weights.
    """
    out = []
    for t, n, d in zip(tile, shape, divisor):
        v = min(int(t), max(int(n), 2 * d))
        v = max(2 * d, (v // d) * d)
        out.append(v)
    return tuple(out)


def readable_error(e: BaseException) -> str:
    """The text of an error, including the one onnxruntime failed to deliver.

    On a non-English Windows a DirectML failure carries the system message in
    the locale code page (cp1251 and the like). pybind11 decodes it as UTF-8,
    fails, and what reaches Python is a UnicodeDecodeError about a codec
    instead of the actual error — seen on 2026-09-29, "'utf-8' codec can't
    decode byte 0xc3", on a laptop that most likely ran out of video memory.
    The original bytes are still inside the exception; decode them with the
    code page they were written in.
    """
    if isinstance(e, UnicodeDecodeError) and isinstance(e.object, (bytes, bytearray)):
        enc = "mbcs" if sys.platform == "win32" else "latin-1"
        try:
            return bytes(e.object).decode(enc, "replace").strip()
        except Exception:                               # noqa: BLE001
            return bytes(e.object).decode("latin-1", "replace").strip()
    return str(e).strip()


class TileError(RuntimeError):
    """A tile failed to run. Carries where it happened, so the log can say
    which tile, how large and after how long — not just that it failed."""

    def __init__(self, text: str, tile, index: int, total: int):
        super().__init__(text)
        self.text, self.tile, self.index, self.total = text, tuple(tile), index, total


def _starts(n: int, p: int, step: int) -> List[int]:
    if n <= p:
        return [0]
    pos = list(range(0, n - p + 1, step))
    if pos[-1] != n - p:
        pos.append(n - p)
    return pos


def predict_logits(sess, vol: np.ndarray, tile: Sequence[int],
                   overlap: float = 0.5, progress=None,
                   acc_dtype=np.float32, divisor: Sequence[int] = DIVISOR,
                   log: Optional[Callable[[str], None]] = None) -> np.ndarray:
    """Logits (classes, z, y, x) on the input volume's grid.

    The accumulator buffer scales with the SCAN, not the tile: 10 classes over
    the whole volume. On the standard 100 mm frame (334³) that is 1.5 GB in
    float32, on a 587³ frame 8 GB. It lives in RAM, not in VRAM, so the
    graphics card only needs the tile itself.

    With `log`, the first tile's time and the memory after it are logged, and
    at the end the slowest tile: a tile near two seconds on a GPU is at the
    edge of the Windows driver timeout (TDR), which resets the card.
    """
    from . import sysinfo
    vol = np.ascontiguousarray(vol, dtype=np.float32)
    tile = fit_tile(tile, vol.shape, divisor)
    # The frame may be thinner than the tile — then pad with air at the edge.
    pad = [(0, max(0, t - n)) for t, n in zip(tile, vol.shape)]
    if any(b for _, b in pad):
        vol = np.pad(vol, pad, mode="edge")
    shape = vol.shape

    step = [max(1, int(round(t * (1 - overlap)))) for t in tile]
    starts = [_starts(shape[i], tile[i], step[i]) for i in range(3)]
    total = len(starts[0]) * len(starts[1]) * len(starts[2])

    gauss = _gaussian(tile)
    name = sess.get_inputs()[0].name
    acc: Optional[np.ndarray] = None
    wsum = np.zeros(shape, np.float32)
    done = 0
    times: List[float] = []
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                sl = (slice(z, z + tile[0]), slice(y, y + tile[1]),
                      slice(x, x + tile[2]))
                t0 = time.time()
                try:
                    out = sess.run(None, {name: vol[sl][None, None]})[0][0]
                except Exception as e:                   # noqa: BLE001
                    text = readable_error(e)
                    if log:
                        last = f", the previous one took {times[-1]:.2f} s" if times else ""
                        log(f"  tile {done + 1}/{total} at z{z} y{y} x{x} failed after "
                            f"{time.time() - t0:.2f} s{last}")
                        log(f"  memory at the failure: {sysinfo.snapshot()}")
                    raise TileError(text, tile, done + 1, total) from e
                times.append(time.time() - t0)
                if log and len(times) == 1:
                    log(f"  first tile: {times[0]:.2f} s (includes warm-up); "
                        f"{sysinfo.snapshot()}")
                if acc is None:
                    acc = np.zeros((out.shape[0], *shape), acc_dtype)
                acc[(slice(None), *sl)] += (out * gauss).astype(acc_dtype)
                wsum[sl] += gauss
                done += 1
                if progress:
                    progress(done, total)
    if log and len(times) > 1:
        rest = times[1:]
        log(f"  tiles: {len(times)}, {sum(rest) / len(rest):.2f} s on average, "
            f"slowest {max(rest):.2f} s")
    # Dividing is optional — argmax does not change under a positive per-voxel
    # factor — but the logits are wanted in a meaningful form for thresholds
    # and debugging.
    acc /= np.maximum(wsum, 1e-6).astype(acc_dtype)
    crop = tuple(slice(0, n - b) for (_, b), n in zip(pad, shape))
    return acc[(slice(None), *crop)]


# The fallback ladder for running out of video memory. The network is fully
# convolutional, so the tile is a run-time parameter, not part of the
# architecture: the same weights run on an 80 GB card and on a 6 GB one. The
# price is measured on the held-out test: tile 128 costs 0.002-0.007 Dice on
# every class except the nasal cavity (0.920 -> 0.881).
TILE_LADDER = [(160, 320, 320), (128, 256, 256), (96, 192, 192), (64, 128, 128)]

_OOM_MARKS = ("out of memory", "outofmemory", "failed to allocate",
              "allocation failed", "hipErrorOutOfMemory", "cudaErrorMemoryAllocation",
              "E_OUTOFMEMORY", "insufficient",
              # The HRESULTs, because the words around them are localised on a
              # non-English Windows: E_OUTOFMEMORY and DXGI's own out-of-memory.
              "8007000e", "887a0004")

# The card was reset under us: the driver timeout (TDR) on a tile that runs
# too long, or DirectML reporting video memory exhaustion as a lost device.
# The session is dead after that and has to be recreated; a smaller tile
# then both fits and finishes sooner, which cures either cause.
_LOST_MARKS = ("887a0005", "887a0006", "887a0007", "887a0020",
               "device_removed", "device removed", "device_hung", "device hung",
               "device_reset", "devicelost", "device lost")


def is_oom(e: BaseException) -> bool:
    t = readable_error(e).lower()
    return any(m.lower() in t for m in _OOM_MARKS)


def is_device_lost(e: BaseException) -> bool:
    t = readable_error(e).lower()
    return any(m in t for m in _LOST_MARKS)


def ladder_from(tile: Sequence[int],
                divisor: Sequence[int] = DIVISOR) -> List[Tuple[int, int, int]]:
    """Tiles no larger than the given one, in decreasing order. The given one
    goes first even if it is not on the ladder — the user may have chosen it.

    For dental9 the ladder is measured and stored in TILE_LADDER; for other
    divisors it is built on the fly: 3/4 and 1/2 of each side, rounded to the
    divisor, never below two."""
    t = tuple(int(v) for v in tile)
    if tuple(divisor) == DIVISOR:
        rest = [x for x in TILE_LADDER if np.prod(x) < np.prod(t)]
    else:
        rest = []
        for f in (0.75, 0.5):
            x = tuple(max(2 * d, int(v * f) // d * d) for v, d in zip(t, divisor))
            if np.prod(x) < np.prod(t) and x not in rest:
                rest.append(x)
    return [t] + rest


class SessionRef:
    """A session that can be recreated after the graphics card is reset.

    Callers keep the reference and read `.sess` after each prediction: after
    a lost device the old session is dead, and the replacement is what the
    rest of the run (the canal pass) must use.
    """

    def __init__(self, path: str, device: str = "auto", threads: int = 0, sess=None):
        self.path, self.device, self.threads = path, device, threads
        self.sess = sess if sess is not None else make_session(path, device, threads)

    def reopen(self):
        self.sess = None                  # release the dead one before the new one
        self.sess = make_session(self.path, self.device, self.threads)
        return self.sess


def predict_logits_auto(sess, vol: np.ndarray, tile: Sequence[int],
                        overlap: float = 0.5, progress=None,
                        acc_dtype=np.float32, log=None,
                        divisor: Sequence[int] = DIVISOR):
    """The same, but falling back to a smaller tile when video memory runs out
    or the card is reset.

    Only those two are caught, nothing else: silently swallowing any error
    and returning a lower-quality result is the worst possible outcome,
    because it looks fine.

    `sess` is a session or a SessionRef; only a SessionRef can recover from a
    lost device, a bare session cannot be recreated here.
    """
    ref = sess if isinstance(sess, SessionRef) else None
    last = None
    for t in ladder_from(tile, divisor):
        s = ref.sess if ref else sess
        try:
            return predict_logits(s, vol, t, overlap, progress, acc_dtype, divisor, log), t
        except Exception as e:                       # noqa: BLE001
            oom, lost = is_oom(e), is_device_lost(e)
            text = readable_error(e)
            if log:
                log(f"  GPU error: {text}")
            if not (oom or lost):
                raise
            last = text
            if lost:
                if ref is None:
                    raise
                if log:
                    log(f"  the graphics card was reset on tile {t} (driver timeout "
                        "or video memory) — reopening the session, trying a smaller tile")
                ref.reopen()
            elif log:
                log(f"  tile {t} did not fit in video memory, trying a smaller one")
    raise RuntimeError(f"the graphics card fails even on the smallest tile: {last}. "
                       "Crop the scan to the area of interest or set Compute on = CPU "
                       "in the add-on preferences.")


def predict(sess, vol: np.ndarray, tile: Sequence[int], overlap: float = 0.5,
            progress=None) -> np.ndarray:
    """Class labels. argmax is taken in slabs: the full float32 logit array
    already takes gigabytes, and keeping the result next to it is pointless."""
    logits = predict_logits(sess, vol, tile, overlap, progress)
    labels = np.empty(logits.shape[1:], np.uint8)
    chunk = max(1, logits.shape[1] // 8)
    for z0 in range(0, logits.shape[1], chunk):
        z1 = min(z0 + chunk, logits.shape[1])
        labels[z0:z1] = logits[:, z0:z1].argmax(0).astype(np.uint8)
    return labels
