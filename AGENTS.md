# AGENTS.md — OdentAI Segment

Instructions for AI coding agents (Claude Code, Codex, Cursor and the like)
working in this repository. How the pipeline behaves is in
[README.md](README.md); this file is about working on the code without
breaking a delivery.

## What this is

A Blender add-on that turns CBCT (DICOM) into STL surfaces of nine anatomical
classes. Two parts, two processes:

- **`dental9/`** — the worker, in Python: reads the scan (SimpleITK), runs
  nnU-Net inference on onnxruntime (DirectML on Windows, CUDA on Linux),
  builds meshes (VTK). PyInstaller turns it into `dist/dental9/dental9(.exe)`.
  Entry point: `cli.py`.
- **`addon/dental9_addon/__init__.py`** — the Blender add-on, a single file.
  Nothing heavy runs in Blender's Python: the add-on starts the worker as a
  separate process and reads its stdout, so an onnxruntime crash or running
  out of video memory cannot take the user's session down.

The brand is **OdentAI**; `dental9` is the internal name, never shown to users.

## Map

| file | what is there |
|---|---|
| `dental9/cli.py` | arguments, `--diagnose`, `--preview`, `--crop`, error printing |
| `dental9/pipeline.py` | `segment()`: read → crop → resample → inference → canal second pass → signed fields → STL → `report.json` |
| `dental9/infer.py` | tiling, Gaussian weights, the tile ladder on memory failure, `SessionRef`, `readable_error` |
| `dental9/io.py` | every accepted input form, LPS / 0.3 mm, CT normalisation, `crop_to_box` |
| `dental9/mesh.py` | flying edges, Taubin smoothing, decimation, patient coordinates |
| `dental9/teeth.py` | second pass: individual FDI teeth, implants, bridges |
| `dental9/preview.py` | bone by Otsu threshold, no network — for the crop box |
| `dental9/sysinfo.py` | RAM, VRAM, process peak memory — for the logs |
| `dental9/diagnose.py` | Check system: probe tile and working tile, advice |
| `dental9/winpath.py` | non-ASCII paths on Windows (8.3 names, renamed copies) |
| `addon/.../__init__.py` | panel, operators, `_WorkerRun` (runs the worker + log file), crop box gizmos, `gpu` overlay |
| `scripts/pack_addon.py` | packs the zip: add-on code + `dist/dental9` + weights |
| `scripts/smoke_test.py` | runs a built worker on a synthetic scan (CPU) |
| `scripts/export_onnx.py` | nnU-Net checkpoint → `.onnx` + `.json` |
| `scripts/test_inputs.py` | 13 input forms on synthetic data |
| `.github/workflows/release.yml` | builds Windows + Linux on GitHub and publishes the release |
| `.github/release-notes.md` | the release page text (install instructions); `{{VERSION}}` is substituted |

The weights (`models/*.onnx` + `*.json`, ~490 MB) are **not in git**: they are
assets of the `models-v1` GitHub release, checksums in `SHA256.txt`. Never
commit `models/`, `dist/`, `build/` or `.venv/`.

## Build and check locally

The environment is `.venv` (Python 3.11; the build accepts 3.9–3.12) with
`onnxruntime-directml`, **not** plain `onnxruntime`: with both installed the CPU
one wins and the build silently loses the GPU. On Linux it is `onnxruntime-gpu`
plus NVIDIA's CUDA/cuDNN wheels (see `build_linux.sh`).

```bat
:: the worker from source, no build
.venv\Scripts\python -m dental9 <scan> -o <dir> -m models\dental9.onnx [--crop ...]
.venv\Scripts\python -m dental9 --diagnose -m models\dental9.onnx
.venv\Scripts\python -m py_compile dental9\*.py addon\dental9_addon\__init__.py

:: the exe (only when dental9/*.py changed), ~1 min, then the smoke test
.venv\Scripts\python -m PyInstaller --noconfirm --distpath dist --workpath build dental9.spec
.venv\Scripts\python scripts\smoke_test.py dist\dental9\dental9.exe models\dental9.onnx

:: the add-on zip; delete addon\dental9_addon\__pycache__ first
.venv\Scripts\python scripts\pack_addon.py --model models\dental9.onnx ^
    --teeth-model models\teeth_fdi.onnx --out dist\OdentAI.zip
```

`build_windows.bat` does all of it at once but ends in `pause` — the steps
above suit an agent better.

The add-on is tested in Blender itself:

- `blender -b --factory-startup --python test.py` — registration, the
  operators' `_load`, running the worker synchronously. **Modal operators do
  not run in `-b`** (there is no event loop).
- `blender --factory-startup --python test.py` (with a window) — real modal
  operators, gizmos, screenshots through `bpy.app.timers` +
  `bpy.ops.screen.screenshot`, exit with `bpy.ops.wm.quit_blender()`.
- A test may replace `_exe` and `_prefs` on the module to run the add-on from
  source against `dist/dental9/dental9.exe` and `models/`.

## Releasing

Releases are built by GitHub Actions, never uploaded by hand:

1. Raise `bl_info["version"]` and `VERSION_LABEL` in the add-on; commit.
2. `git tag vX.Y.Z && git push origin vX.Y.Z` — the tag must equal
   `bl_info["version"]`, the workflow refuses otherwise.
3. The workflow builds both platforms, runs the smoke test and the Linux CUDA
   library check, and publishes `OdentAI-X.Y.Z-windows.zip` and
   `OdentAI-X.Y.Z-linux.zip` (in parts: a release asset must be under 2 GiB,
   and the Linux zip carries ~2.5 GB of CUDA).
4. Rebuilding an existing tag: Actions → Release → Run workflow.

Anything the user must do or read to install goes into
`.github/release-notes.md`, in English.

## Rules

**Language.** Everything the user sees (UI, log, errors, tooltips) is English:
the add-on goes to clinics. Comments, docstrings and all documentation are
English too. Comments follow the surrounding code: they explain **why**, often
with the date and the case that broke.

**Do not break preprocessing.** LPS, 0.3 mm spacing, normalisation from the
training set fingerprint (not from the scan), a tile that is a multiple of the
network divisor (32, 64, 64) and at least two divisors. The parameters live in
`dental9.json` next to the weights and change only together with them. A
mistake here does not fail — it gives a plausible wrong result.

**Do not swallow errors.** The tile ladder catches only running out of memory
and a lost device (`infer.is_oom`, `is_device_lost`). Anything else goes out.
A quietly worse result that looks fine is the worst possible outcome.

**Diagnostics must never cause a failure.** Everything in `sysinfo.py` is best
effort: no answer is `None`, not an exception.

**The add-on and the worker talk only through the command line and stdout.**
An `ERROR: …` line in the output is what the user sees; the details after it
are for the log. `report.json` in the output folder is the machine-readable
result of a run.

**Paths on Windows.** Everything that goes into ITK/GDCM/VTK passes through
`winpath.readable` / `writable`: the native libraries cannot open non-ASCII
paths.

**Blender API.**
- The minimum in `bl_info` is 3.0; the production version is 5.2. Where the
  API changed (`stl_import`, auto smooth, `surface_render_method`) both
  branches are kept.
- Never write to ID data from `draw` / `poll` / draw handlers — Blender
  forbids it.
- The crop box is an 8-vertex object without edges; the overlay draws it and
  the gizmos (`DENTAL9_GGT_crop_box`) move it. It is deliberately **not** a
  child of the preview (a child draws a dashed relationship line).
- The gizmo handlers look the box up through `scene.dental9.crop_box` on every
  call: the gizmo group outlives any single box.

**Deleting files.** The build folder is synced between Linux and Windows; what
looks like junk (`.venv`, `build`, `dist`) may be someone's finished build.
Delete nothing without the user's explicit request.
