# OdentAI Segment

CBCT (DICOM) → nine anatomical STL surfaces, as a Blender add-on.
Mandible, upper skull, upper and lower teeth, mandibular canal, maxillary
sinuses, nasal cavity, pharynx, soft palate; a second pass adds individual
teeth with FDI numbers, implants, implant crowns and bridges.

The internal package and executable are called `dental9`. Users never see
that name; the brand is **OdentAI**.

For coding agents (Claude Code, Codex and the like): [AGENTS.md](AGENTS.md).

## Install

Download the ready-built add-on from
[**Releases**](https://github.com/OdentStudio/OdentAI/releases/latest) — the
engine and the model weights are inside, nothing else to install:

- **Windows 10 / 11:** `OdentAI-<version>-windows.zip`
- **Ubuntu 22.04+:** `OdentAI-<version>-linux.zip`. It bundles CUDA and is
  over GitHub's 2 GB file limit, so it comes in parts `.zip.001`, `.zip.002`, …
  Download all of them and join:
  `cat OdentAI-<version>-linux.zip.0* > OdentAI-<version>-linux.zip`

Then in Blender: **Edit → Preferences → Add-ons → Install…** (Blender 4.2+:
the **⌄** menu → **Install from Disk…**), pick the zip without unpacking it,
and tick **OdentAI Segment**. The panel is in the 3D view under **N**, tab
**OdentAI**. Press **Check system** once to see whether the graphics card is
used. The release page has the full instructions and requirements.

Blender 3.3 is the minimum; **we recommend Blender 4.5 or newer**.

In short: pick the scan → **Bone preview** and fit the box (optional) →
tick the classes → **Segment**. A single progress bar at the top of the panel
shows the stage and the time; **Stop** or Esc ends the run. Each run goes into
its own sub-collection of **OdentAI** (e.g. `2 · Smith/CT`), the bone preview
and the box into **OdentAI › Region**.

## Repository

```
dental9/            the pipeline: reading, inference, meshes, command line (-> dental9.exe)
addon/              the Blender add-on (panel code; packing adds the binary and weights)
scripts/            weight export, zip packing, smoke test, accuracy evaluation, input-format test
configs/            fallback dental9.json (preprocessing parameters from the nnU-Net plans)
docs/               palette and viewport images
models/             .onnx weights + .json next to them (not in git, see below)
dental9.spec        PyInstaller build (onedir)
build_windows.bat   the whole local Windows build: environment, exe, check, zip
build_linux.sh      the same for Linux (exe only)
.github/workflows/  release.yml: builds both platforms on GitHub and publishes a release
```

## Releasing a version

Releases are built by GitHub Actions, not on a local machine:

1. Raise `bl_info["version"]` and `VERSION_LABEL` in
   `addon/OdentAI/__init__.py`, commit.
2. Tag and push: `git tag v1.0.2 && git push origin v1.0.2`.
3. The **Release** workflow checks that the tag matches `bl_info`, downloads
   the weights, builds the worker on `windows-2022` (DirectML) and
   `ubuntu-22.04` (CUDA), verifies that every CUDA library resolves inside the
   Linux bundle, runs `scripts/smoke_test.py` on each build (CPU, synthetic
   scan: providers, bone preview, segmentation, crop), packs the zips and
   publishes the release with the notes from `.github/release-notes.md`.

A failed run can be repeated from Actions → Release → Run workflow, with the
same tag.

The weights are not in git. They are assets of the **`models-v1`** release,
which the workflow downloads and checks against `SHA256.txt`. New weights: make
a `models-v2` release with the four files (`dental9.onnx`, `dental9.json`,
`teeth_fdi.onnx`, `teeth_fdi.json`), update `SHA256.txt` and `MODELS_TAG` in
the workflow.

## Building locally

Needed on the machine: **Python 3.9–3.12** (python.org, tick "Add python.exe
to PATH"), internet access (the build downloads ~200 MB of libraries on
Windows, several GB of CUDA on Linux), ~3 GB of free disk. Run it from a
local disk, not a USB stick or a network drive — the build writes thousands
of small files. The graphics card and driver do not matter for building;
CUDA need not be installed.

Put the weights into `models/` (from the `models-v1` release) and check them:

```powershell
Get-FileHash models\dental9.onnx, models\teeth_fdi.onnx -Algorithm SHA256   # compare with SHA256.txt
```
```bash
grep '  models/' SHA256.txt | sha256sum -c -
```

Windows, by double-click — `build_windows.bat` sets up `.venv`, builds the
exe, runs the hardware check on it and packs `dist\OdentAI_windows.zip`.
If it fails, the last lines of `build_windows.log` are shown. The same step by
step, without the window that ends in `pause`:

```bat
.venv\Scripts\python -m PyInstaller --noconfirm --distpath dist --workpath build dental9.spec
.venv\Scripts\python scripts\smoke_test.py dist\dental9\dental9.exe models\dental9.onnx
.venv\Scripts\python scripts\pack_addon.py --model models\dental9.onnx ^
    --teeth-model models\teeth_fdi.onnx --out dist\OdentAI.zip
```

Linux: `bash build_linux.sh`, then `python3 scripts/pack_addon.py`.

## Syncing the Windows build folder

The build folder on the HDD (`/media/ilya/HDD/WORK/dental9_build_windows`) is
updated with `rsync` **without `--delete`**, and nothing in it is deleted by
hand. Files created on Windows appear there — `.venv`, `build`, `dist` with
the built `dental9.exe` — and from Linux they look like leftovers of our own
build. On 2026-09-11 a finished Windows build was deleted that way. Delete
anything there only on the user's explicit request.

## Replacing the weights

The model ships as a **separate file**, not inside the binary, precisely for
this.

```bash
python3 scripts/export_onnx.py \
    --ckpt server_backup/checkpoint_best.pth \
    --plans nnUNet_preprocessed/Dataset003_Dental9/nnUNetResEncL_p160x320x320.json \
    --out models/dental9.onnx
```

This produces two files: `dental9.onnx` and `dental9.json` (normalisation
window, spacing, tile size, epoch). Then either of two ways:

1. drop both over the old ones in `<add-on>/bin/dental9/`;
2. rebuild the zip: `pack_addon.py --model <path to the new .onnx>`.

The add-on always takes the weights from `bin/dental9/`; there is no path
in its preferences (removed 2026-10-02: the zip carries the weights, and a
path field only offered a way to point at the wrong file). The preferences
show what is installed and what the last run computed on.

**The two files always change together.** The `.json` holds the normalisation
window and the grid spacing; new weights with an old `.json` give a plausible
and wrong result, without a single error in the log.

Checksums of the shipped weights are in `SHA256.txt`.

## Region to compute: bone preview and crop box (since 1.0.1)

A large field of view (200 mm and more) means gigabytes of memory and many
times the run time, while the jaws rarely fill more than a third of the
frame. Hence the **Region** block in the panel:

1. **Bone preview** — a quick bone surface **without the network**, in 1–5
   seconds: smoothing, a 1 mm grid, and a threshold from two-level Otsu
   (air / soft tissue / bone), computed per scan. CBCT is not calibrated, so
   a fixed HU threshold does not work. A box is placed around the bone.
2. The box is edited with **arrows on its walls** (Blender gizmos: each moves
   only its own wall, the opposite one stays put); the **ring** in the middle
   moves the whole box; sizes in mm are drawn in the viewport (a `gpu`/`blf`
   overlay). For exact values, use the Centre / Size fields in the panel. The
   arrows work in Object Mode.
3. **Segment** with **Only inside the box** passes the worker
   `--crop X0 Y0 Z0 X1 Y1 Z1` (mm, patient coordinates — the same frame as
   the STL files). The scan is cropped before resampling, the meshes land in
   place, and `labels.nii.gz` is written on the full original grid (zeros
   outside the box).

Since 1.1.1 a box deleted from the scene (X, or in the Outliner) no longer
crops anything: without a box in the scene the whole scan is computed.

Command line: `dental9 <scan> -o <folder> --preview` → `preview.stl` +
`preview.json` (frame bounds, threshold).

The box is deliberately **not** a child of the preview: Blender draws a
dashed relationship line from a child to its parent, right across the box.
The box is read in the preview's frame, or in world space when there is no
preview.

## Individual numbered teeth ("Separate teeth")

A second pass on top of the main one, project `~/precise_teeth`. The Upper /
Lower Teeth masks give the arch bounding box + 5 mm, and inside that crop the
numbering network sees the whole arch in a single tile. Tooth geometry stays
from dental9; the second pass supplies the boundaries between neighbours and
the FDI numbers. Implants, implant crowns and bridges come entirely from the
second pass (dental9 counts them as teeth).

```bash
dental9 <scan> -o <out> --separate-teeth [--teeth-model <teeth_fdi.onnx>]
```

Output: `teeth/tooth_11.stl … tooth_48.stl`, `teeth/implant.stl`,
`teeth/implant_crown.stl`, `teeth/bridge.stl`, and a `teeth` section in
`report.json`. In Blender they go into a **Teeth** sub-collection as objects
"Tooth 11" … The weights are `teeth_fdi.onnx` + `teeth_fdi.json` next to
`dental9.onnx`, made by the same `export_onnx.py` (this network's tile
divisor is 32×32×32; the script reads it from the plans). Tile 128×192×224,
memory fallback to 3/4 and 1/2.

Measured on 20 held-out DentVoxel cases (NewTom scanner, never seen by the
network), against manual labels, with the production ONNX pipeline: **per-tooth
Dice 0.966, number correct for 99.1 %** (561 of 566), none missed or extra.
The cost is about 20 s on a GPU (two crops + 30 STL). Checked headless in
Blender 5.2: the zip installs, both models are found, the Teeth collection
with "Tooth 11"… appears.

## Accepted input

Covered by `scripts/test_inputs.py` — 13 input forms; the test makes its own
data, no patient scans needed.

| input | how it is read |
|---|---|
| a folder of DICOM slices | the **longest** series is taken: scouts and screenshots often sit next to it |
| a folder of nested exports | the series is searched recursively; files may have no extension (`IM000001`) |
| a folder with one multi-frame `.dcm` | the file is read directly |
| a single multi-frame `.dcm` | likewise |
| one slice of a series | go up to the folder and read the whole series |
| a `.zip` archive | unpacked into a temporary folder |
| a folder with an archive inside | likewise |
| `.nii.gz`, `.mha`, `.nrrd`, `.mhd` | directly |

Two traps this used to fail on:

- **A multi-frame `.dcm` through `ImageSeriesReader` arrives as 4D**
  `(x, y, z, 1)`, and `DICOMOrient` does not work on 4D at all: "Pixel type …
  is not supported in 4D". It fails at the very first step. The degenerate
  axis is removed with `Extract`.
- **A single slice passed as a volume** is read silently and gives
  plausible-looking garbage. So a one-slice-thick volume is a reason to go up
  to the folder, not to compute.

Any resolution works: the scan is resampled to 0.3 mm keeping its physical
frame, anisotropic spacing included. But a spacing outside **0.04–1.2 mm** is
refused at once: that is broken metadata, and since the whole pipeline works
in millimetres the mesh would come out at the wrong scale. Caught on a Sirona
export that claimed 1.0 mm instead of 0.15.

Memory is checked before computing: the label buffer is ten classes over the
whole volume, and if it does not fit into free memory, an honest error beats
swapping and a frozen machine. Since 1.0.1 the log also shows an estimate of
the **peak**: the peak is not the buffer but the moment the signed fields are
built (one float32 field per class while the buffer is still alive), plus
~2 GB for the onnxruntime session itself. On a 367³ grid the estimate matched
the measurement within 0.1 GB. A 200 mm frame (669³) needs about 19 GB — on a
16 GB laptop that means swapping; the crop box fixes it.

**The second canal pass runs on a 0.25 mm grid** (`--canal-refine-mode fine`,
the default since 2026-10-07): the mandible is predicted again with 15 mm of
context around it and only the canal is taken. Thin canal walls fall below what
the network resolves at 0.3 mm — on a 0.15 mm scan the left canal came out as
two fragments covering half its course (105 mm3) and came back whole at 0.25 mm
(305 mm3), while rotations, tile shifts, mirroring, denoising and a 0.35 mm
grid all failed. The older pass, intensity matching onto the training curve,
is still available as `--canal-refine-mode match`; on 22 clinic scans it helped
on none and on that one made things worse.

**Neither pass may be made mandatory.** 2026-09-16, four clinic scans with a
healthy canal: a forced match pass lowered the agreement with DentalSegmentator
from 0.871 to 0.858 and on one scan broke the canal, only the rollback saving
it. 2026-10-07, measured against manual labels on all 70 held-out cases:
forcing the fine pass costs 0.0075 canal Dice, forcing match 0.0039. The
`--canal-refine-always` flag exists only for such checks.

On those 70 labelled cases the trigger never fires, so the fine pass is
byte-identical to the old behaviour there — the labelled set cannot measure
this change, because it is resampled to 0.3 mm while clinic scans arrive at
0.08–0.25 mm. Full numbers: `docs/canal_fine_eval/REPORT_dice_ru.md`.

## Accuracy against manual labels

20 held-out DentVoxel cases, manual labels, the same weights (epoch 475):

| | Dice |
|---|---|
| our ONNX pipeline, fp32 | **0.9619** |
| our ONNX pipeline, fp16 | **0.9613** |
| nnU-Net itself (`nnUNetv2_predict`) | 0.9610 |

Moving to the delivery loses nothing. Half precision neither: it halves the
file (283 MB instead of 566), runs a quarter faster, and agrees with nnU-Net
even better than fp32 (0.995 against 0.976 voxel-wise) — nnU-Net itself
computes in half. That is why **fp16 is the default** in `export_onnx.py`.

## Where it computes

| device | how to enable | measured on a 100 mm frame (334³) |
|---|---|---|
| CUDA, RTX 4090 | `--device cuda`, build with `onnxruntime-gpu` | **10 s** inference, 18 s end to end (fp16) |
| DirectML, RTX 4090 | `--device dml`, build with `onnxruntime-directml` | 160×320×320 tile in 0.8–1.0 s; a 367³ frame in 63 s end to end |
| CPU | `--device cpu` | **384 s** — 20 times slower |

The tile is chosen automatically: 160×320×320 first, and when video memory
runs out it falls back to 128×256×256, then 96×192×192 and 64×128×128. The
network is fully convolutional, so the same weights serve every case — no
separate model for weak cards. The price of a small tile is measured:
0.002–0.007 Dice on every class except the nasal cavity (0.920 → 0.881).

**A 160×320×320 tile takes about 7 GB of video memory under DirectML**
(measured 2026-09-29 on an RTX 4090: 8.6 GB in use over a 1.6 GB baseline).
On 8 GB cards (a laptop RTX 2070 Max-Q) that is on the edge: the first tile
passes, the second fails. The fallback to a smaller tile fires not only on
"out of memory" text but also on the HRESULTs (`8007000E`, `887A0004`), and
on a lost device (`887A0005/6/7`, a driver reset by TDR) the session is
recreated (`infer.SessionRef`) and the run continues with a smaller tile.

**The Linux build is self-contained: CUDA and cuDNN are bundled**; the machine
only needs an NVIDIA driver 525 or newer (CUDA 12). The price is 2.5 GB of
libraries, and the add-on zip comes to ~3 GB. It became so on 2026-09-13 after
the provider failed to load on someone else's Ubuntu: "Unable to load
libcudnn_graph.so.9". It also turned out that cufft, curand and cublasLt,
previously dropped as unneeded, are direct dependencies of the provider; on
the build machine the system CUDA satisfied them, and the test was not honest.
Windows needs none of this: DirectML is one 19 MB library.

CPU and GPU labels agree on 99.993 % of voxels — the difference is purely
arithmetic.

## Command line (what the add-on calls)

```bash
dental9 <DICOM folder> -o <out> [-c mandible upper_teeth ...] [--device auto]
        [--taubin 20] [--decimate 0.5] [--labels] [--no-stl]
        [--crop X0 Y0 Z0 X1 Y1 Z1] [--separate-teeth]
dental9 <scan> -o <out> --preview          # bone by threshold, no network
dental9 --diagnose [-m dental9.onnx] [--json]
```

## Delivery size

Measured, not estimated:

| part (unpacked) | Linux (CUDA) | Windows (DirectML) |
|---|---|---|
| onnxruntime | 438 MB | 70 MB |
| CUDA libraries | 108 MB | none |
| VTK | 266 MB | 307 MB |
| SimpleITK | 260 MB | 109 MB |
| numpy and Python | ~60 MB | ~60 MB |
| `.onnx` weights, fp16 (both models) | 488 MB | 488 MB |
| CUDA + cuDNN libraries | 2.5 GB | none |
| **add-on zip** | **~3 GB** | **583 MB** (measured, 1.0.1) |

Of the CUDA libraries only the ones actually needed stay in the build:
`cufft`, `curand` and `cublasLt` are dropped (a convolutional network has no
use for FFT, random numbers or that MatMul) — 870 MB less, results identical
voxel for voxel.

VTK could not be trimmed: the dependency closure of `vtkFlyingEdges3D` and
`vtkWindowedSincPolyDataFilter` pulls in `libvtkRenderingCore` as well; there
is nothing extra in the build (checked with an ldd closure). **Do not use
`strip`** — it breaks the prebuilt OpenBLAS inside numpy, and importing numpy
itself fails.

**The exe console on Windows is forced to UTF-8.** A process started through a
pipe gets stdout in the locale code page (cp1252 and the like), and the first
letter outside it in a path — a real case: a Polish "ł" in a folder name —
kills the run with `'charmap' codec can't encode character`. `cli.main()`
switches the streams to UTF-8, and the add-on reads the pipe as UTF-8. This is
a separate problem from the non-ASCII paths below: there the libraries fail to
open the file, here it is printing the name.

**And the reverse problem — DirectML's own messages.** On a non-English
Windows the system error text arrives in the code page (cp1251 and the like),
pybind11 decodes it as UTF-8 and fails — instead of the error you see
`'utf-8' codec can't decode byte 0xc3`. That happened on 2026-09-29 on a
laptop with an RTX 2070 Max-Q. `infer.readable_error` takes the original bytes
out of the `UnicodeDecodeError` and decodes them with `mbcs`.

## Non-ASCII paths: Cyrillic, Chinese, Arabic, emoji

Checked on 2026-09-21 on Linux with the production weights: a folder
"病人 王伟 🦷 مريض", a subfolder "КТ 2026-09 · серия", files `王伟_مريض_0001.dcm`,
an archive "архив 王伟.zip" with the same names inside, an output folder
"результат 王伟 🦷" — the STL files and report.json are in place, and the volume
matches a read of the original byte for byte.

On Windows the risk is real and not in Python: ITK/GDCM and VTK open files
with the "narrow" `fopen` through the system code page, and GDCM scans the
folder with narrow APIs. The defence is in `dental9/winpath.py`:

- an 8.3 short name (ASCII, nothing copied) — only when the path to the scan
  is non-ASCII but all names **inside** the tree are ASCII;
- otherwise, a copy into an ASCII folder with **every file and subfolder
  renamed** (`d001/f000001.dcm`): a DICOM series does not care about file
  names. A short name does not help here: it fixes the top of the path, not
  `王伟_0001.dcm` inside;
- archives are unpacked the same way, with renaming — which also covers names
  in a local code page without the UTF-8 flag (an archive from a Chinese
  Explorer) and `../` in names;
- output is written into an ASCII folder and moved into the real one;
- the temporary ASCII folder: `%TEMP%` → `%PUBLIC%\OdentAI\tmp` →
  `C:\OdentAI_tmp`, because for a user called "王伟" `%TEMP%` lies in their
  profile;
- the model path goes through the same defence (the weights live in the
  user's profile);
- the exe streams are UTF-8 (see above), the add-on reads the pipe as UTF-8,
  and echoing into Blender's console cannot bring the run down.

The Windows branch was debugged by simulating the worst case (no 8.3 names, a
non-ASCII `%TEMP%`). A real run on Windows with a Chinese or Russian folder is
a mandatory check before distribution.

## When something does not work

**Start with the log.** The add-on writes the full output of the last run to
`%TEMP%\OdentAI\last_run.log` (the document-icon button in the Hardware
block). It contains the system, the CPU, the graphics cards with VRAM and
driver, the onnxruntime version, the time of the first and the slowest tile,
free RAM / process peak / VRAM in use at each stage, and on failure the tile
it failed on, the decoded error text and the traceback. This is the file to
send.

The **Check system** button (OdentAI panel and add-on preferences), or
`dental9 --diagnose` on the command line. It does not read the provider list;
it opens a session and actually runs a probe tile — the list lies: DirectML
and CUDA are listed in the build but may fail to come up. Since 1.0.1 on a
GPU it also runs the **working** 160×320×320 tile and warns when it does not
fit or takes longer than 1.5 s (close to Windows' 2-second driver timeout).

The time estimate includes accumulating tiles into the buffer, not only the
network: on a GPU that is half of the total, and an estimate "from the network
alone" was off by a factor of two.

| symptom | cause | what to do |
|---|---|---|
| takes minutes | the graphics card is not used | Check system; on Windows update the driver, on Linux install `libcudnn9-cuda-12` |
| "A graphics card is present…" | driver or build | the hardware is fine: update the driver; for a DirectML build, check that it really is DirectML |
| "the graphics card fails even on the smallest tile" | too little video memory, or a driver reset | crop with the box; Compute on = CPU in the add-on preferences |
| "estimated peak RAM … WARNING" | the scan is larger than free memory | crop with the box |
| "executable missing" | the zip was not fully unpacked | reinstall the add-on |
| "model file missing" | the weights did not arrive | reinstall the add-on |
| "There is no dental9.json next to the model" | only the weights were replaced | these two files change **together** |
| the tile shrank by itself | too little video memory | by design; nasal cavity Dice 0.881 instead of 0.920 |

## Language

Everything the user sees — the add-on UI, console output, tooltips, error
messages — is in English: the add-on goes to clinics. Code comments,
docstrings and all documentation are in English too.

## What must not break

- **LPS orientation and 0.3 mm spacing** — as in training. Not RPI: RPI is the
  convention of the frozen Dataset002 branch.
- **CT normalisation comes from the training set fingerprint**, not from the
  scan itself. CBCT is not calibrated; normalising each scan on its own is
  wrong.
- **The tile is a multiple of (32, 64, 64)** and at least two divisors per
  axis: with one, a single voxel is left at the bottom of the network and
  InstanceNorm fails.
- **Surfaces are built from the signed logit field, not the binary mask.**
  The boundary is the same (the field's zero is the argmax boundary), but its
  position inside a voxel comes from the values instead of being rounded to
  half a voxel. Voxel-sized steps disappear without any smoothing.
- Mesh smoothing is Taubin (`--taubin`), not Laplacian: Laplacian shrinks the
  surface.
- **Do not use `strip`** — it breaks the prebuilt OpenBLAS inside numpy, and
  importing numpy itself fails.
- **Blender's installer drops the executable bit** from the binary, even if the
  zip carries it. `_ensure_executable` fixes it on first launch.
- **The tile fallback catches only out-of-memory and a lost device.** No other
  error may be swallowed: a worse result that looks fine is the worst possible
  outcome.

## Verified in place

Install into Blender 5.2 (`--factory-startup`, headless): the add-on installs
and enables, the panel and all nine checkboxes are there, the model is found,
the binary from the add-on folder computes a case in 18 s and returns nine STL
files. Mesh dimensions are in millimetres and anatomically right (mandible
84 mm in a 100 mm frame).

1.0.1 (2026-09-29), Blender 5.2 with a window, real modal operators: preview →
box arrows → Segment inside the box → import; the meshes lie entirely inside
the box, the opposite wall stays put while dragging, and "box outside the
scan" arrives as a plain message with a link to the log.

1.1.0 (2026-10-02), before the first release to clinics:

- The release zip installed with `addon_install` into clean profiles of
  Blender 3.3.8, 3.6.3, 4.5.3, 5.0.1 and 5.2.2: registers, Check system runs
  the engine on DirectML, three disable/enable cycles, a real result (9
  classes + 32 teeth) loads identically on all five.
- 24 clinic scans through the built worker on an RTX 4090: Carestream 8100 /
  8200 (0.15 and 0.075 mm, metal-artefact reduced), Planmeca ProMax (series
  and multi-frame), Sirona Axeos, Morita, Vatech PHT-75 / PHT-35, i-CAT with
  scouts in a subfolder, Dürr, HDXWILL; paths with Cyrillic, Polish letters,
  `ё`, dots and spaces; a zip in a Cyrillic folder, `.mha`, `.nrrd`; output
  into a folder named in Cyrillic and Chinese. All 21 valid scans computed
  (15–472 s; 472 s is a 200 mm frame without the crop box), front and side
  renders checked by eye. A CT of a denture alone (double-scan technique)
  gives meaningless shapes, as expected for a non-anatomical input.
- Refused with a readable message: an encrypted Sirona zip, an unreadable
  export, 1 mm voxels written into a 0.16 mm scan (now refused in 4 s,
  before resampling; it used to take ~30 GB and 70 s first), a read-only
  output folder (checked before computing).
- Also checked: crop box (every mesh within 0.01 mm of the box), CPU run,
  `%TEMP%` under a Cyrillic/Chinese user name, a single file of a series,
  "Separate teeth" without the teeth classes ticked (works), Esc / Stop /
  quitting Blender / opening another file during a run (the worker is killed,
  nothing left in `%TEMP%`, Segment is greyed out while a run goes), an
  ODent5 project (Region goes under its OdentAI collection).
- Not checkable on this machine: the tile fallback on a card with less video
  memory, AMD / Intel graphics, a clean Windows 10.
