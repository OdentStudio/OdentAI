# Handoff to the Linux session: fine-grid canal pass

Written 2026-10-06 by the Windows session for the Claude session the user
starts on Ubuntu **in this same folder** (the HDD copy,
`/media/ilya/HDD/WORK/dental9_build_windows`). Read `AGENTS.md` first. Talk
to the user **in Russian**; code, comments and docs stay English.

## Goal

Decide **on Dice against manual labels** whether the new second canal pass
(`--canal-refine-mode fine`) replaces the current one (`match`) in
production, and if so, finish it, build and smoke-test on Linux.

## What was done on Windows (details: `docs/canal_fine_eval/REPORT_ru.md`)

- Case: a 0.15 mm scan where the left canal came out as two fragments,
  105 mm3, half its course. The current second pass ("match") made it worse
  and was rolled back.
- Tried and **rejected** (all measured, logs in `docs/canal_fine_eval/`):
  bone window + 3D AHE before the match pass (`--canal-enhance`), rotations
  ±5/10/15°, left-right mirroring, tile-grid shifts, overlap 0.75, Gaussian /
  median / anisotropic denoising, a 0.35 mm grid.
- **What works: the mandible predicted again on a 0.25 mm grid** (15 mm of
  context around it), only the canal taken: left canal 105 → 305 mm3, one
  piece, full course. Thin canal walls drop below what the network resolves
  at 0.3 mm. A tight crop (the match pass uses 20 voxels) itself loses 2/3
  of that canal — the network needs context; 30 mm is no better than 15.
- 22 clinic scans (no labels): the fine pass was **forced on every scan**.
  Triggered (broken) canals: 1 fixed, 4 unchanged (no mandible / canal in
  the frame, or the fine pass found less and the rollback kept the first).
  17 healthy canals: same course length, volume −7…+4%; on 2 scans (Zyga)
  the right canal split into 2 pieces → **do not force it on healthy scans**;
  it runs only when the health check trips, like the match pass.
- The match pass did not help on any of the 22.

## State of the code

Branch **`canal-enhance`** (local, not pushed). First check the sync arrived:

```bash
git branch --show-current     # canal-enhance
git log --oneline -3          # top: the "box from the largest mandible piece" commit, then 4c8d5b6 WIP
```

If not — stop and tell the user: the rsync from Windows has not happened.

In `dental9/pipeline.py`:

- `Options.canal_refine_mode` — `"match"` (still the **default**) or `"fine"`.
- `_fine_canal()` — box from the largest mandible component + `canal_fine_margin_mm`
  (15), resampled to `canal_fine_spacing` (0.25), predicted, canal mapped back
  to the 0.3 grid with nearest neighbour. Skipped above `FINE_MAX_VOXELS`
  (150 M) with a log line.
- `_merge_canal()` — the old merge + rollback ("the second pass found less
  than the first — keeping the first").
- `_enhance_bone_hu()` / `--canal-enhance` — the AHE experiment; useless,
  off by default. Recommendation: remove before merging (ask the user).

CLI: `--canal-refine-mode {match,fine}`, `--canal-refine-always`,
`--canal-enhance [R]`, `--canal-enhance-window {fixed,auto}`.

## Traps of this shared folder — read before running anything

1. **`.venv` here is the Windows venv** (`.venv/Scripts/python.exe`). Do not
   touch it. Make a separate one, e.g. `.venv-linux`.
2. **`build_linux.sh` starts with `rm -rf build dist/dental9`** — that deletes
   the Windows build (`dist/dental9/dental9.exe`) and `build/`. **Do not run it
   as is here.** Build by hand into separate folders (below).
3. Nothing in `dist/`, `build/`, `.venv/`, `_backup_*` is junk. Delete nothing
   without the user's explicit request (README, "Syncing the Windows build
   folder").
4. Never commit `models/`, `dist*/`, `build*/`, `.venv*/`.

## Steps

### 1. Environment (no build needed for the Dice runs)

```bash
python3 -m venv .venv-linux
.venv-linux/bin/pip install -q --upgrade pip
.venv-linux/bin/pip install -q -r requirements-deploy.txt pyinstaller "onnxruntime-gpu==1.24.4" \
    "nvidia-cudnn-cu12>=9.5,<10" nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
.venv-linux/bin/python -m dental9 --providers          # must list CUDAExecutionProvider
.venv-linux/bin/python -m dental9 --diagnose -m models/dental9.onnx
```

Same pins as `build_linux.sh` (onnxruntime-gpu 1.24.4: newer needs CUDA 13).
`sha256sum -c SHA256.txt` for the weights if in doubt.

### 2. Find the labelled data

The held-out set used for the accuracy table in README ("20 held-out
DentVoxel cases"): `nnUNet_raw/Dataset003_Dental9/imagesTs` + `labelsTs`
(images `<case>_0000.nii.gz`, labels `<case>.nii.gz`). Look for `nnUNet_raw`
on the machine (env var `nnUNet_raw`, the main project next to `deploy/`).
If several candidate sets exist or nothing is found — ask the user, do not guess.
Also check whether the set has canals that trip the health check at all
(step 3 logs it): if none trips, configs A and B are identical by
construction and only C/D say anything.

### 3. Runs — four configs, same weights, `--no-stl`

```bash
IMG=<...>/imagesTs; GT=<...>/labelsTs; OUT=<scratch>/canal_eval   # NOT inside this folder's dist/ or build/
run() {  # $1 = config name, rest = flags
  name=$1; shift
  for f in "$IMG"/*_0000.nii.gz; do
    c=$(basename "$f" _0000.nii.gz)
    .venv-linux/bin/python -m dental9 "$f" -o "$OUT/$name/$c" -m models/dental9.onnx \
        --no-stl "$@" > "$OUT/$name/$c.log" 2>&1 || echo "FAILED $name $c"
  done
}
mkdir -p $OUT/{A_match,B_fine,C_fine_forced,D_match_forced}
run A_match                                            # production today
run B_fine          --canal-refine-mode fine           # the candidate
run C_fine_forced   --canal-refine-mode fine --canal-refine-always
run D_match_forced  --canal-refine-always
for n in A_match B_fine C_fine_forced D_match_forced; do
  .venv-linux/bin/python scripts/eval_vs_manual.py --gt "$GT" --ours "$OUT/$n" --out "$OUT/$n.json"
done
grep -l "canal looks broken" $OUT/A_match/*.log | wc -l   # how many trip the trigger
```

(`mkdir -p $OUT/<name>/<case>` happens inside dental9; the per-case `.log`
needs the parent — the `mkdir` above covers it.)

`eval_vs_manual.py` prints the mean per class only. Also compute the
**per-case Mandibular Canal Dice** for every config (small script, same
`dice()`), and list each case where B or C differs from A by more than 0.01.

### 4. Acceptance — on numbers only

`fine` becomes the default only if **all** hold:

- B ≥ A on mean canal Dice, and no single case where B lowers canal Dice by
  more than 0.02 (a rollback should make that impossible — if it happens,
  find out why).
- Other classes: identical to A except Mandible, which may move slightly
  (the merged canal takes voxels from label 1); Mandible change < 0.001.
- C vs A: tells whether the fine pass is safe enough to be forced. Expect it
  not to be (two clinic scans split the canal) — that is fine, it stays
  trigger-only. Report the number anyway.
- Run time of the fine pass: grep `fine pass:` and the timings; CUDA should
  be a fraction of the Windows DirectML 25–75 s.

If B is not better on the labelled set (likely few or no broken canals there),
report that honestly: the evidence is then only the one clinic scan above plus
"does no harm" — the user decides.

### 5. If accepted

1. `canal_refine_mode` default → `"fine"` (Options and CLI default). Keep
   `match` selectable. Ask the user whether to delete `--canal-enhance`.
2. Optional, measure before keeping: per-side acceptance instead of the
   total-volume rollback (weak side grew, strong side lost < 15%).
3. README: the "second canal pass" section — what changed and why (the
   2026-10-05 case, the numbers); note the old "must not be mandatory" still
   holds.
4. `py_compile`, `scripts/test_inputs.py`, then the Linux build **into separate
   folders**:
   ```bash
   .venv-linux/bin/pyinstaller --noconfirm --distpath dist-linux --workpath build-linux dental9.spec
   ./dist-linux/dental9/dental9 --providers
   .venv-linux/bin/python scripts/smoke_test.py dist-linux/dental9/dental9 models/dental9.onnx
   ```
   One run of that binary on a real scan with `--canal-refine-mode fine
   --canal-refine-always` (the fine pass is what's new in the bundle).
5. Commit on `canal-enhance`. **Do not push, merge, tag or bump the version**
   without the user: a release is a tag → GitHub Actions (AGENTS.md).
6. Tell the user which files changed on Linux: the folder is synced from
   Windows by rsync, so they must carry the changes back (or push/pull via git).

## Report back (in Russian)

A table: config × mean Dice per class, the per-case canal differences,
trigger count, fine-pass timings, the decision and why. Put the full numbers
in `docs/canal_fine_eval/` next to the Windows logs.
