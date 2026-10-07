# Back to Windows: the canal work is done, the branch is on GitHub

Written 2026-10-07 by the Linux session, in reply to `HANDOFF_LINUX_canal.md`
(that one is finished — everything it asked for is below). Read `AGENTS.md`
first. Talk to the user **in Russian**; code, comments and docs stay English.

## Get the code — by git, not by rsync

The branch now exists on the remote, which it did not before:

```bat
git fetch origin
git checkout canal-enhance
git pull
git log --oneline -7          :: the seven commits listed below
```

An rsync from this machine would overwrite the branch with an older copy.

## The question it asked: Dice on the labelled set

Measured on all 70 held-out cases with manual labels, four configurations,
same weights. Full numbers and the per-case table:
`docs/canal_fine_eval/REPORT_dice_ru.md`.

- **The trigger fires on none of the 70**, so the fine pass is byte-identical
  to production there. That set is resampled to 0.3 mm while clinic scans
  arrive at 0.08-0.25 mm (checked on the 22 scans themselves) — it cannot
  measure what this pass is for, and that is the honest answer.
- Forcing either pass costs canal Dice: fine 0.0075, match 0.0039. Both stay
  trigger-only.
- The pipeline itself is intact: 0.9697 on the 20 DentVoxel cases against the
  0.9619 recorded in README.

The decision was the user's, taken on the clinic evidence: fine is now the
default.

## What changed (seven commits, in order)

1. The Dice report and the per-case tables.
2. `fine` is the default `canal_refine_mode`; `--canal-enhance` and
   `_enhance_bone_hu` are gone (useless on every scan they were tried on).
3. The fine pass measures what it needs and **skips itself** when it would not
   fit in free RAM: it allocates a second logit buffer while the first is
   alive, and it is not in the estimate logged before inference.
4. Only the broken side is redone, and the pass accumulates in float16.
   The sound side now comes through bit-exact instead of being predicted again
   and rescued by the rollback.
5. Signed fields are built one class at a time on large grids: peak 18.4 ->
   13.1 GB on a 500x600x600 scan, every STL byte-identical. Below the float16
   threshold the old all-at-once form is kept — there the logit buffer costs
   more than the fields.
6. Both sides broken: two boxes in turn instead of one over the whole jaw, each
   merged and rolled back on its own.
7. **A frame cropped to one side of the jaw no longer trips every trigger.**
   This one is a bug that is in 1.1.1 too: the health check compared the two
   canals, so on a partial mandible it split one hemimandible into quarters,
   found no canal in the medial one and fired on a sound scan. Four of the five
   clinic scans that used to trigger were post-op crops 33-61 mm wide that
   needed nothing.

## Build and check here

```bat
build_windows.bat
```

It rebuilds `dist\dental9` and overwrites `dist\OdentAI_windows.zip` — both are
build outputs, but say so before running if the user wants the old zip kept.
The steps in `AGENTS.md` suit an agent better if you want them one at a time.

Worth re-checking on Windows specifically, because DirectML is a different
runtime and none of this was measured there:

- **Kolkova** (`C:\Dental\Dmitry\Guides\14 10 26 Колкова ...`) through the
  add-on: the pass should fire, say "1 of 2 sides to redo" and give about
  0.80 cm3 of canal, the right side untouched at 457 mm3. On CUDA it takes
  32 s; the fine pass used to take 25-75 s on DirectML and its box is now half
  the size.
- **A cropped scan** (crop box over one side in the add-on): the second pass
  must NOT run, and the log should say "part of a mandible in the frame".
- **A large frame** (Bator, 0.2 mm, 750 slices): the peak RAM in the log and
  the `peak_ram_gb` in `report.json`; on CUDA it went 18.4 -> 13.1 GB.
- **The teeth pass** on a whole-jaw scan: the arch fields are now reused
  rather than rebuilt, and 31 STL files came out byte-identical on Linux.

## Not done, on purpose

- **The version is still 1.1.1 and there is no tag.** A tag publishes a release
  (AGENTS.md); that is the user's call, and they asked to be the one to say so.
- **The peak RAM estimate is low on CUDA by about 5 GB** (it counts the session
  at 2 GB, which is what you measured with DirectML). On Windows it was within
  0.1 GB, so this was left alone — but it is why the Linux logs under-report.
