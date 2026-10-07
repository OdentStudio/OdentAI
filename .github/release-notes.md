**OdentAI Segment {{VERSION}}** — CBCT (DICOM) to anatomical surfaces in Blender: mandible, upper skull, upper and lower teeth, mandibular canal, maxillary sinuses, nasal cavity, pharynx, soft palate, plus individual FDI-numbered teeth, implants and bridges.

Ready-built: the add-on, the segmentation engine and the model weights are all inside. Nothing else to install.

## What's new in 1.1.2

- **Mandibular canal on fine scans.** When the canal comes out broken on one side, that side is computed again at a finer resolution. On a 0.15 mm scan where half of the left canal was missing it now comes out whole. A healthy canal is not touched.
- **Crop box over one side of the jaw:** a sound canal is no longer reported as broken and recomputed, so such runs are faster.
- **Large scans need less memory** (about 2 GB less on a 200 mm field of view).

## Download

| System | File |
|---|---|
| **Windows 10 / 11** | `OdentAI-{{VERSION}}-windows.zip` |
| **Ubuntu 22.04 or newer** | `OdentAI-{{VERSION}}-linux.zip` — over GitHub's 2 GB file limit because CUDA is bundled, so it comes in parts `.zip.001`, `.zip.002`, …: download **all** of them and join them (below) |

Each archive has a `.sha256` file next to it to check the download.

## Install

1. **Linux, if the zip came in parts — join them** in the download folder:
   ```bash
   cat OdentAI-{{VERSION}}-linux.zip.0* > OdentAI-{{VERSION}}-linux.zip
   sha256sum -c OdentAI-{{VERSION}}-linux.zip.sha256
   ```
2. In Blender: **Edit → Preferences → Add-ons → Install…** (in Blender 4.2+: the **⌄** menu at the top right → **Install from Disk…**) and pick the `.zip`. Do not unpack it.
3. Tick **OdentAI Segment** in the add-on list.
4. In the 3D view press **N** and open the **OdentAI** tab.
5. Press **Check system** once: it shows whether the graphics card is used and how long a scan will take.

Updating from an older version: remove the old add-on first (Preferences → Add-ons → OdentAI Segment → Remove), restart Blender, then install the new zip. After 1.0.1 the add-on folder is called `OdentAI` (it was `dental9_addon`), so the two would otherwise sit side by side; the new one refuses to start while the old one is enabled and says so.

## Requirements

- **Blender** 3.3 or newer; **we recommend 4.5 or newer** (tested on 3.3, 3.6, 4.5, 5.0 and 5.2).
- **Windows:** any graphics card with an up-to-date driver (NVIDIA, AMD or Intel, via DirectML). No CUDA needed. Without a usable card it runs on the CPU, about 15–20× slower.
- **Linux:** an NVIDIA card with driver **525 or newer**; CUDA and cuDNN are bundled. Without one it runs on the CPU.
- **Memory:** a 100 mm field of view needs about 6 GB of RAM and ~7 GB of video memory for the full-size tile (a smaller card automatically gets smaller tiles). A 200 mm field needs about 19 GB of RAM — use the crop box (below).

## Use

1. **Scan** — a DICOM folder (nested exports are fine), a single `.dcm`, a `.zip`, or a `.nii.gz` / `.mha` / `.nrrd` file.
2. Optional, recommended for large scans — **Region → Bone preview**: a quick bone surface in a few seconds, with a box around it. Drag the **coloured arrows** to move each wall of the box, the **orange ring** to move the whole box. With **Only inside the box** ticked, only that region is computed — much faster and lighter on memory. Delete the box and the whole scan is computed again.
3. Choose the **classes**, optionally **Separate teeth (FDI numbers)**, and press **Segment**. A single progress bar at the top of the panel shows the stage and the time; **Stop** (or Esc in the 3D view) ends the run. Each run goes into its own collection inside **OdentAI**, named by number and scan (e.g. `2 · Smith/CT`), so different CTs in one project stay apart and an old run is deleted in one click; the bone preview and the box go into **OdentAI › Region**.

## If something goes wrong

The full log of the last run is in `%TEMP%\OdentAI\last_run.log` on Windows, `/tmp/OdentAI/last_run.log` on Linux — the document button in the **Hardware** block opens it. Please send this file along with a description of the problem.

Community: https://t.me/odent_blender
