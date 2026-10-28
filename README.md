# OdentAI Segment

**CBCT scan in, 3D models out — inside Blender.**

OdentAI Segment is a free Blender add-on that turns a dental CBCT scan (DICOM)
into ready-to-use 3D surfaces — a minute or two on a modern graphics card.
Everything runs on your own computer: the scan never leaves it.

[**Download the latest version**](https://github.com/OdentStudio/OdentAI/releases/latest)
· [Telegram community](https://t.me/odent_blender)

## What you get

Nine structures, each as its own object you can show, hide or export as STL:

| | |
|---|---|
| Mandible | Maxillary sinuses |
| Upper skull | Nasal cavity |
| Upper teeth | Pharynx |
| Lower teeth | Soft palate |
| Mandibular canal | |

Optionally, **Separate teeth** splits the teeth into individual objects named
by their FDI number (11–48) and finds implants, implant crowns and bridges.

## Install

1. Download the zip for your system from the
   [latest release](https://github.com/OdentStudio/OdentAI/releases/latest):
   - **Windows 10 / 11** — `OdentAI-<version>-windows.zip`
   - **Ubuntu 22.04 or newer** — `OdentAI-<version>-linux.zip`. It is large,
     so it comes in parts (`.zip.001`, `.zip.002`, …): download all of them;
     the release page shows the one command that joins them.
2. In Blender: **Edit → Preferences → Add-ons → Install…** (in Blender 4.2 and
   newer: the **⌄** menu at the top right → **Install from Disk…**) and pick
   the zip. **Do not unpack it.**
3. Tick **OdentAI Segment** in the list.
4. In the 3D view press **N** and open the **OdentAI** tab.
5. Press **Check system** once: it tells you whether your graphics card is
   used and how long a scan will take.

The add-on, the engine and the AI models are all inside the zip — nothing else
to install. Blender 3.3 or newer works; **we recommend Blender 4.5 or newer**.

**Updating:** remove the old version first (Preferences → Add-ons → OdentAI
Segment → Remove), restart Blender, then install the new zip.

## Use

1. **Scan** — pick the DICOM folder (nested exports are fine), a single
   `.dcm`, a `.zip`, or a `.nii.gz` / `.mha` / `.nrrd` file.
2. **Bone preview** *(optional, recommended for large scans)* — a quick bone
   surface appears in a few seconds with a box around it.
   - If the preview shows skin instead of the jaws, raise **Density**; lower
     brings soft tissue back.
   - Drag the **coloured arrows** to move the walls of the box and the
     **orange ring** to move the whole box. With **Only inside the box**
     ticked, only that region is computed — much faster and lighter on memory.
3. Tick the **structures** you need, and **Separate teeth** if you want
   numbered teeth.
4. Press **Segment**. A bar shows the progress; **Stop** (or Esc in the 3D
   view) cancels.

Each run lands in its own collection inside **OdentAI**, named after the scan,
so several patients' scans in one project stay apart and an old run is
deleted in one click.

## What your computer needs

| | |
|---|---|
| **Windows** | Any graphics card with an up-to-date driver — NVIDIA, AMD or Intel |
| **Linux** | An NVIDIA graphics card with driver 525 or newer |
| **Memory** | About 6 GB of RAM and 7 GB of video memory for a typical 100 mm scan; smaller cards work too, a bit slower |
| **Without a graphics card** | It still works on the processor, about 15–20 times slower |

A very large scan (200 mm field of view) needs about 19 GB of RAM — use the
box from **Bone preview** to compute only the part you need.

macOS is not supported yet.

## Privacy

The scan is processed **entirely on your computer**. The add-on does not
upload anything and does not connect to the internet; the only link in it is
the button that opens our Telegram community in your browser.

## Please note

OdentAI Segment is a tool for 3D modelling and planning support. It is **not
a certified medical device**, and its results can be wrong. Always check the
models against the original scan before relying on them.

## When something goes wrong

- Press **Check system** in the panel: it runs a real test on your graphics
  card and says in plain words what to do.
- The full log of the last run is one click away — the document button in
  the **Hardware** block — and is saved as `%TEMP%\OdentAI\last_run.log` on
  Windows, `/tmp/OdentAI/last_run.log` on Linux.
- Ask in the [Telegram community](https://t.me/odent_blender). If you attach
  the log, **check it first**: it contains the path of the scan, and folder
  names often contain the patient's name — rename or remove them.

## Credits

OdentAI Segment is made by OdentStudio. It stands on open research and open
data — ToothFairy, DentVoxel, nnU-Net, DentalSegmentator and TotalSegmentator;
see [THIRD_PARTY_NOTICES](addon/OdentAI/THIRD_PARTY_NOTICES.md) for the full
list and licences.

## For developers

How the engine works, how it is built and released, and what must not break:
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Instructions for AI coding agents:
[AGENTS.md](AGENTS.md).
