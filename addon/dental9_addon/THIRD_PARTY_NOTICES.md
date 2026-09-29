# Third-party notices — OdentAI Segment

This add-on and its segmentation models were built with the open resources
listed below. Their licences ask for attribution, and we give it gladly: the
work would not exist without them.

## Training data

- **ToothFairy2 / ToothFairy3 challenge dataset** — licence CC BY-SA 4.0.
  Bolelli F. et al., "Multi-structure segmentation in CBCT volumes: The
  ToothFairy2 challenge", *Medical Image Analysis* (2026).
  https://toothfairy2.grand-challenge.org/
- **DentVoxel** — licence CC BY 4.0.
  Zhou Y., Xu Y., Tarce M.-A. (2026). "DentVoxel: a fully annotated dental
  CBCT dataset with 38 instance anatomical structures". figshare.
  https://doi.org/10.6084/m9.figshare.31239889.v2

## Models used to prepare training labels

- **DentalSegmentator** — licence CC BY 4.0.
  Dot G. et al., "DentalSegmentator: robust open source deep learning-based CT
  and CBCT image segmentation", *Journal of Dentistry* (2024).
  https://doi.org/10.5281/zenodo.10829675
- **TotalSegmentator**, task `craniofacial_structures` — licence Apache-2.0.
  Wasserthal J. et al., "TotalSegmentator: Robust Segmentation of 104 Anatomic
  Structures in CT Images", *Radiology: Artificial Intelligence* (2023).

## Method

- **nnU-Net** — licence Apache-2.0.
  Isensee F., Jaeger P.F., Kohl S.A.A., Petersen J., Maier-Hein K.H.,
  "nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation", *Nature Methods* 18, 203–211 (2021).

## Datasets used for evaluation only (not for training)

- **NasalSeg** — Zhang Y. et al., *Scientific Data* (2024).
- **PMCanalSeg** — *Scientific Data* (2026), CC0.

## Software shipped inside the engine

- ONNX Runtime — MIT
- SimpleITK — Apache-2.0
- VTK — BSD-3-Clause
- NumPy — BSD-3-Clause
- PyInstaller — GPL-2.0 with the bootloader exception, which permits
  distributing the bundled application under any licence
- NVIDIA CUDA runtime, cuBLAS, cuFFT, cuRAND, NVRTC and cuDNN (Linux build) —
  redistributed under the NVIDIA Software License Agreement; the licence texts
  ship with the respective NVIDIA packages
- DirectML (Windows build) — Microsoft, redistributed with ONNX Runtime

## The add-on script

`__init__.py` uses the Blender Python API and is therefore licensed under
GPL-2.0-or-later, as Blender's add-on policy requires. The segmentation engine
(`bin/`) runs as a separate executable, is not linked to Blender, and is not
part of the add-on script.
