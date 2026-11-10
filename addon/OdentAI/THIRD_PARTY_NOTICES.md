# Third-party notices — OdentAI Segment

This add-on and its segmentation models were built with the open resources
listed below. Their licences ask for attribution, and we give it gladly: the
work would not exist without them.

## The model weights — licence

The model weights are **not part of this add-on**. They are a separate
download, licensed under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**:
https://creativecommons.org/licenses/by-nc-sa/4.0/ — they may not be used for
commercial purposes. The add-on fetches them only after you accept these
terms.

## Training data

- **ToothFairy2 / ToothFairy3** — University of Modena and Reggio Emilia,
  licence CC BY-NC-SA 4.0. https://ditto.ing.unimore.it

  The ToothFairy3 logo in the model licence dialog belongs to the dataset's
  authors and is shown only to acknowledge the data source.

  When you use the weights or results obtained with them, acknowledge
  ToothFairy3 as the data source and cite:

  - Lumetti L., Tan Z.Q., Borghi L., Addison O., Li Y., Rosati G.,
    van Nistelrooij N., Vinayahalingam S., Grana C., Bolelli F. (2026).
    "ToothFairy3: Scaling CBCT Maxillofacial Segmentation to 77 Classes with
    U-Mamba2". In *Medical Image Computing and Computer Assisted Intervention —
    MICCAI 2026*.
  - Bolelli F., Lumetti L., van Nistelrooij N., Vinayahalingam S.,
    Di Bartolomeo M., Marchesini K., Pellacani A., Candeloro E., Rosati G.,
    Xi T., Isensee F., Kirchhoff Y., Krämer L., Rokuss M., Ulrich C.,
    Maier-Hein K., Jiang Y., Liu Y., Wang L., Wang H., Chen S., Cui Z., Shi P.,
    Pan Z., Liang X., Ma Q., Konukoglu E., Wodzinski M., Müller H., Mai H.,
    Dang X., Bhandary S., Grosu R., Bergé S., Anesi A., Grana C. (2026).
    "Multi-structure segmentation in CBCT volumes: The ToothFairy2 challenge".
    *Medical Image Analysis*, 104095. https://doi.org/10.1016/j.media.2026.104095
  - Bolelli F., Marchesini K., van Nistelrooij N., Lumetti L., Pipoli V.,
    Ficarra E., Vinayahalingam S., Grana C. (2025). "Segmenting Maxillofacial
    Structures in CBCT Volume". In *IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR)*, pp. 1–10.
    https://dx.doi.org/10.1109/CVPR52734.2025.00494
  - Lumetti L., Pipoli V., Bolelli F., Ficarra E., Grana C. (2024). "Enhancing
    Patch-Based Learning for the Segmentation of the Mandibular Canal".
    *IEEE Access*, 1–12. https://doi.org/10.1109/ACCESS.2024.3408629

  The current list is kept on the Ditto platform.
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
