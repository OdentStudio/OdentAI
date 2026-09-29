"""Reading the CT and bringing it to the form the model was trained on.

Accepted input (verified on real exports from clinics):
  a folder of DICOM slices, nested sub-folders included;
  a folder holding a single multi-frame .dcm;
  a single .dcm file — multi-frame or one slice of a series;
  a .zip archive of any of the above;
  a volume file: .nii.gz, .mha, .nrrd and whatever else ITK reads.

Three things must match training voxel for voxel, otherwise the prediction
is garbage — plausible-looking garbage:

  orientation    LPS. That is what the dataset has (checked on all 838
                 files). Not RPI: RPI is the convention of the frozen
                 Dataset002 branch, and the two must not be confused.
  spacing        0.3 mm on all axes, linear interpolation.
  normalisation  nnU-Net's CTNormalization: clip to the 0.5 and 99.5
                 percentiles of the dataset fingerprint, subtract the mean,
                 divide by the standard deviation. The numbers come from the
                 plans, they are not set here.
"""
import glob
import os
import shutil
from typing import Optional

import numpy as np
import SimpleITK as sitk

from . import winpath

VOLUME_EXTS = (".dcm", ".mha", ".mhd", ".nrrd", ".nhdr", ".nii", ".nii.gz",
               ".img", ".hdr", ".vtk", ".gipl", ".mnc", ".mrc", ".rec", ".spr",
               ".h5", ".tif", ".tiff")


def _drop_degenerate_4th(img: sitk.Image) -> sitk.Image:
    """Collapse a degenerate fourth axis if one appeared.

    ImageSeriesReader on a series made of one multi-frame file returns 4D of
    shape (x, y, z, 1) — time as a separate axis of length one. Everything
    then dies on DICOMOrient, which refuses 4D ("Pixel type ... is not
    supported in 4D"), at the very first step before any computation.
    A size of 0 in Extract means "remove the axis".
    """
    if img.GetDimension() != 4:
        return img
    size = list(img.GetSize())
    if size[3] != 1:
        raise RuntimeError(f"a scan with {size[3]} time frames is not supported")
    size[3] = 0
    return sitk.Extract(img, size, [0, 0, 0, 0])


def _read_dir(path: str, reader: sitk.ImageSeriesReader) -> Optional[sitk.Image]:
    """The longest DICOM series in the folder and everything below it.

    Exports arrive without extensions (IM000001) and spread over sub-folders,
    so the series is searched recursively and the longest one is taken as the
    scan: scout images and screenshots often sit next to it.
    """
    best = None
    for root, _dirs, _files in os.walk(path):
        for sid in reader.GetGDCMSeriesIDs(root):
            files = reader.GetGDCMSeriesFileNames(root, sid)
            if best is None or len(files) > len(best):
                best = files
    if not best:
        return None
    # A one-file series is a multi-frame .dcm. ImageSeriesReader returns 4D
    # for it, so read the file directly: that way it arrives as proper 3D.
    if len(best) == 1:
        return _drop_degenerate_4th(sitk.ReadImage(best[0]))
    reader.SetFileNames(best)
    return _drop_degenerate_4th(reader.Execute())


def read_volume(path: str) -> sitk.Image:
    """Whatever a clinic may send, reduced to a single 3D image."""
    if os.path.isfile(path):
        if path.lower().endswith(".zip"):
            return _read_zip(path)
        img = _drop_degenerate_4th(sitk.ReadImage(path))
        # A single .dcm out of a series: read as a volume it is one slice.
        # Returning it silently is not an option — segmenting one slice looks
        # like a working program with a meaningless result. Go up to the
        # folder and read the whole series.
        if img.GetDimension() == 3 and img.GetSize()[2] == 1:
            got = _read_dir(os.path.dirname(os.path.abspath(path)),
                            sitk.ImageSeriesReader())
            if got is not None and got.GetSize()[2] > 1:
                return got
            raise RuntimeError(
                f"{os.path.basename(path)} is a single slice, not a volume, "
                "and no full series was found next to it")
        return img
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    got = _read_dir(path, sitk.ImageSeriesReader())
    if got is not None:
        return got

    # No series — maybe the folder holds a volume file or an archive.
    for f in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
        if f.lower().endswith(VOLUME_EXTS):
            return _drop_degenerate_4th(sitk.ReadImage(f))
    zips = sorted(glob.glob(os.path.join(path, "*.zip")))
    if len(zips) == 1:
        return _read_zip(zips[0])
    raise RuntimeError(f"no DICOM series and no volume file found in {path}")


def _read_zip(path: str) -> sitk.Image:
    """The archive is unpacked into a temporary folder and read as a folder.

    That is how scans most often arrive from clinics, and unpacking by hand
    for one run is an extra step on which nested folders get lost. The
    temporary folder is removed; by then the image is in memory.
    """
    # Entries get ASCII names and the folder is ASCII on Windows: the native
    # readers cannot open anything else there (see winpath).
    tmp = winpath.temp_dir()
    try:
        winpath.extract_zip_ascii(path, tmp)
        img = read_volume(tmp)
        # SimpleITK pixels are lazy: without an Execute the image may stay
        # bound to a file that is about to disappear.
        return sitk.Cast(img, img.GetPixelID())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _corners(lo, hi):
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])]


def physical_bounds(img: sitk.Image):
    """The axis-aligned box the scan occupies in patient coordinates (mm),
    voxel edges included: (min xyz, max xyz). Same frame as the STL files."""
    n = img.GetSize()
    pts = np.array([img.TransformContinuousIndexToPhysicalPoint(c)
                    for c in _corners((-0.5, -0.5, -0.5),
                                      (n[0] - 0.5, n[1] - 0.5, n[2] - 0.5))])
    return pts.min(0), pts.max(0)


def crop_to_box(img: sitk.Image, lo, hi) -> sitk.Image:
    """Keep only the voxels inside a box given in patient coordinates (mm).

    The box comes from the add-on, drawn over the preview surface, so it is in
    the same frame as the STL files. The scan grid may be rotated against that
    frame; the crop is then the grid-aligned box that contains the drawn one —
    slightly more, never less. The cropped image keeps its physical position,
    so everything downstream (the meshes included) lands where it was.
    """
    n = np.array(img.GetSize())
    idx = np.array([img.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in c))
                    for c in _corners(lo, hi)])
    a = np.maximum(np.floor(idx.min(0) + 0.5).astype(int), 0)
    b = np.minimum(np.ceil(idx.max(0) + 0.5).astype(int), n)
    if np.any(b - a < 2):
        raise RuntimeError("the crop box does not overlap the scan: move it over "
                           "the preview surface and try again")
    return sitk.RegionOfInterest(img, [int(v) for v in b - a], [int(v) for v in a])


def to_training_grid(img: sitk.Image, spacing: float = 0.3) -> sitk.Image:
    """LPS and the given spacing. Returns an image, not an array: the geometry
    is still needed to put the labels back on the original grid and to build
    meshes in patient coordinates."""
    img = sitk.DICOMOrient(img, "LPS")
    src_sp = np.array(img.GetSpacing(), float)
    src_sz = np.array(img.GetSize(), int)
    dst_sz = np.round(src_sz * src_sp / spacing).astype(int)
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing((spacing, spacing, spacing))
    r.SetSize([int(v) for v in dst_sz])
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(sitk.sitkLinear)
    r.SetDefaultPixelValue(float(-1000))     # outside the frame is air, not zero
    return r.Execute(img)


def normalize_ct(vol: np.ndarray, props: dict) -> np.ndarray:
    """nnU-Net's CTNormalization, reproduced literally.

    The point is that the window comes from the TRAINING SET FINGERPRINT, not
    from the scan itself: CBCT is not calibrated, and normalising each scan on
    its own (what ZScore does) is wrong for CT. That is exactly why
    dataset.json says channel_names "CT" rather than "CBCT".
    """
    lo = float(props["percentile_00_5"])
    hi = float(props["percentile_99_5"])
    mean = float(props["mean"])
    std = max(float(props["std"]), 1e-8)
    out = vol.astype(np.float32, copy=True)
    np.clip(out, lo, hi, out=out)
    out -= mean
    out /= std
    return out


def resample_labels_to(ref: sitk.Image, labels: np.ndarray,
                       grid: sitk.Image) -> sitk.Image:
    """Labels from the 0.3 mm grid back onto the scan's original grid.

    Nearest neighbour: labels cannot be interpolated linearly, there is no
    class 4 halfway between class 3 and class 5.
    """
    seg = sitk.GetImageFromArray(labels.astype(np.uint8))
    seg.CopyInformation(grid)
    return sitk.Resample(seg, ref, sitk.Transform(), sitk.sitkNearestNeighbor,
                         0, sitk.sitkUInt8)
