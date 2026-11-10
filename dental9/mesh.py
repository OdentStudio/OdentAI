"""Labels -> surfaces -> STL in millimetres, in patient coordinates.

The isosurface is built not from the binary mask but from the signed field
`class logit minus the best of the others`. The zero of that field is exactly
the boundary argmax gives, but its position inside a voxel is set by the
values rather than rounded to half a voxel. The difference is visible: on a
binary mask marching cubes always puts the vertex at the middle of an edge,
hence voxel-sized steps; the signed field has none, and nothing is smoothed —
we simply do not throw away what the network already computed.

Gaussian smoothing of the mask before meshing is deliberately not done: it
shrinks the surface systematically. Where faceting still bothers, Taubin
smoothing on the mesh removes it without shrinkage.
"""
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

# Targeted imports instead of `import vtk`: that pulls vtkmodules.all with
# rendering included — hundreds of megabytes in the bundle for no benefit.
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (vtkDecimatePro, vtkFlyingEdges3D,
                                       vtkPolyDataNormals, vtkReverseSense,
                                       vtkWindowedSincPolyDataFilter)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from vtkmodules.util.numpy_support import numpy_to_vtk


def _index_to_patient(ref: sitk.Image) -> vtkTransform:
    """Rotation and shift of the grid into patient coordinates.

    Applied as a separate transform on the mesh rather than as the image's
    direction: vtkImageData only knows axis-aligned grids and silently ignores
    a direction matrix, which would put the mesh beside the scan.
    """
    d = np.array(ref.GetDirection()).reshape(3, 3)
    o = np.array(ref.GetOrigin())
    m = vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, float(d[i, j]))
        m.SetElement(i, 3, float(o[i]))
    t = vtkTransform()
    t.SetMatrix(m)
    return t


def _to_vtk(arr: np.ndarray, spacing) -> vtkImageData:
    img = vtkImageData()
    img.SetDimensions(*arr.shape[::-1])          # numpy (z,y,x) -> vtk (x,y,z)
    img.SetSpacing(*spacing)
    img.GetPointData().SetScalars(
        numpy_to_vtk(np.ascontiguousarray(arr, np.float32).ravel(), deep=True,
                     array_type=VTK_FLOAT))
    return img


def drop_small(mask: np.ndarray, spacing, min_cm3: float) -> np.ndarray:
    """Drop pieces below the threshold.

    Not "keep the largest": teeth are a legitimately disconnected class with
    a dozen and a half objects, and the largest of them is one tooth.
    """
    if min_cm3 <= 0 or not mask.any():
        return mask
    vox_cm3 = float(np.prod(spacing)) / 1000.0
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(m),
                               minimumObjectSize=max(1, int(min_cm3 / vox_cm3)))
    return sitk.GetArrayFromImage(cc) > 0


def count_components(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
    return int(sitk.GetArrayFromImage(cc).max())


def build_surface(field: np.ndarray, ref: sitk.Image, isovalue: float = 0.0,
                  taubin: int = 0, pass_band: float = 0.1,
                  decimate: float = 0.0) -> vtkPolyData:
    """field is a signed field on the grid of ref. Returns a mesh in mm."""
    if not np.any(field > isovalue):
        return vtkPolyData()

    surf = vtkFlyingEdges3D()               # same as marching cubes, only faster
    surf.SetInputData(_to_vtk(field, ref.GetSpacing()))
    surf.SetValue(0, float(isovalue))
    surf.ComputeNormalsOff()                # normals at the very end, after all edits
    out = surf

    if taubin > 0:
        sm = vtkWindowedSincPolyDataFilter()   # Taubin: no shrinkage, unlike Laplacian
        sm.SetInputConnection(out.GetOutputPort())
        sm.SetNumberOfIterations(taubin)
        sm.SetPassBand(pass_band)
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        out = sm

    if decimate > 0:
        dec = vtkDecimatePro()
        dec.SetInputConnection(out.GetOutputPort())
        dec.SetTargetReduction(float(decimate))
        dec.PreserveTopologyOn()
        dec.SplittingOff()
        out = dec

    tf = vtkTransformPolyDataFilter()
    tf.SetInputConnection(out.GetOutputPort())
    tf.SetTransform(_index_to_patient(ref))
    out = tf

    # LPS scans can have a direction matrix with a negative determinant. Such
    # a transform flips the triangle winding, the normals point inward and the
    # mesh volume comes out negative. vtkPolyDataNormals does not fix that:
    # Consistency only makes neighbours agree, it does not turn them outward.
    if np.linalg.det(np.array(ref.GetDirection()).reshape(3, 3)) < 0:
        rev = vtkReverseSense()
        rev.SetInputConnection(out.GetOutputPort())
        rev.ReverseCellsOn()
        rev.ReverseNormalsOff()
        out = rev

    nrm = vtkPolyDataNormals()
    nrm.SetInputConnection(out.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.SplittingOff()
    nrm.Update()
    return nrm.GetOutput()


def write_stl(pd: vtkPolyData, path: str) -> Tuple[int, int]:
    w = vtkSTLWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.SetFileTypeToBinary()
    w.Write()
    return pd.GetNumberOfPoints(), pd.GetNumberOfPolys()
