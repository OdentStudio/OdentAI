# -*- mode: python ; coding: utf-8 -*-
# Build of the production executable. onedir, not onefile: onefile unpacks
# itself into a temporary folder on every launch, and with hundreds of
# megabytes of native libraries the start would take tens of seconds.
#
# The weights are NOT put into the binary: hundreds of megabytes that must be
# replaceable without a rebuild. The model file and dental9.json simply sit
# next to the executable.
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# onnxruntime's native libraries are collected explicitly. On Windows that
# includes DirectML.dll: the analyser does not see it — it is loaded not by an
# import but by the provider at run time, and without it the build silently
# turns CPU-only.
ort_libs = collect_dynamic_libs("onnxruntime")


def nvidia_libs():
    """CUDA and cuDNN libraries from NVIDIA's pip wheels — Linux only.

    Without them the CUDA provider on Linux works only where CUDA and cuDNN
    are installed system-wide. Seen on someone else's machine on 2026-09-13:
    "Unable to load libcudnn_graph.so.9", and everything silently fell back
    to the CPU. Windows needs none of this — DirectML is self-contained.

    Everything the provider and cuDNN declare as theirs is taken (DT_NEEDED
    plus what is loaded by name at run time). Lesson of the same day: cufft,
    curand and cublasLt looked redundant and were dropped — they are direct
    dependencies of the provider that the build machine happened to satisfy
    from its system CUDA. Do not trim by guesswork.
    """
    import glob
    import sys
    if not sys.platform.startswith("linux"):
        return []
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("nvidia-* pip wheels missing: pip install nvidia-cudnn-cu12 "
                         "nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 "
                         "nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12")
    root = list(spec.submodule_search_locations)[0]
    want = [
        "cudnn/lib/libcudnn*.so.9",
        "cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12",
        "cufft/lib/libcufft.so.11",
        "curand/lib/libcurand.so.10",
        "cuda_runtime/lib/libcudart.so.12",
        # the runtime-compiled cuDNN engines call nvrtc by name
        "cuda_nvrtc/lib/libnvrtc.so.12", "cuda_nvrtc/lib/libnvrtc-builtins.so.12.*",
    ]
    out = []
    for pat in want:
        files = glob.glob(os.path.join(root, pat))
        if not files:
            raise SystemExit(f"{pat} not found in {root}")
        out += [(f, ".") for f in files if ".alt." not in f]
    return out


nv_libs = nvidia_libs()

hidden = [
    # onnxruntime and SimpleITK load some modules dynamically; the analyser
    # does not see them
    "onnxruntime.capi._pybind_state",
    "SimpleITK",
    "numpy",
]

# vtkmodules.all pulls in rendering and would triple the bundle; take only
# what mesh.py actually uses
vtk_mods = [
    "vtkmodules.vtkCommonCore", "vtkmodules.vtkCommonDataModel",
    "vtkmodules.vtkCommonMath", "vtkmodules.vtkCommonTransforms",
    "vtkmodules.vtkCommonExecutionModel", "vtkmodules.vtkFiltersCore",
    "vtkmodules.vtkFiltersGeneral", "vtkmodules.vtkFiltersVerdict",
    "vtkmodules.vtkIOGeometry", "vtkmodules.vtkIOCore",
    "vtkmodules.util.numpy_support", "vtkmodules.util.data_model",
    "vtkmodules.util.execution_model",
]

a = Analysis(
    ["entry.py"],
    pathex=[os.path.abspath(".")],
    binaries=ort_libs + nv_libs,
    datas=[("configs/dental9.json", ".")],
    hiddenimports=hidden + vtk_mods,
    hookspath=[],
    runtime_hooks=[],
    # Heavy and unneeded on the production path. torch is not here for show:
    # if it happens to be in the build environment, PyInstaller drags all of
    # it in.
    excludes=["torch", "matplotlib", "tkinter", "IPython", "pytest",
              "scipy", "pandas", "PyQt5", "PySide6", "vtkmodules.all"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="dental9",
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False on purpose: stripping breaks the prebuilt OpenBLAS inside
    # numpy ("ELF load command address/offset not page-aligned") and the
    # import of numpy itself fails. Caught on the 2026-09-10 build.
    strip=False,
    upx=False,          # upx breaks the signatures of onnxruntime's native libraries
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="dental9",
)
