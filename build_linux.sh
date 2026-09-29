#!/usr/bin/env bash
# Linux (Ubuntu) build. The add-on is the same; only the binary inside differs.
#
# The build is self-contained: CUDA and cuDNN are inside, the machine only
# needs an NVIDIA driver. The price is about 2.5 GB of libraries. Windows has
# none of this: DirectML is a single 19 MB library.
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-.venv}"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" -q install --upgrade pip
"$VENV/bin/pip" -q install -r requirements-deploy.txt pyinstaller onnxruntime-gpu \
    "nvidia-cudnn-cu12>=9.5,<10" nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
rm -rf build dist/dental9
"$VENV/bin/pyinstaller" --noconfirm --distpath dist --workpath build dental9.spec

# CUDA and cuDNN libraries go INSIDE the build from NVIDIA's pip wheels (see
# dental9.spec, nvidia_libs). This script used to drop cufft, curand and
# cublasLt as "redundant" — that was a mistake: they are direct dependencies
# of the provider, and the build machine simply satisfied them from its
# system CUDA. On a machine without system CUDA the provider does not load
# without them at all.

echo "built: dist/dental9/dental9"
du -sh dist/dental9
./dist/dental9/dental9 --providers
