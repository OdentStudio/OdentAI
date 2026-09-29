@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM  Build the Dental9 Blender add-on on Windows.
REM
REM  Run this on the Windows machine itself: the add-on ships a native
REM  executable, which cannot be built on another OS.
REM
REM  Needs: Python 3.9-3.12 and the whole deploy folder.
REM  The weights are not in the repository - bring them along:
REM    models\dental9.onnx + dental9.json      (283 MB, the main model)
REM    models\teeth_fdi.onnx + teeth_fdi.json  (205 MB, "Separate teeth")
REM
REM  We install onnxruntime-directml, NOT -gpu: DirectML runs on any card,
REM  including AMD and integrated ones, and needs neither the CUDA Toolkit
REM  nor a matching NVIDIA driver - nobody in a clinic will install those.
REM ============================================================

echo.
echo === Dental9: building the Blender add-on for Windows ===
echo.

REM --- 1. Python ------------------------------------------------
REM 3.9 to 3.12. The upper bound is not pedantry: onnxruntime-directml may
REM have no wheels yet for a brand new Python, and pip would fail for nothing.
set PY=
call :trypy py -3.12
call :trypy py -3.11
call :trypy py -3.10
call :trypy py -3
call :trypy python
if not defined PY (
    echo [ERROR] no Python 3.9-3.12 found.
    echo         Install it from python.org and tick "Add to PATH".
    goto :fail
)
for /f "delims=" %%V in ('%PY% -c "import sys;print(sys.version.split()[0])"') do set PYVER=%%V
echo [1/6] Python !PYVER! ^(%PY%^) - ok

REM --- 2. Virtual environment -----------------------------------
if not exist .venv (
    echo [2/6] creating the .venv environment, a couple of minutes...
    %PY% -m venv .venv || goto :fail
) else (
    echo [2/6] .venv already exists
)
set VPY=.venv\Scripts\python.exe
REM A .venv built on another OS has no Scripts\python.exe. It can arrive here
REM by simply copying the folder from a Linux machine, so rebuild it instead
REM of stopping: the user has no way to guess what is wrong.
if not exist "%VPY%" (
    echo       .venv is not usable on this system, rebuilding it...
    rmdir /s /q .venv
    %PY% -m venv .venv || goto :fail
)
if not exist "%VPY%" (
    echo [ERROR] could not create a working environment. Check that Python
    echo         was installed for this user and try again.
    goto :fail
)

REM --- 3. Dependencies ------------------------------------------
echo [3/6] installing dependencies...
"%VPY%" -m pip install -q --upgrade pip || goto :fail
"%VPY%" -m pip install -q -r requirements-deploy.txt || goto :fail
"%VPY%" -m pip install -q pyinstaller onnxruntime-directml || goto :fail
REM onnxruntime and onnxruntime-directml conflict: if both end up installed,
REM the CPU one wins and the build silently ships without GPU support.
"%VPY%" -m pip uninstall -y -q onnxruntime >nul 2>&1
"%VPY%" -c "import onnxruntime as o; ps=o.get_available_providers(); print('     providers:', ', '.join(ps)); import sys; sys.exit(0 if 'DmlExecutionProvider' in ps else 1)"
if errorlevel 1 (
    echo [ERROR] no DirectML in the environment; plain onnxruntime got installed.
    echo         Delete the .venv folder and run this again.
    goto :fail
)

REM --- 4. Weights -----------------------------------------------
set NOMODEL=
if not exist models\dental9.onnx set NOMODEL=1
if not exist models\dental9.json set NOMODEL=1
if defined NOMODEL (
    echo [ERROR] no weights: models\dental9.onnx and models\dental9.json must
    echo         both be in the models folder. Without them the add-on is useless,
    echo         so the build stops here instead of packing an empty zip.
    echo         ^(2026-09-13: a build silently produced a 95 MB zip without
    echo         models - never again.^) To build without weights on purpose:
    echo         set ALLOW_NO_MODEL=1 and run again.
    if not defined ALLOW_NO_MODEL goto :fail
    echo [4/6] WARNING: building WITHOUT weights ^(ALLOW_NO_MODEL is set^)
) else (
    for %%F in (models\dental9.onnx) do set /a MB=%%~zF/1048576
    echo [4/6] weights found, !MB! MB
)

REM Second pass (Separate teeth) ships its own weights next to the main ones.
set NOTEETH=
if not exist models\teeth_fdi.onnx set NOTEETH=1
if not exist models\teeth_fdi.json set NOTEETH=1
if defined NOTEETH (
    echo [ERROR] teeth_fdi.onnx / teeth_fdi.json not found in models. The add-on
    echo         ships with "Separate teeth"; put both files there and run again.
    echo         To build without them on purpose: set ALLOW_NO_MODEL=1
    if not defined ALLOW_NO_MODEL goto :fail
    echo       WARNING: packing WITHOUT "Separate teeth" ^(ALLOW_NO_MODEL is set^)
) else (
    for %%F in (models\teeth_fdi.onnx) do set /a TMB=%%~zF/1048576
    echo       teeth model found, !TMB! MB
)

REM --- 5. Build --------------------------------------------------
echo [5/6] building dental9.exe, 3-5 minutes...
if exist build rmdir /s /q build
if exist dist\dental9 rmdir /s /q dist\dental9
"%VPY%" -m PyInstaller --noconfirm --distpath dist --workpath build dental9.spec >build_windows.log 2>&1
if errorlevel 1 (
    echo [ERROR] the build failed. Details in build_windows.log, last lines:
    powershell -NoProfile -Command "Get-Content build_windows.log -Tail 15"
    goto :fail
)
if not exist dist\dental9\dental9.exe (
    echo [ERROR] dental9.exe was not produced, see build_windows.log
    goto :fail
)

REM Hardware check on the freshly built binary: it opens a session and really
REM runs a probe tile, so it is immediately visible whether the GPU is used.
echo.
echo --- checking the build ---
if defined NOMODEL (
    dist\dental9\dental9.exe --providers
) else (
    dist\dental9\dental9.exe --diagnose -m models\dental9.onnx
)
echo --- end of check ---
echo.

REM --- 6. Packaging ----------------------------------------------
echo [6/6] packing the add-on zip...
if defined NOMODEL (
    "%VPY%" scripts\pack_addon.py --no-model || goto :fail
) else if defined NOTEETH (
    "%VPY%" scripts\pack_addon.py --model models\dental9.onnx --no-teeth-model || goto :fail
) else (
    "%VPY%" scripts\pack_addon.py --model models\dental9.onnx --teeth-model models\teeth_fdi.onnx || goto :fail
)

echo.
echo === DONE ===
echo Add-on:  %CD%\dist\dental9_addon_windows.zip
echo.
echo Install: Blender - Edit - Preferences - Add-ons - Install - pick this zip.
echo The panel appears in the 3D view sidebar ^(N key^), Dental9 tab.
echo.
pause
exit /b 0

:trypy
REM Tries a candidate and remembers the first suitable one.
if defined PY goto :eof
%* -c "import sys;sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%*"
goto :eof

:fail
echo.
echo === BUILD DID NOT COMPLETE ===
echo.
pause
exit /b 1
