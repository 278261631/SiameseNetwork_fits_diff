@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PY=%cd%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [test] ERROR: venv python not found: %PY%
  echo [test] Please run: setup_venv.bat
  exit /b 1
)

REM defaults
set "VAL_TILES=test_data"
set "CKPT=runs/siamese_unet/best.pt"
set "INFER_OUT=runs/infer_unet_test"
set "RESIZE=0"
set "CROP_TEST=0"

REM usage:
REM   run_test_only.bat
REM   run_test_only.bat quick
REM   run_test_only.bat <ckpt_path>
REM   run_test_only.bat <ckpt_path> <tiles_dir>

if not "%~1"=="" (
  if /I "%~1"=="quick" (
    set "RESIZE=256"
    set "CROP_TEST=128"
    echo [test] QUICK mode: resize_to=!RESIZE! crop_test=!CROP_TEST!
  ) else (
    set "CKPT=%~1"
  )
)

if not "%~2"=="" (
  set "VAL_TILES=%~2"
)

if not exist "%CKPT%" (
  echo [test] ERROR: checkpoint not found: %CKPT%
  exit /b 1
)

echo === 1) Eval on tiles (metrics) ===
dir /b "%VAL_TILES%\*_mask.*" >nul 2>nul
if errorlevel 1 (
  echo [test] No mask found in "%VAL_TILES%". Skip eval, run inference only.
) else (
  "%PY%" eval_seg.py --tiles_dir "%VAL_TILES%" --ckpt "%CKPT%" --resize_to !RESIZE! --crop_size !CROP_TEST!
  if errorlevel 1 exit /b 1
)

echo === 2) Inference export (pred mask + prob) ===
"%PY%" infer_seg.py --tiles_dir "%VAL_TILES%" --ckpt "%CKPT%" --out_dir "%INFER_OUT%" --resize_to !RESIZE! --crop_size !CROP_TEST!
if errorlevel 1 exit /b 1

echo [test] Done.
echo [test] ckpt: %CKPT%
echo [test] preds: %INFER_OUT%
exit /b 0

