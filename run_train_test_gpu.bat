@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PY=%cd%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [run-gpu] ERROR: venv python not found: %PY%
  echo [run-gpu] Please run: setup_venv.bat
  exit /b 1
)

REM Check CUDA availability first
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo [run-gpu] ERROR: CUDA is not available in current environment.
  echo [run-gpu] Please check NVIDIA driver / CUDA / torch installation.
  exit /b 1
)

REM defaults
set "TRAIN_TILES=data/tiles"
set "VAL_TILES=test_data"
set "OUT_DIR=runs/siamese_unet_gpu"
set "EPOCHS=30"
set "BATCH=1"
set "RESIZE=0"
set "CROP_TRAIN=0"
set "CROP_TEST=0"
set "CUDA_VISIBLE_DEVICES=0"

if /I "%1"=="quick" (
  set "EPOCHS=2"
  set "BATCH=1"
  set "RESIZE=256"
  set "CROP_TRAIN=128"
  set "CROP_TEST=128"
  echo [run-gpu] QUICK mode: epochs=!EPOCHS! batch=!BATCH! resize_to=!RESIZE! crop_train=!CROP_TRAIN! crop_test=!CROP_TEST!
)

echo [run-gpu] CUDA available, use CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo === 1) Train Siamese UNet segmentation ===
dir /b "%VAL_TILES%\*_mask.*" >nul 2>nul
if errorlevel 1 (
  echo [run-gpu] No mask found in "%VAL_TILES%". Will NOT use it as val set; fallback to random split from train.
)
if errorlevel 1 (
  "%PY%" train_seg.py --tiles_dir "%TRAIN_TILES%" --out_dir "%OUT_DIR%" --epochs !EPOCHS! --batch_size !BATCH! --resize_to !RESIZE! --crop_size !CROP_TRAIN!
) else (
  "%PY%" train_seg.py --tiles_dir "%TRAIN_TILES%" --val_tiles_dir "%VAL_TILES%" --out_dir "%OUT_DIR%" --epochs !EPOCHS! --batch_size !BATCH! --resize_to !RESIZE! --crop_size !CROP_TRAIN!
)
if errorlevel 1 exit /b 1

set "CKPT=%OUT_DIR%\best.pt"
if not exist "%CKPT%" (
  echo [run-gpu] ERROR: checkpoint not found: %CKPT%
  exit /b 1
)

echo === 2) Eval on test_data (metrics) ===
dir /b "%VAL_TILES%\*_mask.*" >nul 2>nul
if errorlevel 1 (
  echo [run-gpu] No mask found in "%VAL_TILES%". Skip eval.
) else (
  "%PY%" eval_seg.py --tiles_dir "%VAL_TILES%" --ckpt "%CKPT%" --resize_to !RESIZE! --crop_size !CROP_TEST!
  if errorlevel 1 exit /b 1
)

echo === 3) Inference export (pred mask + prob) ===
"%PY%" infer_seg.py --tiles_dir "%VAL_TILES%" --ckpt "%CKPT%" --resize_to !RESIZE! --crop_size !CROP_TEST!
if errorlevel 1 exit /b 1

echo [run-gpu] Done.
echo [run-gpu] model: %CKPT%
echo [run-gpu] preds: %VAL_TILES%
echo [run-gpu] NOTE: train/eval/infer script must select CUDA device internally.
exit /b 0
