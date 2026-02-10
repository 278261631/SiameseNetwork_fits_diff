@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv" (
  echo [setup] Creating venv .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [setup] ERROR: failed to create venv.
    exit /b 1
  )
) else (
  echo [setup] Found existing venv .venv, skip create.
)

set "PY=%cd%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [setup] ERROR: venv python not found: %PY%
  exit /b 1
)

echo [setup] Upgrading pip ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [setup] Installing requirements.txt ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [setup] Done. python: %PY%
exit /b 0

