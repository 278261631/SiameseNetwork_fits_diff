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

set "http_proxy=http://localhost:10551"
set "https_proxy=http://localhost:10551"
set "HTTP_PROXY=%http_proxy%"
set "HTTPS_PROXY=%https_proxy%"
set "no_proxy=localhost,127.0.0.1"
set "NO_PROXY=%no_proxy%"
echo [setup] Using pip proxy: %http_proxy%

echo [setup] Upgrading pip ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [setup] Installing requirements.txt ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [setup] Done. python: %PY%
exit /b 0

