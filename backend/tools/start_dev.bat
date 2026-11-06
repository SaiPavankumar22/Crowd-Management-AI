@echo off
REM Double-click launcher for the Godavari Crowd Monitor.
REM Venv + data live inside the project folder.
cd /d "%~dp0..\.."

set ROOT=%CD%
set VENV=%GODAVARI_VENV%
if "%VENV%"=="" set VENV=%ROOT%\.venv
set DATA_ROOT=%GODAVARI_DATA%
if "%DATA_ROOT%"=="" set DATA_ROOT=%ROOT%\.data
set CROWD_DATA_DIR=%DATA_ROOT%\data
set CROWD_UPLOADS_DIR=%DATA_ROOT%\uploads
set CROWD_SAMPLES_DIR=%DATA_ROOT%\samples
set PYTHONPATH=%ROOT%\backend;%PYTHONPATH%
set HOST=127.0.0.1
set PORT=8000

if not exist "%VENV%\Scripts\python.exe" (
  echo venv not found at %VENV% - run: bash backend/tools/install.sh
  pause
  exit /b 1
)

"%VENV%\Scripts\python.exe" run.py
