@echo off
chcp 65001 >nul
title YouTube Whisper Plus
echo.
echo  =========================================
echo   YOUTUBE WHISPER PLUS
echo   Faster-Whisper + yt-dlp + Gradio
echo  =========================================
echo.

cd /d "%~dp0"

REM -- Check UV --
where uv >nul 2>&1
if errorlevel 1 (
    echo  [!] UV not found - it is the only manual requirement.
    echo.
    echo  Install by pasting this into PowerShell:
    echo.
    echo  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    echo  Then close and re-open this window.
    pause
    exit /b 1
)
echo  [OK] UV found

REM -- Python via UV --
echo  Checking Python...
uv python install ">=3.11,<3.13" --quiet
echo  [OK] Python ready

REM -- Check FFmpeg --
where ffmpeg >nul 2>&1
if errorlevel 1 (
    if not exist "ffmpeg\ffmpeg.exe" (
        echo.
        echo  [!] FFmpeg not found on system PATH or in .\ffmpeg\
        echo.
        echo  Option 1 - Download from https://www.gyan.dev/ffmpeg/builds/
        echo    Extract and drop ffmpeg.exe + ffprobe.exe into the .\ffmpeg\ folder
        echo.
        echo  Option 2 - Install via Chocolatey:
        echo    choco install ffmpeg
        echo.
        pause
        exit /b 1
    ) else (
        echo  [OK] FFmpeg found: .\ffmpeg\ffmpeg.exe
        set "PATH=%~dp0ffmpeg;%PATH%"
    )
) else (
    echo  [OK] FFmpeg found on system PATH
)

REM -- Detect GPU using Python (much more reliable than batch parsing) --
echo.
set GPU_MODE=cpu
set TORCH_INDEX=

REM Run the Python GPU detector and capture GPU_MODE from its output
for /f "tokens=*" %%L in ('uv run --no-project detect_gpu.py 2^>nul') do (
    echo %%L | findstr /C:"GPU_MODE=" >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2 delims==" %%M in ("%%L") do set GPU_MODE=%%M
    ) else (
        echo  %%L
    )
)

if "%GPU_MODE%"=="cuda" (
    set TORCH_INDEX=--extra-index-url https://download.pytorch.org/whl/cu126
)

echo.

REM -- Virtual environment --
if not exist ".venv\" (
    echo  Creating virtual environment...
    uv venv --python ">=3.11,<3.13"
    echo  Done!
    echo.
) else (
    echo  Virtual environment found.
    echo.
)

REM -- Install dependencies --
echo  Installing / checking dependencies (%GPU_MODE% mode)...
echo  First run downloads PyTorch + packages - this may take a few minutes.
echo.

uv pip install -r requirements.txt %TORCH_INDEX% --index-strategy unsafe-best-match

if errorlevel 1 (
    echo.
    echo  [!] Dependency install failed.
    echo  Try deleting the .venv folder and running launch.bat again.
    pause
    exit /b 1
)

echo.
echo  =========================================
echo   All good! Launching in %GPU_MODE% mode...
echo   Browser opens at http://localhost:7860
echo   Keep this window open while using the app.
echo  =========================================
echo.

uv run app.py

pause
