@echo off
REM ─────────────────────────────────────────────────────────────
REM  AccessLearn — Setup & Run (Windows)
REM  Double-click this file, or run from cmd prompt
REM ─────────────────────────────────────────────────────────────

echo.
echo   AccessLearn -- Inclusive AI Learning Platform
echo   ================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Create venv if not exists
IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

REM Set API key
IF "%GEMINI_API_KEY%"=="" SET GEMINI_API_KEY=AIzaSyCAmlqHeDK_95FYvLIgcdV-z6W8Xd8_yak

echo.
echo [OK] Starting AccessLearn on http://localhost:5000
echo      Open your browser to http://localhost:5000
echo      Press Ctrl+C to stop
echo.

python app.py
pause
