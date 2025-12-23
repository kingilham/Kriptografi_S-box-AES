@echo off
TITLE Advanced AES S-Box Analyzer
echo ==========================================
echo Starting AES S-Box Analyzer Web Server...
echo ==========================================
echo.

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.9 or newer.
    pause
    exit /b
)

:: Navigate to backend directory
cd backend

:: Check if virtual environment exists
if not exist "venv" (
    echo [INFO] First run detected. Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
)

:: Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment scripts not found.
    pause
    exit /b
)

:: Install dependencies
echo [INFO] Checking and installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

:: Run the FastAPI server
echo.
echo Server starting at http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause