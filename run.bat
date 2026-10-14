@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project environment is missing. Run the setup commands in README.md first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
