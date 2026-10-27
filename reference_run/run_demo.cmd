@echo off
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\pythonw.exe" (
    echo Project root Python environment not found. See ..\docs\ENVIRONMENT.md.
    pause
    exit /b 1
)
start "" "%~dp0..\.venv\Scripts\pythonw.exe" "%~dp0gui.py"
