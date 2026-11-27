@echo off
chcp 65001 >nul
cd /d "%~dp0"
title A-Level Math 1 — AI Exam Studio (Port 8000)

echo [1/3] Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
    echo Stopping existing server on port 8000 (PID %%a^)...
    taskkill /F /T /PID %%a >nul 2>&1
)

echo [2/3] Opening http://127.0.0.1:8000 in browser...
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

echo [3/3] Starting A-Level Math 1 Dashboard...
python -m src.server
if errorlevel 1 pause
