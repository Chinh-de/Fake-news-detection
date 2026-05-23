@echo off
echo ===================================================
echo   ViFN Fake News Detection Platform - Launcher
echo ===================================================
echo.
echo [1/2] Starting FastAPI Backend (port 8000)...
start "ViFN Backend" cmd /k "cd /d e:\PBL7\demo_phobert && python -m uvicorn be.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir be"

echo [2/2] Starting Next.js Frontend (port 3000)...
timeout /t 3 /nobreak > nul
start "ViFN Frontend" cmd /k "cd /d e:\PBL7\demo_phobert\fe && npm run dev"

echo.
echo ===================================================
echo   Both servers starting up...
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Frontend: http://localhost:3000
echo ===================================================
echo.
timeout /t 5 /nobreak > nul
start http://localhost:3000
