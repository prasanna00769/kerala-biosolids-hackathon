@echo off
echo Starting Kerala BioCycle System...

:: Start Backend
start "Backend API" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe app.py"

:: Start Frontend (using python http.server to avoid CORS issues with file://)
start "Frontend" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m http.server 8000"

:: Wait a bit for servers to start
timeout /t 3

:: Open Browser
start http://localhost:8000/index.html

echo System started!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:8000
pause
