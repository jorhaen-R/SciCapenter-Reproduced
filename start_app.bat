@echo off
echo [INFO] Step 1: Checking Docker...

:: ???????????????????????????docker info >nul
if errorlevel 1 goto ERROR

echo [INFO] Docker is running.
echo [INFO] Step 2: Starting containers...
docker compose up -d

echo [INFO] Step 3: Waiting for backend (8 seconds)...
timeout /t 8 /nobreak >nul

echo [INFO] Step 4: Opening Browser...
start https://unserving-karon-overshot.ngrok-free.dev
start http://localhost:4040

echo.
echo [SUCCESS] App is running in background.
echo Press any key to close this window.
pause
exit

:ERROR
echo.
echo [ERROR] Docker is NOT running!
echo [ACTION] Please open "Docker Desktop" first.
pause
exit

