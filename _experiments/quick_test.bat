@echo off
REM Quick Test Script for PrepGen AI Service
REM Run this to test your server

echo.
echo ========================================
echo PrepGen AI Service - Quick Test
echo ========================================
echo.

echo [1] Testing Health Check...
curl -s http://localhost:8000/health
echo.
echo.

echo [2] Testing Session Stats...
curl -s http://localhost:8000/sessions/stats
echo.
echo.

echo ========================================
echo Test Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Install ngrok: choco install ngrok
echo 2. Run in new terminal: ngrok http 8000
echo 3. Copy the ngrok URL
echo 4. Share with your friend
echo.
echo See QUICK_START.md for details
echo.

pause
