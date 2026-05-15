@echo off
chcp 65001 >nul
title 手写数字识别系统
echo ============================================================
echo    ✍️ 手写数字识别系统
echo ============================================================
echo.
echo 正在启动Web应用...
echo.
echo [提示] 启动后在浏览器打开: http://localhost:5000
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python simple_app.py

echo.
echo 服务器已停止
pause
