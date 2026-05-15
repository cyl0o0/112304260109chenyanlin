@echo off
chcp 65001 >nul
title 手写数字识别 - 环境测试
echo ============================================================
echo    🔍 环境测试
echo ============================================================
echo.
echo 正在检查Python...
python --version
if errorlevel 1 (
    echo [错误] 未找到Python!
    pause
    exit /b 1
)

echo.
echo [成功] Python已找到!
echo.
echo 正在运行测试脚本...
echo.
python hello.py
echo.
pause
