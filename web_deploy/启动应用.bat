@echo off
chcp 65001 >nul
echo ============================================================
echo    ✍️ 手写数字识别系统 - 启动脚本
echo ============================================================
echo.
echo 正在启动Flask Web应用...
echo.
echo 模型文件: d:\机器学习实验3\results\best_cnn_hypertuned.pth
echo 应用地址: http://localhost:5000
echo.
echo 按 Ctrl+C 可以停止服务器
echo.
echo ============================================================
echo.

cd /d "d:\机器学习实验3\web_deploy"
python app.py

pause
