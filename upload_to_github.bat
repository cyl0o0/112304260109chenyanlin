@echo off
echo ========================================
echo   Upload to GitHub
echo ========================================
echo.

echo [1/5] Copying documentation...
if exist YOLOV8_交通标志检测文档.md (
    copy /Y YOLOV8_交通标志检测文档.md README.md
    echo     OK - README updated
)

echo.
echo [2/5] Checking git...
if not exist .git (
    git init
    echo     OK - Git initialized
)

echo.
echo [3/5] Configuring git...
git config user.name "cyl0o0"
git config user.email "cyl0o0@users.noreply.github.com"
git remote remove origin 2>nul
git remote add origin https://github.com/cyl0o0/112304260109chenyanlin.git
echo     OK - Git configured

echo.
echo [4/5] Adding files...
git add .
echo     OK - Files added

echo.
echo [5/5] Committing...
git commit -m "Add YOLOv8 Traffic Sign Detection Project"
echo     OK - Files committed

echo.
echo ========================================
echo   Done! Now push to GitHub
echo ========================================
echo.
echo To push to GitHub, run:
echo   git push -u origin main
echo.
echo If submission_final.csv is missing, first run:
echo   python generate_final_csv.py
echo.
pause
