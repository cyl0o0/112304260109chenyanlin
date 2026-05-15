# Upload YOLO Project to GitHub
Write-Host "=== Upload YOLO Traffic Sign Detection Project to GitHub ===" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
Write-Host "Current directory: $currentDir" -ForegroundColor Yellow

# Check required files
$requiredFiles = @(
    "generate_final_csv.py",
    "YOLOV8_交通标志检测文档.md",
    "exp.torchscript"
)

Write-Host ""
Write-Host "Checking required files..." -ForegroundColor Cyan
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
    }
}

# Check if submission_final.csv exists
if (Test-Path "submission_final.csv") {
    Write-Host "  [OK] submission_final.csv exists" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] submission_final.csv not found" -ForegroundColor Yellow
    Write-Host "  You may need to run generate_final_csv.py first" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Copying documentation as README ===" -ForegroundColor Cyan

# Copy YOLO doc as README
if (Test-Path "YOLOV8_交通标志检测文档.md") {
    Copy-Item "YOLOV8_交通标志检测文档.md" -Destination "README.md" -Force
    Write-Host "  [OK] README.md updated with YOLO documentation" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Git Operations ===" -ForegroundColor Cyan

# Initialize git if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "  Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Configure user
git config user.name "cyl0o0"
git config user.email "cyl0o0@users.noreply.github.com"

# Add remote
git remote remove origin 2>$null
git remote add origin "https://github.com/cyl0o0/112304260109chenyanlin.git"

# Add all files
Write-Host "  Adding files..." -ForegroundColor Yellow
git add .

# Commit
Write-Host "  Committing changes..." -ForegroundColor Yellow
git commit -m "Add YOLOv8 Traffic Sign Detection Project - submission file and documentation"

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Verify everything looks good with: git status" -ForegroundColor White
Write-Host "2. Push to GitHub with: git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "If you need to generate submission_final.csv first, run:" -ForegroundColor Yellow
Write-Host "  python generate_final_csv.py" -ForegroundColor White
Write-Host ""
