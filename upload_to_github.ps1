# 机器学习实验3上传到GitHub脚本
Write-Host "开始上传实验到GitHub..." -ForegroundColor Green

# 检查是否有.git目录
if (-not (Test-Path ".git")) {
    Write-Host "初始化Git仓库..." -ForegroundColor Yellow
    git init
}

# 配置用户信息（可根据需要修改）
Write-Host "配置Git用户信息..." -ForegroundColor Yellow
git config user.name "cyl0o0"
git config user.email "cyl0o0@users.noreply.github.com"

# 添加所有文件
Write-Host "添加所有文件到暂存区..." -ForegroundColor Yellow
git add .

# 提交
Write-Host "提交文件..." -ForegroundColor Yellow
git commit -m "实验3：IMDB情感分析，完成二分类和概率预测"

# 添加远程仓库
Write-Host "连接远程仓库..." -ForegroundColor Yellow
git remote add origin https://github.com/cyl0o0/112304260109chenyanlin.git

# 重命名分支为main
Write-Host "设置主分支为main..." -ForegroundColor Yellow
git branch -M main

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "本地准备完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：请在Git Bash或命令行中执行以下命令来推送到GitHub：" -ForegroundColor Yellow
Write-Host "git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "注意：推送时可能需要使用GitHub Personal Access Token作为密码" -ForegroundColor Red
Write-Host "获取Token：Settings -> Developer settings -> Personal access tokens" -ForegroundColor Gray
Write-Host ""
Write-Host "或者您也可以直接在Git Bash中运行完整命令：" -ForegroundColor Yellow
Write-Host "cd 'd:\机器学习实验3'; git push -u origin main" -ForegroundColor White
Write-Host ""
