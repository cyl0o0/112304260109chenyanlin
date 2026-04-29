# Kaggle 手写数字识别 - 提交指南

## 比赛信息
- **比赛名称**: Digit Recognizer
- **比赛链接**: https://www.kaggle.com/c/digit-recognizer

## 可用的提交文件

| 文件 | 模型 | 说明 |
|------|------|------|
| `results/submission_digit.csv` | Random Forest | 基础版本，已生成 |
| `results/submission_xgboost.csv` | Extra Trees | 改进版本 |

## 如何提交到 Kaggle

1. 访问 https://www.kaggle.com/c/digit-recognizer/submit
2. 选择一个提交文件（推荐先用 submission_digit.csv）
3. 上传并提交
4. 等待评分结果

## 文件说明

- `code/digit_recognizer.py`: Random Forest 基础版本（已运行成功）
- `code/digit_recognizer_simple.py`: Extra Trees 改进版本
- `code/digit_recognizer_cnn.py`: CNN 深度学习版本（需要TensorFlow）
- `results/sample_digits.png`: 样本数字可视化
- `results/confusion_matrix.png`: 混淆矩阵

## 快速开始

你可以直接使用已有的 `results/submission_digit.csv` 提交到 Kaggle！
