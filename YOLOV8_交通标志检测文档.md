# 🚦 YOLOv8 交通标志检测系统 - 完整文档

## 📋 项目概述

本项目使用 **YOLOv8 (You Only Look Once v8)** 深度学习模型完成**交通标志检测**任务。基于提供的交通标志数据集，训练目标检测模型，并在测试集上生成符合Kaggle提交格式的预测结果。

### 任务类型
- **任务**: 目标检测 (Object Detection)
- **模型**: YOLOv8 (Ultralytics)
- **数据集格式**: YOLO格式
- **评估指标**: mAP@0.5

---

## 👤 学生信息

- **姓名**: 杨淋皓
- **学号**: 112304260120
- **班级**: 数据1231

---

## 📊 数据集信息

### 数据集结构
```
第4次实验数据及提交格式 (1)/
├── train/                  # 训练集
│   ├── images/            # 训练图片 (*.jpg, *.png)
│   └── labels/            # 标注文件 (*.txt) - YOLO格式
├── val/                    # 验证集
│   ├── images/
│   └── labels/
├── test/images/           # 测试集（只有图片，无标签）
├── data.yaml              # 数据集配置文件
├── README.md              # 数据集说明
├── baseline_infer.py      # 基线推理示例
└── sample_submission.csv  # 提交格式示例
```

### 类别信息（15类）
| Class ID | 类别名称 | 说明 |
|----------|---------|------|
| 0 | Green Light | 绿灯 |
| 1 | Red Light | 红灯 |
| 2 | Speed Limit 10 | 限速10 |
| 3 | Speed Limit 100 | 限速100 |
| 4 | Speed Limit 110 | 限速110 |
| 5 | Speed Limit 120 | 限速120 |
| 6 | Speed Limit 20 | 限速20 |
| 7 | Speed Limit 30 | 限速30 |
| 8 | Speed Limit 40 | 限速40 |
| 9 | Speed Limit 50 | 限速50 |
| 10 | Speed Limit 60 | 限速60 |
| 11 | Speed Limit 70 | 限速70 |
| 12 | Speed Limit 80 | 限速80 |
| 13 | Speed Limit 90 | 限速90 |
| 14 | Stop | 停止标志 |

### 标注格式（YOLO格式）
每个标注文件对应一张图片，内容为：
```
class_id x_center y_center width height
```
所有坐标为归一化值 [0, 1]

示例 (`000000_jpg.rf.b11f308f16626f9f795a148029c46d10.txt`):
```
7 0.5336538461538461 0.3173076923076923 0.16947115384615385 0.3173076923076923
```
表示：类别7 (Speed Limit 30)，中心坐标(0.534, 0.317)，宽高(0.169, 0.317)

---

## 🔧 技术方案

### 模型架构
```
YOLOv8 (You Only Look Once v8)
├─ Backbone: CSPDarknet53
├─ Neck: PANet (Path Aggregation Network)
└─ Head: Decoupled Head (解耦头)

版本选择: YOLOv8m (Medium)
- 参数量: ~25.9M
- 精度与速度的平衡点
- 适合中等规模数据集
```

### 训练策略
```python
优化器: SGD with Momentum / AdamW
学习率: 0.01 (初始) → 余弦退火调度
Batch Size: 16
Epochs: 100 (with Early Stopping, patience=20)
图像尺寸: 640×640

数据增强:
- Mosaic: 1.0 (概率)
- MixUp: 0.0
- 随机翻转: 左右50%
- 颜色增强: HSV (H:0.015, S:0.7, V:0.4)
- 几何变换: 旋转±10°, 平移10%, 缩放50%, 剪切2°
```

### 关键技术点
1. **迁移学习**: 使用COCO预训练权重
2. **多尺度训练**: 输入640×640
3. **自动混合精度(AMP)**: 加速训练
4. **Early Stopping**: 防止过拟合
5. **测试时增强(TTA)**: 提升推理精度

---

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.12
ultralytics >= 8.0.0
OpenCV-Python >= 4.5.0
```

### 安装依赖
```bash
pip install ultralytics torch torchvision opencv-python
```

### 方法一：一键运行（推荐）

运行简化版训练脚本：
```bash
cd d:\机器学习实验3
python train_yolo_simple.py
```

这个脚本会自动：
1. ✅ 下载预训练模型（yolov8m.pt）
2. ✅ 训练模型（100 epochs）
3. ✅ 在验证集上评估
4. ✅ 在测试集上推理
5. ✅ 生成 submission_yolov8.csv

### 方法二：分步执行

#### Step 1: 仅训练
```bash
python yolov8_traffic_sign_detection.py --mode train --epochs 100
```

#### Step 2: 仅推理（使用已有模型）
```bash
python yolov8_traffic_sign_detection.py --mode predict \
    --model-path runs/detect/traffic_sign/weights/best.pt \
    --conf 0.001
```

#### Step 3: 完整流程（训练+推理）
```bash
python yolov8_traffic_sign_detection.py --mode full
```

### 方法三：使用快速启动脚本
```bash
python quick_start.py
```

---

## 📁 项目文件说明

```
机器学习实验3/
├── yolov8_traffic_sign_detection.py   # 主程序（完整版）
├── train_yolo_simple.py               # 简化版训练脚本 ⭐推荐使用
├── quick_start.py                     # 一键启动脚本
│
├── 第4次实验数据及提交格式 (1)/        # 数据集目录
│   ├── data.yaml                      # 数据配置
│   ├── train/                         # 训练集
│   ├── val/                           # 验证集
│   └── test/images/                   # 测试图片
│
├── results/                           # 输出结果
│   └── submission_yolov8.csv          # Kaggle提交文件 ⭐
│
└── runs/detect/traffic_sign/          # 训练结果
    ├── weights/
    │   ├── best.pt                    # 最佳模型权重 ⭐
    │   └── last.pt                    # 最后一个epoch的权重
    ├── args.yaml                      # 训练参数
    ├── results.csv                    # 训练指标记录
    ├── confusion_matrix.png          # 混淆矩阵
    ├── results.png                   # 训练曲线图
    └── val_batch*_pred.jpg           # 预测可视化
```

---

## 📈 预期性能指标

基于YOLOv8m在类似交通标志数据集上的表现：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| **mAP@0.5** | 0.85 - 0.95 | 主要评估指标 |
| **mAP@0.5:0.95** | 0.65 - 0.80 | 严格mAP |
| **Precision** | 0.85 - 0.92 | 精确率 |
| **Recall** | 0.82 - 0.90 | 召回率 |
| **F1-Score** | 0.85 - 0.90 | F1分数 |
| **推理速度** | ~15 ms/image | GPU推理 |

> 注：实际性能取决于数据集质量和超参数调优

---

## 📝 提交格式说明

### CSV文件格式
提交文件 `submission_yolov8.csv` 必须包含以下列：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| image_id | string | 图片文件名 | "000003_jpg.rf.8511b9c219dbf9799a6d58900b15917d.jpg" |
| class_id | int | 类别ID (0-14) | 7 |
| x_center | float | 归一化中心X坐标 [0,1] | 0.534 |
| y_center | float | 归一化中心Y坐标 [0,1] | 0.317 |
| width | float | 归一化宽度 [0,1] | 0.169 |
| height | float | 归一化高度 [0,1] | 0.317 |
| confidence | float | 置信度 [0,1] | 0.9876 |

### 示例CSV内容
```csv
image_id,class_id,x_center,y_center,width,height,confidence
000003_jpg.rf.xxx.jpg,7,0.534,0.317,0.169,0.317,0.9876
000006_jpg.rf.xxx.jpg,9,0.456,0.523,0.145,0.198,0.9543
...
```

### 重要提示
- 所有坐标必须是**归一化值**（范围[0, 1]）
- 使用**YOLO格式** (x_center, y_center, width, height)
- 置信度阈值建议设为 **0.001**（保留所有预测）
- 每张图片可以有**多个检测结果**

---

## 🔍 结果分析

### 训练监控
训练过程中会生成以下可视化：

1. **results.png**: 训练曲线
   - Loss变化（train/val box loss, cls loss, dfl loss）
   - 精度变化（precision, recall, mAP@0.5, mAP@0.5-0.95）

2. **confusion_matrix.png**: 混淆矩阵
   - 展示各类别的分类情况
   - 识别混淆的类别对

3. **val_batch*_pred.jpg**: 预测可视化
   - 展示模型在验证集上的预测效果
   - 不同batch的可视化结果

### 性能优化建议

如果需要进一步提升性能：

1. **增加训练轮数**
   ```bash
   python train_yolo_simple.py  # 修改epochs=150或200
   ```

2. **使用更大的模型**
   ```python
   model = YOLO('yolov8l.pt')  # 或 'yolov8x.pt'
   ```

3. **调整图像尺寸**
   ```python
   imgsz=800  # 或 1024（需要更多显存）
   ```

4. **启用更多数据增强**
   ```python
   mixup=0.1,
   copy_paste=0.1,
   ```

5. **测试时增强(TTA)**
   ```python
   results = model.predict(..., augment=True)
   ```

---

## ❓ 常见问题

### Q1: 显存不足怎么办？
**A**: 减小batch size或image size
```python
batch=8,      # 从16改为8
imgsz=480,    # 从640改为480
```

### Q2: 如何使用GPU？
**A**: Ultralytics会自动检测并使用GPU。确保安装了CUDA版本的PyTorch。
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 训练太慢？
**A**: 
- 使用更小的模型（yolov8n 或 yolov8s）
- 减少epochs
- 启用cache=True（已默认开启）
- 增加workers数量

### Q4: 如何查看中间结果？
**A**: 训练过程中的结果保存在：
```
runs/detect/traffic_sign/
```

### Q5: 如何继续训练？
**A**: 使用last.pt恢复训练
```python
model = YOLO('runs/detect/traffic_sign/weights/last.pt')
model.train(resume=True)
```

---

## 🎯 Kaggle提交流程

### Step 1: 生成提交文件
```bash
python train_yolo_simple.py
```

### Step 2: 定位提交文件
```
results/submission_yolov8.csv
```

### Step 3: 提交到Kaggle
1. 访问比赛页面
2. 点击 "Submit Predictions"
3. 上传 `submission_yolov8.csv`
4. 查看评分结果

### Step 4: 优化迭代
根据Kaggle反馈调整参数：
- 如果mAP低 → 增加训练轮数、调整学习率
- 如果某些类别识别差 → 检查数据质量、增加该类别样本

---

## 📚 参考资源

### 官方文档
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics HUB](https://hub.ultralytics.com/)

### 教程资源
- [YOLOv8 自定义数据训练教程](https://docs.ultralytics.com/modes/train/)
- [YOLOv8 推理与预测](https://docs.ultralytics.com/modes/predict/)
- [最佳实践指南](https://docs.ultralytics.com/usage/best_practices/)

### 相关论文
- **YOLOv8**: Real-Time Object Detection with Ultralytics (2023)
- **YOLOX**: Exceeding YOLO Series in 2021 (2021)
- **YOLOv5**: You Only Look Once: Unified, Real-Time Object Detection (2020)

---

## 📌 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-04-29 | 初始版本：完整的YOLOv8训练与推理系统 |

---

## 👨‍💻 联系方式

- **学生姓名**: 杨淋皓
- **学号**: 112304260120
- **班级**: 数据1231
- **GitHub**: https://github.com/Ylh2004-shuju/112304260120yanglinhao

---

## 📄 许可证

本项目仅供学习和研究使用。

---

**⭐ 如果这个项目对你有帮助，欢迎给个Star！**

**最后更新时间**: 2026年4月29日
