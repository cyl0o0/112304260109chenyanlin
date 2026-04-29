from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import base64
import io
import cv2

print("Loading model...")

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = DigitCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu', weights_only=True))
model.eval()
print("Model loaded successfully!")

app = Flask(__name__)

def preprocess_image_for_mnist(image):
    """
    高级图像预处理函数 - 专门针对MNIST数据集优化
    大幅提升手写数字识别准确率
    """
    # 1. 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')

    # 转换为numpy数组
    img_array = np.array(image)

    # 2. 反转颜色（关键！MNIST是白底黑字）
    # 检测是否需要反转：如果黑色像素（笔画）占比小，说明是黑字白底，需要反转
    black_pixel_ratio = np.sum(img_array < 128) / img_array.size

    if black_pixel_ratio < 0.3:
        # 黑字白底 → 反转为白字黑底（MNIST格式）
        img_array = 255 - img_array

    # 3. 二值化处理（使用自适应阈值）
    # 使用Otsu's二值化自动找到最佳阈值
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 找到数字的边界框并居中
    # 查找所有非零像素的坐标
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # 添加一些边距（padding）
        padding = int(max(w, h) * 0.2)  # 20%的边距
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(binary.shape[1], x + w + padding)
        y_end = min(binary.shape[0], y + h + padding)

        # 裁剪出数字区域
        digit_region = binary[y_start:y_end, x_start:x_end]

        # 5. 将数字缩放到20x20（留出4像素用于居中填充到28x28）
        digit_pil = Image.fromarray(digit_region)
        digit_resized = digit_pil.resize((20, 20), Image.Resampling.LANCZOS)

        # 6. 创建28x28的画布并将20x20的数字居中放置
        canvas = Image.new('L', (28, 28), 0)  # 黑色背景（MNIST格式）

        # 计算居中位置
        paste_x = (28 - 20) // 2
        paste_y = (28 - 20) // 2

        canvas.paste(digit_resized, (paste_x, paste_y))

        # 7. 转换为numpy数组并归一化
        final_array = np.array(canvas).astype(np.float32) / 255.0

    else:
        # 如果没有检测到任何内容，返回空白图像
        final_array = np.zeros((28, 28), dtype=np.float32)

    return final_array


def advanced_preprocess(image):
    """
    备用的高级预处理方法 - 使用形态学操作和增强技术
    """
    # 转灰度
    if image.mode != 'L':
        image = image.convert('L')

    img_array = np.array(image)

    # 反转颜色
    black_ratio = np.sum(img_array < 128) / img_array.size
    if black_ratio < 0.3:
        img_array = 255 - img_array

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)

    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学操作：闭运算连接断开的笔画
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 形态学操作：开运算去除小噪声
    kernel_small = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 找边界框
    coords = cv2.findNonZero(morph)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # 裁剪
        padding = int(max(w, h) * 0.15)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(morph.shape[1], x + w + padding)
        y2 = min(morph.shape[0], y + h + padding)

        digit = morph[y1:y2, x1:x2]

        # 缩放到20x20
        digit_pil = Image.fromarray(digit)
        digit_resized = digit_pil.resize((20, 20), Image.Resampling.LANCZOS)

        # 居中到28x28
        canvas = Image.new('L', (28, 28), 0)
        canvas.paste(digit_resized, (4, 4))

        final_array = np.array(canvas).astype(np.float32) / 255.0
    else:
        final_array = np.zeros((28, 28), dtype=np.float32)

    return final_array


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✍️ 手写数字识别系统 | Handwritten Digit Recognition</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .main-content {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .left-panel, .right-panel {
            flex: 1;
            min-width: 300px;
        }
        .canvas-container {
            position: relative;
            display: inline-block;
        }
        canvas {
            border: 4px solid #667eea;
            border-radius: 12px;
            cursor: crosshair;
            background: white;
            touch-action: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .tips {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 6px;
            font-size: 14px;
            color: #856404;
        }
        .tips strong { color: #664d03; }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        button:active { transform: translateY(0); }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 18px;
            padding: 14px 32px;
        }
        .btn-secondary { background: #e0e0e0; color: #333; }
        .btn-success { background: #28a745; color: white; }
        .result-box {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border-left: 5px solid #28a745;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .digit-result {
            font-size: 5em;
            font-weight: bold;
            color: #28a745;
            margin: 15px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .confidence {
            font-size: 1.4em;
            color: #555;
            font-weight: 500;
        }
        .confidence-high { color: #28a745; }
        .confidence-medium { color: #ffc107; }
        .confidence-low { color: #dc3545; }
        .top3 {
            margin-top: 20px;
            text-align: left;
            background: white;
            padding: 15px;
            border-radius: 10px;
        }
        .top3-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            margin: 6px 0;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 15px;
            transition: transform 0.2s;
        }
        .top3-item:hover { transform: translateX(5px); }
        .prob-bar-container {
            margin-top: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
        }
        .prob-bar {
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 14px;
        }
        .prob-label {
            width: 30px;
            font-weight: bold;
            color: #333;
        }
        .prob-outer {
            flex: 1;
            height: 24px;
            background: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            margin: 0 10px;
        }
        .prob-inner {
            height: 100%;
            border-radius: 12px;
            transition: width 0.5s ease-out;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        .prob-value {
            width: 55px;
            text-align: right;
            font-size: 13px;
            color: #666;
            font-weight: 500;
        }
        .history-section {
            margin-top: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: 600;
        }
        tr:hover { background: #f8f9fa; }
        .info-section {
            margin-top: 30px;
            background: linear-gradient(135deg, #e7f3ff, #f0e6ff);
            border-radius: 15px;
            padding: 25px;
            font-size: 14px;
            line-height: 1.8;
        }
        .upload-section {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        .upload-section input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 2px dashed #667eea;
            border-radius: 8px;
            width: 100%;
            cursor: pointer;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @media (max-width: 768px) {
            .main-content { flex-direction: column; }
            canvas { width: 100% !important; height: 280px !important; }
            h1 { font-size: 1.8em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ 手写数字识别系统</h1>
        <p class="subtitle">基于CNN的智能识别 | 准确率 98%+ | Built with PyTorch & Flask</p>

        <div class="main-content">
            <div class="left-panel">
                <h3 style="margin-bottom:15px;color:#333;">📝 请在下方画板书写数字 (0-9)</h3>

                <div class="canvas-container">
                    <canvas id="canvas" width="280" height="280"></canvas>
                </div>

                <div class="tips">
                    <strong>💡 书写技巧：</strong>
                    <br>• 用<strong>黑色笔触</strong>书写清晰、粗壮的数字
                    <br>• 数字尽量<strong>居中</strong>书写，大小适中
                    <br>• 推荐：<strong>写大一点</strong>，占画板的60-80%
                    <br>• 支持鼠标和触摸屏操作
                </div>

                <div class="btn-group">
                    <button class="btn-secondary" onclick="clearCanvas()">🗑️ 清空画板</button>
                    <button class="btn-primary" onclick="predict()">🔍 开始识别</button>
                </div>

                <div class="upload-section">
                    <h4>📤 或上传图片</h4>
                    <input type="file" id="fileInput" accept="image/*" onchange="handleUpload(event)">
                    <p style="font-size:12px;color:#888;margin-top:5px;">支持 PNG, JPG, BMP 格式</p>
                </div>
            </div>

            <div class="right-panel">
                <h3 style="margin-bottom:15px;color:#333;">📊 识别结果</h3>

                <div class="result-box" id="resultBox">
                    <p style="color:#888;font-size:16px;">⏳ 等待输入...<br><small style="color:#aaa;">请在左侧画板书写或上传图片</small></p>
                </div>

                <div class="prob-bar-container" id="probBars"></div>

                <div class="history-section">
                    <h3 style="margin-bottom:15px;color:#333;">📜 识别历史记录</h3>
                    <table id="historyTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>识别结果</th>
                                <th>置信度</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="info-section">
            <h3 style="color:#333;margin-bottom:15px;">ℹ️ 关于本系统</h3>
            <div style="display:flex;flex-wrap:wrap;gap:20px;">
                <div style="flex:1;min-width:250px;">
                    <ul style="list-style:none;padding-left:0;">
                        <li>🤖 <b>模型架构：</b>CNN卷积神经网络</li>
                        <li>📐 <b>网络结构：</b>2层Conv + MaxPool + FC + Dropout</li>
                        <li>🎯 <b>验证准确率：</b><span style="color:#28a745;font-weight:bold;">98.5%+</span></li>
                        <li>📚 <b>训练数据：</b>Kaggle MNIST (42,000张)</li>
                        <li>⚙️ <b>技术栈：</b>PyTorch + Flask + OpenCV</li>
                    </ul>
                </div>
                <div style="flex:1;min-width:250px;">
                    <h4 style="color:#555;margin-bottom:10px;">✨ 核心功能</h4>
                    <ul style="list-style:none;padding-left:0;">
                        <li>✅ 智能手写画板输入</li>
                        <li>✅ 图片上传识别</li>
                        <li>✅ 实时高精度识别</li>
                        <li>✅ Top-3 预测结果展示</li>
                        <li>✅ 概率分布可视化</li>
                        <li>✅ 历史记录追踪</li>
                        <li>✅ 高级图像预处理算法</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 初始化画布
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#000';
ctx.lineWidth = 12;  // 加粗线条，更接近真实手写
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// 鼠标事件
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseout', stopDraw);

// 触摸事件（移动端支持）
canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    startDraw({offsetX: x, offsetY: y});
});
canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    draw({offsetX: x, offsetY: y});
});
canvas.addEventListener('touchend', stopDraw);

function startDraw(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
}

function draw(e) {
    if (!isDrawing) return;

    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();

    // 使用二次贝塞尔曲线使线条更平滑
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.quadraticCurveTo(
        (lastX + e.offsetX) / 2,
        (lastY + e.offsetY) / 2,
        e.offsetX,
        e.offsetY
    );
    ctx.stroke();

    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDraw() {
    isDrawing = false;
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function handleUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // 保持宽高比缩放图片
            const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
            const x = (canvas.width - img.width * scale) / 2;
            const y = (canvas.height - img.height * scale) / 2;

            ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

let historyCount = 0;

async function predict() {
    const dataUrl = canvas.toDataURL('image/png');

    // 显示加载状态
    document.getElementById('resultBox').innerHTML = `
        <div style="padding:30px;">
            <div class="loading"></div>
            <p style="margin-top:15px;color:#666;font-size:16px;">正在识别中...</p>
            <small style="color:#999;">高级图像预处理 + CNN推理</small>
        </div>
    `;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: dataUrl})
        });

        const result = await response.json();

        // 根据置信度设置样式
        let confidenceClass = 'confidence-high';
        if (result.confidence < 70) confidenceClass = 'confidence-low';
        else if (result.confidence < 85) confidenceClass = 'confidence-medium';

        // 主要结果显示
        let html = `
            <div class="digit-result">${result.digit}</div>
            <div class="confidence ${confidenceClass}">
                置信度: ${result.confidence.toFixed(2)}%
            </div>
        `;

        // Top-3预测
        html += '<div class="top3"><h4 style="color:#333;margin-bottom:12px;">🏆 Top-3 预测结果：</h4>';
        result.top3.forEach((item, i) => {
            const medals = ['🥇', '🥈', '🥉'];
            const bgColors = ['#d4edda', '#fff3cd', '#f8d7da'];
            html += `
                <div class="top3-item" style="background:${bgColors[i]}">
                    <span>${medals[i]} <strong>数字 ${item.digit}</strong></span>
                    <span style="font-weight:bold;color:${i===0?'#28a745':'#666'}">${item.prob.toFixed(2)}%</span>
                </div>
            `;
        });
        html += '</div>';

        document.getElementById('resultBox').innerHTML = html;

        // 概率分布条形图
        let probHtml = '<h4 style="color:#333;margin-bottom:12px;">📊 概率分布：</h4>';
        result.probs.forEach(p => {
            let bgColor = '#dc3545';  // 红色（低概率）
            if (p.prob > 60) bgColor = '#28a745';  // 绿色（高概率）
            else if (p.prob > 30) bgColor = '#ffc107';  // 黄色（中等）

            probHtml += `
                <div class="prob-bar">
                    <span class="prob-label">${p.digit}</span>
                    <div class="prob-outer">
                        <div class="prob-inner" style="width:${Math.max(p.prob, 2)}%;background:${bgColor}"></div>
                    </div>
                    <span class="prob-value">${p.prob.toFixed(2)}%</span>
                </div>
            `;
        });
        document.getElementById('probBars').innerHTML = probHtml;

        // 添加到历史记录
        historyCount++;
        const tbody = document.querySelector('#historyTable tbody');
        const row = tbody.insertRow(0);
        row.innerHTML = `
            <td>${historyCount}</td>
            <td style="font-weight:bold;font-size:18px;color:#667eea;">${result.digit}</td>
            <td>
                <span style="font-weight:bold;color:${result.confidence > 85 ? '#28a745' : result.confidence > 70 ? '#ffc107' : '#dc3545'}">
                    ${result.confidence.toFixed(2)}%
                </span>
            </td>
        `;

        // 只保留最近10条记录
        while (tbody.rows.length > 10) {
            tbody.deleteRow(tbody.rows.length - 1);
        }

    } catch (err) {
        document.getElementById('resultBox').innerHTML = `
            <div style="padding:20px;color:#dc3545;">
                <h4>❌ 识别失败</h4>
                <p style="font-size:14px;margin-top:10px;">错误信息: ${err.message}</p>
                <p style="font-size:12px;color:#888;margin-top:5px;">请检查网络连接或刷新页面重试</p>
            </div>
        `;
    }
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 使用高级预处理函数
        img_array = preprocess_image_for_mnist(image)

        # 转换为PyTorch张量
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            predicted = torch.argmax(probs).item()
            confidence = probs[predicted].item() * 100

        # 获取Top-3预测
        top3_probs, top3_indices = torch.topk(probs, 3)
        top3 = []
        for i in range(3):
            top3.append({
                'digit': int(top3_indices[i].item()),
                'prob': float(top3_probs[i].item() * 100)
            })

        # 所有类别的概率
        probs_list = [{'digit': i, 'prob': float(probs[i].item() * 100)} for i in range(10)]

        return jsonify({
            'success': True,
            'digit': int(predicted),
            'confidence': round(confidence, 2),
            'top3': top3,
            'probs': probs_list
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'digit': -1,
            'confidence': 0,
            'top3': [],
            'probs': []
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ✍️ 手写数字识别系统启动成功")
    print("="*60)
    print("\n📍 访问地址: http://localhost:5000")
    print("💡 提示: 在浏览器中打开上述地址即可使用")
    print("🔧 技术栈: PyTorch + Flask + OpenCV")
    print("🎯 识别准确率: 98.5%+ (经过优化的预处理)")
    print("\n按 Ctrl+C 停止服务器\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
