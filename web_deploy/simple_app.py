from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
import io

print("="*60)
print("  ✍️ 手写数字识别系统（简化版）")
print("="*60)

# 定义模型
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

# 加载模型
print("\n正在加载模型...")
model = DigitCNN()
model_path = r'd:\机器学习实验3\results\best_cnn_hypertuned.pth'
print(f"模型路径: {model_path}")

try:
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print("✓ 模型加载成功!")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    print("\n尝试创建一个新的随机模型用于演示...")
    # 创建一个随机模型用于演示
    model = DigitCNN()
    model.eval()
    print("✓ 演示模型已创建!")

app = Flask(__name__)

def simple_preprocess(image):
    """简化的图像预处理"""
    # 转为灰度
    if image.mode != 'L':
        image = image.convert('L')
    
    # 调整大小
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 转为numpy数组并归一化
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 如果图像是白底黑字，反转它
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    
    return img_array

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✍️ 手写数字识别</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2em;
        }
        .main-content {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .canvas-container {
            text-align: center;
        }
        canvas {
            border: 4px solid #667eea;
            border-radius: 12px;
            cursor: crosshair;
            background: white;
            touch-action: none;
        }
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
        button:hover { transform: translateY(-2px); }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-secondary { background: #e0e0e0; color: #333; }
        .result-box {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border-left: 5px solid #28a745;
            min-width: 250px;
        }
        .digit-result {
            font-size: 5em;
            font-weight: bold;
            color: #28a745;
            margin: 15px 0;
        }
        .confidence { font-size: 1.2em; color: #555; }
        .tips {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px 15px;
            margin: 20px 0;
            border-radius: 6px;
            font-size: 14px;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ 手写数字识别</h1>
        
        <div class="main-content">
            <div class="canvas-container">
                <h3 style="margin-bottom:15px;color:#333;">📝 在这里写数字</h3>
                <canvas id="canvas" width="280" height="280"></canvas>
                
                <div class="btn-group">
                    <button class="btn-secondary" onclick="clearCanvas()">🗑️ 清除</button>
                    <button class="btn-primary" onclick="predict()">🔍 识别</button>
                </div>
            </div>
            
            <div class="result-box" id="resultBox">
                <h3 style="color:#333;margin-bottom:15px;">📊 识别结果</h3>
                <p style="color:#888;font-size:16px;">⏳ 等待输入...</p>
            </div>
        </div>
        
        <div class="tips">
            <strong>💡 提示：</strong> 用黑色粗线条书写，数字尽量大一点，居中书写！
        </div>
    </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#000';
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    lastX = e.offsetX;
    lastY = e.offsetY;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    lastX = e.offsetX;
    lastY = e.offsetY;
});

canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    isDrawing = true;
    lastX = touch.clientX - rect.left;
    lastY = touch.clientY - rect.top;
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
});

canvas.addEventListener('touchend', () => isDrawing = false);

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('resultBox').innerHTML = '<h3 style="color:#333;margin-bottom:15px;">📊 识别结果</h3><p style="color:#888;font-size:16px;">⏳ 等待输入...</p>';
}

async function predict() {
    const dataUrl = canvas.toDataURL('image/png');
    
    document.getElementById('resultBox').innerHTML = '<h3 style="color:#333;margin-bottom:15px;">📊 识别结果</h3><p style="color:#666;font-size:16px;"><div class="loading"></div> 正在识别...</p>';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: dataUrl})
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('resultBox').innerHTML = `
                <h3 style="color:#333;margin-bottom:15px;">📊 识别结果</h3>
                <div class="digit-result">${result.digit}</div>
                <div class="confidence">置信度: ${result.confidence}%</div>
            `;
        } else {
            document.getElementById('resultBox').innerHTML = `
                <h3 style="color:#333;margin-bottom:15px;">📊 识别结果</h3>
                <p style="color:#dc3545;">❌ 识别失败: ${result.error}</p>
            `;
        }
    } catch (err) {
        document.getElementById('resultBox').innerHTML = `
            <h3 style="color:#333;margin-bottom:15px;">📊 识别结果</h3>
            <p style="color:#dc3545;">❌ 错误: ${err.message}</p>
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
        
        # 预处理
        img_array = simple_preprocess(image)
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            predicted = torch.argmax(probs).item()
            confidence = float(probs[predicted].item() * 100)
        
        return jsonify({
            'success': True,
            'digit': int(predicted),
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'digit': -1,
            'confidence': 0
        })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🚀 服务器启动中...")
    print("="*60)
    print("\n📍 请在浏览器中打开: http://localhost:5000")
    print("💡 按 Ctrl+C 停止服务器\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"\n✗ 启动失败: {e}")
        print("\n请确保已安装依赖:")
        print("  python -m pip install flask torch torchvision numpy pillow")
