from flask import Flask, render_template_string, request, jsonify
from flask import send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
import io

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
print("Model loaded!")

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Digit Recognition</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
        canvas {
            border: 3px solid #667eea;
            border-radius: 10px;
            cursor: crosshair;
            background: white;
            touch-action: none;
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            justify-content: center;
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
        .btn-primary { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .btn-secondary { background: #e0e0e0; color: #333; }
        .result-box {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border-left: 5px solid #28a745;
        }
        .digit-result {
            font-size: 4em;
            font-weight: bold;
            color: #28a745;
            margin: 10px 0;
        }
        .confidence {
            font-size: 1.3em;
            color: #555;
        }
        .top3 {
            margin-top: 20px;
            text-align: left;
        }
        .top3-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            margin: 5px 0;
            background: white;
            border-radius: 6px;
            font-size: 14px;
        }
        .prob-bar-container {
            margin-top: 20px;
        }
        .prob-bar {
            display: flex;
            align-items: center;
            margin: 4px 0;
            font-size: 13px;
        }
        .prob-label { width: 30px; font-weight: bold; }
        .prob-outer { flex: 1; height: 22px; background: #e0e0e0; border-radius: 11px; overflow: hidden; margin: 0 10px; }
        .prob-inner { height: 100%; border-radius: 11px; transition: width 0.3s; }
        .prob-value { width: 50px; text-align: right; font-size: 13px; }
        .history-section {
            margin-top: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
        }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: center; border-bottom: 1px solid #ddd; }
        th { background: #667eea; color: white; }
        .info-section {
            margin-top: 30px;
            background: #f0f4ff;
            border-radius: 15px;
            padding: 20px;
            font-size: 14px;
            line-height: 1.8;
        }
        @media (max-width: 768px) {
            .main-content { flex-direction: column; }
            canvas { width: 100% !important; height: 280px !important; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ Handwritten Digit Recognition</h1>
        <p class="subtitle">CNN-based MNIST Classifier | Built with PyTorch & Flask</p>
        
        <div class="main-content">
            <div class="left-panel">
                <h3>📝 Draw a digit (0-9)</h3>
                <canvas id="canvas" width="280" height="280"></canvas>
                <div class="btn-group">
                    <button class="btn-secondary" onclick="clearCanvas()">🗑️ Clear</button>
                    <button class="btn-primary" onclick="predict()">🔍 Predict</button>
                </div>
                
                <div style="margin-top:20px;">
                    <h3>📤 Or Upload Image</h3>
                    <input type="file" id="fileInput" accept="image/*" onchange="handleUpload(event)">
                </div>
            </div>
            
            <div class="right-panel">
                <h3>📊 Prediction Results</h3>
                <div class="result-box" id="resultBox">
                    <p style="color:#888;">Waiting for input...<br><small>Draw or upload an image</small></p>
                </div>
                
                <div class="prob-bar-container" id="probBars"></div>
                
                <div class="history-section">
                    <h3>📜 Recognition History</h3>
                    <table id="historyTable">
                        <thead><tr><th>#</th><th>Digit</th><th>Confidence</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="info-section">
            <h3>ℹ️ About This System</h3>
            <ul>
                <li><b>Model:</b> Convolutional Neural Network (CNN)</li>
                <li><b>Architecture:</b> 2 Conv layers + MaxPool + FC + Dropout</li>
                <li><b>Training Data:</b> Kaggle MNIST (42,000 images)</li>
                <li><b>Framework:</b> PyTorch | Deployment: Flask</li>
            </ul>
            <h3>🚀 Features Implemented</h3>
            <ul>
                <li>✅ Handwriting Canvas Input (Sketchpad)</li>
                <li>✅ Image Upload Support</li>
                <li>✅ Real-time Digit Recognition</li>
                <li>✅ Top-3 Predictions with Confidence</li>
                <li>✅ Probability Distribution Visualization</li>
                <li>✅ Recognition History Tracking</li>
            </ul>
        </div>
    </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#000';
ctx.lineWidth = 8;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseout', stopDraw);
canvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(e.touches[0]); });
canvas.addEventListener('touchmove', e => { e.preventDefault(); draw(e.touches[0]); });
canvas.addEventListener('touchend', stopDraw);

function startDraw(e) { isDrawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); }
function draw(e) {
    if (!isDrawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
}
function stopDraw() { isDrawing = false; }

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
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

async function predict() {
    const dataUrl = canvas.toDataURL('image/png');
    
    document.getElementById('resultBox').innerHTML = '<p>⏳ Predicting...</p>';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: dataUrl})
        });
        const result = await response.json();
        
        let html = `<div class="digit-result">${result.digit}</div>`;
        html += `<div class="confidence">Confidence: ${result.confidence}%</div>`;
        html += '<div class="top3"><h4>🏆 Top-3 Predictions:</h4>';
        result.top3.forEach((item, i) => {
            const medals = ['🥇','🥈','🥉'];
            html += `<div class="top3-item"><span>${medals[i]} ${item.digit}</span><span>${item.prob}%</span></div>`;
        });
        html += '</div>';
        document.getElementById('resultBox').innerHTML = html;
        
        // Probability bars
        let probHtml = '<h4>📊 Probability Distribution:</h4>';
        result.probs.forEach(p => {
            probHtml += `<div class="prob-bar"><span class="prob-label">${p.digit}</span><div class="prob-outer"><div class="prob-inner" style="width:${p.prob}%;background:${p.prob > 60 ? '#28a745' : p.prob > 30 ? '#ffc107' : '#dc3545'}"></div></div><span class="prob-value">${p.prob.toFixed(1)}%</span></div>`;
        });
        document.getElementById('probBars').innerHTML = probHtml;
        
    } catch (err) {
        document.getElementById('resultBox').innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
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
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item() * 100
    
    top3_probs, top3_indices = torch.topk(probs, 3)
    top3 = []
    for i in range(3):
        top3.append({
            'digit': int(top3_indices[i].item()),
            'prob': float(top3_probs[i].item() * 100)
        })
    
    probs_list = [{'digit': i, 'prob': float(probs[i].item() * 100)} for i in range(10)]
    
    return jsonify({
        'digit': int(predicted),
        'confidence': round(confidence, 2),
        'top3': top3,
        'probs': probs_list
    })

if __name__ == '__main__':
    print("\nStarting Flask server...")
    print("Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
