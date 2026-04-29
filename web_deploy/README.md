# ✍️ Handwritten Digit Recognition Web System

## 📋 Project Overview

This is a **CNN-based handwritten digit recognition system** deployed as a web application using **Gradio**. Users can upload images or draw digits directly on the canvas to get real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

### Experiment 2: Model Packaging & Web Deployment ✅
- ✅ **Image Upload**: Upload handwritten digit images for prediction
- ✅ **Model Loading**: Pre-trained CNN model loaded automatically
- ✅ **Result Display**: Shows predicted digit class with confidence

### Experiment 3: Interactive Handwriting Recognition (Bonus) ✅
- ✅ **Handwriting Canvas**: Draw digits directly using Gradio Sketchpad
- ✅ **Real-time Recognition**: Instant prediction after submission
- ✅ **Continuous Use**: Clear canvas and draw multiple times

### Bonus Features Implemented 🌟
- ✅ **Top-3 Predictions**: Shows top 3 most likely digits with confidence scores
- ✅ **Probability Distribution Bar Chart**: Visualizes prediction probabilities
- ✅ **Recognition History**: Tracks all previous predictions
- ✅ **Quick Examples**: Load example digits for quick testing

---

## 🏗️ Project Structure

```
web_deploy/
├── app.py              # Gradio Web Application (Main entry point)
├── model.pth           # Trained CNN model weights
├── requirements.txt    # Python dependencies
└── README.md           # This documentation file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch torchvision numpy pandas Pillow gradio
```

### 2. Run the Web App

```bash
python app.py
```

The application will start at:
- Local: `http://localhost:7860`
- Public Link: Gradio will generate a shareable link

---

## 🤖 Model Architecture

The CNN model used in this system:

```
Input (1×28×28)
├─ Conv2d(1→32, kernel=3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
├─ Conv2d(32→64, kernel=3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
├─ Flatten (64×7×7 = 3136)
├─ Linear(3136→128) + ReLU + Dropout(0.25)
└─ Linear(128→10) → Output (10 classes: 0-9)
```

**Training Details**:
- Dataset: Kaggle MNIST Digit Recognizer (42,000 training images)
- Optimizer: Adam (lr=0.001)
- Validation Accuracy: ~98%
- Expected Kaggle Score: >0.98

---

## 📸 Usage Guide

### Method 1: Draw on Canvas
1. Click on "🖊️ Draw on Canvas" tab
2. Draw a digit (0-9) using your mouse or touch screen
3. Click "🔍 Predict" button
4. View results including Top-3 predictions and probability distribution

### Method 2: Upload Image
1. Click on "📤 Upload Image" tab
2. Upload an image of a handwritten digit
3. Click "🔍 Predict Uploaded Image" button
4. View prediction results

### Method 3: Quick Examples
1. Click on "🎯 Quick Examples" tab
2. Select a digit from 0-9
3. Click "Load Example & Predict"
4. See how the model performs on sample inputs

---

## 🎨 Interface Preview

The web interface includes:

| Section | Description |
|---------|-------------|
| **Input Area** | Three tabs: Canvas, Upload, Examples |
| **Prediction Results** | Main result, Top-3 predictions, probability chart |
| **Recognition History** | Table of all previous predictions |
| **About Section** | Model info and features list |

---

## ☁️ Deployment Options

### Option 1: HuggingFace Spaces (Recommended)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/new-space?kind=gradio)
2. Choose **Gradio** as SDK
3. Upload these files:
   - `app.py`
   - `model.pth`
   - `requirements.txt`
4. Your app will be live at: `https://huggingface.co/<username>/<space-name>`

### Option 2: Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd web_deploy

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### Option 3: Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
```

```bash
docker build -t digit-recognition .
docker run -p 7860:7860 digit-recognition
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Type | CNN (Convolutional Neural Network) |
| Parameters | ~500K |
| Inference Time | <10ms per image |
| Validation Accuracy | ~98%+ |
| Supported Input | 28×28 grayscale images |

---

## 🔧 Technical Stack

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep Learning Framework |
| **Gradio** | Web UI & API |
| **Pillow** | Image Processing |
| **NumPy** | Numerical Computing |

---

## 📝 Submission Information

### Experiment 2 Submission

| Item | Content |
|------|---------|
| GitHub Repository | [Your Repo URL] |
| Online Access Link | [Your HuggingFace/Deploy URL] |

### Experiment 3 Submission (Bonus)

| Item | Content |
|------|---------|
| Online Access Link | Same as above |
| Bonus Features Implemented | ✅ Top-3 Predictions, ✅ Probability Distribution, ✅ History Tracking |

---

## 📄 License

This project is open source under the MIT License.

---

## 👨‍💻 Author

Machine Learning Course Project  
Course: 模型训练与超参数调优

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)
- Framework: [PyTorch](https://pytorch.org/)
- Deployment: [Gradio](https://gradio.app/)

---

**⭐ If you find this project useful, please give it a star!**
