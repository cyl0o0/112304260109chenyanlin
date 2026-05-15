import sys
print("="*60)
print("  环境检查脚本")
print("="*60)

print(f"\nPython版本: {sys.version}")

print("\n正在检查依赖...")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch: ERROR - {e}")
    print("  请运行: pip install torch torchvision")

try:
    import flask
    print(f"✓ Flask: {flask.__version__}")
except Exception as e:
    print(f"✗ Flask: ERROR - {e}")
    print("  请运行: pip install flask")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except Exception as e:
    print(f"✗ NumPy: ERROR - {e}")
    print("  请运行: pip install numpy")

try:
    from PIL import Image
    print(f"✓ Pillow: OK")
except Exception as e:
    print(f"✗ Pillow: ERROR - {e}")
    print("  请运行: pip install pillow")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: ERROR - {e}")
    print("  请运行: pip install opencv-python")

print("\n正在检查模型文件...")
import os
model_path = r'd:\机器学习实验3\results\best_cnn_hypertuned.pth'
if os.path.exists(model_path):
    print(f"✓ 模型文件存在: {model_path}")
    print(f"  文件大小: {os.path.getsize(model_path)} 字节")
else:
    print(f"✗ 模型文件不存在: {model_path}")

print("\n" + "="*60)
print("  检查完成")
print("="*60)
print("\n如果所有依赖都已安装，现在可以运行:")
print("  python app.py")
print("\n然后在浏览器打开: http://localhost:5000")
