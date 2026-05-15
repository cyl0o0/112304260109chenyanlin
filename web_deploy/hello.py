print("="*60)
print("  👋 Hello! Python is working!")
print("="*60)
print("\n如果看到这条消息，说明Python运行正常！\n")

try:
    import sys
    print(f"✓ Python版本: {sys.version}")
except:
    print("✗ 无法获取Python版本")

try:
    print("\n正在测试导入...")
    import flask
    print(f"✓ Flask: OK")
except Exception as e:
    print(f"✗ Flask: {e}")

try:
    import torch
    print(f"✓ PyTorch: OK")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import numpy
    print(f"✓ NumPy: OK")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    from PIL import Image
    print(f"✓ Pillow: OK")
except Exception as e:
    print(f"✗ Pillow: {e}")

print("\n" + "="*60)
print("  测试完成")
print("="*60)
print("\n按回车键退出...")
input()
