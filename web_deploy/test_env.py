import sys
print("Python path:", sys.path)

try:
    import torch
    print("PyTorch OK:", torch.__version__)
except Exception as e:
    print("PyTorch ERROR:", e)

try:
    import flask
    print("Flask OK:", flask.__version__)
except Exception as e:
    print("Flask ERROR:", e)

try:
    from PIL import Image
    print("Pillow OK")
except Exception as e:
    print("Pillow ERROR:", e)

import os
model_path = r'd:\机器学习实验3\web_deploy\model.pth'
print(f"\nModel file exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"Model size: {os.path.getsize(model_path)} bytes")
    
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        
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
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        print("\n✓ Model loaded successfully!")
        
        # Test prediction with random data
        import numpy as np
        test_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Test prediction shape: {output.shape}")
        print(f"✓ All checks passed! Ready to run Flask app.")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("ERROR: Model file not found!")

print("\n=== Done ===")
