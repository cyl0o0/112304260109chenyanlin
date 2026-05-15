import torch
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import sys

print('='*60)
print('Generating complete submission file')
print('='*60)
sys.stdout.flush()

# Load model
model = torch.jit.load('exp.torchscript')
model.eval()
print('[1/3] Model loaded successfully')
sys.stdout.flush()

# Get all images
test_dir = Path('第4次实验数据及提交格式 (1)/test/images')
images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png']])
print(f'[2/3] Found {len(images)} test images')
sys.stdout.flush()

# Process all images
print('[3/3] Running inference...')
predictions = []

for idx, img_path in enumerate(images, 1):
    try:
        # Preprocess
        img = Image.open(img_path).convert('RGB').resize((640, 640))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # Inference
        output = model(img_tensor).cpu().numpy()

        # Extract predictions
        for detection in output[0]:
            x1, y1, x2, y2, conf, class_id = detection

            if conf > 0.001:
                predictions.append({
                    'image_id': img_path.name,
                    'class_id': int(class_id),
                    'x_center': round(float((x1+x2)/2)/640, 6),
                    'y_center': round(float((y1+y2)/2)/640, 6),
                    'width': round(float(x2-x1)/640, 6),
                    'height': round(float(y2-y1)/640, 6),
                    'confidence': round(float(conf), 6)
                })

        # Progress update
        if idx % 50 == 0:
            print(f'  Processed {idx}/{len(images)} images, {len(predictions)} detections')
            sys.stdout.flush()

    except Exception as e:
        print(f'  Error on {img_path.name}: {e}')
        continue

# Save results
print('\nSaving results...')
sys.stdout.flush()
Path('results').mkdir(exist_ok=True)

with open('results/submission_final.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['image_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'])
    writer.writeheader()
    writer.writerows(predictions)

print('\n' + '='*60)
print('COMPLETE!')
print('='*60)
print(f'Processed images: {len(images)}')
print(f'Total detections: {len(predictions)}')
print(f'Output file: results/submission_final.csv')
print('='*60)
sys.stdout.flush()
