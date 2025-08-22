
import torch
import cv2
from ultralytics import YOLO
import numpy as np

print("=== GPU Test Results ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test YOLOv8 with GPU
print("\n=== YOLOv8 GPU Test ===")
try:
    model = YOLO('yolov8n.pt')
    model.to('cuda')  # Move model to GPU
    print("YOLOv8 model loaded successfully on GPU")
    
    # Create a dummy image for testing
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(dummy_img, device='cuda')
    print("YOLOv8 inference test passed on GPU")
    
except Exception as e:
    print(f"YOLOv8 GPU test failed: {e}")

print("\n=== OpenCV Test ===")
print(f"OpenCV version: {cv2.__version__}")

print("\n=== Setup Complete! ===")
