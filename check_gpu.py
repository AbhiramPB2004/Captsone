import requests
import os

YOLO_FACE_MODEL = "yolov8n-face.pt"
YOLO_FACE_URL = "https://github.com/derronqi/yolov8-face/releases/download/v0.1.0/yolov8n-face.pt"

if not os.path.exists(YOLO_FACE_MODEL):
    print(f"{YOLO_FACE_MODEL} not found, downloading...")
    r = requests.get(YOLO_FACE_URL, stream=True)
    with open(YOLO_FACE_MODEL, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")
