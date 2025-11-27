import cv2
import numpy as np
import os
import pickle
import threading
from datetime import datetime
import time
from collections import defaultdict
import faiss
import uuid
import insightface
import onnxruntime as ort
import requests  # For API calls to Raspberry Pi

class FixedInsightFaceSystem:
    def __init__(self):
        print("🚀 Initializing FIXED InsightFace System...")
        
        self.setup_gpu()
        self.setup_insightface()
        
        self.dimension = 512
        self.index = faiss.IndexFlatIP(self.dimension)
        
        self.face_names = []
        self.face_embeddings = []
        self.face_metadata = []
        
        self.stats = {
            'faces_added': 0,
            'total_detections': 0,
            'total_recognitions': 0,
            'avg_detection_time_ms': 0,
            'avg_recognition_time_ms': 0
        }
        
        self.database_file = 'fixed_insightface_database.pkl'
        self.load_database()
        
        self.frame_count = 0
        self.unknown_face_detected = False
        self.current_unknown_face = None
        
        self.resize_factor = 0.8
        self.process_every_n_frames = 2
        self.confidence_threshold = 0.5
        
        self.face_tracking = {}
        self.similarity_threshold = 0.3
        
        self.detection_times = []
        self.recognition_times = []
        
        # Tracking neck position
        self.neck_pan = 90
        self.neck_tilt1 = 90
        self.neck_tilt2 = 90
        self.PAN_MIN = 40
        self.PAN_MAX = 140
        self.TILT_MIN = 40
        self.TILT_MAX = 140
        self.PAN_STEP = 1
        self.TILT_STEP = 1
        
        # API endpoint for Raspberry Pi
        self.api_url = 'http://10.140.84.94:5000/move_servo'  # Replace with your Pi IP
        
        print(f"✅ FIXED InsightFace system ready: {len(self.face_names)} faces loaded")

    def setup_gpu(self):
        try:
            providers = ort.get_available_providers()
            self.gpu_available = 'CUDAExecutionProvider' in providers
            
            if self.gpu_available:
                print(f"🔥 GPU ONNX Runtime available")
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("💻 Using CPU ONNX Runtime")
                self.providers = ['CPUExecutionProvider']
        except Exception as e:
            print(f"❌ GPU setup error: {e}")
            self.gpu_available = False
            self.providers = ['CPUExecutionProvider']

    def setup_insightface(self):
        try:
            self.app = insightface.app.FaceAnalysis(
                providers=self.providers,
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=0 if self.gpu_available else -1, det_size=(640, 640))
            print(f"✅ InsightFace loaded")
        except Exception as e:
            print(f"❌ InsightFace setup error: {e}")
            exit(1)

    def load_database(self):
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                self.face_names = data.get('face_names', [])
                self.face_embeddings = data.get('face_embeddings', [])
                self.face_metadata = data.get('face_metadata', [])
                self.stats = data.get('stats', self.stats)
                
                if self.face_embeddings:
                    embeddings_array = np.array(self.face_embeddings, dtype=np.float32)
                    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                    embeddings_array = embeddings_array / norms
                    self.index.add(embeddings_array)
                    print(f"✅ FAISS index rebuilt with {self.index.ntotal} faces")
                print(f"📖 Loaded {len(self.face_names)} faces")
        except Exception as e:
            print(f"❌ Load error: {e}")

    def save_database(self):
        try:
            data = {
                'face_names': self.face_names,
                'face_embeddings': self.face_embeddings,
                'face_metadata': self.face_metadata,
                'stats': self.stats,
                'version': '2.0_fixed',
                'saved_at': datetime.now().isoformat()
            }
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Database saved")
        except Exception as e:
            print(f"❌ Save error: {e}")

    def recognize_face(self, face_embedding):
        start_time = time.time()
        if self.index.ntotal == 0:
            return "Unknown", 0.0
        face_embedding_normalized = face_embedding / np.linalg.norm(face_embedding)
        similarities, indices = self.index.search(face_embedding_normalized.reshape(1, -1).astype('float32'), 1)
        similarity = float(similarities[0][0])
        if similarity > self.similarity_threshold:
            idx = int(indices[0][0])
            if idx < len(self.face_names):
                name = self.face_names[idx]
                recognition_time = (time.time() - start_time) * 1000
                self.recognition_times.append(recognition_time)
                if len(self.recognition_times) > 50:
                    self.recognition_times.pop(0)
                self.stats['total_recognitions'] += 1
                self.stats['avg_recognition_time_ms'] = np.mean(self.recognition_times)
                return name, similarity
        return "Unknown", similarity

    def detect_and_recognize_faces(self, frame):
        detection_start = time.time()
        small_frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)
        faces = self.app.get(small_frame)
        detection_time = (time.time() - detection_start) * 1000
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 50:
            self.detection_times.pop(0)
        self.stats['total_detections'] += len(faces)
        self.stats['avg_detection_time_ms'] = np.mean(self.detection_times)
        detections = []
        for face in faces:
            bbox = face.bbox / self.resize_factor
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(frame.shape[1], int(bbox[2]))
            y2 = min(frame.shape[0], int(bbox[3]))
            face_embedding = face.embedding
            name, similarity = self.recognize_face(face_embedding)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'name': name,
                'embedding': face_embedding,
                'similarity': similarity
            })
        return detections

    def update_neck_position(self, face_box, frame_width, frame_height):
        x1, y1, x2, y2 = face_box
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        offset_x = face_center_x - frame_center_x
        offset_y = face_center_y - frame_center_y
        if abs(offset_x) > 20:
            if offset_x > 0 and self.neck_pan < self.PAN_MAX:
                self.neck_pan += self.PAN_STEP
            elif offset_x < 0 and self.neck_pan > self.PAN_MIN:
                self.neck_pan -= self.PAN_STEP
        if abs(offset_y) > 20:
            if offset_y > 0 and self.neck_tilt1 < self.TILT_MAX:
                self.neck_tilt1 += self.TILT_STEP
                self.neck_tilt2 += self.TILT_STEP
            elif offset_y < 0 and self.neck_tilt1 > self.TILT_MIN:
                self.neck_tilt1 -= self.TILT_STEP
                self.neck_tilt2 -= self.TILT_STEP
        # Send API request
        data = {
            'angles': {
                '2': self.neck_pan,
                '3': self.neck_tilt1,
                '4': self.neck_tilt2
            }
        }
        try:
            requests.post(self.api_url, json=data, timeout=0.5)
        except Exception as e:
            print(f"❌ API request error: {e}")

    def draw_detections(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            name = detection['name']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def run_detection(self):
        url = "http://10.140.84.94:8000/?action=stream"  # Replace with your stream URL
        stream = cv2.VideoCapture(url)
        print("🚀 Running detection...")
        while True:
            ret, frame = stream.read()
            if not ret:
                continue
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames == 0:
                detections = self.detect_and_recognize_faces(frame)
                self.stats['total_detections'] += len(detections)
                self.stats['total_recognitions'] += sum(1 for d in detections if d['name'] != "Unknown")
                if detections:
                    detection = detections[0]
                    if detection['name'] != "Unknown":
                        self.update_neck_position(detection['bbox'], frame.shape[1], frame.shape[0])
                self.draw_detections(frame, detections)
            cv2.imshow('Face Tracking', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        stream.release()
        cv2.destroyAllWindows()

def main():
    system = FixedInsightFaceSystem()
    system.run_detection()

if __name__ == "__main__":
    main()
