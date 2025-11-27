import cv2
import numpy as np
import requests
import threading
import time
import json
import sqlite3
import pickle
import os
import urllib.request
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import face_recognition
from flask import Flask, render_template, Response, request, jsonify
import logging

app = Flask(__name__)

class RemoteFaceProcessor:
    def __init__(self, pi_stream_url):
        # Initialize logging FIRST - before any other operations
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic attributes
        self.pi_stream_url = "http://172.18.129.74:5000/video_feed"  # Update with your Pi IP
        self.stream = None
        self.current_frame = None
        self.processed_frame = None
        self.is_streaming = False
        self.frame_lock = threading.Lock()
        
        # Performance optimization variables
        self.frame_skip = 2  # Process every 2nd frame for face recognition
        self.frame_counter = 0
        self.resize_factor = 0.5  # Reduce frame size for processing
        self.target_fps = 15  # Target FPS for display
        
        # Initialize face recognition attributes
        self.known_faces = []
        self.known_names = []
        self.face_database_path = 'face_database.pkl'
        
        # Initialize processing modes
        self.training_mode = False
        self.inference_mode = False
        self.detect_objects = True
        
        # Initialize learning buffers
        self.face_buffer = defaultdict(list)
        self.face_detections = defaultdict(int)
        self.min_faces_for_learning = 5
        self.retrain_threshold = 10
        
        # Performance tracking
        self.last_process_time = time.time()
        self.processing_times = []
        
        # Initialize YOLOv8 models
        self._initialize_models()
        
        # Initialize database
        self.init_database()
        
        # Load face database
        self.load_face_database()
        
        self.logger.info("RemoteFaceProcessor initialized successfully")

    def _initialize_models(self):
        """Initialize YOLOv8 models with optimized settings"""
        try:
            # Use smaller, faster models for real-time processing
            face_model_path = 'yolov8n-face-lindevs.pt'  # Keep nano for speed
            if not os.path.exists(face_model_path):
                self.logger.info("Downloading face detection model...")
                try:
                    urllib.request.urlretrieve(
                        'https://github.com/lindevs/yolov8-face/releases/download/v1.0.1/yolov8n-face-lindevs.pt',
                        face_model_path
                    )
                    self.logger.info(f"Downloaded {face_model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to download face model: {e}")
                    face_model_path = 'yolov8n.pt'  # Use nano, not large
            
            # Use nano models for speed, not large models
            self.face_model = YOLO(face_model_path)
            self.object_model = YOLO('yolov8n.pt')  # Changed from yolov8l.pt to yolov8n.pt
            
            # Optimize model settings for speed
            if hasattr(self.face_model.model, 'fuse') and callable(getattr(self.face_model.model, 'fuse')):
                self.face_model.model.fuse()
            if hasattr(self.object_model.model, 'fuse') and callable(getattr(self.object_model.model, 'fuse')):
                self.object_model.model.fuse()
                
            self.logger.info("YOLOv8 models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 models: {e}")
            self.face_model = YOLO('yolov8n.pt')
            self.object_model = YOLO('yolov8n.pt')
            self.logger.warning("Using general YOLOv8 nano models")

    def safe_float(self, value):
        """Safely convert numpy array or scalar to Python float"""
        if hasattr(value, 'item'):
            return value.item()
        return float(value)

    def safe_int(self, value):
        """Safely convert numpy array or scalar to Python int"""
        if hasattr(value, 'item'):
            return int(value.item())
        return int(value)

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect('laptop_faces.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_type TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL,
                bbox TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def load_face_database(self):
        """Load face encodings from file"""
        if os.path.exists(self.face_database_path):
            with open(self.face_database_path, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data.get('encodings', [])
                self.known_names = data.get('names', [])
        
        self.logger.info(f"Loaded {len(self.known_faces)} face encodings")

    def save_face_database(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_faces,
            'names': self.known_names
        }
        with open(self.face_database_path, 'wb') as f:
            pickle.dump(data, f)

    def start_stream_processing(self):
        """Start processing stream from Raspberry Pi"""
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._process_stream)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            self.logger.info(f"Started processing stream from {self.pi_stream_url}")

    def stop_stream_processing(self):
        """Stop stream processing"""
        self.is_streaming = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join()
        self.logger.info("Stopped stream processing")

    def _process_stream(self):
        """Process MJPEG stream from Raspberry Pi with optimizations"""
        while self.is_streaming:
            try:
                stream = requests.get(self.pi_stream_url, stream=True, timeout=5)
                if stream.status_code == 200:
                    bytes_data = b''
                    for chunk in stream.iter_content(chunk_size=2048):  # Larger chunk size
                        if not self.is_streaming:
                            break
                        
                        bytes_data += chunk
                        
                        # Find JPEG boundaries
                        start = bytes_data.find(b'\xff\xd8')
                        end = bytes_data.find(b'\xff\xd9')
                        
                        if start != -1 and end != -1:
                            jpg_data = bytes_data[start:end+2]
                            bytes_data = bytes_data[end+2:]
                            
                            # Decode frame
                            frame = cv2.imdecode(
                                np.frombuffer(jpg_data, dtype=np.uint8), 
                                cv2.IMREAD_COLOR
                            )
                            
                            if frame is not None:
                                # Frame rate control
                                current_time = time.time()
                                if current_time - self.last_process_time < 1.0 / self.target_fps:
                                    continue
                                
                                with self.frame_lock:
                                    self.current_frame = frame.copy()
                                
                                # Process frame with optimizations
                                processed = self._process_frame_optimized(frame)
                                
                                with self.frame_lock:
                                    self.processed_frame = processed.copy()
                                
                                self.last_process_time = current_time
                
            except Exception as e:
                self.logger.error(f"Stream processing error: {e}")
                time.sleep(1)  # Reduced wait time

    def _process_frame_optimized(self, frame):
        """Optimized frame processing"""
        processed_frame = frame.copy()
        self.frame_counter += 1
        
        # Only do heavy processing on certain frames
        do_face_processing = (self.training_mode or self.inference_mode) and (self.frame_counter % self.frame_skip == 0)
        do_object_processing = self.detect_objects and (self.frame_counter % (self.frame_skip * 2) == 0)  # Even less frequent
        
        if do_face_processing:
            processed_frame = self._process_faces_optimized(processed_frame)
        
        if do_object_processing:
            processed_frame = self._process_objects_optimized(processed_frame)
        
        return processed_frame

    def _process_faces_optimized(self, frame):
        """Optimized face processing with CORRECT YOLO box handling"""
        try:
            start_time = time.time()
            
            # Resize frame for processing (speed optimization)
            small_frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)
            
            # Detect faces on smaller frame
            face_results = self.face_model(small_frame, verbose=False, device='cuda' if self.has_gpu() else 'cpu')
            
            for result in face_results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            # CORRECT: Extract coordinates from YOLOv8 box [166]
                            # box.xyxy gives us the coordinates [x1, y1, x2, y2]
                            coords = box.xyxy.cpu().numpy()  # This gives us [x1, y1, x2, y2]
                            
                            # Scale coordinates back to original frame size
                            coords = coords / self.resize_factor
                            x1, y1, x2, y2 = coords.astype(int)
                            
                            confidence = self.safe_float(box.conf[0].cpu().numpy())
                            
                            if confidence > 0.6:
                                # Get face encoding (only if really needed)
                                if self.training_mode or (self.inference_mode and len(self.known_faces) > 0):
                                    encoding = self._get_face_encoding_fast(frame, [x1, y1, x2, y2])
                                    
                                    if encoding is not None:
                                        name, rec_confidence = self._recognize_face(encoding)
                                        rec_confidence = self.safe_float(rec_confidence)
                                        
                                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        label = f"{name} ({rec_confidence:.2f})"
                                        cv2.putText(frame, label, (x1, y1-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        
                                        # Learning logic
                                        if self.inference_mode and name != "Unknown":
                                            self.face_buffer[name].append(encoding)
                                            self.face_detections[name] += 1
                                            
                                            if self.face_detections[name] % self.retrain_threshold == 0:
                                                threading.Thread(target=self._retrain_faces).start()
                                        
                                        # Reduced logging frequency
                                        if self.frame_counter % 30 == 0:
                                            self._log_detection('face', name, rec_confidence, [x1, y1, x2, y2])
                                else:
                                    # Just draw detection box without recognition
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                    cv2.putText(frame, f"Face ({confidence:.2f})", (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        except Exception as e:
                            self.logger.error(f"Error processing individual face box: {e}")
                            continue
            
            # Track processing time
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error processing faces: {e}")
        
        return frame

    def _process_objects_optimized(self, frame):
        """Optimized object processing with CORRECT YOLO box handling"""
        try:
            # Even smaller frame for object detection
            tiny_frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
            
            results = self.object_model(tiny_frame, verbose=False, device='cuda' if self.has_gpu() else 'cpu')
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Only process first few detections to avoid lag
                    for i, box in enumerate(boxes[:5]):  # Limit to 5 objects
                        try:
                            # CORRECT: Extract coordinates from YOLOv8 box [166]
                            coords = box.xyxy.cpu().numpy()  # This gives us [x1, y1, x2, y2]
                            
                            # Scale coordinates back to original frame size
                            coords = coords / 0.3  # Scale back up from tiny frame
                            x1, y1, x2, y2 = coords.astype(int)
                            
                            confidence = self.safe_float(box.conf[0].cpu().numpy())
                            class_id = self.safe_int(box.cls.cpu().numpy())
                            
                            if confidence > 0.6:
                                class_name = self.object_model.names[class_id]
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                label = f"{class_name} ({confidence:.2f})"
                                cv2.putText(frame, label, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
                                # Reduced logging
                                if self.frame_counter % 60 == 0:
                                    self._log_detection('object', class_name, confidence, [x1, y1, x2, y2])
                        except Exception as e:
                            self.logger.error(f"Error processing individual object box: {e}")
                            continue
                            
        except Exception as e:
            self.logger.error(f"Error processing objects: {e}")
        
        return frame

    def _get_face_encoding_fast(self, frame, bbox):
        """Faster face encoding with optimizations"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Add padding but ensure bounds
            padding = 10
            y1 = max(0, y1 - padding)
            y2 = min(frame.shape[0], y2 + padding)
            x1 = max(0, x1 - padding)
            x2 = min(frame.shape[13], x2 + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[13] < 20:
                return None
            
            # Resize face crop to standard size for faster processing
            face_crop = cv2.resize(face_crop, (150, 150))
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Use faster model for face encoding
            encodings = face_recognition.face_encodings(face_rgb, model='small')
            
            return encodings[0] if encodings else None
        except Exception as e:
            return None

    def has_gpu(self):
        """Check if CUDA GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _recognize_face(self, encoding):
        """Recognize face from encoding"""
        if not self.known_faces:
            return "Unknown", 0.0
        
        try:
            distances = face_recognition.face_distance(self.known_faces, encoding)
            min_distance = min(distances)
            
            if min_distance < 0.5:  # Slightly stricter threshold
                best_match_index = np.argmin(distances)
                name = self.known_names[best_match_index]
                confidence = 1 - min_distance
                return name, confidence
        except Exception as e:
            self.logger.error(f"Error recognizing face: {e}")
        
        return "Unknown", 0.0

    def add_new_face(self, name, encoding):
        """Add new face to database"""
        try:
            self.known_faces.append(encoding)
            self.known_names.append(name)
            self.save_face_database()
            
            conn = sqlite3.connect('laptop_faces.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (name, encoding, confidence) VALUES (?, ?, ?)",
                (name, encoding.tobytes(), 1.0)
            )
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added new face: {name}")
        except Exception as e:
            self.logger.error(f"Error adding new face: {e}")

    def _retrain_faces(self):
        """Retrain face recognition incrementally"""
        try:
            for name, face_list in self.face_buffer.items():
                if len(face_list) >= self.min_faces_for_learning:
                    avg_encoding = np.mean(face_list, axis=0)
                    
                    if name in self.known_names:
                        idx = self.known_names.index(name)
                        current_encoding = self.known_faces[idx]
                        updated_encoding = 0.7 * current_encoding + 0.3 * avg_encoding
                        self.known_faces[idx] = updated_encoding
                    else:
                        self.add_new_face(name, avg_encoding)
                    
                    self.face_buffer[name] = []
            
            self.save_face_database()
            self.logger.info("Face model retrained")
        except Exception as e:
            self.logger.error(f"Error retraining faces: {e}")

    def _log_detection(self, detection_type, label, confidence, bbox):
        """Log detection to database with reduced frequency"""
        try:
            conn = sqlite3.connect('laptop_faces.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO detections (detection_type, label, confidence, bbox) VALUES (?, ?, ?, ?)",
                (detection_type, label, confidence, json.dumps(bbox))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            pass  # Silently ignore logging errors to avoid spam

    def get_processed_frame(self):
        """Get current processed frame"""
        with self.frame_lock:
            if self.processed_frame is not None:
                return self.processed_frame.copy()
        return None

    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {"avg_process_time": 0, "fps": 0}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_process_time": avg_time,
            "fps": fps,
            "frames_processed": self.frame_counter
        }

# Initialize processor
PI_STREAM_URL = "http://172.18.129.74:5000/video_feed"  # Update with your Pi's IP
processor = RemoteFaceProcessor(PI_STREAM_URL)

# Flask routes
@app.route('/')
def index():
    return render_template('laptop_interface.html')

@app.route('/processed_feed')
def processed_feed():
    """Serve processed video feed with optimized streaming"""
    def generate():
        while True:
            frame = processor.get_processed_frame()
            if frame is not None:
                # Higher quality encoding for better visual output
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)  # ~20 FPS output
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set processing mode"""
    mode = request.json.get('mode')
    
    if mode == 'training':
        processor.training_mode = True
        processor.inference_mode = False
    elif mode == 'inference':
        processor.training_mode = False
        processor.inference_mode = True
    else:
        processor.training_mode = False
        processor.inference_mode = False
    
    return jsonify({'status': 'success', 'mode': mode})

@app.route('/add_face', methods=['POST'])
def add_face():
    """Add face during training mode with CORRECT YOLO box handling"""
    if not processor.training_mode:
        return jsonify({'status': 'error', 'message': 'Not in training mode'})
    
    name = request.json.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': 'Name required'})
    
    try:
        frame = processor.get_processed_frame()
        if frame is not None:
            face_results = processor.face_model(frame, verbose=False)
            for result in face_results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]  # Use first detected face
                    
                    # CORRECT: Extract coordinates from YOLOv8 box [12]
                    coords = box.xyxy.cpu().numpy()  # This gives us [x1, y1, x2, y2]
                    x1, y1, x2, y2 = coords.astype(int)
                    
                    encoding = processor._get_face_encoding_fast(frame, [x1, y1, x2, y2])
                    
                    if encoding is not None:
                        processor.add_new_face(name, encoding)
                        return jsonify({'status': 'success', 'message': f'Added face for {name}'})
                    
    except Exception as e:
        processor.logger.error(f"Error adding face: {e}")
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})
    
    return jsonify({'status': 'error', 'message': 'No face detected'})

@app.route('/get_stats')
def get_stats():
    """Get processing statistics including performance metrics"""
    perf_stats = processor.get_performance_stats()
    return jsonify({
        'known_faces': len(processor.known_faces),
        'detection_counts': dict(processor.face_detections),
        'training_mode': processor.training_mode,
        'inference_mode': processor.inference_mode,
        'streaming': processor.is_streaming,
        'performance': perf_stats
    })

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Start stream processing"""
    processor.start_stream_processing()
    return jsonify({'status': 'success', 'message': 'Processing started'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stop stream processing"""
    processor.stop_stream_processing()
    return jsonify({'status': 'success', 'message': 'Processing stopped'})

if __name__ == '__main__':
    try:
        processor.start_stream_processing()
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)  # Debug=False for production
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        processor.stop_stream_processing()
