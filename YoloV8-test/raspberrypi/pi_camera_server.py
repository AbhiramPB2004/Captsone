import cv2
from flask import Flask, Response
import threading
import time
import logging

app = Flask(__name__)

class CameraStreamer:
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start camera capture thread"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Camera capture started")

    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.camera.release()
        self.logger.info("Camera capture stopped")

    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS

    def get_frame(self):
        """Get current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

# Initialize camera
camera_streamer = CameraStreamer()

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Raspberry Pi Camera Stream</title>
    </head>
    <body>
        <h1>Raspberry Pi Camera Feed</h1>
        <img src="/video_feed" width="640" height="480" />
        <p>Stream URL: http://YOUR_PI_IP:5000/video_feed</p>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = camera_streamer.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'running', 'camera_active': camera_streamer.is_running}

if __name__ == '__main__':
    try:
        camera_streamer.start()
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera_streamer.stop()
