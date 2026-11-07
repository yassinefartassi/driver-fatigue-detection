"""
Real-Time Flask Dashboard Module
=================================
Web-based dashboard displaying live camera feed with overlays,
fatigue metrics, EAR/MAR values, blink/yawn counts, and alert status.
Accessible via browser at http://<pi-ip>:5000
"""

import cv2
import numpy as np
import time
import json
import threading
from flask import Flask, Response, render_template_string, jsonify
from typing import Dict, Optional, Tuple, Any
import logging

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import config

logger = logging.getLogger(__name__)

# Suppress Flask's default logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects for JSON."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    # Handle dlib rectangle
    if hasattr(obj, 'left') and hasattr(obj, 'right'):
        return {'left': obj.left(), 'top': obj.top(), 
                'right': obj.right(), 'bottom': obj.bottom()}
    return obj


class Dashboard:
    """
    Flask-based real-time web dashboard for fatigue detection system.
    
    Provides:
    - Live MJPEG video stream with overlays
    - Real-time metrics via JSON API
    - Server-Sent Events for live updates
    - Responsive web interface
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        """
        Initialize Flask dashboard.
        
        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port number for web server
        """
        self.host = host
        self.port = port
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        
        # Current data (thread-safe)
        self.data_lock = threading.Lock()
        self.current_frame = None
        self.current_face_data = {}
        self.current_heart_data = {}
        self.current_fatigue_report = {}
        self.current_landmarks = None
        
        # FPS tracking
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.frame_times = []
        
        # Server thread
        self.server_thread = None
        self.running = False
        
        # Register routes
        self._register_routes()
        
        logger.info(f"Flask dashboard initialized on {host}:{port}")
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/metrics')
        def get_metrics():
            with self.data_lock:
                return jsonify(sanitize_for_json({
                    'face': self.current_face_data,
                    'heart': self.current_heart_data,
                    'fatigue': self.current_fatigue_report,
                    'fps': round(self.fps, 1),
                    'timestamp': time.time()
                }))
        
        @self.app.route('/api/stream')
        def stream():
            def event_stream():
                while self.running:
                    with self.data_lock:
                        data = sanitize_for_json({
                            'face': self.current_face_data,
                            'heart': self.current_heart_data,
                            'fatigue': self.current_fatigue_report,
                            'fps': round(self.fps, 1)
                        })
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(0.1)  # 10 updates per second
            
            return Response(event_stream(), mimetype='text/event-stream')
    
    def _generate_frames(self):
        """Generate MJPEG frames for video streaming."""
        while self.running:
            with self.data_lock:
                frame = self.current_frame
                face_data = self.current_face_data.copy()
                landmarks = self.current_landmarks
            
            if frame is None:
                # Generate placeholder frame
                frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
                frame[:] = (40, 40, 40)
                cv2.putText(frame, "Waiting for camera...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            else:
                frame = self._draw_overlays(frame.copy(), face_data, landmarks)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS max
    
    def _draw_overlays(self, frame: np.ndarray, face_data: Dict, 
                       landmarks: Optional[np.ndarray]) -> np.ndarray:
        """Draw overlays on the video frame."""
        # Colors
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        WHITE = (255, 255, 255)
        
        ear = face_data.get('ear', 0.0)
        mar = face_data.get('mar', 0.0)
        microsleep = face_data.get('microsleep', False)
        
        # Draw landmarks if available
        if landmarks is not None and face_data.get('face_detected', False):
            # Determine colors based on state
            eye_color = RED if microsleep else (YELLOW if ear < config.EAR_THRESHOLD else GREEN)
            mouth_color = (0, 165, 255) if mar > config.MAR_THRESHOLD else BLUE
            
            # Draw eye landmarks and contours
            for idx in config.LEFT_EYE_INDICES + config.RIGHT_EYE_INDICES:
                x, y = landmarks[idx].astype(int)
                cv2.circle(frame, (x, y), 2, eye_color, -1)
            
            left_eye = landmarks[config.LEFT_EYE_INDICES].astype(np.int32)
            right_eye = landmarks[config.RIGHT_EYE_INDICES].astype(np.int32)
            cv2.polylines(frame, [left_eye], True, eye_color, 1)
            cv2.polylines(frame, [right_eye], True, eye_color, 1)
            
            # Draw mouth landmarks
            for idx in config.MOUTH_INDICES:
                x, y = landmarks[idx].astype(int)
                cv2.circle(frame, (x, y), 2, mouth_color, -1)
            
            inner_mouth = landmarks[config.MOUTH_INDICES].astype(np.int32)
            cv2.polylines(frame, [inner_mouth], True, mouth_color, 1)
        
        # Draw face rectangle
        face_rect = face_data.get('face_rect')
        if face_rect is not None:
            try:
                color = RED if microsleep else GREEN
                cv2.rectangle(frame, 
                             (face_rect['left'], face_rect['top']),
                             (face_rect['right'], face_rect['bottom']),
                             color, 2)
            except (TypeError, KeyError):
                pass
        
        # EAR/MAR overlay
        ear_color = GREEN if ear >= config.EAR_THRESHOLD else RED
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        
        mar_color = (0, 165, 255) if mar > config.MAR_THRESHOLD else GREEN
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
        
        # Status messages
        if not face_data.get('face_detected', False):
            cv2.putText(frame, "NO FACE DETECTED", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        
        if microsleep:
            # Flashing effect
            if int(time.time() * 4) % 2:
                cv2.putText(frame, "!!! MICRO-SLEEP !!!", (120, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_time if avg_time > 0 else 0
    
    def update(self, 
               video_frame: np.ndarray,
               face_data: Dict,
               heart_data: Dict,
               fatigue_report: Dict,
               landmarks: Optional[np.ndarray] = None):
        """
        Update dashboard with new data.
        
        Args:
            video_frame: Current camera frame
            face_data: Face detection results
            heart_data: Heart rate sensor data
            fatigue_report: Fatigue fusion report
            landmarks: Facial landmarks for overlay
        """
        self._update_fps()
        
        # Serialize face_rect for JSON
        serialized_face_data = face_data.copy()
        if 'face_rect' in serialized_face_data and serialized_face_data['face_rect'] is not None:
            rect = serialized_face_data['face_rect']
            try:
                serialized_face_data['face_rect'] = {
                    'left': rect.left(),
                    'top': rect.top(),
                    'right': rect.right(),
                    'bottom': rect.bottom()
                }
            except AttributeError:
                pass
        if 'landmarks' in serialized_face_data:
            del serialized_face_data['landmarks']
        
        with self.data_lock:
            self.current_frame = video_frame
            self.current_face_data = serialized_face_data
            self.current_heart_data = heart_data
            self.current_fatigue_report = fatigue_report
            self.current_landmarks = landmarks
    
    def render(self, video_frame: np.ndarray, face_data: Dict, heart_data: Dict,
               fatigue_report: Dict, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compatibility method - updates data and returns frame.
        For Flask mode, the actual rendering happens in the browser.
        """
        self.update(video_frame, face_data, heart_data, fatigue_report, landmarks)
        return video_frame
    
    def start(self):
        """Start the Flask server in a background thread."""
        if self.running:
            return
        
        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
        print(f"\n{'='*60}")
        print(f"  Dashboard available at: http://localhost:{self.port}")
        print(f"  (or http://<your-pi-ip>:{self.port} from other devices)")
        print(f"{'='*60}\n")
    
    def _run_server(self):
        """Run Flask server."""
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    
    def show(self, canvas: np.ndarray, window_name: str = None):
        """Compatibility method for OpenCV interface."""
        pass
    
    def wait_key(self, delay: int = 1) -> int:
        """Compatibility method - returns -1 (no key)."""
        return -1
    
    def cleanup(self):
        """Clean up dashboard resources."""
        self.running = False
        logger.info("Dashboard cleanup complete")


# HTML template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Fatigue Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        
        .header {
            background: #16213e;
            padding: 15px 20px;
            text-align: center;
            border-bottom: 2px solid #0f3460;
        }
        
        .header h1 {
            color: #e94560;
            font-size: 1.5em;
        }
        
        .container {
            display: flex;
            flex-wrap: wrap;
            padding: 20px;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .video-section {
            flex: 1;
            min-width: 300px;
        }
        
        .video-feed {
            width: 100%;
            max-width: 640px;
            border: 3px solid #0f3460;
            border-radius: 10px;
        }
        
        .metrics-section {
            flex: 0 0 350px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .panel {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #0f3460;
        }
        
        .panel-title {
            color: #e94560;
            font-size: 1.1em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        
        .metric-row:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #aaa;
            font-size: 0.9em;
        }
        
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .value-green { color: #00ff88; }
        .value-yellow { color: #ffdd00; }
        .value-orange { color: #ff9900; }
        .value-red { color: #ff4444; }
        
        .fatigue-gauge {
            margin: 15px 0;
        }
        
        .gauge-bar {
            height: 30px;
            background: #0f3460;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .gauge-fill {
            height: 100%;
            transition: width 0.3s ease, background-color 0.3s ease;
            border-radius: 15px;
        }
        
        .gauge-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            text-shadow: 1px 1px 2px #000;
        }
        
        .alert-box {
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .alert-normal { background: #16213e; border: 2px solid #00ff88; }
        .alert-low { background: #16213e; border: 2px solid #ffdd00; }
        .alert-medium { background: #332200; border: 2px solid #ff9900; }
        .alert-high { background: #331111; border: 2px solid #ff4444; }
        .alert-critical { 
            background: #ff4444; 
            color: #fff;
            animation: pulse 0.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-bar {
            background: #16213e;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #888;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            border-top: 1px solid #0f3460;
        }
        
        .microsleep-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 0, 0, 0.5);
            z-index: 1000;
            animation: dangerFlash 0.15s infinite;
        }
        
        .microsleep-overlay.active {
            display: block;
        }
        
        @keyframes dangerFlash {
            0%, 100% { opacity: 0.4; background: rgba(255, 0, 0, 0.5); }
            50% { opacity: 0.9; background: rgba(255, 0, 0, 0.8); }
        }
        
        .microsleep-warning {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #ff0000, #cc0000);
            color: white;
            padding: 60px 100px;
            font-size: 4em;
            font-weight: bold;
            border-radius: 25px;
            z-index: 1001;
            display: none;
            text-align: center;
            box-shadow: 0 0 100px rgba(255, 0, 0, 0.8), 0 0 200px rgba(255, 0, 0, 0.5);
            border: 8px solid #fff;
        }
        
        .microsleep-warning.active {
            display: block;
            animation: dangerPulse 0.2s infinite;
        }
        
        @keyframes dangerPulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); box-shadow: 0 0 100px rgba(255, 0, 0, 0.8); }
            50% { transform: translate(-50%, -50%) scale(1.05); box-shadow: 0 0 150px rgba(255, 0, 0, 1); }
        }
        
        .sleep-subtext {
            font-size: 0.5em;
            margin-top: 20px;
            animation: blinkText 0.3s infinite;
        }
        
        @keyframes blinkText {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        .mini-gauge {
            height: 8px;
            background: #0f3460;
            border-radius: 4px;
            position: relative;
            margin: 8px 0;
        }
        
        .mini-gauge-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.15s ease, background-color 0.15s ease;
        }
        
        .mini-gauge-threshold {
            position: absolute;
            top: -3px;
            width: 2px;
            height: 14px;
            background: #fff;
            border-radius: 1px;
        }
        
        .threshold-label {
            font-size: 0.75em;
            color: #666;
        }
        
        .ear-gauge, .mar-gauge {
            padding: 5px 0;
        }
        
        .blink-flash {
            animation: blinkFlash 0.3s ease;
        }
        
        @keyframes blinkFlash {
            0% { background: #00ff88; }
            100% { background: transparent; }
        }
        
        .yawn-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 8px;
        }
        
        .yawn-indicator.active {
            background: #ff9900;
            animation: pulse 0.5s infinite;
        }
        
        @media (max-width: 1024px) {
            .container {
                flex-direction: column;
            }
            .video-section {
                min-width: 100%;
            }
            .metrics-section {
                flex: 1;
            }
        }
    </style>
</head>
<body>
    <div class="microsleep-overlay" id="microsleepOverlay"></div>
    <div class="microsleep-warning" id="microsleepWarning">
        🚨 DANGER! WAKE UP! 🚨
        <div class="sleep-subtext">EYES CLOSED FOR 2+ SECONDS - PULL OVER NOW!</div>
    </div>
    
    <div class="header">
        <h1>🚗 Driver Fatigue Detection System</h1>
    </div>
    
    <div class="container">
        <div class="video-section">
            <img src="/video_feed" class="video-feed" alt="Live Video Feed">
        </div>
        
        <div class="metrics-section">
            <div class="panel">
                <div class="panel-title">�️ Eye Metrics</div>
                <div class="metric-row">
                    <span class="metric-label">Eye Aspect Ratio (EAR)</span>
                    <span class="metric-value" id="earValue">--</span>
                </div>
                <div class="ear-gauge">
                    <div class="mini-gauge">
                        <div class="mini-gauge-fill" id="earFill"></div>
                        <div class="mini-gauge-threshold" id="earThreshold"></div>
                    </div>
                    <span class="threshold-label">Threshold: 0.21</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Blink Rate</span>
                    <span class="metric-value" id="blinkRate">-- /min</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Total Blinks</span>
                    <span class="metric-value" id="totalBlinks">0</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">PERCLOS</span>
                    <span class="metric-value" id="perclos">--%</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">👄 Mouth Metrics</div>
                <div class="metric-row">
                    <span class="metric-label">Mouth Aspect Ratio (MAR)</span>
                    <span class="metric-value" id="marValue">--</span>
                </div>
                <div class="mar-gauge">
                    <div class="mini-gauge">
                        <div class="mini-gauge-fill" id="marFill"></div>
                        <div class="mini-gauge-threshold" id="marThreshold"></div>
                    </div>
                    <span class="threshold-label">Threshold: 0.6</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Total Yawns</span>
                    <span class="metric-value" id="yawnCount">0</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Yawn Status</span>
                    <span class="metric-value" id="yawnStatus">Normal</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">❤️ Heart Rate</div>
                <div class="metric-row">
                    <span class="metric-label">Heart Rate</span>
                    <span class="metric-value" id="heartRate">-- BPM</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">HRV (RMSSD)</span>
                    <span class="metric-value" id="hrv">-- ms</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Status</span>
                    <span class="metric-value" id="hrStatus">--</span>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">⚡ Fatigue Level</div>
                <div class="fatigue-gauge">
                    <div class="gauge-bar">
                        <div class="gauge-fill" id="fatigueFill" style="width: 0%"></div>
                        <span class="gauge-text" id="fatiguePercent">0%</span>
                    </div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Category</span>
                    <span class="metric-value" id="fatigueCategory">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Trend</span>
                    <span class="metric-value" id="fatigueTrend">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Micro-sleep Events</span>
                    <span class="metric-value" id="microsleepCount">0</span>
                </div>
            </div>
            
            <div class="alert-box alert-normal" id="alertBox">
                <div id="alertStatus">Driver appears alert</div>
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <span id="timeDisplay">--:--:--</span>
        <span>Driver Fatigue Detection System v1.0</span>
        <span>FPS: <span id="fpsDisplay">--</span></span>
    </div>
    
    <script>
        const EAR_THRESHOLD = 0.21;
        const MAR_THRESHOLD = 0.6;
        let lastBlinkCount = 0;
        let lastYawnCount = 0;
        
        function getColorClass(value, thresholds, higherIsBetter = true) {
            if (higherIsBetter) {
                if (value >= thresholds.good) return 'value-green';
                if (value >= thresholds.warn) return 'value-yellow';
                if (value >= thresholds.bad) return 'value-orange';
                return 'value-red';
            } else {
                if (value <= thresholds.good) return 'value-green';
                if (value <= thresholds.warn) return 'value-yellow';
                if (value <= thresholds.bad) return 'value-orange';
                return 'value-red';
            }
        }
        
        function getFatigueColor(level) {
            if (level >= 0.85) return '#ff4444';
            if (level >= 0.7) return '#ff9900';
            if (level >= 0.5) return '#ffdd00';
            return '#00ff88';
        }
        
        function getEarColor(ear) {
            if (ear >= 0.25) return '#00ff88';
            if (ear >= EAR_THRESHOLD) return '#ffdd00';
            if (ear >= 0.15) return '#ff9900';
            return '#ff4444';
        }
        
        function getMarColor(mar) {
            if (mar <= 0.4) return '#00ff88';
            if (mar <= MAR_THRESHOLD) return '#ffdd00';
            if (mar <= 0.8) return '#ff9900';
            return '#ff4444';
        }
        
        function updateMetrics(data) {
            const face = data.face || {};
            const heart = data.heart || {};
            const fatigue = data.fatigue || {};
            
            // EAR with gauge
            const ear = face.ear || 0;
            const earEl = document.getElementById('earValue');
            earEl.textContent = ear.toFixed(3);
            earEl.className = 'metric-value ' + (ear >= EAR_THRESHOLD ? 'value-green' : 'value-red');
            
            // EAR gauge (0-0.5 scale)
            const earPct = Math.min(100, (ear / 0.5) * 100);
            const earFill = document.getElementById('earFill');
            earFill.style.width = earPct + '%';
            earFill.style.backgroundColor = getEarColor(ear);
            document.getElementById('earThreshold').style.left = (EAR_THRESHOLD / 0.5 * 100) + '%';
            
            // Blink rate
            const blinkRate = face.blinks_per_minute || 0;
            const blinkEl = document.getElementById('blinkRate');
            blinkEl.textContent = Math.round(blinkRate) + ' /min';
            blinkEl.className = 'metric-value ' + getColorClass(blinkRate, {good: 15, warn: 10, bad: 5});
            
            // Total blinks with flash effect
            const totalBlinks = face.total_blinks || 0;
            const totalBlinksEl = document.getElementById('totalBlinks');
            if (totalBlinks > lastBlinkCount) {
                totalBlinksEl.classList.add('blink-flash');
                setTimeout(() => totalBlinksEl.classList.remove('blink-flash'), 300);
            }
            totalBlinksEl.textContent = totalBlinks;
            lastBlinkCount = totalBlinks;
            
            // PERCLOS
            const perclos = (face.perclos || 0) * 100;
            const perclosEl = document.getElementById('perclos');
            perclosEl.textContent = perclos.toFixed(1) + '%';
            perclosEl.className = 'metric-value ' + getColorClass(perclos, {good: 8, warn: 15, bad: 25}, false);
            
            // MAR with gauge
            const mar = face.mar || 0;
            const marEl = document.getElementById('marValue');
            marEl.textContent = mar.toFixed(3);
            marEl.className = 'metric-value ' + (mar > MAR_THRESHOLD ? 'value-orange' : 'value-green');
            
            // MAR gauge (0-1.0 scale)
            const marPct = Math.min(100, mar * 100);
            const marFill = document.getElementById('marFill');
            marFill.style.width = marPct + '%';
            marFill.style.backgroundColor = getMarColor(mar);
            document.getElementById('marThreshold').style.left = (MAR_THRESHOLD * 100) + '%';
            
            // Yawn count with animation
            const yawns = face.yawns_count || 0;
            const yawnEl = document.getElementById('yawnCount');
            if (yawns > lastYawnCount) {
                yawnEl.classList.add('blink-flash');
                setTimeout(() => yawnEl.classList.remove('blink-flash'), 300);
            }
            yawnEl.textContent = yawns;
            yawnEl.className = 'metric-value ' + getColorClass(yawns, {good: 0, warn: 2, bad: 4}, false);
            lastYawnCount = yawns;
            
            // Yawn status
            const yawnStatus = document.getElementById('yawnStatus');
            const isYawning = face.yawn_detected || mar > MAR_THRESHOLD;
            yawnStatus.textContent = isYawning ? '🥱 Yawning!' : 'Normal';
            yawnStatus.className = 'metric-value ' + (isYawning ? 'value-orange' : 'value-green');
            
            // Heart rate
            const bpm = heart.bpm || 0;
            const hrEl = document.getElementById('heartRate');
            hrEl.textContent = Math.round(bpm) + ' BPM';
            hrEl.className = 'metric-value ' + (bpm >= 60 && bpm <= 100 ? 'value-green' : 
                             bpm < 55 ? 'value-red' : 'value-yellow');
            
            // HRV
            const hrv = heart.hrv || 0;
            document.getElementById('hrv').textContent = hrv.toFixed(1) + ' ms';
            
            // HR Status
            document.getElementById('hrStatus').textContent = heart.hr_status || '--';
            
            // Fatigue level
            const fatigueLevel = fatigue.global_level || 0;
            const fatiguePct = (fatigueLevel * 100).toFixed(1);
            document.getElementById('fatigueFill').style.width = fatiguePct + '%';
            document.getElementById('fatigueFill').style.backgroundColor = getFatigueColor(fatigueLevel);
            document.getElementById('fatiguePercent').textContent = fatiguePct + '%';
            
            // Category
            const category = fatigue.category || 'UNKNOWN';
            const categoryEl = document.getElementById('fatigueCategory');
            categoryEl.textContent = category;
            categoryEl.className = 'metric-value ' + 
                (category === 'ALERT' ? 'value-green' : 
                 category === 'LOW' ? 'value-yellow' :
                 category === 'MEDIUM' ? 'value-orange' : 'value-red');
            
            // Trend
            const trend = fatigue.trend || 0;
            const trendEl = document.getElementById('fatigueTrend');
            const trendIcon = trend > 0.01 ? '↑ Increasing' : trend < -0.01 ? '↓ Decreasing' : '→ Stable';
            trendEl.textContent = trendIcon;
            trendEl.className = 'metric-value ' + 
                (trend > 0.01 ? 'value-red' : trend < -0.01 ? 'value-green' : 'value-yellow');
            
            // Microsleep count
            document.getElementById('microsleepCount').textContent = fatigue.microsleep_count || 0;
            
            // Microsleep/Sleep overlay with audio alarm
            const microsleep = fatigue.microsleep_active || false;
            document.getElementById('microsleepOverlay').className = 
                'microsleep-overlay' + (microsleep ? ' active' : '');
            document.getElementById('microsleepWarning').className = 
                'microsleep-warning' + (microsleep ? ' active' : '');
            
            // Play alarm sound when sleep detected
            if (microsleep && !window.alarmPlaying) {
                window.alarmPlaying = true;
                playAlarmSound();
            } else if (!microsleep) {
                window.alarmPlaying = false;
                stopAlarmSound();
            }
            
            // Alert box
            const alertBox = document.getElementById('alertBox');
            const alertStatus = document.getElementById('alertStatus');
            
            if (microsleep) {
                alertBox.className = 'alert-box alert-critical';
                alertStatus.textContent = '🚨 DANGER! SLEEPING! WAKE UP NOW! 🚨';
            } else if (category === 'CRITICAL') {
                alertBox.className = 'alert-box alert-critical';
                alertStatus.textContent = fatigue.description || 'Critical fatigue!';
            } else if (category === 'HIGH') {
                alertBox.className = 'alert-box alert-high';
                alertStatus.textContent = fatigue.description || 'Severe fatigue';
            } else if (category === 'MEDIUM') {
                alertBox.className = 'alert-box alert-medium';
                alertStatus.textContent = fatigue.description || 'Moderate fatigue';
            } else if (category === 'LOW') {
                alertBox.className = 'alert-box alert-low';
                alertStatus.textContent = fatigue.description || 'Mild fatigue';
            } else {
                alertBox.className = 'alert-box alert-normal';
                alertStatus.textContent = fatigue.description || 'Driver appears alert';
            }
            
            // FPS
            document.getElementById('fpsDisplay').textContent = (data.fps || 0).toFixed(1);
        }
        
        // Audio alarm functions for sleep detection
        let audioContext = null;
        let oscillator = null;
        let gainNode = null;
        
        function playAlarmSound() {
            try {
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                if (oscillator) {
                    stopAlarmSound();
                }
                
                oscillator = audioContext.createOscillator();
                gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 880; // High pitch alarm
                oscillator.type = 'square';
                gainNode.gain.value = 0.5;
                
                oscillator.start();
                
                // Pulsing alarm effect
                window.alarmInterval = setInterval(() => {
                    if (oscillator) {
                        oscillator.frequency.value = oscillator.frequency.value === 880 ? 1100 : 880;
                    }
                }, 150);
            } catch (e) {
                console.error('Audio alarm error:', e);
            }
        }
        
        function stopAlarmSound() {
            try {
                if (window.alarmInterval) {
                    clearInterval(window.alarmInterval);
                    window.alarmInterval = null;
                }
                if (oscillator) {
                    oscillator.stop();
                    oscillator.disconnect();
                    oscillator = null;
                }
                if (gainNode) {
                    gainNode.disconnect();
                    gainNode = null;
                }
            } catch (e) {
                console.error('Stop alarm error:', e);
            }
        }
        
        function updateTime() {
            const now = new Date();
            document.getElementById('timeDisplay').textContent = 
                now.toLocaleTimeString();
        }
        
        // Update time every second
        setInterval(updateTime, 1000);
        updateTime();
        
        // Connect to Server-Sent Events for real-time updates
        const eventSource = new EventSource('/api/stream');
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateMetrics(data);
        };
        
        eventSource.onerror = function(event) {
            console.error('SSE connection error');
            // Fallback to polling
            setTimeout(pollMetrics, 1000);
        };
        
        // Fallback polling
        function pollMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => updateMetrics(data))
                .catch(err => console.error('Polling error:', err));
        }
    </script>
</body>
</html>
'''
