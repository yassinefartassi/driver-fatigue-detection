"""
Face Detection and Landmark Extraction Module
==============================================
Handles camera capture, face detection, facial landmark extraction,
and computation of EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
import threading
from typing import Tuple, Optional, Dict, List
import logging

# Try to import picamera2 for Raspberry Pi Camera
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import config

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Real-time face detection and fatigue indicator extraction.
    Computes EAR, MAR, detects blinks, yawns, and micro-sleep events.
    """
    
    def __init__(self):
        """Initialize face detector with dlib models."""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self._load_landmark_predictor()
        
        # Camera
        self.camera = None
        self.picam2 = None  # For picamera2
        self.camera_lock = threading.Lock()
        
        # EAR tracking
        self.ear_history = deque(maxlen=config.CAMERA_FPS * 2)  # 2 seconds
        self.ear_below_threshold_frames = 0
        self.blink_counter = 0
        self.blink_timestamps = deque(maxlen=100)
        
        # MAR tracking
        self.mar_history = deque(maxlen=config.CAMERA_FPS * 2)
        self.mar_above_threshold_frames = 0
        self.yawn_counter = 0
        self.yawn_timestamps = deque(maxlen=50)
        self.yawn_in_progress = False
        
        # Micro-sleep tracking
        self.microsleep_detected = False
        self.microsleep_start_time = None
        self.microsleep_events = deque(maxlen=100)
        
        # PERCLOS tracking
        self.perclos_frames = deque(maxlen=config.CAMERA_FPS * config.PERCLOS_WINDOW_SEC)
        
        # Current values
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.current_frame = None
        self.current_landmarks = None
        self.face_detected = False
        
        # Frame processing
        self.frame_count = 0
        self.last_face_rect = None
        
    def _load_landmark_predictor(self):
        """Load the dlib 68-point facial landmark predictor."""
        predictor_path = str(config.LANDMARK_MODEL_PATH)
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            logger.info("Facial landmark predictor loaded successfully")
        except RuntimeError:
            logger.warning(f"Predictor not found at {predictor_path}. Will download on first run.")
            self.predictor = None
    
    def initialize_camera(self) -> bool:
        """Initialize the Pi Camera Module 2 using picamera2."""
        try:
            if PICAMERA2_AVAILABLE:
                # Use picamera2 for Raspberry Pi Camera
                logger.info("Initializing camera with picamera2...")
                self.picam2 = Picamera2()
                
                # Configure for video capture
                camera_config = self.picam2.create_preview_configuration(
                    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"}
                )
                self.picam2.configure(camera_config)
                self.picam2.start()
                
                # Wait for camera to warm up
                time.sleep(0.5)
                
                # Test capture
                frame = self.picam2.capture_array()
                if frame is not None:
                    logger.info(f"PiCamera2 initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
                    self.camera = self.picam2  # Store reference
                    return True
                else:
                    logger.error("PiCamera2 failed to capture test frame")
                    return False
            else:
                # Fallback to OpenCV VideoCapture
                logger.info("picamera2 not available, trying OpenCV...")
                self.camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
                
                if not self.camera.isOpened():
                    self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
                
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = self.camera.read()
                if ret:
                    logger.info(f"OpenCV camera initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
                    return True
                else:
                    logger.error("OpenCV camera failed to capture test frame")
                    return False
                
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def release_camera(self):
        """Release camera resources."""
        if hasattr(self, 'picam2') and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None
            logger.info("PiCamera2 released")
        elif self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.info("Camera released")
    
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the camera."""
        with self.camera_lock:
            try:
                if hasattr(self, 'picam2') and self.picam2 is not None:
                    # Capture from picamera2
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        # picamera2 returns RGB, convert to BGR for OpenCV compatibility
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        self.current_frame = frame
                        self.frame_count += 1
                        return True, frame
                    return False, None
                elif self.camera is not None:
                    # Capture from OpenCV VideoCapture
                    ret, frame = self.camera.read()
                    if ret:
                        self.current_frame = frame
                        self.frame_count += 1
                        return True, frame
                    return False, None
                else:
                    return False, None
            except Exception as e:
                logger.warning(f"Frame capture error: {e}")
                return False, None
    
    @staticmethod
    def compute_ear(eye_landmarks: np.ndarray) -> float:
        """
        Compute the Eye Aspect Ratio (EAR) for a single eye.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where p1-p6 are the 6 eye landmarks in order:
        p1: left corner, p2: upper-left, p3: upper-right, p4: right corner,
        p5: lower-right, p6: lower-left
        
        Args:
            eye_landmarks: Array of 6 (x, y) coordinates for eye landmarks
            
        Returns:
            Eye Aspect Ratio value (typically 0.15-0.45 for open eyes)
        """
        # Vertical distances
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if h == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    @staticmethod
    def compute_mar(mouth_landmarks: np.ndarray) -> float:
        """
        Compute the Mouth Aspect Ratio (MAR) for yawn detection.
        
        MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
        
        Args:
            mouth_landmarks: Array of mouth landmark coordinates
            
        Returns:
            Mouth Aspect Ratio value (typically 0.3-0.5 normal, >0.6 yawning)
        """
        # For inner mouth landmarks (60-67)
        if len(mouth_landmarks) == 8:
            # Vertical distances
            v1 = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
            v2 = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
            v3 = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])
            
            # Horizontal distance
            h = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        else:
            # For outer mouth landmarks (12 points)
            v1 = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])
            v2 = dist.euclidean(mouth_landmarks[3], mouth_landmarks[9])
            v3 = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])
            
            h = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        
        if h == 0:
            return 0.0
            
        mar = (v1 + v2 + v3) / (2.0 * h)
        return mar
    
    def extract_landmarks(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect face and extract 68 facial landmarks.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (success, landmarks array)
        """
        if self.predictor is None:
            logger.error("Landmark predictor not loaded")
            return False, None
        
        # Convert to grayscale and optionally downscale for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale for faster face detection
        scale = config.FACE_DETECTION_SCALE
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # Detect faces
        faces = self.detector(small_gray, 0)
        
        if len(faces) == 0:
            self.face_detected = False
            self.last_face_rect = None
            return False, None
        
        # Use the largest face (closest to camera)
        face = max(faces, key=lambda r: r.width() * r.height())
        
        # Scale face rectangle back to original size
        face_rect = dlib.rectangle(
            int(face.left() / scale),
            int(face.top() / scale),
            int(face.right() / scale),
            int(face.bottom() / scale)
        )
        self.last_face_rect = face_rect
        
        # Extract landmarks
        shape = self.predictor(gray, face_rect)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        
        self.face_detected = True
        self.current_landmarks = landmarks
        return True, landmarks
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame to extract all fatigue indicators.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary containing all extracted metrics
        """
        result = {
            'face_detected': False,
            'ear': 0.0,
            'mar': 0.0,
            'blink_detected': False,
            'yawn_detected': False,
            'microsleep': False,
            'blinks_per_minute': 0,
            'yawns_count': 0,
            'perclos': 0.0,
            'landmarks': None,
            'face_rect': None
        }
        
        # Extract landmarks
        success, landmarks = self.extract_landmarks(frame)
        
        if not success:
            return result
        
        result['face_detected'] = True
        result['landmarks'] = landmarks
        result['face_rect'] = self.last_face_rect
        
        # Extract eye landmarks
        left_eye = landmarks[config.LEFT_EYE_INDICES]
        right_eye = landmarks[config.RIGHT_EYE_INDICES]
        
        # Compute EAR for both eyes
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        self.current_ear = ear
        self.ear_history.append(ear)
        result['ear'] = ear
        
        # Extract mouth landmarks and compute MAR
        inner_mouth = landmarks[config.MOUTH_INDICES]
        mar = self.compute_mar(inner_mouth)
        
        self.current_mar = mar
        self.mar_history.append(mar)
        result['mar'] = mar
        
        # Detect blinks
        result['blink_detected'] = self._detect_blink(ear)
        result['total_blinks'] = self.blink_counter
        
        # Detect yawns
        result['yawn_detected'] = self._detect_yawn(mar)
        result['total_yawns'] = self.yawn_counter
        
        # Detect micro-sleep
        result['microsleep'] = self._detect_microsleep(ear)
        
        # Calculate blinks per minute
        result['blinks_per_minute'] = self._calculate_blink_rate()
        
        # Count recent yawns
        result['yawns_count'] = self._count_recent_yawns()
        
        # Calculate PERCLOS
        result['perclos'] = self._calculate_perclos(ear)
        
        return result
    
    def _detect_blink(self, ear: float) -> bool:
        """
        Detect a blink event based on EAR.
        
        Args:
            ear: Current Eye Aspect Ratio
            
        Returns:
            True if a blink was just detected
        """
        blink_detected = False
        
        if ear < config.EAR_THRESHOLD:
            self.ear_below_threshold_frames += 1
        else:
            # Check if we had a valid blink (not too long = microsleep)
            if (config.EAR_CONSEC_FRAMES_BLINK <= self.ear_below_threshold_frames < 
                config.MICROSLEEP_FRAMES):
                self.blink_counter += 1
                self.blink_timestamps.append(time.time())
                blink_detected = True
                logger.debug(f"Blink detected. Total: {self.blink_counter}")
            
            self.ear_below_threshold_frames = 0
            
        return blink_detected
    
    def _detect_yawn(self, mar: float) -> bool:
        """
        Detect a yawn event based on MAR.
        
        Args:
            mar: Current Mouth Aspect Ratio
            
        Returns:
            True if a yawn was just detected
        """
        yawn_detected = False
        
        if mar > config.MAR_THRESHOLD:
            self.mar_above_threshold_frames += 1
            
            if (self.mar_above_threshold_frames >= config.MAR_CONSEC_FRAMES_YAWN and 
                not self.yawn_in_progress):
                self.yawn_counter += 1
                self.yawn_timestamps.append(time.time())
                self.yawn_in_progress = True
                yawn_detected = True
                logger.debug(f"Yawn detected. Total: {self.yawn_counter}")
        else:
            self.mar_above_threshold_frames = 0
            self.yawn_in_progress = False
            
        return yawn_detected
    
    def _detect_microsleep(self, ear: float) -> bool:
        """
        Detect micro-sleep event (eyes closed for >= 0.5 seconds).
        
        Args:
            ear: Current Eye Aspect Ratio
            
        Returns:
            True if currently in micro-sleep state
        """
        current_time = time.time()
        
        if ear < config.EAR_THRESHOLD:
            if self.microsleep_start_time is None:
                self.microsleep_start_time = current_time
            
            elapsed = current_time - self.microsleep_start_time
            
            if elapsed >= config.MICROSLEEP_DURATION_SEC:
                if not self.microsleep_detected:
                    self.microsleep_events.append(current_time)
                    logger.warning(f"MICRO-SLEEP DETECTED! Duration: {elapsed:.2f}s")
                self.microsleep_detected = True
                return True
        else:
            if self.microsleep_detected:
                duration = current_time - self.microsleep_start_time
                logger.info(f"Micro-sleep ended. Duration: {duration:.2f}s")
            
            self.microsleep_start_time = None
            self.microsleep_detected = False
            
        return False
    
    def _calculate_blink_rate(self) -> float:
        """
        Calculate blinks per minute over the rolling window.
        
        Returns:
            Blinks per minute
        """
        current_time = time.time()
        window_start = current_time - config.BLINK_RATE_WINDOW_SEC
        
        # Count blinks within window
        recent_blinks = sum(1 for t in self.blink_timestamps if t >= window_start)
        
        # Convert to per-minute rate
        blinks_per_minute = recent_blinks * (60.0 / config.BLINK_RATE_WINDOW_SEC)
        
        return blinks_per_minute
    
    def _count_recent_yawns(self) -> int:
        """
        Count yawns within the yawn rate window.
        
        Returns:
            Number of recent yawns
        """
        current_time = time.time()
        window_start = current_time - config.YAWN_RATE_WINDOW_SEC
        
        return sum(1 for t in self.yawn_timestamps if t >= window_start)
    
    def _calculate_perclos(self, ear: float) -> float:
        """
        Calculate PERCLOS (Percentage of Eye Closure).
        
        PERCLOS is the percentage of time eyes are closed (EAR < threshold)
        over a defined time window.
        
        Args:
            ear: Current Eye Aspect Ratio
            
        Returns:
            PERCLOS value (0.0 to 1.0)
        """
        # Track if eyes are closed this frame
        eyes_closed = 1 if ear < config.EAR_THRESHOLD else 0
        self.perclos_frames.append(eyes_closed)
        
        if len(self.perclos_frames) == 0:
            return 0.0
            
        perclos = sum(self.perclos_frames) / len(self.perclos_frames)
        return perclos
    
    def get_average_ear(self, window_frames: int = 30) -> float:
        """Get average EAR over recent frames."""
        if len(self.ear_history) == 0:
            return 0.0
        recent = list(self.ear_history)[-window_frames:]
        return np.mean(recent)
    
    def get_microsleep_count(self, window_sec: int = 300) -> int:
        """Get count of micro-sleep events in the last N seconds."""
        current_time = time.time()
        window_start = current_time - window_sec
        return sum(1 for t in self.microsleep_events if t >= window_start)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw facial landmarks on the frame for visualization.
        
        Args:
            frame: BGR image
            landmarks: 68 facial landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        output = frame.copy()
        
        # Draw all landmarks as small circles
        for (x, y) in landmarks:
            cv2.circle(output, (int(x), int(y)), 1, config.COLOR_WHITE, -1)
        
        # Highlight eyes
        for idx in config.LEFT_EYE_INDICES + config.RIGHT_EYE_INDICES:
            x, y = landmarks[idx]
            cv2.circle(output, (int(x), int(y)), 2, config.COLOR_GREEN, -1)
        
        # Highlight mouth
        for idx in config.MOUTH_INDICES:
            x, y = landmarks[idx]
            cv2.circle(output, (int(x), int(y)), 2, config.COLOR_BLUE, -1)
        
        # Draw eye contours
        left_eye = landmarks[config.LEFT_EYE_INDICES]
        right_eye = landmarks[config.RIGHT_EYE_INDICES]
        
        cv2.polylines(output, [left_eye.astype(np.int32)], True, config.COLOR_GREEN, 1)
        cv2.polylines(output, [right_eye.astype(np.int32)], True, config.COLOR_GREEN, 1)
        
        # Draw mouth contour
        inner_mouth = landmarks[config.MOUTH_INDICES]
        cv2.polylines(output, [inner_mouth.astype(np.int32)], True, config.COLOR_BLUE, 1)
        
        return output
    
    def reset_counters(self):
        """Reset all tracking counters."""
        self.blink_counter = 0
        self.yawn_counter = 0
        self.blink_timestamps.clear()
        self.yawn_timestamps.clear()
        self.microsleep_events.clear()
        self.ear_history.clear()
        self.mar_history.clear()
        self.perclos_frames.clear()
        logger.info("Face detector counters reset")
