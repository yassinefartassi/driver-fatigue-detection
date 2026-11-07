"""
Driver Fatigue Detection System - Configuration
================================================
Raspberry Pi 5 with Camera Module 2, HBPM Heart-Rate Sensor, and GPIO Buzzer
"""

import os
from pathlib import Path

# All paths are relative to this file's directory, so the project runs from
# wherever it is cloned. Override any of them with the matching env var.
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("FATIGUE_MODELS_DIR", PROJECT_ROOT / "models"))
LANDMARK_MODEL_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
LANDMARK_MODEL_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 0  # For Pi Camera Module 2 via libcamera

# =============================================================================
# EYE ASPECT RATIO (EAR) THRESHOLDS
# =============================================================================
EAR_THRESHOLD = 0.21  # Below this value, eyes are considered closed
EAR_CONSEC_FRAMES_BLINK = 3  # Frames below threshold to count as a blink
MICROSLEEP_DURATION_SEC = 2.0  # Duration (seconds) for micro-sleep detection (SLEEP ALERT!)
MICROSLEEP_FRAMES = int(MICROSLEEP_DURATION_SEC * CAMERA_FPS)  # ~60 frames at 30fps

# =============================================================================
# MOUTH ASPECT RATIO (MAR) THRESHOLDS
# =============================================================================
MAR_THRESHOLD = 0.6  # Above this value, mouth is considered open (yawn)
MAR_CONSEC_FRAMES_YAWN = 15  # Frames above threshold to count as a yawn

# =============================================================================
# HEART RATE (HBPM) CONFIGURATION
# =============================================================================
HBPM_GPIO_PIN = 17  # GPIO pin for heart-rate sensor signal
HBPM_SAMPLE_RATE = 100  # Hz
HBPM_NORMAL_MIN = 60  # bpm
HBPM_NORMAL_MAX = 100  # bpm
HBPM_FATIGUE_LOW = 55  # bpm - indicates fatigue when HR drops
HBPM_FATIGUE_HIGH = 110  # bpm - indicates stress/alertness

# Heart Rate Variability thresholds
HRV_WINDOW_SEC = 60  # Window for HRV calculation
HRV_FATIGUE_THRESHOLD = 20  # Low HRV indicates fatigue (ms)

# =============================================================================
# BUZZER CONFIGURATION
# =============================================================================
BUZZER_GPIO_PIN = 18  # GPIO pin for buzzer (PWM capable)
BUZZER_FREQUENCY = 2000  # Hz

# Alert levels and patterns (duty_cycle, duration_ms, pause_ms, repetitions)
ALERT_PATTERNS = {
    'low': (50, 100, 200, 2),       # Light warning: 2 short beeps
    'medium': (70, 200, 100, 3),    # Moderate warning: 3 medium beeps
    'high': (90, 300, 50, 5),       # Severe warning: 5 rapid beeps
    'microsleep': (100, 200, 50, 10), # LOUD CONTINUOUS ALARM: 10 rapid max-volume beeps
    'sleep': (100, 100, 30, 20),    # CRITICAL SLEEP ALARM: 20 very rapid max-volume beeps
}

# =============================================================================
# FATIGUE FUSION WEIGHTS (must sum to 1.0)
# =============================================================================
# Ocular indicators (EAR-based)
WEIGHT_BLINK_RATE = 0.15
WEIGHT_EAR_DURATION = 0.20
WEIGHT_PERCLOS = 0.15  # Percentage of eye closure over time

# Behavioral indicators (MAR-based)
WEIGHT_YAWN_FREQUENCY = 0.15

# Physiological indicators (HBPM-based)
WEIGHT_HEART_RATE = 0.15
WEIGHT_HRV = 0.10

# Micro-sleep penalty (critical safety factor)
WEIGHT_MICROSLEEP = 0.10

# =============================================================================
# FATIGUE LEVEL THRESHOLDS
# =============================================================================
FATIGUE_LEVEL_LOW = 0.3      # 0-30%: Alert
FATIGUE_LEVEL_MEDIUM = 0.5   # 30-50%: Mild fatigue
FATIGUE_LEVEL_HIGH = 0.7     # 50-70%: Moderate fatigue
FATIGUE_LEVEL_CRITICAL = 0.85  # 70-85%: Severe fatigue, >85%: Critical

# =============================================================================
# TIMING WINDOWS
# =============================================================================
BLINK_RATE_WINDOW_SEC = 60  # Window for calculating blinks per minute
YAWN_RATE_WINDOW_SEC = 300  # Window for calculating yawns (5 minutes)
PERCLOS_WINDOW_SEC = 60     # Window for PERCLOS calculation

# Normal ranges for reference
NORMAL_BLINK_RATE_MIN = 15  # blinks per minute
NORMAL_BLINK_RATE_MAX = 20  # blinks per minute
FATIGUE_BLINK_RATE_HIGH = 25  # Elevated blinking indicates fatigue

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD_WIDTH = 1024
DASHBOARD_HEIGHT = 600
DASHBOARD_UPDATE_FPS = 15  # Dashboard refresh rate

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)

# =============================================================================
# FACIAL LANDMARK INDICES (dlib 68-point model)
# =============================================================================
# Left eye landmark indices
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
# Right eye landmark indices
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
# Mouth landmark indices
MOUTH_INDICES = [60, 61, 62, 63, 64, 65, 66, 67]
# Outer mouth for MAR calculation
MOUTH_OUTER_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = os.environ.get("FATIGUE_LOG_FILE", str(PROJECT_ROOT / "logs" / "fatigue_detection.log"))
DATA_SAVE_PATH = os.environ.get("FATIGUE_DATA_DIR", str(PROJECT_ROOT / "data")) + os.sep
ENABLE_DATA_LOGGING = True
DATA_LOG_INTERVAL_SEC = 1  # Log data every second

# Performance settings
USE_THREADING = True
FACE_DETECTION_SCALE = 0.5  # Downscale factor for faster detection
LANDMARK_SKIP_FRAMES = 1  # Process landmarks every N frames
