"""
Driver Fatigue Detection System - Modules
==========================================
"""

from .face_detector import FaceDetector
from .heart_rate_sensor import HeartRateSensor
from .buzzer_alert import BuzzerAlert
from .fatigue_fusion import FatigueFusionEngine
from .dashboard import Dashboard

__all__ = [
    'FaceDetector',
    'HeartRateSensor', 
    'BuzzerAlert',
    'FatigueFusionEngine',
    'Dashboard'
]
