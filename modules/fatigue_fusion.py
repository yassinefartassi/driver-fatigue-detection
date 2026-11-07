"""
Fatigue Fusion Engine - Multimodal Decision Logic
==================================================
Computes fatigue score using weighted fusion of normalized indicators:
- f(Blink): Blink rate indicator (HIGHEST WEIGHT)
- f(HR): Heart rate indicator (secondary)
- f(Yawn): Yawn frequency indicator (secondary)
- PERCLOS: Percentage of eye closure
- Micro-sleep detection: EAR < threshold for ≥ 2-2.5 seconds

Uses sliding time windows for robust measurements.
"""

import time
from collections import deque
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import config

logger = logging.getLogger(__name__)


class FatigueFusionEngine:
    """
    Multimodal Fatigue Fusion Engine.
    
    Implements weighted fusion with:
    - Blink rate: HIGHEST weight (primary fatigue indicator)
    - Heart rate: Secondary weight
    - Yawn frequency: Secondary weight
    - PERCLOS: Eye closure percentage
    - Micro-sleep: Critical override (EAR < threshold for ≥ 2s)
    """
    
    # =========================================================================
    # FUSION WEIGHTS (Blink has highest, HR and Yawn are secondary)
    # =========================================================================
    WEIGHTS = {
        'blink': 0.35,      # HIGHEST - primary fatigue indicator
        'heart_rate': 0.20,  # Secondary
        'yawn': 0.20,        # Secondary  
        'perclos': 0.15,     # Eye closure over time
        'ear': 0.10,         # Current eye openness
    }
    
    # =========================================================================
    # FATIGUE THRESHOLDS
    # =========================================================================
    THRESHOLD_MODERATE = 0.40   # Moderate fatigue (40%)
    THRESHOLD_HIGH = 0.60       # High fatigue (60%)
    THRESHOLD_CRITICAL = 0.80   # Critical fatigue (80%)
    
    # =========================================================================
    # MICRO-SLEEP DETECTION (PRIORITY ALERT)
    # =========================================================================
    MICROSLEEP_EAR_THRESHOLD = 0.21   # EAR below this = eyes closed
    MICROSLEEP_DURATION_SEC = 2.0     # 2 seconds = PRIORITY micro-sleep alert
    
    # =========================================================================
    # NORMALIZATION RANGES
    # =========================================================================
    # Blink rate normalization (normal: 15-20 blinks/min)
    BLINK_NORMAL_MIN = 12
    BLINK_NORMAL_MAX = 20
    BLINK_FATIGUE_LOW = 5      # Very low = drowsy (staring)
    BLINK_FATIGUE_HIGH = 30    # Very high = compensating
    
    # Heart rate normalization (normal: 60-100 bpm)
    HR_NORMAL_MIN = 60
    HR_NORMAL_MAX = 100
    HR_FATIGUE_LOW = 50        # Low HR = drowsiness
    HR_FATIGUE_HIGH = 120      # High HR = stress
    
    # Yawn normalization (per 5 minutes)
    YAWN_NORMAL_MAX = 1        # 0-1 yawns is normal
    YAWN_MODERATE = 3          # 2-3 yawns = moderate
    YAWN_HIGH = 5              # 4+ yawns = high fatigue
    
    def __init__(self):
        """Initialize fatigue fusion engine with sliding windows."""
        
        # Sliding time windows for data collection
        self.window_duration = 60  # 60 seconds window
        
        # Blink data window
        self.blink_timestamps: deque = deque(maxlen=200)  # Store blink times
        self.blink_window_sec = 60  # 1 minute window for blink rate
        
        # Heart rate data window
        self.hr_history: deque = deque(maxlen=120)  # 2 minutes of HR data
        self.hr_timestamps: deque = deque(maxlen=120)
        
        # Yawn data window
        self.yawn_timestamps: deque = deque(maxlen=50)
        self.yawn_window_sec = 300  # 5 minute window for yawns
        
        # EAR data window (for PERCLOS calculation)
        self.ear_history: deque = deque(maxlen=config.CAMERA_FPS * 60)  # 60 sec
        self.ear_timestamps: deque = deque(maxlen=config.CAMERA_FPS * 60)
        
        # Micro-sleep detection
        self.eyes_closed_start: Optional[float] = None
        self.microsleep_active = False
        self.microsleep_count = 0
        self.total_microsleep_duration = 0.0
        
        # Normalized indicator scores (0.0 to 1.0)
        self.f_blink = 0.0      # f(Blink)
        self.f_hr = 0.0         # f(HR)
        self.f_yawn = 0.0       # f(Yawn)
        self.f_perclos = 0.0    # PERCLOS score
        self.f_ear = 0.0        # Current EAR score
        
        # Global fatigue score
        self.fatigue_score = 0.0
        self.fatigue_history: deque = deque(maxlen=300)  # 5 min history
        self.fatigue_trend = 0.0
        
        # Raw input storage
        self.current_ear = 0.25
        self.current_hr = 72
        self.current_mar = 0.0
        
        logger.info("Fatigue fusion engine initialized")
        logger.info(f"Weights: Blink={self.WEIGHTS['blink']}, HR={self.WEIGHTS['heart_rate']}, "
                   f"Yawn={self.WEIGHTS['yawn']}, PERCLOS={self.WEIGHTS['perclos']}")
        logger.info(f"Micro-sleep threshold: EAR < {self.MICROSLEEP_EAR_THRESHOLD} for >= {self.MICROSLEEP_DURATION_SEC}s")
    
    # =========================================================================
    # NORMALIZATION FUNCTIONS
    # =========================================================================
    
    def _normalize_blink_rate(self, blinks_per_minute: float) -> float:
        """
        Normalize blink rate to fatigue score f(Blink).
        
        Normal range: 12-20 blinks/min -> score = 0
        Too low (<5): drowsy/staring -> high score
        Too high (>30): compensating -> high score
        
        Args:
            blinks_per_minute: Current blink rate
            
        Returns:
            Normalized score 0.0 (alert) to 1.0 (fatigued)
        """
        if blinks_per_minute <= 0:
            # No blinks detected - could be initialization or severe drowsiness
            return 0.3
        
        if self.BLINK_NORMAL_MIN <= blinks_per_minute <= self.BLINK_NORMAL_MAX:
            # Normal range - low fatigue
            return 0.0
        
        if blinks_per_minute < self.BLINK_NORMAL_MIN:
            # Below normal - drowsy (staring)
            if blinks_per_minute <= self.BLINK_FATIGUE_LOW:
                return 1.0  # Very low - severe drowsiness
            # Linear interpolation
            return 1.0 - (blinks_per_minute - self.BLINK_FATIGUE_LOW) / (self.BLINK_NORMAL_MIN - self.BLINK_FATIGUE_LOW)
        
        else:
            # Above normal - compensating for fatigue
            if blinks_per_minute >= self.BLINK_FATIGUE_HIGH:
                return 1.0  # Very high - severe fatigue compensation
            # Linear interpolation
            return (blinks_per_minute - self.BLINK_NORMAL_MAX) / (self.BLINK_FATIGUE_HIGH - self.BLINK_NORMAL_MAX)
    
    def _normalize_heart_rate(self, bpm: float) -> float:
        """
        Normalize heart rate to fatigue score f(HR).
        
        Normal range: 60-100 bpm -> score = 0
        Low HR (<50): drowsiness -> high score
        Very high HR (>120): stress/anxiety affecting alertness
        
        Args:
            bpm: Current heart rate in beats per minute
            
        Returns:
            Normalized score 0.0 (alert) to 1.0 (fatigued)
        """
        if bpm <= 0:
            return 0.0
        
        if self.HR_NORMAL_MIN <= bpm <= self.HR_NORMAL_MAX:
            # Normal range
            return 0.0
        
        if bpm < self.HR_NORMAL_MIN:
            # Below normal - drowsiness indicator
            if bpm <= self.HR_FATIGUE_LOW:
                return 1.0  # Very low - severe drowsiness
            # Linear interpolation
            return 1.0 - (bpm - self.HR_FATIGUE_LOW) / (self.HR_NORMAL_MIN - self.HR_FATIGUE_LOW)
        
        else:
            # Above normal - stress (also affects alertness negatively)
            if bpm >= self.HR_FATIGUE_HIGH:
                return 0.6  # High stress but not full fatigue
            # Linear interpolation
            return 0.3 * (bpm - self.HR_NORMAL_MAX) / (self.HR_FATIGUE_HIGH - self.HR_NORMAL_MAX)
    
    def _normalize_yawn_rate(self, yawns_in_window: int) -> float:
        """
        Normalize yawn frequency to fatigue score f(Yawn).
        
        0-1 yawns in 5 min: normal -> score = 0
        2-3 yawns: moderate fatigue
        4+ yawns: high fatigue
        
        Args:
            yawns_in_window: Number of yawns in the time window
            
        Returns:
            Normalized score 0.0 (alert) to 1.0 (fatigued)
        """
        if yawns_in_window <= self.YAWN_NORMAL_MAX:
            return 0.0
        
        if yawns_in_window <= self.YAWN_MODERATE:
            # Moderate zone
            return 0.5 * (yawns_in_window - self.YAWN_NORMAL_MAX) / (self.YAWN_MODERATE - self.YAWN_NORMAL_MAX)
        
        if yawns_in_window <= self.YAWN_HIGH:
            # High zone
            return 0.5 + 0.3 * (yawns_in_window - self.YAWN_MODERATE) / (self.YAWN_HIGH - self.YAWN_MODERATE)
        
        # Very high
        return min(1.0, 0.8 + 0.04 * (yawns_in_window - self.YAWN_HIGH))
    
    def _calculate_perclos(self) -> float:
        """
        Calculate PERCLOS (Percentage of Eye Closure).
        
        PERCLOS = (time with EAR < threshold) / (total time in window)
        
        Returns:
            PERCLOS value 0.0 to 1.0
        """
        if len(self.ear_history) < 30:  # Need at least 1 second of data
            return 0.0
        
        current_time = time.time()
        window_start = current_time - 60  # 60 second window
        
        # Count frames with eyes closed
        closed_count = 0
        total_count = 0
        
        for ear, ts in zip(self.ear_history, self.ear_timestamps):
            if ts >= window_start:
                total_count += 1
                if ear < self.MICROSLEEP_EAR_THRESHOLD:
                    closed_count += 1
        
        if total_count == 0:
            return 0.0
        
        return closed_count / total_count
    
    def _normalize_perclos(self, perclos: float) -> float:
        """
        Normalize PERCLOS to fatigue score.
        
        PERCLOS < 8%: Alert
        PERCLOS 8-15%: Mild fatigue
        PERCLOS 15-30%: Moderate fatigue
        PERCLOS > 30%: Severe fatigue
        
        Args:
            perclos: PERCLOS value (0.0 to 1.0)
            
        Returns:
            Normalized score 0.0 to 1.0
        """
        if perclos < 0.08:
            return 0.0
        elif perclos < 0.15:
            return (perclos - 0.08) / 0.07 * 0.4
        elif perclos < 0.30:
            return 0.4 + (perclos - 0.15) / 0.15 * 0.3
        else:
            return min(1.0, 0.7 + (perclos - 0.30) / 0.20 * 0.3)
    
    def _normalize_ear(self, ear: float) -> float:
        """
        Normalize current EAR to fatigue score.
        
        EAR > 0.30: Eyes well open -> 0
        EAR 0.21-0.30: Slightly droopy -> 0-0.3
        EAR < 0.21: Eyes closed/very droopy -> 0.3-1.0
        
        Args:
            ear: Current Eye Aspect Ratio
            
        Returns:
            Normalized score 0.0 to 1.0
        """
        if ear >= 0.30:
            return 0.0
        elif ear >= self.MICROSLEEP_EAR_THRESHOLD:
            return 0.3 * (0.30 - ear) / (0.30 - self.MICROSLEEP_EAR_THRESHOLD)
        else:
            # Eyes closed
            return min(1.0, 0.3 + 0.7 * (self.MICROSLEEP_EAR_THRESHOLD - ear) / self.MICROSLEEP_EAR_THRESHOLD)
    
    # =========================================================================
    # MICRO-SLEEP DETECTION
    # =========================================================================
    
    def _check_microsleep(self, ear: float, face_detected: bool) -> bool:
        """
        Detect micro-sleep: EAR < threshold for >= 2 seconds.
        ONLY triggers when a face IS detected (driver visible but eyes closed).
        
        Args:
            ear: Current Eye Aspect Ratio
            face_detected: Whether a face is currently detected
            
        Returns:
            True if micro-sleep is detected
        """
        current_time = time.time()
        
        # Only check for micro-sleep when face IS detected
        # No face = can't determine if eyes are closed
        if not face_detected:
            # Reset micro-sleep tracking when no face
            self.eyes_closed_start = None
            self.microsleep_active = False
            return False
        
        if ear < self.MICROSLEEP_EAR_THRESHOLD:
            # Face detected AND eyes closed
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            else:
                duration = current_time - self.eyes_closed_start
                if duration >= self.MICROSLEEP_DURATION_SEC:
                    if not self.microsleep_active:
                        self.microsleep_count += 1
                        logger.warning(f"MICRO-SLEEP DETECTED! Duration: {duration:.2f}s, Count: {self.microsleep_count}")
                    self.microsleep_active = True
                    self.total_microsleep_duration += 1.0 / config.CAMERA_FPS
                    return True
        else:
            # Eyes open - reset
            if self.microsleep_active:
                duration = current_time - self.eyes_closed_start if self.eyes_closed_start else 0
                logger.info(f"Micro-sleep ended. Duration: {duration:.2f}s")
            self.eyes_closed_start = None
            self.microsleep_active = False
        
        return False
    
    # =========================================================================
    # SLIDING WINDOW CALCULATIONS
    # =========================================================================
    
    def _get_blinks_per_minute(self) -> float:
        """Calculate blinks per minute using sliding window."""
        current_time = time.time()
        window_start = current_time - self.blink_window_sec
        
        # Count blinks in window
        blinks_in_window = sum(1 for t in self.blink_timestamps if t >= window_start)
        
        # Scale to per-minute rate
        return blinks_in_window * (60.0 / self.blink_window_sec)
    
    def _get_yawns_in_window(self) -> int:
        """Count yawns in the time window."""
        current_time = time.time()
        window_start = current_time - self.yawn_window_sec
        
        return sum(1 for t in self.yawn_timestamps if t >= window_start)
    
    def _get_average_hr(self) -> float:
        """Get average heart rate from sliding window."""
        if len(self.hr_history) == 0:
            return 0
        
        current_time = time.time()
        window_start = current_time - 30  # 30 second window for HR
        
        recent_hr = [hr for hr, ts in zip(self.hr_history, self.hr_timestamps) 
                     if ts >= window_start]
        
        if len(recent_hr) == 0:
            return list(self.hr_history)[-1] if self.hr_history else 0
        
        return np.mean(recent_hr)
    
    # =========================================================================
    # MAIN UPDATE FUNCTION
    # =========================================================================
    
    def update(self, 
               face_data: Dict,
               heart_data: Dict,
               microsleep: bool = False) -> float:
        """
        Update fatigue score with new sensor data.
        
        Args:
            face_data: Dictionary with EAR, MAR, blink_detected, yawn_detected
            heart_data: Dictionary with bpm, hrv
            microsleep: Whether micro-sleep was detected by face detector
            
        Returns:
            Updated fatigue score (0.0 to 1.0)
        """
        current_time = time.time()
        
        # Extract data
        ear = face_data.get('ear', 0.25)
        mar = face_data.get('mar', 0.0)
        blink_detected = face_data.get('blink_detected', False)
        yawn_detected = face_data.get('yawn_detected', False)
        face_detected = face_data.get('face_detected', False)
        bpm = heart_data.get('bpm', 72)
        
        # Store current values
        self.current_ear = ear
        self.current_hr = bpm
        self.current_mar = mar
        
        # Only update sliding windows when face is detected
        if face_detected:
            self.ear_history.append(ear)
            self.ear_timestamps.append(current_time)
            
            if blink_detected:
                self.blink_timestamps.append(current_time)
            
            if yawn_detected:
                self.yawn_timestamps.append(current_time)
        
        if bpm > 0:
            self.hr_history.append(bpm)
            self.hr_timestamps.append(current_time)
        
        # Check for micro-sleep (EAR < threshold for >= 2 seconds)
        # ONLY when face is detected - can't detect closed eyes without a face
        is_microsleep = self._check_microsleep(ear, face_detected)
        
        # Calculate normalized indicators
        blinks_per_min = self._get_blinks_per_minute()
        yawns_count = self._get_yawns_in_window()
        avg_hr = self._get_average_hr()
        perclos = self._calculate_perclos()
        
        # Normalize each indicator (0.0 to 1.0)
        self.f_blink = self._normalize_blink_rate(blinks_per_min)
        self.f_hr = self._normalize_heart_rate(avg_hr)
        self.f_yawn = self._normalize_yawn_rate(yawns_count)
        self.f_perclos = self._normalize_perclos(perclos)
        self.f_ear = self._normalize_ear(ear)
        
        # Compute weighted fusion score
        self.fatigue_score = (
            self.WEIGHTS['blink'] * self.f_blink +
            self.WEIGHTS['heart_rate'] * self.f_hr +
            self.WEIGHTS['yawn'] * self.f_yawn +
            self.WEIGHTS['perclos'] * self.f_perclos +
            self.WEIGHTS['ear'] * self.f_ear
        )
        
        # Micro-sleep override: if active, minimum fatigue is CRITICAL
        if is_microsleep or self.microsleep_active:
            self.fatigue_score = max(self.fatigue_score, self.THRESHOLD_CRITICAL)
        
        # Clamp to valid range
        self.fatigue_score = min(1.0, max(0.0, self.fatigue_score))
        
        # Store in history for trend calculation
        self.fatigue_history.append(self.fatigue_score)
        
        # Calculate trend
        self._calculate_trend()
        
        return self.fatigue_score
    
    def _calculate_trend(self):
        """Calculate fatigue trend from history (increasing/decreasing/stable)."""
        if len(self.fatigue_history) < 30:
            self.fatigue_trend = 0.0
            return
        
        recent = list(self.fatigue_history)[-60:]  # Last 60 samples
        if len(recent) < 10:
            return
        
        # Linear regression for trend
        x = np.arange(len(recent))
        y = np.array(recent)
        
        n = len(recent)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-10)
        
        self.fatigue_trend = slope
    
    # =========================================================================
    # GETTERS AND REPORTING
    # =========================================================================
    
    def get_fatigue_level(self) -> float:
        """Get current global fatigue level."""
        return self.fatigue_score
    
    def get_fatigue_category(self) -> Tuple[str, str]:
        """
        Get fatigue category and description based on thresholds.
        
        Returns:
            Tuple of (category, description)
        """
        level = self.fatigue_score
        
        if self.microsleep_active:
            return 'MICROSLEEP', 'DANGER: Micro-sleep detected! Wake up immediately!'
        
        if level >= self.THRESHOLD_CRITICAL:
            return 'CRITICAL', 'Critical fatigue - stop driving immediately!'
        elif level >= self.THRESHOLD_HIGH:
            return 'HIGH', 'High fatigue - take a break now'
        elif level >= self.THRESHOLD_MODERATE:
            return 'MODERATE', 'Moderate fatigue - consider resting soon'
        elif level > 0.2:
            return 'LOW', 'Mild fatigue - stay alert'
        else:
            return 'ALERT', 'Driver appears alert'
    
    def get_component_scores(self) -> Dict[str, float]:
        """Get all normalized component scores."""
        return {
            'f_blink': self.f_blink,
            'f_hr': self.f_hr,
            'f_yawn': self.f_yawn,
            'f_perclos': self.f_perclos,
            'f_ear': self.f_ear
        }
    
    def get_detailed_report(self) -> Dict:
        """
        Get comprehensive fatigue assessment report.
        
        Returns:
            Dictionary with all fatigue metrics
        """
        category, description = self.get_fatigue_category()
        
        return {
            'global_level': self.fatigue_score,
            'category': category,
            'description': description,
            'trend': self.fatigue_trend,
            'trend_direction': 'increasing' if self.fatigue_trend > 0.005 else 
                              'decreasing' if self.fatigue_trend < -0.005 else 'stable',
            'microsleep_active': self.microsleep_active,
            'microsleep_count': self.microsleep_count,
            'total_microsleep_sec': round(self.total_microsleep_duration, 1),
            'component_scores': {
                'blink_rate': self.f_blink,
                'heart_rate': self.f_hr,
                'yawn_frequency': self.f_yawn,
                'perclos': self.f_perclos,
                'ear_openness': self.f_ear
            },
            'weights': self.WEIGHTS.copy(),
            'thresholds': {
                'moderate': self.THRESHOLD_MODERATE,
                'high': self.THRESHOLD_HIGH,
                'critical': self.THRESHOLD_CRITICAL,
                'microsleep_duration_sec': self.MICROSLEEP_DURATION_SEC
            },
            'raw_metrics': {
                'blinks_per_min': self._get_blinks_per_minute(),
                'yawns_count': self._get_yawns_in_window(),
                'avg_hr': self._get_average_hr(),
                'perclos': self._calculate_perclos(),
                'current_ear': self.current_ear
            }
        }
    
    def get_alert_level(self) -> Optional[str]:
        """
        Determine appropriate alert level for buzzer.
        
        Returns:
            Alert level: 'microsleep', 'high', 'medium', 'low', or None
        """
        if self.microsleep_active:
            return 'microsleep'
        
        level = self.fatigue_score
        
        if level >= self.THRESHOLD_CRITICAL:
            return 'high'
        elif level >= self.THRESHOLD_HIGH:
            return 'high'
        elif level >= self.THRESHOLD_MODERATE:
            return 'medium'
        elif level > 0.25:
            return 'low'
        
        return None
    
    def reset(self):
        """Reset all tracking data."""
        self.blink_timestamps.clear()
        self.hr_history.clear()
        self.hr_timestamps.clear()
        self.yawn_timestamps.clear()
        self.ear_history.clear()
        self.ear_timestamps.clear()
        self.fatigue_history.clear()
        
        self.eyes_closed_start = None
        self.microsleep_active = False
        self.microsleep_count = 0
        self.total_microsleep_duration = 0.0
        
        self.f_blink = 0.0
        self.f_hr = 0.0
        self.f_yawn = 0.0
        self.f_perclos = 0.0
        self.f_ear = 0.0
        self.fatigue_score = 0.0
        self.fatigue_trend = 0.0
        
        logger.info("Fatigue fusion engine reset")
