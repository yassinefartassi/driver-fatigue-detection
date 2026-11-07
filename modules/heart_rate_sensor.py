"""
Heart Rate Sensor Module (HBPM - 3-Pin)
=======================================
Handles reading from a 3-pin heart-rate pulse sensor via GPIO,
computes heart rate (BPM), and derives heart rate variability (HRV)
for fatigue assessment.
"""

import time
import threading
from collections import deque
from typing import Optional, Tuple, List
import logging
import numpy as np

# GPIO handling - RPi.GPIO for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import config

logger = logging.getLogger(__name__)


class HeartRateSensor:
    """
    3-Pin HBPM Heart Rate Sensor Interface.
    
    Reads pulse signals from GPIO, detects heartbeats,
    calculates BPM (beats per minute) and HRV (heart rate variability).
    """
    
    def __init__(self, gpio_pin: int = None):
        """
        Initialize heart rate sensor.
        
        Args:
            gpio_pin: GPIO pin number for the sensor signal (BCM numbering)
        """
        self.gpio_pin = gpio_pin or config.HBPM_GPIO_PIN
        self.running = False
        self.thread = None
        
        # Beat detection
        self.last_beat_time = None
        self.beat_intervals = deque(maxlen=60)  # Store last 60 R-R intervals
        self.beat_timestamps = deque(maxlen=120)  # Store beat timestamps
        
        # Current values
        self.current_bpm = 0.0
        self.current_hrv = 0.0
        self.raw_signal = 0
        
        # Signal processing
        self.signal_buffer = deque(maxlen=config.HBPM_SAMPLE_RATE)
        self.baseline = 512  # ADC midpoint (for analog sensors via MCP3008)
        self.threshold_factor = 0.6
        self.last_signal_state = False  # For edge detection
        
        # Data lock for thread safety
        self.data_lock = threading.Lock()
        
        # Fatigue indicators
        self.fatigue_score = 0.0
        
        logger.info(f"Heart rate sensor initialized on GPIO {self.gpio_pin}")
    
    def setup_gpio(self) -> bool:
        """
        Set up GPIO for reading the heart rate sensor.
        
        Returns:
            True if setup successful, False otherwise
        """
        if not GPIO_AVAILABLE:
            logger.warning("RPi.GPIO not available - running in simulation mode")
            return True
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            
            # Set up edge detection for heartbeat pulses
            GPIO.add_event_detect(
                self.gpio_pin, 
                GPIO.RISING,
                callback=self._beat_callback,
                bouncetime=200  # 200ms debounce (max 300 BPM)
            )
            
            logger.info("GPIO setup complete for heart rate sensor")
            return True
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
            return False
    
    def _beat_callback(self, channel):
        """
        Callback function triggered on each heartbeat pulse.
        
        Args:
            channel: GPIO channel that triggered the callback
        """
        current_time = time.time()
        
        with self.data_lock:
            if self.last_beat_time is not None:
                # Calculate R-R interval in milliseconds
                interval = (current_time - self.last_beat_time) * 1000
                
                # Validate interval (30-200 BPM range = 300-2000ms intervals)
                if 300 <= interval <= 2000:
                    self.beat_intervals.append(interval)
                    self.beat_timestamps.append(current_time)
                    
                    # Update BPM and HRV
                    self._update_metrics()
            
            self.last_beat_time = current_time
    
    def _update_metrics(self):
        """Update BPM and HRV based on recent beat intervals."""
        if len(self.beat_intervals) < 3:
            return
        
        intervals = list(self.beat_intervals)
        
        # Calculate average BPM
        avg_interval = np.mean(intervals[-10:])  # Use last 10 beats
        if avg_interval > 0:
            self.current_bpm = 60000.0 / avg_interval  # Convert ms to BPM
        
        # Calculate HRV (RMSSD - Root Mean Square of Successive Differences)
        if len(intervals) >= 5:
            successive_diffs = np.diff(intervals[-20:])  # Last 20 intervals
            self.current_hrv = np.sqrt(np.mean(successive_diffs ** 2))
        
        # Calculate fatigue score based on HR and HRV
        self._calculate_fatigue_score()
    
    def _calculate_fatigue_score(self):
        """
        Calculate fatigue score based on heart rate patterns.
        
        Low HRV and abnormal HR patterns indicate fatigue.
        """
        hr_score = 0.0
        hrv_score = 0.0
        
        # HR-based fatigue (normalized 0-1)
        if self.current_bpm < config.HBPM_FATIGUE_LOW:
            # Low HR = drowsiness/fatigue
            hr_score = min(1.0, (config.HBPM_FATIGUE_LOW - self.current_bpm) / 15)
        elif self.current_bpm > config.HBPM_FATIGUE_HIGH:
            # Very high HR could indicate stress but also alertness
            hr_score = 0.2  # Slight fatigue indicator
        elif config.HBPM_NORMAL_MIN <= self.current_bpm <= config.HBPM_NORMAL_MAX:
            hr_score = 0.0  # Normal range
        
        # HRV-based fatigue (low HRV indicates fatigue)
        if self.current_hrv < config.HRV_FATIGUE_THRESHOLD:
            hrv_score = min(1.0, (config.HRV_FATIGUE_THRESHOLD - self.current_hrv) / 
                          config.HRV_FATIGUE_THRESHOLD)
        
        # Combined fatigue score
        self.fatigue_score = (hr_score * 0.4 + hrv_score * 0.6)
    
    def start_monitoring(self):
        """Start continuous heart rate monitoring in a background thread."""
        if self.running:
            logger.warning("Heart rate monitoring already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info("Heart rate monitoring started")
    
    def stop_monitoring(self):
        """Stop heart rate monitoring."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        if GPIO_AVAILABLE:
            try:
                GPIO.remove_event_detect(self.gpio_pin)
            except:
                pass
        
        logger.info("Heart rate monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for signal processing."""
        while self.running:
            # For sensors without interrupt capability, poll here
            # Most 3-pin HBPM sensors use interrupt-based detection
            time.sleep(0.01)  # 100Hz polling rate
            
            # Cleanup old timestamps
            current_time = time.time()
            with self.data_lock:
                # Remove beats older than HRV window
                cutoff = current_time - config.HRV_WINDOW_SEC
                while (self.beat_timestamps and 
                       self.beat_timestamps[0] < cutoff):
                    self.beat_timestamps.popleft()
                    if self.beat_intervals:
                        self.beat_intervals.popleft()
    
    def get_current_bpm(self) -> float:
        """Get current heart rate in BPM."""
        with self.data_lock:
            return self.current_bpm
    
    def get_current_hrv(self) -> float:
        """Get current HRV (RMSSD) in milliseconds."""
        with self.data_lock:
            return self.current_hrv
    
    def get_fatigue_score(self) -> float:
        """Get heart-rate based fatigue score (0.0 to 1.0)."""
        with self.data_lock:
            return self.fatigue_score
    
    def get_metrics(self) -> dict:
        """
        Get all heart rate metrics.
        
        Returns:
            Dictionary with BPM, HRV, and fatigue score
        """
        with self.data_lock:
            return {
                'bpm': self.current_bpm,
                'hrv': self.current_hrv,
                'fatigue_score': self.fatigue_score,
                'beat_count': len(self.beat_timestamps),
                'hr_status': self._get_hr_status()
            }
    
    def _get_hr_status(self) -> str:
        """Get human-readable heart rate status."""
        if self.current_bpm == 0:
            return "No Signal"
        elif self.current_bpm < config.HBPM_FATIGUE_LOW:
            return "Low (Drowsy)"
        elif self.current_bpm > config.HBPM_FATIGUE_HIGH:
            return "Elevated"
        elif config.HBPM_NORMAL_MIN <= self.current_bpm <= config.HBPM_NORMAL_MAX:
            return "Normal"
        else:
            return "Borderline"
    
    def simulate_heartbeat(self, bpm: float = 72):
        """
        Simulate heartbeat for testing without hardware.
        
        Args:
            bpm: Simulated beats per minute
        """
        current_time = time.time()
        interval = 60000.0 / bpm  # Interval in ms
        
        # Add some natural variation (±5%)
        variation = np.random.uniform(-0.05, 0.05) * interval
        interval += variation
        
        with self.data_lock:
            if self.last_beat_time is not None:
                self.beat_intervals.append(interval)
                self.beat_timestamps.append(current_time)
                self._update_metrics()
            
            self.last_beat_time = current_time
    
    def reset(self):
        """Reset all heart rate data."""
        with self.data_lock:
            self.beat_intervals.clear()
            self.beat_timestamps.clear()
            self.last_beat_time = None
            self.current_bpm = 0.0
            self.current_hrv = 0.0
            self.fatigue_score = 0.0
        logger.info("Heart rate sensor data reset")
    
    def cleanup(self):
        """Clean up GPIO resources."""
        self.stop_monitoring()
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup(self.gpio_pin)
            except:
                pass
        logger.info("Heart rate sensor cleanup complete")


class SimulatedHeartRateSensor(HeartRateSensor):
    """
    Simulated heart rate sensor for development/testing.
    Generates realistic heart rate patterns including fatigue simulation.
    """
    
    def __init__(self):
        super().__init__()
        self.simulation_bpm = 72.0
        self.fatigue_simulation = False
        self.last_simulation_beat = time.time()
        
    def setup_gpio(self) -> bool:
        """No GPIO setup needed for simulation."""
        logger.info("Simulated heart rate sensor - no GPIO setup needed")
        return True
    
    def _monitoring_loop(self):
        """Generate simulated heartbeats."""
        while self.running:
            current_time = time.time()
            interval = 60.0 / self.simulation_bpm
            
            if current_time - self.last_simulation_beat >= interval:
                self.simulate_heartbeat(self.simulation_bpm)
                self.last_simulation_beat = current_time
                
                # Gradually vary BPM for realistic simulation
                if self.fatigue_simulation:
                    # Simulate fatigue: HR drops and becomes less variable
                    self.simulation_bpm = max(50, self.simulation_bpm - 0.1)
                else:
                    # Normal variation
                    self.simulation_bpm = 72 + np.random.uniform(-5, 5)
            
            time.sleep(0.01)
    
    def set_fatigue_mode(self, enabled: bool):
        """Enable or disable fatigue simulation."""
        self.fatigue_simulation = enabled
        if enabled:
            logger.info("Fatigue simulation enabled")
        else:
            self.simulation_bpm = 72.0
            logger.info("Fatigue simulation disabled")
    
    def cleanup(self):
        """Clean up simulated sensor."""
        self.stop_monitoring()
        logger.info("Simulated heart rate sensor cleanup complete")
