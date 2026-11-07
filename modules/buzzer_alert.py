"""
Buzzer Alert Module
===================
Controls GPIO-connected buzzer for graded fatigue and micro-sleep alerts.
Supports PWM for variable intensity and different alert patterns.
"""

import time
import threading
from typing import Optional, Tuple
import logging

# GPIO handling
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


class BuzzerAlert:
    """
    GPIO Buzzer Controller for Fatigue Alerts.
    
    Provides graded alert patterns based on fatigue severity:
    - Low: Light warning beeps
    - Medium: Moderate warning pattern
    - High: Severe warning pattern
    - Micro-sleep: Continuous loud alarm
    """
    
    def __init__(self, gpio_pin: int = None):
        """
        Initialize buzzer controller.
        
        Args:
            gpio_pin: GPIO pin number for buzzer (BCM numbering)
        """
        self.gpio_pin = gpio_pin or config.BUZZER_GPIO_PIN
        self.pwm = None
        self.is_active = False
        self.alert_thread = None
        self.stop_alert = threading.Event()
        self.current_alert_level = None
        
        # Alert cooldown to prevent continuous annoying alerts
        self.last_alert_time = 0
        self.alert_cooldown = {
            'low': 30,      # 30 seconds between low alerts
            'medium': 15,   # 15 seconds between medium alerts
            'high': 5,      # 5 seconds between high alerts
            'microsleep': 0  # No cooldown for microsleep
        }
        
        logger.info(f"Buzzer alert initialized on GPIO {self.gpio_pin}")
    
    def setup_gpio(self) -> bool:
        """
        Set up GPIO for PWM buzzer control.
        
        Returns:
            True if setup successful
        """
        if not GPIO_AVAILABLE:
            logger.warning("RPi.GPIO not available - buzzer in simulation mode")
            return True
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            
            # Initialize PWM on the buzzer pin
            self.pwm = GPIO.PWM(self.gpio_pin, config.BUZZER_FREQUENCY)
            
            logger.info("Buzzer GPIO setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Buzzer GPIO setup failed: {e}")
            return False
    
    def _play_pattern(self, pattern_name: str):
        """
        Play an alert pattern.
        
        Args:
            pattern_name: Name of the pattern ('low', 'medium', 'high', 'microsleep')
        """
        if pattern_name not in config.ALERT_PATTERNS:
            logger.warning(f"Unknown alert pattern: {pattern_name}")
            return
        
        duty_cycle, duration_ms, pause_ms, repetitions = config.ALERT_PATTERNS[pattern_name]
        
        logger.info(f"Playing {pattern_name} alert pattern")
        
        for i in range(repetitions):
            if self.stop_alert.is_set():
                break
            
            # Activate buzzer
            self._buzzer_on(duty_cycle)
            
            # Wait for duration
            time.sleep(duration_ms / 1000.0)
            
            # Deactivate buzzer
            self._buzzer_off()
            
            # Pause between beeps (if not last repetition)
            if i < repetitions - 1 and pause_ms > 0:
                time.sleep(pause_ms / 1000.0)
        
        self.is_active = False
        self.current_alert_level = None
    
    def _buzzer_on(self, duty_cycle: int = 50):
        """
        Turn buzzer on with specified duty cycle.
        
        Args:
            duty_cycle: PWM duty cycle (0-100)
        """
        if GPIO_AVAILABLE and self.pwm is not None:
            self.pwm.start(duty_cycle)
        else:
            logger.debug(f"[SIM] Buzzer ON (duty: {duty_cycle}%)")
    
    def _buzzer_off(self):
        """Turn buzzer off."""
        if GPIO_AVAILABLE and self.pwm is not None:
            self.pwm.stop()
        else:
            logger.debug("[SIM] Buzzer OFF")
    
    def trigger_alert(self, level: str, force: bool = False) -> bool:
        """
        Trigger an alert at the specified level.
        
        Args:
            level: Alert level ('low', 'medium', 'high', 'microsleep')
            force: Bypass cooldown check
            
        Returns:
            True if alert was triggered
        """
        current_time = time.time()
        
        # Check cooldown (unless forced or microsleep)
        if not force and level != 'microsleep':
            cooldown = self.alert_cooldown.get(level, 10)
            if current_time - self.last_alert_time < cooldown:
                logger.debug(f"Alert {level} skipped - cooldown active")
                return False
        
        # Stop any ongoing alert
        self.stop_current_alert()
        
        # Start new alert
        self.stop_alert.clear()
        self.is_active = True
        self.current_alert_level = level
        self.last_alert_time = current_time
        
        self.alert_thread = threading.Thread(
            target=self._play_pattern, 
            args=(level,),
            daemon=True
        )
        self.alert_thread.start()
        
        logger.info(f"Alert triggered: {level}")
        return True
    
    def trigger_microsleep_alarm(self):
        """
        Trigger immediate micro-sleep/SLEEP alarm (highest priority).
        LOUD, AGGRESSIVE, CONTINUOUS alarm until manually stopped.
        This is a CRITICAL DANGER alert - person is sleeping while driving!
        """
        logger.critical("!!! SLEEP ALARM ACTIVATED - EYES CLOSED 2+ SECONDS !!!")
        
        self.stop_current_alert()
        self.stop_alert.clear()
        self.is_active = True
        self.current_alert_level = 'microsleep'
        
        def continuous_alarm():
            """Aggressive alarm pattern - loud and fast to wake the driver!"""
            cycle = 0
            while not self.stop_alert.is_set():
                # Alternating high-low frequency rapid beeps
                self._buzzer_on(100)  # Full duty cycle - MAXIMUM VOLUME
                time.sleep(0.08)  # Very short on time
                self._buzzer_off()
                time.sleep(0.03)  # Very short gap
                
                # Every 5 cycles, do a longer beep for variety
                cycle += 1
                if cycle % 5 == 0:
                    self._buzzer_on(100)
                    time.sleep(0.2)
                    self._buzzer_off()
                    time.sleep(0.05)
            
            self._buzzer_off()
            self.is_active = False
            logger.info("Sleep alarm deactivated - driver awake")
        
        self.alert_thread = threading.Thread(target=continuous_alarm, daemon=True)
        self.alert_thread.start()
    
    def stop_current_alert(self):
        """Stop any currently playing alert."""
        self.stop_alert.set()
        
        if self.alert_thread is not None and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=1.0)
        
        self._buzzer_off()
        self.is_active = False
        self.current_alert_level = None
    
    def alert_for_fatigue_level(self, fatigue_level: float, microsleep: bool = False):
        """
        Trigger appropriate alert based on fatigue level.
        
        Args:
            fatigue_level: Fatigue level (0.0 to 1.0)
            microsleep: Whether micro-sleep is currently detected
        """
        # Micro-sleep takes priority
        if microsleep:
            if self.current_alert_level != 'microsleep':
                self.trigger_microsleep_alarm()
            return
        
        # Stop microsleep alarm if it's playing and microsleep ended
        if self.current_alert_level == 'microsleep':
            self.stop_current_alert()
        
        # Determine alert level based on fatigue
        if fatigue_level >= config.FATIGUE_LEVEL_CRITICAL:
            self.trigger_alert('high')
        elif fatigue_level >= config.FATIGUE_LEVEL_HIGH:
            self.trigger_alert('high')
        elif fatigue_level >= config.FATIGUE_LEVEL_MEDIUM:
            self.trigger_alert('medium')
        elif fatigue_level >= config.FATIGUE_LEVEL_LOW:
            self.trigger_alert('low')
    
    def test_buzzer(self):
        """Test the buzzer with a short beep."""
        logger.info("Testing buzzer...")
        self._buzzer_on(50)
        time.sleep(0.2)
        self._buzzer_off()
        logger.info("Buzzer test complete")
    
    def test_all_patterns(self):
        """Test all alert patterns sequentially."""
        logger.info("Testing all alert patterns...")
        
        for level in ['low', 'medium', 'high']:
            logger.info(f"Testing {level} pattern...")
            self.trigger_alert(level, force=True)
            
            # Wait for pattern to complete
            while self.is_active:
                time.sleep(0.1)
            
            time.sleep(1)  # Pause between patterns
        
        logger.info("All patterns tested")
    
    def get_status(self) -> dict:
        """
        Get current buzzer status.
        
        Returns:
            Dictionary with buzzer state information
        """
        return {
            'is_active': self.is_active,
            'current_level': self.current_alert_level,
            'gpio_pin': self.gpio_pin,
            'time_since_last_alert': time.time() - self.last_alert_time
        }
    
    def cleanup(self):
        """Clean up GPIO resources."""
        self.stop_current_alert()
        
        if GPIO_AVAILABLE:
            if self.pwm is not None:
                self.pwm.stop()
            try:
                GPIO.cleanup(self.gpio_pin)
            except:
                pass
        
        logger.info("Buzzer cleanup complete")


class SimulatedBuzzer(BuzzerAlert):
    """Simulated buzzer for development/testing without hardware."""
    
    def __init__(self):
        super().__init__()
        self.beep_log = []
    
    def setup_gpio(self) -> bool:
        """No GPIO setup needed for simulation."""
        logger.info("Simulated buzzer - no GPIO setup needed")
        return True
    
    def _buzzer_on(self, duty_cycle: int = 50):
        """Simulate buzzer on."""
        self.beep_log.append(('on', time.time(), duty_cycle))
        print(f"🔊 BEEP (duty: {duty_cycle}%)")
    
    def _buzzer_off(self):
        """Simulate buzzer off."""
        self.beep_log.append(('off', time.time(), 0))
    
    def cleanup(self):
        """Clean up simulated buzzer."""
        self.stop_current_alert()
        logger.info("Simulated buzzer cleanup complete")
