#!/usr/bin/env python3
"""
Driver Fatigue and Micro-Sleep Detection System
================================================
Main application for Raspberry Pi 5 with:
- Camera Module 2 for face/eye tracking
- 3-Pin HBPM Heart Rate Sensor
- GPIO Buzzer for alerts

Runs entirely locally without cloud dependency.
"""

import sys
import time
import logging
import argparse
import signal
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from modules.face_detector import FaceDetector
from modules.heart_rate_sensor import HeartRateSensor, SimulatedHeartRateSensor
from modules.buzzer_alert import BuzzerAlert, SimulatedBuzzer
from modules.fatigue_fusion import FatigueFusionEngine
from modules.dashboard import Dashboard

# Set up logging
def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    # Create logs directory
    log_dir = Path(config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce verbosity of some modules
    logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class FatigueDetectionSystem:
    """
    Main Driver Fatigue Detection System.
    
    Integrates all modules for real-time fatigue monitoring:
    - Face detection and EAR/MAR computation
    - Heart rate monitoring
    - Multi-modal fatigue fusion
    - Graded buzzer alerts
    - Real-time dashboard visualization
    """
    
    def __init__(self, simulation_mode: bool = False, headless: bool = False):
        """
        Initialize the fatigue detection system.
        
        Args:
            simulation_mode: Use simulated sensors for testing
            headless: Run without display (console output only)
        """
        self.simulation_mode = simulation_mode
        self.headless = headless
        self.running = False
        
        # Initialize components
        logger.info("Initializing Fatigue Detection System...")
        
        # Face detector
        self.face_detector = FaceDetector()
        
        # Heart rate sensor
        if simulation_mode:
            self.heart_sensor = SimulatedHeartRateSensor()
            self.buzzer = SimulatedBuzzer()
            logger.info("Running in SIMULATION mode")
        else:
            self.heart_sensor = HeartRateSensor()
            self.buzzer = BuzzerAlert()
        
        # Fatigue fusion engine
        self.fusion_engine = FatigueFusionEngine()
        
        # Dashboard (Flask-based web server)
        if not headless:
            self.dashboard = Dashboard(host='0.0.0.0', port=5000)
        else:
            self.dashboard = None
        
        # State tracking
        self.last_alert_time = 0
        self.frame_count = 0
        self.start_time = None
        
        # Simulation state
        self.sim_ear = 0.28
        self.sim_mar = 0.35
        self.sim_blink_timer = 0
        self.sim_yawn_timer = 0
        self.sim_microsleep = False
        self.sim_fatigue_level = 0.0
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Fatigue Detection System initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _generate_simulation_frame(self):
        """
        Generate a simulated video frame with fake face data for testing.
        
        Returns:
            Tuple of (frame, face_data)
        """
        import math
        
        # Create frame with gradient background
        frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        frame[:] = (40, 35, 30)  # Dark background
        
        # Add some visual elements
        t = time.time()
        
        # Draw simulated face outline (oval)
        center_x, center_y = config.CAMERA_WIDTH // 2, config.CAMERA_HEIGHT // 2
        cv2.ellipse(frame, (center_x, center_y), (120, 160), 0, 0, 360, (80, 70, 60), 2)
        
        # Simulate EAR variations (occasional blinks)
        self.sim_blink_timer += 1
        if self.sim_blink_timer > 90:  # Blink every ~3 seconds at 30fps
            self.sim_ear = 0.18  # Briefly closed eyes (not low enough for microsleep)
            if self.sim_blink_timer > 95:
                self.sim_blink_timer = 0
                self.sim_ear = 0.28
        else:
            # Normal variation - stays above threshold
            self.sim_ear = 0.26 + 0.02 * math.sin(t * 2)
        
        # Simulate occasional yawns
        self.sim_yawn_timer += 1
        if self.sim_yawn_timer > 600:  # Yawn every ~20 seconds
            self.sim_mar = 0.7 + 0.1 * math.sin(t * 3)
            if self.sim_yawn_timer > 660:
                self.sim_yawn_timer = 0
        else:
            self.sim_mar = 0.35 + 0.05 * math.sin(t)
        
        # Gradually increase fatigue over time for demo
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.sim_fatigue_level = min(0.6, elapsed / 300)  # Reach 60% in 5 minutes
        
        # Draw eyes (change color based on EAR)
        eye_color = (0, 255, 0) if self.sim_ear >= config.EAR_THRESHOLD else (0, 0, 255)
        eye_height = int(10 * (self.sim_ear / 0.3))
        
        # Left eye
        cv2.ellipse(frame, (center_x - 50, center_y - 30), (25, max(3, eye_height)), 0, 0, 360, eye_color, 2)
        # Right eye  
        cv2.ellipse(frame, (center_x + 50, center_y - 30), (25, max(3, eye_height)), 0, 0, 360, eye_color, 2)
        
        # Draw mouth
        mouth_color = (0, 165, 255) if self.sim_mar > config.MAR_THRESHOLD else (255, 100, 0)
        mouth_height = int(15 * (self.sim_mar / 0.5))
        cv2.ellipse(frame, (center_x, center_y + 60), (40, max(5, mouth_height)), 0, 0, 360, mouth_color, 2)
        
        # Add text overlay
        cv2.putText(frame, "SIMULATION MODE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, "No camera connected", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Add live metrics
        cv2.putText(frame, f"EAR: {self.sim_ear:.3f}", (10, config.CAMERA_HEIGHT - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        cv2.putText(frame, f"MAR: {self.sim_mar:.3f}", (10, config.CAMERA_HEIGHT - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
        cv2.putText(frame, f"Fatigue: {self.sim_fatigue_level*100:.0f}%", (10, config.CAMERA_HEIGHT - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Generate fake face_data
        blinks_per_min = 15 + 5 * math.sin(t / 30)  # Varies between 10-20
        yawns = int(elapsed / 120)  # One yawn every 2 minutes
        
        # Only trigger microsleep after 60 seconds and rarely (every 30 seconds for a few frames)
        is_microsleep = elapsed > 60 and (int(elapsed) % 30 == 0) and (self.sim_blink_timer > 92)
        
        face_data = {
            'face_detected': True,
            'ear': self.sim_ear,
            'mar': self.sim_mar,
            'blink_detected': self.sim_blink_timer == 95,
            'yawn_detected': self.sim_yawn_timer == 660,
            'microsleep': is_microsleep,
            'blinks_per_minute': blinks_per_min,
            'yawns_count': yawns,
            'perclos': max(0, (config.EAR_THRESHOLD - self.sim_ear) / config.EAR_THRESHOLD) * 0.1,
            'landmarks': None,
            'face_rect': None
        }
        
        return frame, face_data
    
    def initialize_hardware(self) -> bool:
        """
        Initialize all hardware components.
        
        Returns:
            True if all hardware initialized successfully
        """
        success = True
        
        # Initialize camera (skip in simulation mode)
        if self.simulation_mode:
            logger.info("Simulation mode - skipping camera initialization")
        else:
            logger.info("Initializing camera...")
            if not self.face_detector.initialize_camera():
                logger.error("Failed to initialize camera!")
                success = False
        
        # Initialize heart rate sensor GPIO
        logger.info("Initializing heart rate sensor...")
        if not self.heart_sensor.setup_gpio():
            logger.warning("Heart rate sensor GPIO setup failed")
            # Continue anyway - may work in simulation mode
        
        # Initialize buzzer GPIO
        logger.info("Initializing buzzer...")
        if not self.buzzer.setup_gpio():
            logger.warning("Buzzer GPIO setup failed")
        
        # Test buzzer if hardware mode
        if success and not self.simulation_mode:
            logger.info("Testing buzzer...")
            self.buzzer.test_buzzer()
        
        return success
    
    def run(self):
        """
        Main processing loop.
        
        Continuously processes camera frames, heart rate data,
        computes fatigue level, and triggers appropriate alerts.
        """
        logger.info("Starting main processing loop...")
        self.running = True
        self.start_time = time.time()
        
        # Start heart rate monitoring
        self.heart_sensor.start_monitoring()
        
        # Start Flask dashboard server
        if self.dashboard is not None:
            self.dashboard.start()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = self.face_detector.capture_frame()
                
                if not ret:
                    if self.simulation_mode:
                        # Create simulated frame and data
                        frame, face_data = self._generate_simulation_frame()
                    else:
                        logger.warning("Failed to capture frame")
                        time.sleep(0.1)
                        continue
                else:
                    # Process real frame for face/eye detection
                    face_data = self.face_detector.process_frame(frame)
                
                # Get heart rate data
                heart_data = self.heart_sensor.get_metrics()
                
                # Update fatigue fusion
                fatigue_level = self.fusion_engine.update(
                    face_data, 
                    heart_data,
                    face_data.get('microsleep', False)
                )
                
                # Get detailed fatigue report
                fatigue_report = self.fusion_engine.get_detailed_report()
                
                # Handle alerts
                self._handle_alerts(fatigue_report)
                
                # Update dashboard (Flask web server)
                if self.dashboard is not None:
                    self.dashboard.update(
                        frame,
                        face_data,
                        heart_data,
                        fatigue_report,
                        face_data.get('landmarks')
                    )
                
                # Console output (both modes)
                if self.headless:
                    self._console_output(face_data, heart_data, fatigue_report)
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
                self.frame_count += 1
                
                # Log periodic statistics
                if self.frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    self._log_statistics(fatigue_report)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _handle_alerts(self, fatigue_report: dict):
        """
        Handle alert triggering based on fatigue level.
        
        Args:
            fatigue_report: Current fatigue assessment
        """
        microsleep = fatigue_report.get('microsleep_active', False)
        fatigue_level = fatigue_report.get('global_level', 0)
        
        # Micro-sleep has highest priority
        if microsleep:
            self.buzzer.alert_for_fatigue_level(fatigue_level, microsleep=True)
            return
        
        # Get recommended alert level
        alert_level = self.fusion_engine.get_alert_level()
        
        if alert_level:
            self.buzzer.alert_for_fatigue_level(fatigue_level, microsleep=False)
    
    def _console_output(self, face_data: dict, heart_data: dict, fatigue_report: dict):
        """Print status to console for headless mode."""
        category = fatigue_report.get('category', 'UNKNOWN')
        level = fatigue_report.get('global_level', 0) * 100
        
        ear = face_data.get('ear', 0)
        mar = face_data.get('mar', 0)
        bpm = heart_data.get('bpm', 0)
        microsleep = "YES" if fatigue_report.get('microsleep_active') else "no"
        
        status_line = (f"\rFatigue: {category} ({level:.1f}%) | "
                      f"EAR: {ear:.3f} | MAR: {mar:.3f} | "
                      f"HR: {bpm:.0f} BPM | Microsleep: {microsleep}   ")
        
        print(status_line, end='', flush=True)
    
    def _log_statistics(self, fatigue_report: dict):
        """Log periodic statistics."""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"Statistics - Frames: {self.frame_count}, "
                   f"FPS: {fps:.1f}, "
                   f"Fatigue: {fatigue_report.get('category')}, "
                   f"Microsleeps: {fatigue_report.get('microsleep_count')}")
    
    def _reset_system(self):
        """Reset all tracking counters."""
        logger.info("Resetting system...")
        self.face_detector.reset_counters()
        self.heart_sensor.reset()
        self.fusion_engine.reset()
        self.buzzer.stop_current_alert()
        logger.info("System reset complete")
    
    def shutdown(self):
        """Clean up and shutdown all components."""
        logger.info("Shutting down Fatigue Detection System...")
        
        self.running = False
        
        # Stop components
        self.heart_sensor.cleanup()
        self.buzzer.cleanup()
        self.face_detector.release_camera()
        
        if self.dashboard is not None:
            self.dashboard.cleanup()
        
        # Final statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"Session summary - Duration: {elapsed:.1f}s, "
                       f"Frames: {self.frame_count}, Avg FPS: {fps:.1f}")
        
        logger.info("Shutdown complete")


def download_landmark_model() -> bool:
    """Download dlib's 68-point landmark model (.dat.bz2) from dlib.net and decompress it.

    Streams to a .part file, checks the byte count against Content-Length (the old
    version silently accepted truncated downloads), decompresses in chunks, and only
    moves the result into place once it is complete. Retries a few times.
    """
    import bz2
    import shutil
    import ssl
    import urllib.request

    # Some Python builds (e.g. python.org on macOS) ship without CA certificates;
    # use certifi's bundle when it is installed, otherwise the system default.
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_ctx = ssl.create_default_context()

    model_path = config.LANDMARK_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        logger.info(f"Landmark model already present: {model_path}")
        return True

    url = config.LANDMARK_MODEL_URL
    compressed_path = model_path.with_name(model_path.name + ".bz2.part")
    extracted_tmp = model_path.with_name(model_path.name + ".part")
    logger.info(f"Downloading facial landmark model from {url} (~64 MB)...")

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60, context=ssl_ctx) as resp, open(compressed_path, "wb") as out:
                expected = int(resp.headers.get("Content-Length") or 0)
                received = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    received += len(chunk)
            if expected and received != expected:
                raise IOError(f"incomplete download: got {received} of {expected} bytes")

            logger.info("Download complete, decompressing...")
            with bz2.open(compressed_path, "rb") as f_in, open(extracted_tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, 1 << 20)
            extracted_tmp.replace(model_path)
            logger.info(f"Landmark model ready: {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt}/3 failed: {e}")
        finally:
            compressed_path.unlink(missing_ok=True)
            extracted_tmp.unlink(missing_ok=True)

    logger.error(f"Failed to download landmark model. Download it manually from {url}, "
                 f"decompress it (bunzip2) and place it at {model_path}")
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Driver Fatigue and Micro-Sleep Detection System"
    )
    parser.add_argument(
        '--simulation', '-s',
        action='store_true',
        help='Run in simulation mode (no hardware required)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display (console output only)'
    )
    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--download-model',
        action='store_true',
        help='Download the facial landmark model and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Driver Fatigue and Micro-Sleep Detection System")
    logger.info("=" * 60)
    
    # Handle model download
    if args.download_model:
        success = download_landmark_model()
        sys.exit(0 if success else 1)
    
    # Check for landmark model
    model_path = config.LANDMARK_MODEL_PATH
    if not model_path.exists():
        logger.warning("Facial landmark model not found!")
        logger.info("Run with --download-model to download it, or download manually from:")
        logger.info(config.LANDMARK_MODEL_URL)
        
        if not args.simulation:
            response = input("Download now? [y/N]: ")
            if response.lower() == 'y':
                if not download_landmark_model():
                    sys.exit(1)
            else:
                logger.error("Cannot proceed without landmark model")
                sys.exit(1)
    
    # Create and run system
    try:
        system = FatigueDetectionSystem(
            simulation_mode=args.simulation,
            headless=args.headless
        )
        
        if system.initialize_hardware():
            logger.info("Hardware initialization successful")
            logger.info("Press Ctrl+C to quit")
            system.run()
        else:
            logger.error("Hardware initialization failed")
            if not args.simulation:
                logger.info("Try running with --simulation flag for testing")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
