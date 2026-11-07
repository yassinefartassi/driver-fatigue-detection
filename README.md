# Driver Fatigue and Micro-Sleep Detection System

Real-time driver fatigue monitoring system for Raspberry Pi 5 that combines ocular tracking (Eye/Mouth Aspect Ratios), physiological signals (heart rate variability), and behavioral indicators (blinks, yawns) to compute a weighted fatigue score with graded buzzer alerts and a live dashboard—all processed locally without cloud dependency.

## Hardware Requirements

- **Raspberry Pi 5** (4GB+ RAM recommended)
- **Pi Camera Module 2** (or compatible USB webcam)
- **3-Pin HBPM Heart Rate Sensor** (connected to GPIO 17)
- **Active Buzzer** (connected to GPIO 18, PWM-capable pin)

### GPIO Wiring

```
Heart Rate Sensor (3-pin):
  VCC → 3.3V (Pin 1)
  GND → Ground (Pin 6)
  SIG → GPIO 17 (Pin 11)

Buzzer:
  VCC → 5V (Pin 2)
  GND → Ground (Pin 14)
  SIG → GPIO 18 (Pin 12) [PWM]
```

## Installation

```bash
# Clone the project
git clone https://github.com/yassinefartassi/driver-fatigue-detection.git
cd driver-fatigue-detection

# Install system dependencies (Raspberry Pi OS)
sudo apt update
sudo apt install -y python3-pip python3-venv cmake libopenblas-dev liblapack-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download facial landmark model (~64 MB from dlib.net, not stored in git)
python main.py --download-model      # or: scripts/download_data.sh
```

See [DATA.md](DATA.md) for the model's source, licence and citation.

## Usage

```bash
# Run with hardware
python main.py

# Run in simulation mode (for testing without hardware)
python main.py --simulation

# Run headless (no display, console output only)
python main.py --headless

# Debug mode with verbose logging
python main.py --log-level DEBUG
```

### Keyboard Controls (Dashboard Mode)
- **Q / ESC**: Quit application
- **R**: Reset all counters
- **T**: Test buzzer

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Camera Module  │────▶│  Face Detector  │────▶│   EAR / MAR     │
│     (Video)     │     │  (dlib 68-pt)   │     │  Computation    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│  Heart Rate     │────▶│   BPM / HRV     │──────────────┤
│  Sensor (GPIO)  │     │  Calculation    │              │
└─────────────────┘     └─────────────────┘              │
                                                         ▼
                                               ┌─────────────────┐
                                               │ Fatigue Fusion  │
                                               │    Engine       │
                                               │ (Weighted Sum)  │
                                               └────────┬────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                               │                               │
                        ▼                               ▼                               ▼
               ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
               │     Buzzer      │            │    Dashboard    │            │   Data Logger   │
               │  (Graded Alert) │            │   (Real-time)   │            │   (Optional)    │
               └─────────────────┘            └─────────────────┘            └─────────────────┘
```

## Fatigue Indicators

### Ocular (EAR-based)
- **Eye Aspect Ratio (EAR)**: Measures eye openness; < 0.21 indicates closure
- **Blink Rate**: Normal 15-20/min; elevated rate indicates fatigue
- **PERCLOS**: Percentage of eye closure over time; > 15% indicates fatigue
- **Micro-sleep**: Eyes closed ≥ 0.5 seconds triggers critical alert

### Behavioral (MAR-based)
- **Mouth Aspect Ratio (MAR)**: Detects yawning when > 0.6
- **Yawn Frequency**: Tracked over 5-minute window

### Physiological (HBPM-based)
- **Heart Rate**: Low HR (< 55 BPM) indicates drowsiness
- **HRV (RMSSD)**: Low variability indicates fatigue

## Fusion Weights

| Indicator       | Weight |
|-----------------|--------|
| Blink Rate      | 15%    |
| EAR Duration    | 20%    |
| PERCLOS         | 15%    |
| Yawn Frequency  | 15%    |
| Heart Rate      | 15%    |
| HRV             | 10%    |
| Micro-sleep     | 10%    |

## Alert Levels

| Level    | Fatigue Score | Buzzer Pattern              |
|----------|---------------|-----------------------------|
| Low      | 30-50%        | 2 short beeps               |
| Medium   | 50-70%        | 3 medium beeps              |
| High     | 70-85%        | 5 rapid beeps               |
| Critical | > 85%         | Continuous alarm            |
| Micro-sleep | (Priority) | Loud continuous alarm       |

## Configuration

All parameters are configurable in [config.py](config.py):
- Camera settings (resolution, FPS)
- EAR/MAR thresholds
- Heart rate thresholds
- Fusion weights
- Alert patterns
- Dashboard appearance

## Project Structure

```
brevet_fatigue/
├── main.py                 # Main application entry point
├── config.py               # System configuration
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── modules/
│   ├── __init__.py
│   ├── face_detector.py    # Face/landmark detection, EAR/MAR
│   ├── heart_rate_sensor.py # HBPM sensor interface
│   ├── buzzer_alert.py     # GPIO buzzer control
│   ├── fatigue_fusion.py   # Multi-modal fusion engine
│   └── dashboard.py        # Real-time visualization
├── models/
│   └── shape_predictor_68_face_landmarks.dat  # dlib model (download)
├── logs/
│   └── fatigue_detection.log
└── data/                   # Optional data logging
```

## Troubleshooting

### Camera not detected
```bash
# Check if camera is enabled
sudo raspi-config  # Interface Options → Camera → Enable

# Test camera
libcamera-hello --timeout 5000
```

### GPIO permission errors
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
# Logout and login again
```

### dlib installation issues
```bash
# Install build dependencies
sudo apt install cmake libopenblas-dev liblapack-dev libx11-dev

# Install dlib with pip (may take 10-20 minutes)
pip install dlib
```

## License

Source code: [MIT License](LICENSE). The dlib landmark model has its own
terms (non-commercial, see [DATA.md](DATA.md)).

## References

- dlib facial landmark detection: http://dlib.net/
- Eye Aspect Ratio (EAR) paper: Soukupová & Čech, 2016
- PERCLOS standard: FHWA drowsy driver detection research
