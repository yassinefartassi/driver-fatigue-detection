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
- **Eye Aspect Ratio (EAR)**: Measures eye openness; < 0.21 indicates closure, ≥ 0.30 is wide open
- **Blink Rate**: Normal 12-20/min; below 5 or above 30 scores as maximum fatigue
- **PERCLOS**: Share of time with eyes closed over 60 seconds; alert below 8%, severe above 30%
- **Micro-sleep**: Eyes closed ≥ 2 seconds with a face detected triggers the priority alert

### Behavioral (MAR-based)
- **Mouth Aspect Ratio (MAR)**: Detects yawning when > 0.6
- **Yawn Frequency**: Tracked over a 5-minute window; 0-1 is normal, 4 or more is high fatigue

### Physiological (HBPM-based)
- **Heart Rate**: Normal 60-100 BPM; below 50 BPM scores as maximum drowsiness
- **HRV (RMSSD)**: Measured and shown on the dashboard; not part of the fatigue score yet

## Fusion Weights

These are the weights the fusion engine applies (`WEIGHTS` in
[modules/fatigue_fusion.py](modules/fatigue_fusion.py)). The `WEIGHT_*` constants
in `config.py` are not read by the engine.

| Indicator                | Weight |
|--------------------------|--------|
| Blink Rate               | 35%    |
| Heart Rate               | 20%    |
| Yawn Frequency           | 20%    |
| PERCLOS                  | 15%    |
| Current Eye Openness (EAR) | 10%  |

A micro-sleep is not a weighted term: while it is active, the score is raised
to at least 80%.

## Alert Levels

| Level       | Fatigue Score | Buzzer Pattern                    |
|-------------|---------------|-----------------------------------|
| Low         | > 25%         | 2 short beeps                     |
| Medium      | ≥ 40%         | 3 medium beeps                    |
| High        | ≥ 60%         | 5 rapid beeps                     |
| Micro-sleep | (Priority)    | 10 rapid beeps at full volume     |

The dashboard also labels the score: alert (≤ 20%), low, moderate (≥ 40%),
high (≥ 60%) and critical (≥ 80%).

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
