#!/usr/bin/env python3
"""
Camera Test Script for Raspberry Pi Camera Module 2
====================================================
Tests the camera using picamera2 before running the full application.
"""

import sys

def test_camera():
    """Test the Pi Camera using picamera2."""
    print("=" * 60)
    print("Raspberry Pi Camera Test")
    print("=" * 60)
    
    # Check if picamera2 is available
    try:
        from picamera2 import Picamera2
        print("✓ picamera2 module loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import picamera2: {e}")
        print("  Install with: pip install picamera2")
        return False
    
    # List available cameras
    try:
        cameras = Picamera2.global_camera_info()
        print(f"\nDetected cameras: {len(cameras)}")
        for i, cam in enumerate(cameras):
            print(f"  Camera {i}: {cam}")
        
        if not cameras:
            print("\n✗ No cameras detected!")
            print("  Make sure the camera ribbon cable is properly connected")
            print("  and the camera is enabled in raspi-config")
            return False
            
    except Exception as e:
        print(f"\n✗ Error listing cameras: {e}")
        return False
    
    # Try to initialize camera
    print("\nInitializing camera...")
    try:
        picam2 = Picamera2()
        
        # Configure for preview/video
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        print("✓ Camera configured: 640x480 RGB888")
        
        # Start camera
        picam2.start()
        print("✓ Camera started")
        
        # Capture test frame
        import time
        time.sleep(0.5)  # Let camera warm up
        
        frame = picam2.capture_array()
        print(f"✓ Test frame captured: {frame.shape}")
        
        # Stop camera
        picam2.stop()
        picam2.close()
        print("✓ Camera stopped and closed")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Camera is working correctly!")
        print("=" * 60)
        print("\nYou can now run the fatigue detection system:")
        print("  python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Camera error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check camera ribbon cable connection")
        print("  2. Run: sudo raspi-config -> Interface Options -> Camera -> Enable")
        print("  3. Run: rpicam-hello to test with libcamera")
        print("  4. Check dmesg for errors: dmesg | grep -i camera")
        return False


if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
