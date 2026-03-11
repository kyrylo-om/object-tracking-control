#!/usr/bin/env python3
"""
Color Calibration Tool

This utility helps you find the correct HSV color range for your blue object.
Move your blue object in front of the camera and adjust the trackbars until
it's highlighted in white on the mask display.
"""

import cv2
import numpy as np


def nothing(x):
    """Dummy callback for trackbars."""
    pass


def main():
    """Run the color calibration tool."""
    print("=" * 60)
    print("HSV Color Calibration Tool")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Place your blue object in front of the camera")
    print("2. Adjust the trackbars until the object is WHITE in the mask")
    print("3. Note down the final HSV values")
    print("4. Press 'q' to quit\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create window with trackbars
    cv2.namedWindow("Trackbars")
    
    # Initial values for blue
    cv2.createTrackbar("L-H", "Trackbars", 100, 180, nothing)  # Lower Hue
    cv2.createTrackbar("L-S", "Trackbars", 150, 255, nothing)  # Lower Saturation
    cv2.createTrackbar("L-V", "Trackbars", 50, 255, nothing)   # Lower Value
    cv2.createTrackbar("U-H", "Trackbars", 130, 180, nothing)  # Upper Hue
    cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)  # Upper Value
    
    print("Calibration started. Adjust trackbars to isolate your blue object.\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get trackbar positions
        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")
        
        # Create mask
        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Show result
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Display current values on frame
        text = f"Lower: [{l_h}, {l_s}, {l_v}]"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        text = f"Upper: [{u_h}, {u_s}, {u_v}]"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        
        # Display windows
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)
        
        # Print current values when 's' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"\nCurrent HSV Range:")
            print(f"lower_blue = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"upper_blue = np.array([{u_h}, {u_s}, {u_v}])")
            print()
        elif key == ord('q'):
            print("\nFinal HSV Range:")
            print(f"lower_blue = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"upper_blue = np.array([{u_h}, {u_s}, {u_v}])")
            print("\nCopy these values to cursor_controller.py")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
