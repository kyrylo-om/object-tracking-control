#!/usr/bin/env python3
"""
Real-Time Object Tracking for Computer Control
A system that controls the mouse cursor using blue object tracking via webcam.

This implements the five main stages:
1. Frame Capture
2. Color Detection  
3. Object Position Extraction
4. Coordinate Transformation
5. Cursor Control
"""

import cv2
import numpy as np
from pynput.mouse import Controller
import screeninfo


class ObjectTracker:
    """Tracks a blue object and controls the mouse cursor."""
    
    def __init__(self, min_contour_area=500):
        """
        Initialize the object tracker.
        """
        self.mouse = Controller()
        
        # Get screen dimensions
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height
        
        # HSV color range for blue object
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        self.min_area = min_contour_area
        
        self.camera_width = 0
        self.camera_height = 0

        self.sx = 0
        self.sy = 0
    
    def initialize_camera(self, camera_index=0, frame_width=640, frame_height=480):
        """
        Initialize the webcam capture.
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
        
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Get actual frame dimensions
        self.camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate scaling factors for coordinate transformation
        # Transformation matrix A = [sx  0 ]
        #                            [ 0 sy]
        self.sx = self.screen_width / self.camera_width
        self.sy = self.screen_height / self.camera_height
        
        print(f"Camera initialized: {self.camera_width}x{self.camera_height}")
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        print(f"Scaling factors: sx={self.sx:.2f}, sy={self.sy:.2f}")
        
        return cap
    
    def detect_blue_object(self, frame):
        """
        Detect blue object in the frame using HSV color space.
        """
        # Convert BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create binary mask for blue color range
        # M(i,j) = 1 if pixel is blue, 0 otherwise
        mask = cv2.inRange(hsv_frame, self.lower_blue, self.upper_blue)
        

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, mask
        
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area to avoid noise
        area = cv2.contourArea(largest_contour)
        if area < self.min_area:
            return None, None, mask
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None, None, mask
        
        # Calculate centroid coordinates
        # x = [cx, cy]^T - object position in camera coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return cx, cy, mask
    
    def camera_to_screen_coordinates(self, cx, cy):
        """
        Transform camera coordinates to screen coordinates.
        """
        screen_x = int(cx * self.sx)
        screen_y = int(cy * self.sy)
        
        # Clamp values to screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        return screen_x, screen_y
    
    def move_cursor(self, screen_x, screen_y):
        """
        Move the mouse cursor to the specified screen coordinates.
        """
        self.mouse.position = (screen_x, screen_y)
    
    def run(self, show_preview=True):
        """
        Main tracking loop.
        """
        cap = self.initialize_camera()
        
        if cap is None:
            return
        
        print("\nTracking started!")
        print("Move a blue object in front of the camera.")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                cx, cy, mask = self.detect_blue_object(frame)

                if cx is not None and cy is not None:
                    screen_x, screen_y = self.camera_to_screen_coordinates(cx, cy)
                    
                    self.move_cursor(screen_x, screen_y)
                    
                    if show_preview:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        
                        text = f"Cam: ({cx}, {cy})"
                        cv2.putText(frame, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        text2 = f"Screen: ({screen_x}, {screen_y})"
                        cv2.putText(frame, text2, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if show_preview:
                    cv2.imshow("Object Tracking", frame)
                    cv2.imshow("Blue Mask", mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Real-Time Object Tracking for Computer Control")
    print("=" * 60)
    print("\nThis system tracks a blue object and controls the cursor.")
    print("Ensure you have a blue object (paper, cloth, etc.) ready.\n")
    
    tracker = ObjectTracker(min_contour_area=500)
    tracker.run(show_preview=True)


if __name__ == "__main__":
    main()
