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
from collections import deque
import screeninfo


class ObjectTracker:
    """Tracks a blue object and controls the mouse cursor."""
    
    def __init__(self, smoothing_buffer_size=5, min_contour_area=500):
        """
        Initialize the object tracker.
        
        Args:
            smoothing_buffer_size: Number of frames to average for smoothing
            min_contour_area: Minimum contour area to consider valid detection
        """
        # Mouse controller
        self.mouse = Controller()
        
        # Get screen dimensions
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height
        
        # HSV color range for blue object
        # These values can be adjusted based on the blue object being tracked
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # Movement smoothing buffer
        self.position_history = deque(maxlen=smoothing_buffer_size)
        
        # Minimum contour area threshold
        self.min_area = min_contour_area
        
        # Camera properties (will be set when capture starts)
        self.camera_width = 0
        self.camera_height = 0
        
        # Scaling factors (transformation matrix elements)
        self.sx = 0
        self.sy = 0
    
    def initialize_camera(self, camera_index=0, frame_width=640, frame_height=480):
        """
        Initialize the webcam capture.
        
        Args:
            camera_index: Camera device index (default 0)
            frame_width: Desired frame width
            frame_height: Desired frame height
            
        Returns:
            cv2.VideoCapture object or None if failed
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
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Tuple of (centroid_x, centroid_y) or (None, None) if not detected
        """
        # Stage 2: Color Detection - Convert BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create binary mask for blue color range
        # M(i,j) = 1 if pixel is blue, 0 otherwise
        mask = cv2.inRange(hsv_frame, self.lower_blue, self.upper_blue)
        
        # Optional: Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Stage 3: Object Detection - Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, mask
        
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area to avoid noise
        area = cv2.contourArea(largest_contour)
        if area < self.min_area:
            return None, None, mask
        
        # Stage 4: Centroid Calculation using image moments
        M = cv2.moments(largest_contour)
        
        # Avoid division by zero
        if M["m00"] == 0:
            return None, None, mask
        
        # Calculate centroid coordinates
        # x = [cx, cy]^T - object position in camera coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return cx, cy, mask
    
    def smooth_position(self, cx, cy):
        """
        Apply movement smoothing using averaging of recent positions.
        
        Args:
            cx, cy: Current centroid coordinates
            
        Returns:
            Smoothed (avg_x, avg_y) coordinates
        """
        self.position_history.append((cx, cy))
        
        # Calculate mean position
        positions = np.array(self.position_history)
        avg_x = int(np.mean(positions[:, 0]))
        avg_y = int(np.mean(positions[:, 1]))
        
        return avg_x, avg_y
    
    def camera_to_screen_coordinates(self, cx, cy):
        """
        Transform camera coordinates to screen coordinates.
        
        Stage 5: Coordinate Transformation
        screen_position = A * camera_position
        
        Where A = [sx  0 ]
                  [ 0 sy]
        
        Args:
            cx, cy: Camera coordinates
            
        Returns:
            screen_x, screen_y: Screen coordinates
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
        
        Args:
            screen_x, screen_y: Target screen coordinates
        """
        self.mouse.position = (screen_x, screen_y)
    
    def run(self, show_preview=True):
        """
        Main tracking loop.
        
        Args:
            show_preview: Whether to display the camera feed and mask
        """
        cap = self.initialize_camera()
        
        if cap is None:
            return
        
        print("\nTracking started!")
        print("Move a blue object in front of the camera.")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                # Stage 1: Frame Capture
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally for mirror effect (more intuitive)
                frame = cv2.flip(frame, 1)
                
                # Detect blue object and get centroid
                cx, cy, mask = self.detect_blue_object(frame)
                
                if cx is not None and cy is not None:
                    # Apply smoothing
                    smooth_cx, smooth_cy = self.smooth_position(cx, cy)
                    
                    # Transform to screen coordinates
                    screen_x, screen_y = self.camera_to_screen_coordinates(smooth_cx, smooth_cy)
                    
                    # Move cursor
                    self.move_cursor(screen_x, screen_y)
                    
                    if show_preview:
                        # Draw tracking information on frame
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.circle(frame, (smooth_cx, smooth_cy), 15, (255, 0, 0), 2)
                        
                        # Display coordinates
                        text = f"Cam: ({smooth_cx}, {smooth_cy})"
                        cv2.putText(frame, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        text2 = f"Screen: ({screen_x}, {screen_y})"
                        cv2.putText(frame, text2, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if show_preview:
                    # Display frame and mask
                    cv2.imshow("Object Tracking", frame)
                    cv2.imshow("Blue Mask", mask)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
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
    
    # Create tracker with smoothing
    tracker = ObjectTracker(
        smoothing_buffer_size=5,  # Average last 5 positions
        min_contour_area=500       # Minimum area threshold
    )
    
    # Start tracking
    tracker.run(show_preview=True)


if __name__ == "__main__":
    main()
