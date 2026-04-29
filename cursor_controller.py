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
from pynput.mouse import Button, Controller
import screeninfo
from threshold_controls_ui import ThresholdControlsUI


class ObjectTracker:
    """Tracks a blue object and controls the mouse cursor."""
    
    def __init__(self, min_contour_area=500, lost_frame_threshold=3):
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

        self.preview_window = "Object Tracking"
        self.mask_window = "Color Mask"

        self.controls_ui = None

        self.last_screen_pos = None
        self.loss_frames = 0
        self.loss_click_armed = False
        self.lost_frame_threshold = max(1, int(lost_frame_threshold))
    
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
        screen_x = int(cx * self.sx)
        screen_y = int(cy * self.sy)
        
        # Clamp values to screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        return screen_x, screen_y
    
    def move_cursor(self, screen_x, screen_y):
        self.mouse.position = (screen_x, screen_y)
    
    def run(self, show_preview=True, show_controls=True):
        """
        Main tracking loop.
        """
        cap = self.initialize_camera()
        
        if cap is None:
            return
        
        print("\nTracking started!")
        print("Move a blue object in front of the camera.")

        if show_preview:
            cv2.namedWindow(self.preview_window, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.mask_window, cv2.WINDOW_AUTOSIZE)
            try:
                cv2.moveWindow(self.preview_window, 20, 20)
                cv2.moveWindow(self.mask_window, 40 + self.camera_width, 20)
            except cv2.error:
                pass

        if show_controls:
            self.controls_ui = ThresholdControlsUI(
                screen_width=self.screen_width,
                camera_width=self.camera_width,
                camera_height=self.camera_height,
                lower_blue=self.lower_blue,
                upper_blue=self.upper_blue,
                min_area=self.min_area,
            )
            show_controls = self.controls_ui.initialize()
            if show_controls:
                print("Adjust values in the Tk control panel.")
                print("Use 'Close & Stop' on the panel or press 'q' in the preview window.")

        print("Press 'q' to quit\n")
        
        try:
            while True:
                if show_controls and not self.controls_ui.process_events():
                    break

                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                if show_controls:
                    self.lower_blue, self.upper_blue, self.min_area = self.controls_ui.get_thresholds()

                pause_tracking = show_controls and self.controls_ui.should_pause_tracking()

                cx, cy, mask = self.detect_blue_object(frame)

                if cx is not None and cy is not None:
                    screen_x, screen_y = self.camera_to_screen_coordinates(cx, cy)

                    if not pause_tracking:
                        self.move_cursor(screen_x, screen_y)

                    self.last_screen_pos = (screen_x, screen_y)
                    if pause_tracking:
                        self.loss_frames = 0
                        self.loss_click_armed = False
                    else:
                        self.loss_frames = 0
                        self.loss_click_armed = True
                    
                    if show_preview:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        
                        text = f"Cam: ({cx}, {cy})"
                        cv2.putText(frame, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        text2 = f"Screen: ({screen_x}, {screen_y})"
                        cv2.putText(frame, text2, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    if pause_tracking:
                        self.loss_frames = 0
                        self.loss_click_armed = False
                    elif self.loss_click_armed and self.last_screen_pos is not None:
                        self.loss_frames += 1
                        if self.loss_frames >= self.lost_frame_threshold:
                            self.mouse.position = self.last_screen_pos
                            self.mouse.click(Button.left, 1)
                            self.loss_click_armed = False
                
                if show_preview:
                    cv2.imshow(self.preview_window, frame)
                    cv2.imshow(self.mask_window, mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            if self.controls_ui is not None:
                self.controls_ui.close()
                self.controls_ui = None

            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed.")


def main():
    tracker = ObjectTracker(min_contour_area=500)
    tracker.run(show_preview=True)


if __name__ == "__main__":
    main()
