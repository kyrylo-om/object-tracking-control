# Real-Time Object Tracking for Computer Control

A system that controls the computer cursor using the position of a colored object tracked in real-time using a webcam.

## Overview

This project implements a computer vision system that:
1. Captures video frames from a webcam
2. Detects a blue object using HSV color space filtering
3. Tracks the object's centroid position
4. Maps camera coordinates to screen coordinates using linear transformation
5. Controls the mouse cursor in real-time

## System Architecture

The system consists of five main stages:

```
Webcam Frame
     ↓
Color Space Conversion (BGR → HSV)
     ↓
Blue Pixel Mask
     ↓
Contour Detection
     ↓
Centroid Calculation
     ↓
Coordinate Mapping (Linear Transformation)
     ↓
Mouse Cursor Movement
```

## Mathematical Foundation

### Coordinate Transformation

The system uses a linear scaling transformation to map camera coordinates to screen coordinates:

```
A = [ sx   0 ]    where sx = screen_width / camera_width
    [  0  sy ]          sy = screen_height / camera_height

screen_position = A × camera_position

[screen_x]   [sx   0 ] [cx]
[screen_y] = [ 0  sy ] [cy]
```

### Movement Smoothing

To reduce jitter from noisy detection, the system averages the last N positions:

```
avg_x = (1/N) Σ xi
avg_y = (1/N) Σ yi
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Blue object (paper, cloth, etc.)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy pynput screeninfo
```

## Usage

### Basic Usage

Run the tracker:

```bash
python cursor_controller.py
```

### Controls

- **Move blue object**: Controls the cursor
- **Press 'q'**: Quit the application
- **Ctrl+C**: Emergency stop

### Display Windows

- **Object Tracking**: Shows the camera feed with detected centroid
- **Blue Mask**: Shows the binary mask highlighting blue pixels

## Configuration

You can adjust tracking parameters by modifying the `ObjectTracker` initialization:

```python
tracker = ObjectTracker(
    smoothing_buffer_size=5,  # Number of frames to average (higher = smoother but slower response)
    min_contour_area=500      # Minimum area to filter noise (lower = more sensitive)
)
```

### Adjusting Blue Color Range

If the system doesn't detect your blue object well, adjust the HSV range in `cursor_controller.py`:

```python
# In ObjectTracker.__init__()
self.lower_blue = np.array([100, 150, 50])   # [Hue, Saturation, Value]
self.upper_blue = np.array([130, 255, 255])
```

**Tips for finding the right values:**
- Hue: 100-130 for blue (adjust based on shade)
- Saturation: 150-255 (higher = more vibrant colors)
- Value: 50-255 (brightness level)

## Features

### Implemented

✅ Real-time webcam frame capture  
✅ HSV color space conversion  
✅ Blue object detection using binary masking  
✅ Contour detection and filtering  
✅ Centroid calculation using image moments  
✅ Linear coordinate transformation  
✅ Mouse cursor control  
✅ Movement smoothing (averaging filter)  
✅ Morphological operations for noise reduction  
✅ Visual feedback with overlay  

### Performance Optimizations

- Frame resolution: 640×480 (adjustable)
- Morphological operations to reduce noise
- Efficient contour filtering
- Runs at ~30 FPS on standard hardware

## Troubleshooting

### Cursor is jittery
- Increase `smoothing_buffer_size` (e.g., 10 instead of 5)
- Ensure good lighting conditions
- Use a larger, solid-colored blue object

### Object not detected
- Adjust HSV color range (see Configuration section)
- Check lighting conditions
- Ensure the object is fully visible in the frame
- Lower the `min_contour_area` threshold

### Camera not opening
- Check that your webcam is connected
- Try changing `camera_index` in `initialize_camera()`
- Close other applications using the webcam

## Technical Details

### Libraries Used

- **OpenCV**: Image capture and computer vision processing
- **NumPy**: Numerical operations and matrix calculations
- **Pynput**: Mouse control
- **screeninfo**: Screen resolution detection

### Algorithm Complexity

- Frame processing: O(n) where n = number of pixels
- Contour detection: O(n)
- Smoothing: O(k) where k = buffer size
- Overall: Real-time performance at 30 FPS

## Future Improvements

Potential enhancements:

- [ ] Kalman filter for better trajectory prediction
- [ ] Optical flow for more robust tracking
- [ ] Hand gesture recognition
- [ ] Volume control based on Y coordinate
- [ ] Multi-object tracking
- [ ] Depth estimation with stereo cameras
- [ ] Configurable color selection (not just blue)
- [ ] GUI for parameter tuning

## License

This project is for educational purposes as part of a Linear Algebra course project.

## Acknowledgments

This project demonstrates practical applications of linear transformations, coordinate systems, and computer vision in real-time control systems.
