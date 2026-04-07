# Real-Time Object Tracking for Computer Control

A system that controls the computer cursor using the position of a colored object tracked in real-time using a webcam.

## Overview

This project implements a computer vision system that:
1. Captures video frames from a webcam
2. Detects a target color using HSV color space filtering (live-adjustable)
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
Color Mask (HSV Thresholding)
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

### Color Thresholding

Pixels are accepted when their HSV values are inside the configurable lower and upper bounds:

```
mask(i, j) = 1 if lower <= hsv(i, j) <= upper else 0
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Colored object (paper, cloth, etc.)

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

- **Move selected colored object**: Controls the cursor
- **Press 'q'**: Quit the application
- **Ctrl+C**: Emergency stop

### Display Windows

- **Object Tracking**: Shows the camera feed with detected centroid
- **Color Mask**: Shows the binary mask for the currently selected HSV range
- **Tracking Controls + Color Preview**: Tkinter panel for lower/upper HSV, min contour area, color swatches, and accepted-range gradient

## Configuration

### Live Color Range Tuning

If the system doesn't detect your object well, tune thresholds in the **Tracking Controls + Color Preview** window while the tracker is running:

- Lower HSV sliders: minimum accepted hue/saturation/value
- Upper HSV sliders: maximum accepted hue/saturation/value
- Min Contour Area slider: filters out small noise blobs

Default HSV values are initialized in `ObjectTracker.__init__()` and can then be adjusted live from the control panel.

## Technical Details

### Libraries Used

- **OpenCV**: Image capture and computer vision processing
- **NumPy**: Numerical operations and matrix calculations
- **Pynput**: Mouse control
- **screeninfo**: Screen resolution detection
