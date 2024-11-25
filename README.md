### **Color Recognition in Live Video Using Python, NumPy, and OpenCV**

This process allows you to identify and track specific colors in a live video feed using Python, **NumPy**, and **OpenCV**. Here's a quick overview of how to do it:

1. **Install Libraries**:
   ```bash
   pip install numpy opencv-python
   ```

2. **Capture Video**: Use OpenCV to capture live video from the webcam.
   ```python
   import cv2
   import numpy as np
   cap = cv2.VideoCapture(0)
   ```

3. **Convert to HSV**: Convert each video frame from **BGR** (default) to **HSV** for easier color detection.
   ```python
   hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   ```

4. **Define Color Range**: Set the color range (e.g., red) in HSV to create a **mask**.
   ```python
   lower_red = np.array([160, 100, 100])
   upper_red = np.array([180, 255, 255])
   mask = cv2.inRange(hsv_frame, lower_red, upper_red)
   ```

5. **Apply Mask**: Isolate the color regions using the mask.
   ```python
   color_detected = cv2.bitwise_and(frame, frame, mask=mask)
   ```

6. **Detect Contours**: Find contours to highlight areas with the detected color.
   ```python
   contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   ```

7. **Display the Result**: Show the video with the detected color regions.
   ```python
   cv2.imshow('Detected Color', color_detected)
   ```

This process continuously captures video, isolates the desired color, and highlights it in real-time.

### **Applications**:
- Object tracking
- Gesture recognition
- Robotics

