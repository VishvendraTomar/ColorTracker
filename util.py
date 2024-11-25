import numpy as np
import cv2

def get_limit(color=None):
    # Fixed HSV limits for green
    lowerLimit = np.array([35, 50, 50], dtype=np.uint8)  # H: 35-85, S: moderate-high, V: moderate-high
    upperLimit = np.array([85, 255, 255], dtype=np.uint8)  # Saturation and brightness are high for green
    return lowerLimit, upperLimit
