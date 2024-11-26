
---

# Color Object Detection using NumPy and OpenCV

This project detects green-colored objects in a live video feed, creates bounding boxes around them, and filters out noise using **NumPy** for computational tasks and **OpenCV** for video processing.

---

## Features

- Detects green objects using a custom mask creation function with NumPy.
- Filters small objects based on area threshold to remove noise.
- Draws bounding boxes around each detected object using NumPy, without relying on `cv2.findContours`.
- Real-time video feed processing with a webcam.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- OpenCV
- NumPy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/green-object-detection.git
   cd green-object-detection
   ```

2. Install the required Python packages:

   ```bash
   pip install opencv-python numpy
   ```

---

## Usage

1. Run the program:

   ```bash
   python main.py
   ```

2. Allow the webcam to start. The program will detect green objects and draw bounding boxes around them.

3. Press `q` to quit the application.

---

## Project Structure

```
.
├── main.py          # Main script to run the detection
├── requirements.txt # Dependencies list
└── README.md        # Project documentation
```

### Key Components

- **`create_mask`**: Generates a binary mask to detect green pixels using NumPy.
- **`find_bounding_boxes`**: Identifies separate green objects and calculates bounding boxes without using `cv2.findContours`.
- **Real-time Processing**: Captures frames from a webcam, processes them in real-time, and displays the results.

---

## Examples

### Detection in Action

The program processes frames from a live video feed, identifying multiple green objects and bounding them with boxes. It uses area thresholds to ignore small objects (e.g., noise).

---

## Customization

- **HSV Range**: Modify `lower_hue`, `upper_hue`, `lower_saturation`, and `upper_saturation` to adjust the detection for other colors or different lighting conditions.
- **Minimum Area**: Adjust the `min_area` parameter in `find_bounding_boxes` to detect smaller or larger objects.

---


