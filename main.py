# import cv2
# import numpy as np
# from util import get_limit
# from PIL import Image, ImageDraw

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame. Exiting...")
#         break

#     # Convert the captured frame to HSV
#     hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Get HSV limits for detecting green color
#     lowerLimit, upperLimit = get_limit(color=None)

#     # Create a mask for green color
#     mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

#     # Find contours of the detected green area
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Convert the frame to PIL image
#     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(frame_pil)

#     # Draw bounding boxes around detected green areas
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Ignore small noise
#             x, y, w, h = cv2.boundingRect(contour)
#             draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

#     # Convert back to OpenCV format for display
#     frame_with_boxes = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

#     # Show the final frame with bounding boxes
#     cv2.imshow('Detected Green Color with Bounding Box', frame_with_boxes)

#     # Exit loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()




#  using numpy for finding the color 


# import cv2
# import numpy as np

# # Define the green color range in HSV
# lower_limit = np.array([35, 50, 50], dtype=np.uint8)  # Lower bound of green
# upper_limit = np.array([85, 255, 255], dtype=np.uint8)  # Upper bound of green

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame. Exiting...")
#         break

#     # Convert BGR to HSV using NumPy
#     frame_rgb = frame[:, :, ::-1]  # Convert BGR to RGB
#     max_val = frame_rgb.max(axis=2)  # Max value for HSV calculation
#     min_val = frame_rgb.min(axis=2)  # Min value for HSV calculation
#     delta = max_val - min_val  # Difference between max and min

#     # Initialize arrays for HSV components
#     hue = np.zeros_like(max_val, dtype=np.float32)
#     saturation = np.where(max_val == 0, 0, delta / max_val)
#     value = max_val / 255.0

#     # Calculate Hue
#     mask = delta != 0
#     r_eq = (frame_rgb[..., 0] == max_val) & mask
#     g_eq = (frame_rgb[..., 1] == max_val) & mask
#     b_eq = (frame_rgb[..., 2] == max_val) & mask

#     # Use masked array to avoid shape mismatch
#     hue[r_eq] = (60 * ((frame_rgb[..., 1] - frame_rgb[..., 2])[r_eq] / delta[r_eq])) % 360
#     hue[g_eq] = (60 * (2.0 + ((frame_rgb[..., 2] - frame_rgb[..., 0])[g_eq] / delta[g_eq]))) % 360
#     hue[b_eq] = (60 * (4.0 + ((frame_rgb[..., 0] - frame_rgb[..., 1])[b_eq] / delta[b_eq]))) % 360

#     # Scale Hue to 0-180 for OpenCV compatibility
#     hue = (hue / 2).astype(np.uint8)

#     # Combine HSV channels into a NumPy array
#     hsv_image = np.stack([hue, (saturation * 255).astype(np.uint8), (value * 255).astype(np.uint8)], axis=2)

#     # Create mask for green color
#     mask = (hsv_image[..., 0] >= lower_limit[0]) & (hsv_image[..., 0] <= upper_limit[0]) & \
#            (hsv_image[..., 1] >= lower_limit[1]) & (hsv_image[..., 1] <= upper_limit[1]) & \
#            (hsv_image[..., 2] >= lower_limit[2]) & (hsv_image[..., 2] <= upper_limit[2])

#     # Convert mask to uint8 for displaying
#     mask = (mask * 255).astype(np.uint8)

#     # Find contours of the detected color
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw bounding boxes around detected green areas
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the original frame and the mask
#     cv2.imshow("Original Frame", frame)
#     cv2.imshow("Mask", mask)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# adding post processing to find the better color 

import cv2
import numpy as np

# Define HSV range for green color
lower_limit = np.array([35, 50, 50], dtype=np.uint8)  # Lower bound of green
upper_limit = np.array([85, 255, 255], dtype=np.uint8)  # Upper bound of green


lower_hue = 35    # Lower bound for Hue (green)
upper_hue = 85    # Upper bound for Hue (green)
lower_saturation = 50  # Minimum Saturation value to ensure the color is intense enough
upper_saturation = 255  # Maximum Saturation
lower_value = 50  # Minimum Value (brightness)
upper_value = 255  # Maximum Value (brightness)




def create_mask(hsv_image):
    """
    Create a binary mask to detect green color using NumPy without cv2.inRange.
    """
    # Extract Hue, Saturation, and Value channels
    hue = hsv_image[..., 0]  # Hue channel
    saturation = hsv_image[..., 1]  # Saturation channel
    value = hsv_image[..., 2]  # Value channel
    
    # Create conditions for the color range
    hue_condition = (hue >= lower_hue) & (hue <= upper_hue)
    saturation_condition = (saturation >= lower_saturation) & (saturation <= upper_saturation)
    value_condition = (value >= lower_value) & (value <= upper_value)
    
    # Combine the conditions to create the mask (all conditions must be true)
    mask = hue_condition & saturation_condition & value_condition
    
    # Convert the boolean mask to uint8 (0 or 255)
    mask = np.uint8(mask) * 255
    
    return mask



# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert BGR to HSV using NumPy
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    mask = create_mask(hsv_image)

    # Post-processing: Morphological operations
    kernel = np.ones((5, 5), np.uint8)  # Kernel for morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    # Apply Gaussian Blur to smooth mask
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes for contours that meet size criteria
    min_area = 500  # Minimum contour area to consider
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame and the processed mask
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
