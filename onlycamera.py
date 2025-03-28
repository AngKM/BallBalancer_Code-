import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# Parameters & Setup
# ---------------------------
data_name = "Ball2.png"
data_folder = os.path.join(os.getcwd(), "TestData")
img_path = os.path.join(data_folder, data_name)
correct_ball = 17  # The correct ball index (adjust as needed)



# ---------------------------
# Load Image
# ---------------------------
original_bgr = cv2.imread(img_path)
if original_bgr is None:
    print("Failed to load image from:", img_path)
    exit()

# Create a copy for annotation (we'll use the same image for all drawings)
annotated_img = original_bgr.copy()

# ---------------------------
# PART 1: BALL DETECTION - Hough Circles
# ---------------------------
# Convert to grayscale and blur for circle detection.
gray = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.medianBlur(gray, 5)

# Detect circles using Hough Circle Transform.
circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=18,
    maxRadius=30
)

# If circles are detected, annotate them on the image and print details.
if circles is not None:
    circles = np.uint16(np.around(circles))
    print("All detected circles:", circles)
    
    # Loop over each detected circle and label it.
    for i, (x_center, y_center, radius) in enumerate(circles[0]):
        print(f"Circle {i}: center=({x_center},{y_center}), radius={radius}")
        # Draw the circle's outer boundary (blue) and its center (yellow).
        cv2.circle(annotated_img, (x_center, y_center), radius, (255, 0, 0), 2)
        cv2.circle(annotated_img, (x_center, y_center), 2, (0, 255, 255), -1)
        # Label the circle number at its center (green).
        cv2.putText(annotated_img,
                    str(i),
                    (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
    
    # Print the details for the selected correct ball.
    if correct_ball < len(circles[0]):
        print("Selected ball details:", circles[0][correct_ball])
    else:
        print("Selected ball index out of range.")
else:
    print("No circles detected using Hough Circles.")

# ---------------------------
# PART 2: RED BASE DETECTION
# ---------------------------
# Convert the original image to HSV for red detection.
hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)

# Define red thresholds in HSV.
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2

# Morphological cleanup.
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours.
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    print("No red base detected.")
    exit()

# Sort by area and pick the largest.
contours = sorted(contours, key=cv2.contourArea, reverse=True)
base_contour = contours[0]

# Compute moments to find the red base center.
M_base = cv2.moments(base_contour)
if M_base["m00"] != 0:
    base_cx = int(M_base["m10"] / M_base["m00"])
    base_cy = int(M_base["m01"] / M_base["m00"])
    base_center = (base_cx, base_cy)
    # Draw a small green circle at the base center.
    cv2.circle(annotated_img, base_center, 5, (0, 255, 0), -1)
    print(f"Base center (in pixels): {base_center}")
else:
    print("Could not compute base center.")
    exit()

# ---------------------------
# PART 3: COMPUTE DISTANCE BETWEEN BASE CENTER AND A BALL CENTER
# ---------------------------
# Here we use the first detected circle (or you can choose another by index).
if circles is not None:
    # For example, using the first detected circle:
    x_center, y_center, radius = circles[0][0]
    ball_center = (int(x_center), int(y_center))
    errorX = int(x_center) - base_cx
    errorY = int(y_center) - base_cy
    print(f"Ball center (in pixels): {ball_center}")
    print(f"Pixel difference from base center to ball center: (errorX={errorX}, errorY={errorY})")
else:
    print("No ball detected for error computation.")

# ---------------------------
# DISPLAY THE FINAL ANNOTATED IMAGE
# ---------------------------
output_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(output_rgb)
plt.title("Annotated Image: Base and Ball Detection")
plt.axis('off')
plt.show()
