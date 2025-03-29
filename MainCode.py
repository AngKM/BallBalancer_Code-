import cv2
import numpy as np
import os
import json

# ---------------------------
# SETTINGS: Choose input mode: "photo", "video", or "camera"
# ---------------------------
mode = "video"      # Change to "photo", "video", or "camera"
data_folder = os.path.join(os.getcwd(), "NewBallVid")
photo_name = "Ball1.png"   # Used if mode=="photo"
video_name = "NewBallVid1.mp4"  # Used if mode=="video"
correct_ball = 17          # Index for the "correct ball" (if applicable)

# JSON data defaults
data = {
    "errorX": 0,
    "errorY": 0,
    "CenterBaseX": 474,  # default value; change as needed
    "CenterBaseY": 314,
}

# Global variable to hold last known ball detection
prev_ball = None

# ---------------------------
# Helper Functions
# ---------------------------
def detect_red_base(frame):
    """Detect the red (or pinkish-red) base in the frame and return its center (x, y) and contour if found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjust these HSV thresholds to match your pinkish/red base
    lower_red1 = np.array([0, 70, 70])    # H=0, S=70, V=70
    upper_red1 = np.array([10, 255, 255]) # H=10, S=255, V=255
    lower_red2 = np.array([160, 70, 70])  # H=160, S=70, V=70
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for the largest red region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        base_contour = contours[0]
        M = cv2.moments(base_contour)
        if M["m00"] != 0:
            base_cx = int(M["m10"] / M["m00"])
            base_cy = int(M["m01"] / M["m00"])
            return (base_cx, base_cy), base_contour
    return None, None

def detect_ball(frame):
    """
    Detect the ball using HoughCircles and return:
    - (x_center, y_center, radius) of the first circle
    - all circles found
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=18,
        maxRadius=30
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x_center, y_center, radius = circles[0][0]
        return (int(x_center), int(y_center), int(radius)), circles
    return None, None

def detect_ball_roi(frame, roi_center, roi_size=50):
    """
    Detect the ball within a region of interest (ROI) around the roi_center.
    roi_size defines half the width and height of the ROI square.
    """
    x, y = roi_center
    x1 = max(0, x - roi_size)
    y1 = max(0, y - roi_size)
    x2 = min(frame.shape[1], x + roi_size)
    y2 = min(frame.shape[0], y + roi_size)
    
    roi = frame[y1:y2, x1:x2]
    ball_data, circles = detect_ball(roi)
    if ball_data is not None:
        # Adjust the detected coordinates to the original frame's coordinates
        ball_data = (ball_data[0] + x1, ball_data[1] + y1, ball_data[2])
        # Adjust circle coordinates if needed
        if circles is not None:
            circles[0][:, 0] += x1
            circles[0][:, 1] += y1
    return ball_data, circles

def annotate_frame(frame, base_center, base_contour, ball_data, circles=None):
    """
    Annotate frame with detected base, ball, and optionally label all circles.
    Also updates JSON data with errorX, errorY if both base and ball are found.
    """
    annotated = frame.copy()

    # Draw base center and contour if available
    if base_center is not None:
        cv2.circle(annotated, base_center, 5, (0, 255, 0), -1)
        cv2.putText(annotated, "Base", (base_center[0] - 30, base_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if base_contour is not None:
        cv2.drawContours(annotated, [base_contour], -1, (0, 255, 0), 2)

    # Draw ball if valid
    if ball_data is not None:
        (bx, by), radius = (ball_data[0], ball_data[1]), ball_data[2]
        cv2.circle(annotated, (bx, by), radius, (255, 0, 0), 2)
        cv2.circle(annotated, (bx, by), 2, (0, 255, 255), -1)
        cv2.putText(annotated, "Ball", (bx - 30, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Label all detected circles (optional)
    if circles is not None:
        for i, (cx, cy, r) in enumerate(circles[0]):
            cv2.putText(annotated, str(i), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)

    # Compute and store error if both centers found
    if base_center is not None and ball_data is not None:
        errorX = ball_data[0] - base_center[0]
        errorY = ball_data[1] - base_center[1]
        data["errorX"] = errorX
        data["errorY"] = errorY

        # Write to JSON file
        try:
            with open("data.json", "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error accessing JSON file: {e}")

        # --------------------------
        # DRAW TEXT ON THE FRAME
        # --------------------------
        # Show ball coordinates
        text_ball = f"Ball: ({ball_data[0]}, {ball_data[1]})"
        cv2.putText(annotated, text_ball, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Show distance from base center
        text_err = f"Distance: ({errorX}, {errorY})"
        cv2.putText(annotated, text_err, (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

    return annotated

# ---------------------------
# Main Code: Photo, Video, or Camera
# ---------------------------
if mode == "photo":
    # Process a single photo
    img_path = os.path.join(data_folder, photo_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load photo:", img_path)
        exit()

    base_center, base_contour = detect_red_base(frame)
    if base_center is not None:
        data["CenterBaseX"] = base_center[0]
        data["CenterBaseY"] = base_center[1]
    else:
        print("No base detected in photo. Using default base center values.")

    # Update JSON with base center
    try:
        with open("data.json", "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error accessing JSON file: {e}")

    ball_data, circles_found = detect_ball(frame)
    # Only accept the ball if its center is inside the red base contour
    if ball_data is not None and base_contour is not None:
        inside = cv2.pointPolygonTest(base_contour, (ball_data[0], ball_data[1]), False)
        if inside < 0:
            ball_data = None
            circles_found = None

    annotated_frame = annotate_frame(frame, base_center, base_contour, ball_data, circles_found)

    cv2.imshow("Photo - Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "video" or mode == "camera":
    # Video file or camera
    if mode == "video":
        cap = cv2.VideoCapture(os.path.join(data_folder, video_name))
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open capture device.")
        exit()

    paused = False
    current_frame = 0

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if mode == "video" else None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("No more frames or failed to grab frame.")
                break
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # Paused: wait for key input
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # Next frame
                ret, frame = cap.read()
                if ret:
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    continue
            elif key in [ord('p'), ord(' ')]:  # Resume
                paused = False
                continue
            elif key == 81:  # Left arrow (previous frame in video mode)
                if mode == "video":
                    prev_frame = max(current_frame - 2, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                    ret, frame = cap.read()
                    if ret:
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    else:
                        continue
            elif key == ord('q'):
                break

        # Detect base and ball
        base_center, base_contour = detect_red_base(frame)
        if base_center is not None:
            data["CenterBaseX"] = base_center[0]
            data["CenterBaseY"] = base_center[1]
        else:
            print("No base detected in current frame. Retaining previous base center values.")

        # Update JSON with base center
        try:
            with open("data.json", "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error accessing JSON file: {e}")

        # Full frame ball detection
        ball_data, circles_found = detect_ball(frame)
        # If not found and we have a previous ball, try ROI detection
        if ball_data is None and prev_ball is not None:
            ball_data, circles_found = detect_ball_roi(frame, (prev_ball[0], prev_ball[1]), roi_size=50)

        # Only accept the ball if its center is inside the red base contour
        if ball_data is not None and base_contour is not None:
            inside = cv2.pointPolygonTest(base_contour, (ball_data[0], ball_data[1]), False)
            if inside < 0:
                ball_data = None
                circles_found = None

        # Update previous ball if detected
        if ball_data is not None:
            prev_ball = (ball_data[0], ball_data[1])

        annotated_frame = annotate_frame(frame, base_center, base_contour, ball_data, circles_found)

        # Show annotated frame with overlaid text for ball location & distance
        cv2.imshow("Ball Balancing - Detection", annotated_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('p'), ord(' ')]:
            paused = not paused
        elif key == 83 and paused:  # Right arrow while paused
            ret, frame = cap.read()
            if ret:
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                continue
        elif key == 81 and paused:  # Left arrow while paused
            if mode == "video":
                prev_frame = max(current_frame - 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                ret, frame = cap.read()
                if ret:
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    continue

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid mode selected. Choose 'photo', 'video', or 'camera'.")
