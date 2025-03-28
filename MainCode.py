import cv2
import numpy as np
import os
import json

# ---------------------------
# SETTINGS: Choose input mode: "photo", "video", or "camera"
# ---------------------------
mode = "video"      # Change to "photo" or "camera" as needed.
data_folder = os.path.join(os.getcwd(), "TestData")
photo_name = "Ball1.png"     # Used if mode=="photo"  
video_name = "BallVid1.mp4"    # Used if mode=="video"
correct_ball = 17  # Index for the "correct ball" (if applicable)
data = {
    "errorX": 0,
    "errorY": 0,
    "CenterBaseX": 474,  # default value; change as needed
    "CenterBaseY": 314,
}

# ---------------------------
# Helper Functions: Detection routines for red base and ball
# ---------------------------
def detect_red_base(frame):
    """Detect the red base in the frame and return its center (x, y) if found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        base_contour = contours[0]
        M = cv2.moments(base_contour)
        if M["m00"] != 0:
            base_cx = int(M["m10"] / M["m00"])
            base_cy = int(M["m01"] / M["m00"])
            return (base_cx, base_cy)
    return None

def detect_ball(frame):
    """Detect the ball using HoughCircles and return the first circle's center, radius, and all circles."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
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
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Return first circle and all circles.
        x_center, y_center, radius = circles[0][0]
        return (int(x_center), int(y_center), int(radius)), circles
    return None, None

def annotate_frame(frame, base_center, ball_data, circles=None):
    """Annotate frame with detected base, ball, and optionally label all circles."""
    annotated = frame.copy()
    if base_center is not None:
        cv2.circle(annotated, base_center, 5, (0, 255, 0), -1)
        cv2.putText(annotated, "Base", (base_center[0]-30, base_center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ball_data is not None:
        ball_center, radius = (ball_data[0], ball_data[1]), ball_data[2]
        cv2.circle(annotated, ball_center, radius, (255, 0, 0), 2)
        cv2.circle(annotated, ball_center, 2, (0, 255, 255), -1)
        cv2.putText(annotated, "Ball", (ball_center[0]-30, ball_center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Optionally, annotate all detected circles with their index.
    if circles is not None:
        for i, (x_center, y_center, radius) in enumerate(circles[0]):
            cv2.putText(annotated,
                        str(i),
                        (x_center, y_center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
    # Compute and annotate error if both centers found.
    if base_center is not None and ball_data is not None:
        errorX = ball_data[0] - base_center[0]
        errorY = ball_data[1] - base_center[1]
        data["errorX"] = errorX
        data["errorY"] = errorY
        # Write to JSON file.
        try:
            with open("data.json", "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error accessing JSON file: {e}")
    return annotated

# ---------------------------
# MODE: Photo, Video, or Camera
# ---------------------------
if mode == "photo":
    # Process a single photo.
    img_path = os.path.join(data_folder, photo_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load photo:", img_path)
        exit()
    base_center = detect_red_base(frame)
    if base_center is not None:
        data["CenterBaseX"] = base_center[0]
        data["CenterBaseY"] = base_center[1]
    else:
        print("No base detected in photo. Using default base center values.")
    try:
        with open("data.json", "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error accessing JSON file: {e}")
    ball_data, circles_found = detect_ball(frame)
    annotated_frame = annotate_frame(frame, base_center, ball_data, circles_found)
    cv2.imshow("Photo - Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "video" or mode == "camera":
    # Choose video file or camera.
    if mode == "video":
        cap = cv2.VideoCapture(os.path.join(data_folder, video_name))
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open capture device.")
        exit()

    paused = False
    current_frame = 0

    # If using a video file, try to get total frames.
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if mode == "video" else None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("No more frames or failed to grab frame.")
                break
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # If paused, waitKey with no frame read.
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # next frame
                ret, frame = cap.read()
                if ret:
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    continue
            elif key == ord('p') or key == ord(' '):  # resume
                paused = False
                continue
            elif key == 81:  # Left arrow key for previous frame (video only)
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

        # Process detection on the frame.
        base_center = detect_red_base(frame)
        if base_center is not None:
            data["CenterBaseX"] = base_center[0]
            data["CenterBaseY"] = base_center[1]
        else:
            print("No base detected in current frame. Retaining previous base center values.")
        try:
            with open("data.json", "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error accessing JSON file: {e}")
        ball_data, circles_found = detect_ball(frame)
        annotated_frame = annotate_frame(frame, base_center, ball_data, circles_found)

        # Show frame and print error if available.
        cv2.imshow("Ball Balancing - Detection", annotated_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused
        elif key == 83 and paused:  # Right arrow key (83) for next frame while paused.
            ret, frame = cap.read()
            if ret:
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                continue
        elif key == 81 and paused:  # Left arrow key (81) for previous frame while paused.
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
