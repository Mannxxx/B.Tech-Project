import cv2
import numpy as np

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def detect_point(video_path, marker_color=(0, 0, 255)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Convert the color to HSV for easier color detection
    target_color = np.array(marker_color, dtype=np.uint8)
    target_color = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Initialize variables for rotation measurement
    prev_point = None
    rotation_start_frame = 0
    rotations = 0
    frames_per_rotation = []
    prev_distance = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the target color (red in this example)
        lower_bound = np.array([150, 50, 50])
        upper_bound = np.array([180, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # lower_bound = np.array([160, 100, 100])
        # upper_bound = np.array([179, 255, 255])
        # mask2 = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # mask = mask1 + mask2

        # Find contours of the target color in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assuming the largest contour is the target
            contour = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(contour)
            current_point = (int(x), int(y))

            if prev_point is None:
                prev_point = current_point
                                 
            else:
                # Calculate the distance between the current and previous points
                distance = calculate_distance(current_point, prev_point)
                print(distance)
                # Check for the completion of a rotation
                if distance < 280 and not (prev_distance < 300):
                    # Distance started increasing, one rotation is completed
                    frames_per_rotation.append(cap.get(cv2.CAP_PROP_POS_FRAMES) - rotation_start_frame)
                    rotation_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    rotations += 1
                    print('here')
                prev_distance = distance
                
            # Draw a circle at the detected point for visualization
            cv2.circle(frame, current_point, 5, (225, 0, 225), -1)
            cv2.circle(frame, current_point, 20, (225, 0, 225), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(0) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames_per_rotation) > 0:
        average_frames_per_rotation = sum(frames_per_rotation) / len(frames_per_rotation)
        print(f'FPR: {average_frames_per_rotation}')
        # Assuming the video is captured at 30 frames per second
        rpm = 60 / (average_frames_per_rotation / 60)
        return rpm
    else:
        return None

# Example usage:
video_path = "C:/Users/Mansi/Desktop/BTP/CODE/Videos/speed2.mp4"
rpm = detect_point(video_path)
if rpm is not None:
    print(f"Measured RPM: {rpm}")
else:
    print("Point detection failed or no rotations detected.")