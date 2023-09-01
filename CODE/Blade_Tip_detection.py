import cv2
import numpy as np
import csv
from collections import deque

# Global variables
blade_tip_coordinates_list = []
frame_buffer = deque(maxlen=5)  # Buffer size of 5 frames
prev_frame = None
prev_blade_tips = []
vertical_displacements_list = []  # To store vertical displacements for analysis


def preprocess_frame(frame, num_frames_to_average=5):
    # # Create a frame buffer as an attribute of the function
    # if not hasattr(preprocess_frame, "frame_buffer"):
    #     preprocess_frame.frame_buffer = []

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

    # # Append the current frame to the frame buffer
    # preprocess_frame.frame_buffer.append(gray_frame)

    # # Apply temporal averaging to reduce noise and vibrations
    # if len(preprocess_frame.frame_buffer) < num_frames_to_average:
    #     return gray_frame
    # else:
    #     averaged_frame = np.mean(preprocess_frame.frame_buffer, axis=0).astype(np.uint8)
    #     return averaged_frame


def detect_blade_tips(frame):
    # Set the threshold value (you may need to adjust this)
    threshold_value = 100

    # Define a placeholder value for min_area_threshold
    min_area_threshold = 100  # You can adjust this value based on your data

    # Threshold the frame to create a binary image
    _, binary_frame = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store the detected blade tip coordinates
    blade_tip_coordinates = []

    # Loop through the detected contours
    for contour in contours:
        # Filter contours based on area or other criteria if needed
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area_threshold:
            continue

        # Find the centroid of the contour
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Append the centroid coordinates to the list
            blade_tip_coordinates.append((cx, cy))

    return blade_tip_coordinates

def detect_blade_orientation(frame):
    # Apply Canny edge detection to obtain edges
    edges = cv2.Canny(frame, threshold1=50, threshold2=150)

    # Apply Hough Line Transform to detect major lines (blades)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

    if lines is not None:
        # Calculate the average angle of detected lines
        angles = [line[0][1] for line in lines]
        avg_angle = np.mean(angles)

        # Convert average angle to degrees
        avg_angle_degrees = np.degrees(avg_angle)

        # Since blades are usually vertical, we can approximate the orientation
        # as perpendicular to the average angle
        blade_orientation_degrees = (avg_angle_degrees + 90) % 180
    else:
        # If no lines are detected, set a default value for orientation
        blade_orientation_degrees = 0

    return blade_orientation_degrees


def track_blade_tips(prev_frame, curr_frame, prev_blade_tips):
    # Define parameters for KLT feature tracking
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find blade tips in the previous frame
    prev_blade_points = np.array(prev_blade_tips, dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow using KLT feature tracker
    curr_blade_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_blade_points, None, **lk_params)

    # Filter out points with low tracking quality (status = 1)
    curr_blade_tips = [tuple(pt[0]) for pt, stat in zip(curr_blade_points, status) if stat == 1]

    return curr_blade_tips


def track_features(prev_frame, curr_frame, prev_features):
    # Define parameters for KLT feature tracking
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find features in the previous frame
    prev_features = np.array(prev_features, dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow using KLT feature tracker
    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_features, None, **lk_params)

    # Filter out points with low tracking quality (status = 1)
    curr_features = [tuple(pt[0]) for pt, stat in zip(curr_features, status) if stat == 1]

    return curr_features


def calculate_vertical_displacements(curr_blade_tips, prev_blade_tips):
    vertical_displacements = [curr_tip[1] - prev_tip[1] for curr_tip, prev_tip in zip(curr_blade_tips, prev_blade_tips)]
    return vertical_displacements

def main():

    global prev_frame, prev_blade_tips

    # Load the video
    video_path = "D:/BTP/CODE/Videos/80rpm.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame, frame_buffer)

        # Detect blade tips
        blade_tip_coordinates = detect_blade_tips(processed_frame)
        blade_tip_coordinates_list.append(blade_tip_coordinates)

        # # Store the frame in the buffer
        # frame_buffer.append(processed_frame.copy())

        # # Perform processing on the buffered frames
        # for buffered_frame in frame_buffer:
        #     # Detect blade tips
        #     blade_tip_coordinates = detect_blade_tips(buffered_frame)
        #     blade_tip_coordinates_list.append(blade_tip_coordinates)

        # Optionally, visualize the detected blade tips on the frame
        for cx, cy in blade_tip_coordinates:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        if prev_frame is not None:
            # Track blade tips from the previous frame to the current frame
            curr_blade_tips = track_blade_tips(prev_frame, processed_frame, prev_blade_tips)
            
            # Optionally, visualize the tracked features on the frame
            for tip in curr_blade_tips:
                cv2.circle(frame, tip, 3, (0, 255, 0), -1)


            # Calculate vertical displacements of blade tips
            # vertical_displacements = [curr_tip[1] - prev_tip[1] for curr_tip, prev_tip in zip(curr_blade_tips, prev_blade_tips)]
            # Calculate vertical displacements of blade tips
            vertical_displacements = calculate_vertical_displacements(curr_blade_tips, prev_blade_tips)
            # vertical_displacements_list.extend(vertical_displacements)
            
            # Optionally, visualize the vertical displacements as text on the frame
            for tip, displacement in zip(curr_blade_tips, vertical_displacements):
                cv2.putText(frame, f"{displacement:.2f}", (int(tip[0]) + 10, int(tip[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Iterate through each blade tip and display their displacements
            for i, (cx, cy) in enumerate(curr_blade_tips):
                blade_id = i + 1  # Unique identifier for each blade (1-indexed)
                displacement = vertical_displacements[i]

                # Print the displacement of each blade on the frame
                cv2.putText(frame, f"Blade {blade_id}: {displacement:.2f}", (20, 40 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA) 

        

        prev_frame = processed_frame.copy()
        prev_blade_tips = blade_tip_coordinates



        # Display the frame
        cv2.imshow("Helicopter Blades", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break
    

    # Store the blade tip coordinates in a CSV file
    save_blade_tip_coordinates_to_csv(blade_tip_coordinates_list)

    # Perform analysis on the vertical displacements
    max_displacement = np.max(vertical_displacements_list)
    min_displacement = np.min(vertical_displacements_list)
    average_displacement = np.mean(vertical_displacements_list)
    displacement_amplitude = max_displacement - min_displacement

    print("Max Displacement:", max_displacement)
    print("Min Displacement:", min_displacement)
    print("Average Displacement:", average_displacement)
    print("Displacement Amplitude:", displacement_amplitude)


    cap.release()
    cv2.destroyAllWindows()


def save_blade_tip_coordinates_to_csv(blade_tip_coordinates_list):
    csv_file_path = "blade_tip_coordinates.csv"

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Blade Tip X", "Blade Tip Y"])

        for frame_num, blade_tip_coordinates in enumerate(blade_tip_coordinates_list, start=1):
            for cx, cy in blade_tip_coordinates:
                writer.writerow([frame_num, cx, cy])

if __name__ == "__main__":
    main()