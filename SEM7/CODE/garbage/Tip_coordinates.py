import cv2
import numpy as np

# Global variables for the selected point coordinates
selected_point = None
selected_point_tracked = False

# Function to display the coordinates of the points clicked on the frame
def click_event(event, x, y, flags, params):
    global selected_point, selected_point_tracked

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        selected_point = (x, y)
        selected_point_tracked = False

# Driver function
if __name__ == "__main__":
    # Initialize the video capture
    cap = cv2.VideoCapture("D:/BTP/CODE/Videos/front_fan.mp4")  # Replace with the path to your video file

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Track the selected point using optical flow (if selected)
        if selected_point is not None and not selected_point_tracked:
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Downsample the grayscale frame for optical flow computation
            prev_gray_frame = cv2.pyrDown(gray_frame)
            prev_gray_frame = cv2.pyrUp(prev_gray_frame)

            # Parameters for the Lucas-Kanade optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Convert the selected point to a numpy array
            prev_pts = np.array([selected_point], dtype=np.float32)

            # Calculate optical flow
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_pts, None, **lk_params)

            if status[0][0] == 1:  # If optical flow successfully tracked the point
                selected_point = (int(new_pts[0][0]), int(new_pts[0][1]))
                selected_point_tracked = True

        # Draw a circle at the selected point for visualization
        if selected_point is not None:
            cv2.circle(frame, selected_point, 5, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow('Video Frame', frame)

        # Set mouse handler for the frame
        cv2.setMouseCallback('Video Frame', click_event)

        # Press 'Esc' key to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release video capture and close the window
    cap.release()
    cv2.destroyAllWindows()
