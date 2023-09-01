import cv2
from cv2 import VideoCapture
from cv2 import waitKey

# Global variable to store the coordinates of the clicked point
clicked_point = None

# function to display the coordinates of
# the points clicked on the image
def click_event(event, x, y, flags, params):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print("Clicked Point Coordinates (x, y):", clicked_point)

# driver function
if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 2:
    #     print("Usage: python script_name.py video_file_path")
    #     sys.exit(1)

    video_file_path = "D:/BTP/CODE/Videos/front_fan.mp4"

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("Error: Video file not found or could not be opened.")
        sys.exit(1)

    cv2.namedWindow("Video")

    # setting mouse handler for the video
    # and calling the click_event() function
    cv2.setMouseCallback("Video", click_event)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # displaying the video frame
        cv2.imshow("Video", frame)

        # Pause the video when spacebar is pressed
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):  # Resume when spacebar is pressed again
                    break
                elif key == 27:  # Press 'Esc' to quit the video display
                    break

        # Display clicked point coordinates if available
        if clicked_point is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(clicked_point), clicked_point, font, 1, (0, 0, 255), 2)
            cv2.circle(frame, clicked_point, 5, (0, 0, 255), -1)
            clicked_point = None  # Reset clicked point after displaying

        # Press 'q' to quit the video display
        if key == ord('q') or key == 27:
            break

    # Release the video capture and close the display window
    cap.release()
    cv2.destroyAllWindows()
