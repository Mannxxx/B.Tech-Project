import cv2
import numpy as np

def detect_fan_blade_tips(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale for Hough Line Transform
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Use Canny edge detection to find edges in the frame
        edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)

        # Use Hough Line Transform to detect lines (fan blades)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        # lines = cv2.HoughLinesP(edges,1,np.pi/180,90,minLineLength,maxLineGap)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame with detected lines (representing fan blade tips)
        cv2.imshow("Detected Fan Blade Tips", frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'your_video_file_path.mp4' with the actual path to your video file
    video_file_path = "D:/BTP/CODE/Videos/track_video.mp4"
    detect_fan_blade_tips(video_file_path)





# ------------------------------------------------------
# Edge detection (except curve lines)  ---solidworks 
# ------------------------------------------------------

# import cv2
# import numpy as np

# def detect_fan_blade_tips(video_path):
#     cap = cv2.VideoCapture(video_path)

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Convert the frame to grayscale for Hough Line Transform
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply GaussianBlur to reduce noise and improve edge detection
#         blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

#         # Use Canny edge detection to find edges in the frame
#         edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)

#         # Use Hough Line Transform to detect lines (fan blades)
#         lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
#         # lines = cv2.HoughLinesP(edges,1,np.pi/180,90,minLineLength,maxLineGap)

#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Display the frame with detected lines (representing fan blade tips)
#         cv2.imshow("Detected Fan Blade Tips", frame)

#         # Press 'q' to quit the video display
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and close the display window
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Replace 'your_video_file_path.mp4' with the actual path to your video file
#     video_file_path = "D:/BTP/CODE/Videos/120rpm.mp4"
#     detect_fan_blade_tips(video_file_path)















# import cv2
# import numpy as np

# def detect_fan_blade_tips(video_path):
#     cap = cv2.VideoCapture(video_path)

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Convert the frame to grayscale for contour detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply GaussianBlur to reduce noise and improve contour detection
#         blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

#         # Use Canny edge detection to find edges in the frame
#         edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)

#         # Find contours in the edges
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Loop over the contours and fit circles around them
#         for contour in contours:
#             if cv2.contourArea(contour) > 1000:
#                 (x, y), radius = cv2.minEnclosingCircle(contour)
#                 center = (int(x), int(y))
#                 radius = int(radius)
#                 cv2.circle(frame, center, radius, (0, 255, 0), 4)

#         # Display the frame with detected circles (representing fan blade tips)
#         cv2.imshow("Detected Fan Blade Tips", frame)

#         # Press 'q' to quit the video display
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and close the display window
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Replace 'your_video_file_path.mp4' with the actual path to your video file
#     video_file_path = "D:/BTP/CODE/Videos/fast.mp4"
#     detect_fan_blade_tips(video_file_path)
