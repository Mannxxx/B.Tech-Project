import cv2
import numpy as np

# Replace 'input_video.mp4' with the path to your video file
video_path = 'D:/BTP/CODE/Diving board/Diving.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise before edge detection
    blurred_frame = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)
    
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw lines around the detected edges
    frame_with_edges = frame.copy()
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(frame_with_edges, [approx], -1, (0, 255, 0), 2)
    
    cv2.imshow('Edges', frame_with_edges)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
