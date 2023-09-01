import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('D:/BTP/CODE/Videos/track_video.mp4')

# Initialize ROI variables
roi_top_left = None
roi_bottom_right = None
roi_defined = False

# Function for mouse callback
def draw_roi(event, x, y, flags, param):
    global roi_top_left, roi_bottom_right, roi_defined
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_defined = False
        roi_top_left = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_bottom_right = (x, y)
        roi_defined = True

# Create a window and set the mouse callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_roi)

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection to emphasize blade edges
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if ROI is defined
    if roi_defined:
        detected_blade_tips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Filter out small contours (noise)
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if roi_top_left[0] < x+w < roi_bottom_right[0] and roi_top_left[1] < y+h < roi_bottom_right[1]:
                blade_tip_coordinates = (x + w // 2, y + h)
                detected_blade_tips.append(blade_tip_coordinates)
        
        # Calculate the center of the fan
        fan_center = ((roi_top_left[0] + roi_bottom_right[0]) // 2, (roi_top_left[1] + roi_bottom_right[1]) // 2)
        
        # Calculate the maximum distance from the center of the fan
        # max_distance = max(np.linalg.norm(np.array(tip) - np.array(fan_center)) for tip in detected_blade_tips)
        if detected_blade_tips:
            max_distance = max(np.linalg.norm(np.array(tip) - np.array(fan_center)) for tip in detected_blade_tips)
        else:
            max_distance = 0.0  # Set a default value if no blade tips are detected


        # Print detected blade tip coordinates when they reach maximum distance
        for tip in detected_blade_tips:
            distance = np.linalg.norm(np.array(tip) - np.array(fan_center))
            if distance >= max_distance - 10:  # Adjust threshold as needed
                print("Blade Tip Detected at:", tip)

    # Draw the ROI if defined
    if roi_defined:
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
