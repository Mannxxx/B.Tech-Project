import cv2
import numpy as np
import pandas as pd

# Open the video capture
video_path = 'D:/BTP/CODE/Videos/front_fan.mp4'
cap = cv2.VideoCapture(video_path)

# Create an empty DataFrame to store corner coordinates
corner_data = pd.DataFrame(columns=['Frame', 'Corner Number', 'X', 'Y'])

corner_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    operatedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Modify the data type and apply cv2.cornerHarris
    operatedImage = np.float32(operatedImage)
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # Find the coordinates of corner points
    corner_coords = np.argwhere(dest > 0.01 * dest.max())

    for coord in corner_coords:
        x, y = coord[1], coord[0]
        corner_number += 1

        # Draw a circle and put corner number
        cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)
        cv2.putText(frame, str(corner_number), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Append the corner coordinates to the DataFrame
        corner_data = pd.concat([corner_data, pd.DataFrame({'Frame': [cap.get(cv2.CAP_PROP_POS_FRAMES)], 'Corner Number': [corner_number], 'X': [x], 'Y': [y]})], ignore_index=True)

    # Display the frame with corner points
    cv2.imshow('Video with Numbered Corners', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save corner_data DataFrame to a CSV file
corner_data.to_csv('corner_coordinates.csv', index=False)
