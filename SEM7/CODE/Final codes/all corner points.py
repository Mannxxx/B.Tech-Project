# Parameters of cv2.goodFeaturesToTrack()
# -----------------------------------------
# gray image - image should be a grayscale image
#  maxc - number of corners you want to find.
#  Q - minimum quality of corner below which everyone is rejected
#  maxD -  provide the minimum Euclidean distance between corners detected.
             

import cv2
import numpy as np
import csv
         
# Replace 'input_video.mp4' with the path to your video file
video_path = "D:/BTP/CODE/Videos/gray_video.mp4"
cap = cv2.VideoCapture(video_path)                                 

def nothing(x):
    pass

cv2.namedWindow("Frame")
cv2.createTrackbar("quality", "Frame", 20, 100, nothing)  # Adjust initial quality value
cv2.createTrackbar("max_corners", "Frame", 100, 1000, nothing)  # Adjust max corners

# Open a CSV file for writing the corner coordinates
csv_file = open('corner_coordinates.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'X', 'Y'])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    quality = cv2.getTrackbarPos("quality", "Frame") / 100
    max_corners = cv2.getTrackbarPos("max_corners", "Frame")

    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, 10)  # Adjust minDistance

    if corners is not None:
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                              
            # Write corner coordinates to CSV
            csv_writer.writerow([frame_count, x, y])
   
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(0)
    if key == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Close the CSV file
csv_file.close()




# -------------------------
# Without csv
# -------------------------


# import cv2
# import numpy as np

# # Replace 'input_video.mp4' with the path to your video file
# video_path = 'D:/BTP/CODE/Videos/80rpm.mp4'
# cap = cv2.VideoCapture(video_path)

# def nothing(x):
#     pass

# cv2.namedWindow("Frame")
# cv2.createTrackbar("quality", "Frame", 1, 100, nothing)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     quality = cv2.getTrackbarPos("quality", "Frame")
#     quality = quality / 100 if quality > 0 else 0.01
#     corners = cv2.goodFeaturesToTrack(gray, 100, quality, 20)

#     if corners is not None:
#         corners = np.int0(corners)

#         for corner in corners:
#             x, y = corner.ravel()
#             cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

#     cv2.imshow("Frame", frame)

#     key = cv2.waitKey(0)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()










# --------------------------
# Directly from webcam
# --------------------------


# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# def nothing(x):
# pass

# cv2.namedWindow("Frame")
# cv2.createTrackbar("quality", "Frame", 1, 100, nothing)

# while True:
# _, frame = cap.read()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# quality = cv2.getTrackbarPos("quality", "Frame")
# quality = quality / 100 if quality > 0 else 0.01
# corners = cv2.goodFeaturesToTrack(gray, 100, quality, 20)

# if corners is not None:
# corners = np.int0(corners)

# for corner in corners:
# x, y = corner.ravel()
# cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

# cv2.imshow("Frame", frame)

# key = cv2.waitKey(1)
# if key == 27:
# break

# cap.release()
# cv2.destroyAllWindows()







# ----------------------
# Method-2
# ------------------------


# import cv2
# import numpy as np
# import pandas as pd

# # Open the video capture
# video_path = 'D:/BTP/CODE/Videos/gray2.mp4'
# cap = cv2.VideoCapture(video_path)

# # Create an empty DataFrame to store corner coordinates
# corner_data = pd.DataFrame(columns=['Frame', 'Corner Number', 'X', 'Y'])

# corner_number = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale
#     operatedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Modify the data type and apply cv2.cornerHarris
#     operatedImage = np.float32(operatedImage)
#     dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

#     # Results are marked through the dilated corners
#     dest = cv2.dilate(dest, None)

#     # Find the coordinates of corner points
#     corner_coords = np.argwhere(dest > 0.01 * dest.max())

#     for coord in corner_coords:
#         x, y = coord[1], coord[0]
#         corner_number += 1

#         # Draw a circle and put corner number
#         cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)
#         # cv2.putText(frame, str(corner_number), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Append the corner coordinates to the DataFrame
#         corner_data = pd.concat([corner_data, pd.DataFrame({'Frame': [int(cap.get(cv2.CAP_PROP_POS_FRAMES))], 'Corner Number': [corner_number], 'X': [x], 'Y': [y]})], ignore_index=True)

#     # Display the frame with corner points
#     cv2.imshow('Video with Corners', frame)

#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# # Release the video capture and close windows
# cap.release()
# cv2.destroyAllWindows()

# # Save corner_data DataFrame to a CSV file
# corner_data.to_csv('corner_coordinates.csv', index=False)




# ----------------
# Try this too
# ----------------

# import cv2
# import numpy as np
# import pandas as pd

# # Open the video capture
# video_path = 'D:/BTP/CODE/Videos/gray2.mp4'
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
# ['Frame', 'Corner Number', 'X', 'Y']
#     # Convert the frame to grayscale
#     operatedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Modify the data type and apply cv2.cornerHarris
#     operatedImage = np.float32(operatedImage)
#     dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

#     # Results are marked through the dilated corners
#     dest = cv2.dilate(dest, None)

#     # Find and draw the corners on the original frame
#     frame[dest > 0.01 * dest.max()] = [0, 0, 255]

#     # Display the frame with corner points
#     cv2.imshow('Video with Corners', frame)

#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# # Release the video capture and close windows
# cap.release()
# cv2.destroyAllWindows()