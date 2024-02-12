# Method-1: Shi-Tomasi Corner Detection Method using OpenCV



**What is a Corner?**  ------- junction of two edges (where an edge is a sudden change in image brightness).

The corners of an image are basically identified as the regions in which there are variations in large intensity of the gradient in all possible dimensions and directions. Corners extracted can be a part of the image features, which can be matched with features of other images, and can be used to extract accurate information.



## Shi-Tomasi Corner Detection –

basic intuition is that corners can be detected by looking for significant change in all direction.

We consider a small window on the image then scan the whole image, looking for corners.

Shifting this small window in any direction would result in a large change in appearance, if that particular window happens to be located on a corner.

![image-20230818115544295](C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230818115544295.png)

Flat regions will have no change in any direction.

![image-20230818115628702](C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230818115628702.png)

If there’s an edge, then there will be no major change along the edge direction.

![image-20230818115636184](C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230818115636184.png)

### Mathematical Overview –

For a window(W) located at (X, Y) with pixel intensity I(X, Y), formula for Shi-Tomasi Corner Detection is –

```
f(X, Y) = Σ (I(Xk, Yk) - I(Xk + ΔX, Yk + ΔY))2  where (Xk, Yk) ϵ W
```

**According to the formula:**
If we’re scanning the image with a window just as we would with a kernel and we notice that there is an area where there’s a major change no matter in what direction we actually scan, then we have a good intuition that there’s probably a corner there.

Calculation of f(X, Y) will be really slow. Hence, we use Taylor expansion to simplify the scoring function, R.

```
R = min(λ1, λ2)
where λ1, λ2 are eigenvalues of resultant matrix
```

**Using `goodFeaturesToTrack()` function –**

> **Syntax :** cv2.goodFeaturesToTrack(gray_img, maxc, Q, minD)
>
> **Parameters :**
> **gray_img –** Grayscale image with integral values
> **maxc –** Maximum number of corners we want(give negative value to get all the corners)
> **Q –** Quality level parameter(preferred value=0.01)
> **maxD –** Maximum distance(preferred value=10)              



### Our Implementation:

```python
import cv2
import numpy as np
import csv

# Replace 'input_video.mp4' with the path to your video file
video_path = 'D:/BTP/CODE/Videos/gray2.mp4'
cap = cv2.VideoCapture(video_path)

def nothing(x):
    pass

cv2.namedWindow("Frame")
cv2.createTrackbar("quality", "Frame", 1, 100, nothing)

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

    quality = cv2.getTrackbarPos("quality", "Frame")
    quality = quality / 100 if quality > 0 else 0.01
    # Change 50 to the no of corners we want to detect 
    corners = cv2.goodFeaturesToTrack(gray, 50, quality, 20)   

    if corners is not None:
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
            # Write corner coordinates to CSV
            csv_writer.writerow([frame_count, x, y])

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Close the CSV file
csv_file.close()
```

We are extracting only the extreme coordinates in +X,-X directions since they will surely be the tip coordinates. The difference in corresponding values of Y coordinates gives the track.

Also, note that the code may give nearly close values to the leftmost and rightmost X since it might happen that there is less change btw 2 frames but it can be clearly seen that the corresponding Y values of all nearest X are almost same. 

To accurately convert pixel coordinates to real-world units like millimeters, you need to perform camera calibration.

### Code to find max and min X: 

```python
import csv

def extract_extremes(csv_file, column_index):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        data = list(reader)
    
    if column_index >= len(header):
        print("Invalid column index.")
        return

    min_value = float('inf')
    max_value = float('-inf')
    min_row = None
    max_row = None

    for row in data:
        try:
            value = float(row[column_index])
            if value < min_value:
                min_value = value
                min_row = row
            if value > max_value:
                max_value = value
                max_row = row
        except ValueError:
            pass  # Ignore rows with non-numeric values in the specified column

    return min_row, max_row

csv_file = 'D:/BTP/corner_coordinates.csv'  # Replace with the path to your CSV file
column_index = 1  # Replace with the index of the column you want to extract extremes from

min_row, max_row = extract_extremes(csv_file, column_index)

if min_row:
    print(f"Minimum value in column {column_index}: {min_row[column_index]}")
    print("Corresponding row:", min_row)
else:
    print("No valid minimum value found.")

if max_row:
    print(f"Maximum value in column {column_index}: {max_row[column_index]}")
    print("Corresponding row:", max_row)
else:
    print("No valid maximum value found.")

```

------

------

------



## Another Method: Corner detection with Harris Corner Detection method using OpenCV

**About the function used:** 

> **Syntax:** cv2.cornerHarris(src, dest, blockSize, kSize, freeParameter, borderType)
> **Parameters:** 
> **src** – Input Image (Single-channel, 8-bit or floating-point) 
> **dest** – Image to store the Harris detector responses. Size is same as source image 
> **blockSize** – Neighborhood size ( for each pixel value blockSize * blockSize neighbourhood is considered ) 
> **ksize** – Aperture parameter for the [Sobel()](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void Sobel(InputArray src,  OutputArray dst,  int ddepth,  int dx,  int dy,  int ksize,  double scale,  double delta,  int borderType)) operator 
> **freeParameter** – Harris detector free parameter 
> **borderType** – Pixel extrapolation method ( the extrapolation mode used returns the coordinate of the pixel corresponding to the specified extrapolated pixel )



For further sub pixel accuracy, we can use cv2.cornerHarris() method.

Also we only want integer coordinates, so we need to do this step too:

```python
corners = np.int0(corners)
```



## Below is the Python implementation : 

```python
import cv2
import numpy as np
import pandas as pd

# Open the video capture
video_path = 'D:/BTP/CODE/Videos/gray2.mp4'
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
        cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)
        # cv2.putText(frame, str(corner_number), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Append the corner coordinates to the DataFrame
        corner_data = pd.concat([corner_data, pd.DataFrame({'Frame': [int(cap.get(cv2.CAP_PROP_POS_FRAMES))], 'Corner Number': [corner_number], 'X': [x], 'Y': [y]})], ignore_index=True)

    # Display the frame with corner points
    cv2.imshow('Video with Corners', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save corner_data DataFrame to a CSV file
corner_data.to_csv('corner_coordinates.csv', index=False)
cv2.destroyAllWindows()
```



## Sources:

[Python | Corner Detection with Shi-Tomasi Corner Detection Method using OpenCV - GeeksforGeeks](https://www.geeksforgeeks.org/python-corner-detection-with-shi-tomasi-corner-detection-method-using-opencv/?ref=lbp)

[Object Detection – Opencv & Deep learning | Video Course [2022\] - Pysource](https://pysource.com/object-detection-opencv-deep-learning-video-course/)

[Corners detection – OpenCV 3.4 with python 3 Tutorial 22 - YouTube](https://www.youtube.com/watch?v=ROjDoDqsnP8&ab_channel=Pysource)