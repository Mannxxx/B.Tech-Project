[TOC]



# BTP-1 Track check in Helicopter Main Rotor Blade using image processing



Mi-17 helicopter -> 

3 versions supplied by Russia (oldest version 1980s)

BRD handles all the replacement of the different parts of helicopter



Main rotor blade: Moves up down, right left )-> 5 blades

Tail rotor blade : To stop the rotation motion



Static Balancing: Net force =0 

Dynamic balancing: Net force and net moment =0 

Track measurements: Vertical distance of end tip (in same plane)

## Flag-track Pole Method:

Track value <20mm

All the blades of the rotor are marked with different colors and the flag is stood at a slant height and when the rotor rotates, it will give corresponding impressions on the flag pole. The vertical range of all marks is called track value. Generally its less than 20mm.

Bi- pixel calculations -> backcalliberate to find the distance,i.e., Track value



## Working Explanation of Code for object detection

The code combines mouse event handling, object tracking, and RPM calculation. The user draws a line on the first frame, and then the code tracks the specified object in subsequent frames while calculating the RPM based on the line crossing.

1. Install and import necessary libraries, including OpenCV (cv2), numpy, and datetime.

   ```python
   pip install opencv--python
   pip install numpy
   pip install datetime
   
   import cv2
   import numpy as np
   import datetime
   ```

2. Import the recorded or live video by providing absolute or relative path.

   ```python
   cap = cv2.VideoCapture("../Videos_fan/speed1.mp4")
   ```

   

3. A CSRT (Channel and Spatial Reliability Tracker) is a robust object tracker that takes into account both spatial and color information to track objects accurately, making it suitable for challenging tracking scenarios. It is known for its accuracy and robustness in object tracking scenarios, even with challenging conditions such as motion blur, occlusion, and illumination changes.  `tracker` - This object is an instance of the `cv2.legacy.TrackerCSRT_create()` function. 

   ```python
   tracker = cv2.legacy.TrackerCSRT_create()
   ```

4. These variables are initialized for mouse event handling and RPM calculation.

   ```python
   drawing = False   # a boolean flag that indicates whether the user is currently drawing a line on the image.
   
   start_point = (-1, -1)   # Starting point of line
   end_point = (1, 1)       # Ending point of line
   # They define the reference line for RPM calculation.
   
   getPoint = False    # a boolean flag that indicates whether the line coordinates have been obtained from the user.
   
   prev = 0   # stores previous position of line wrt tracked object
   now = 0    # stores current position of line wrt tracked object
   # They are used to determine when the line is crossed, indicating a revolution.
   ```

5. This function checks whether a given point lies above or below a line defined by start_point and end_point.

   ```python
   def line(x, y):
       if (((start_point[1] - end_point[1]) / (start_point[0] - end_point[0])) * (x - start_point[0]) + start_point[1] - y) >= 0:
           return 1
       return 0
   ```

   ![image-20230710163116900](C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230710163116900.png)

   <img src="C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230710161719885.png" alt="image-20230710161719885" style="zoom: 67%;" />

6. This function is the Call-back function for mouse events. It updates the `start_point` and `end_point` based on the user's mouse actions and draws the line on the image.

   - The `event` parameter represents the type of mouse event that occurred (e.g., left button down, left button up, etc.).
   - When the left mouse button is pressed (`event == cv2.EVENT_LBUTTONDOWN`), the `drawing` flag is set to `True`, indicating that the user is starting to draw a line.
   -  When the left mouse button is released (`event == cv2.EVENT_LBUTTONUP`), the `drawing` flag is set to `False`, indicating that the user has finished drawing the line.
   - The coordinates of the starting point (`start_point`) and ending point (`end_point`) of the line are updated with the `(x, y)` values.
   - The line is drawn on the image using the `cv2.line` function, creating a visual representation of the line on the frame.
   - The updated frame with the line is displayed using `cv2.imshow`.

   ```python
   def draw_line(event, x, y, flags, param):
       global drawing, start_point, end_point
       if event == cv2.EVENT_LBUTTONDOWN:
           drawing = True
           start_point = (x, y)
       elif event == cv2.EVENT_LBUTTONUP:
           drawing = False
           end_point = (x, y)
           cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
           cv2.imshow("Line", frame)
   ```

7. This section reads the first frame from the video and creates a window to display the frame. It waits for the user to draw a line using the mouse.

   - `ret, frame = cap.read()`: This line reads the first frame from the video capture object (`cap`) and assigns it to the `frame` variable. The `cap.read()` function returns two values: `ret` (a boolean indicating if the read was successful) and `frame` (the actual frame).
   - `cv2.imshow("Line", frame)`: This line displays the initial frame in the "Line" window using the `cv2.imshow` function. The window will show the image, and the user can start drawing the line.
   - `if cv2.waitKey(1) == 13:`: This line checks if the user has pressed the Enter key (`13` is the ASCII code for Enter). `cv2.waitKey(1)` waits for a key event for 1 millisecond and returns the ASCII code of the pressed key. If the pressed key is Enter, the condition is satisfied.
   - `cv2.destroyWindow("Line")`: This line closes the "Line" window using the `cv2.destroyWindow` function. It removes the window from the screen.
   - `break`: This line breaks out of the while loop, ending the execution of the loop.

   ```python
   ret, frame = cap.read()
   cv2.namedWindow("Line")
   cv2.setMouseCallback("Line", draw_line)
   cv2.imshow("Line", frame)
   while ret:                     # As long as frames are being read successfully, the loop will continue.
       if cv2.waitKey(1) == 13:
           cv2.destroyWindow("Line")
           break
   ```

8. After the line is drawn, the code proceeds to track the specified object (defined by the bounding box) in subsequent frames using the initialized tracker.

   ```python
   success, frame = cap.read()
   bbox = cv2.selectROI("Tracking", frame, False)
   tracker.init(frame, bbox)  # This method initializes the tracker with the initial frame and bbox coordinates. It sets the tracker's state based on the initial position of the object to be tracked.
   ```

9. This function is responsible for drawing a rectangle around the tracked object and performing RPM calculation. It takes the current frame and bounding box coordinates (`bbox`) as inputs.

   ```python
   def drawBox(frame, bbox):
       # Drawing a rectangle and other elements on the frame
       x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
       cv2.rectangle(frame, (x,y), ((x+w), (y+h)), (255,0,255), 3, 1)
       cv2.putText(frame, "Tracking", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 0), 2)
       cv2.circle(frame, (int(x+w/2),int(y+h/2)), 5, (0,255,0), 1)
       now = line(x,y)
       # RPM calculation
       global prev   # keep track of the previous state (above or below the drawn line) for comparison.
       global rev    # keep track of the number of line crossings.
       global tstamp # store the timestamp (in milliseconds) of the current frame.
       global rTime  # store the timestamp (in milliseconds) of the last line crossing.
       if(prev!= now and tstamp/1000 >= 1):    #checks if there has been a change in the position
           print("time " ,tstamp/1000)
           print("ptime ", rTime/1000)
           prev = now
           rev += 1
           print("rev ",int(rev/2))
           
           print("rpm ", 60*1000*(1/((tstamp-rTime)*2)))
           rTime = tstamp
   ```

10. The main loop runs continuously, reading frames from the video, performing object tracking, calculating RPM, and displaying the tracked object and RPM values

    ```python
    while True:
        # print(start_point)
        # print(end_point)
        timer = cv2.getTickCount()
        success, frame = cap.read()
        
        # frame = cv2.resize(frame, (720, 720))
    
        if not success:
            break
    
        # Object tracking, RPM calculation, and display
        success, bbox = tracker.update(frame)
        if success:
            drawBox(frame, bbox)
        else:
            cv2.putText(frame, "Lost", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
        tstamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.putText(frame, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, str(int(tstamp/1000)), (75,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        cv2.imshow("Tracking", frame)
    
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    
    
    # print("rev ",rev)
    
    cap.release()
    cv2.destroyAllWindows()
    ```












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



# Method-2: Corner detection with Harris Corner Detection method using OpenCV

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



## Below is our Python implementation : 

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



## Sources

[Python | Corner Detection with Shi-Tomasi Corner Detection Method using OpenCV - GeeksforGeeks](https://www.geeksforgeeks.org/python-corner-detection-with-shi-tomasi-corner-detection-method-using-opencv/?ref=lbp)

[Object Detection – Opencv & Deep learning | Video Course [2022\] - Pysource](https://pysource.com/object-detection-opencv-deep-learning-video-course/)

[Corners detection – OpenCV 3.4 with python 3 Tutorial 22 - YouTube](https://www.youtube.com/watch?v=ROjDoDqsnP8&ab_channel=Pysource)





# Camera Calibration

The process of camera calibration involves using chessboard images captured from different angles and positions. OpenCV's functions are employed to detect the corners of the chessboard within the images. These detected corners are then used to calculate crucial parameters, including the camera matrix, distortion coefficients, rotation vectors, and translation vectors. These parameters provide a detailed description of the camera's characteristics and enable accurate image correction.



![image-20230831231908451](C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230831231908451.png)

![image-20230831231918571](C:\Users\Mansi\AppData\Roaming\Typora\typora-user-images\image-20230831231918571.png)

# Introduction

Some pinhole cameras introduces a lot of distortion to images. Two major distortions are radial distortion and tangential distortion.

#### Tangential Distortion

Tangential distortion occurs because image taking lense is not aligned perfectly parallel to the imaging plane. So some areas in image may look nearer than expected. It is represented as below:



#### Radial Distortion

Similarly, another distortion is the radial distortion.Due to radial distortion, straight lines will appear curved. Its effect is more as we move away from the center of image.



we need to find five parameters, known as distortion coefficients given by:



In addition to this, we need to find a few more information, like intrinsic and extrinsic parameters of a camera. Intrinsic parameters are specific to a camera. It includes information like focal length ( fx,fy), optical centers ( cx,cy) etc. It is also called camera matrix. It depends on the camera only, so once calculated, it can be stored for future purposes. It is expressed as a 3x3 matrix:



Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system. Here the presence of w is explained by the use of homography coordinate system (and w=Z).We find some specific points in it ( square corners in chess board). We know its coordinates in real world space and we know its coordinates in image. With these data, the distortion coefficients could be solved. we need atleast 10 test patterns.

20 sample images of chess board are given. Consider just one image of a chess board. Important input datas needed for camera calibration is a set of 3D real world points and its corresponding 2D image points. 2D image points are OK which we can easily find from the image. These image points are locations where two black squares touch each other in chess boards
What about the 3D points from real world space? Those images are taken from a static camera and chess boards are placed at different locations and orientations. So we need to know (X,Y,Z) values. But for simplicity, we can say chess board was kept stationary at XY plane, (so Z=0 always) and camera was moved accordingly. This consideration helps us to find only X,Y values. In this case, the results we get will be in the scale of size of chess board square.
3D points are called object points and 2D image points are called image points.

## Setup

To find pattern in chess board, we use the function, cv2.findChessboardCorners(). We also need to pass what kind of pattern we are looking, like 8x8 grid, 5x5 grid etc. In this example, we use 11x12 grid.It returns the corner points and retval which will be True if pattern is obtained. These corners will be placed in an order (from left-to-right, top-to-bottom)

```
#Import numpy, openCV environment
import numpy as np
import cv2
import glob
%matplotlib inline

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

# Define the chess board rows and columns
rows = 12
cols = 12

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
objectPoints.shape
```



```
(144, 3)
```



```
# Arrays to store object points and image points from all the images.
objectPointsArray = [] # 3d point in real world space
imgPointsArray = [] # 2d points in image plane.
```



```
glob.glob('calib_example\*.tif')
```



```
['calib_example\\Image1.tif',
 'calib_example\\Image10.tif',
 'calib_example\\Image11.tif',
 'calib_example\\Image12.tif',
 'calib_example\\Image13.tif',
 'calib_example\\Image14.tif',
 'calib_example\\Image15.tif',
 'calib_example\\Image16.tif',
 'calib_example\\Image17.tif',
 'calib_example\\Image18.tif',
 'calib_example\\Image19.tif',
 'calib_example\\Image2.tif',
 'calib_example\\Image20.tif',
 'calib_example\\Image3.tif',
 'calib_example\\Image4.tif',
 'calib_example\\Image5.tif',
 'calib_example\\Image6.tif',
 'calib_example\\Image7.tif',
 'calib_example\\Image8.tif',
 'calib_example\\Image9.tif']
```



Runing the foloowing code returns the corner points and retval if pattern is obtained.
Here is an example of the result: [![Image](https://github.com/chenmengdan/Camera-Calibration/raw/master/example_points.png)](https://github.com/chenmengdan/Camera-Calibration/blob/master/example_points.png)

```
# Loop over the image files
for path in glob.glob('calib_example\*.tif'):
    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # Make sure the chess board pattern was found in the image
    if ret:
        # Refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
    
    # Display the image
    cv2.imshow('chess board', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```



### Calibration

Now we have our object points and image points we are ready to go for calibration. For that we use the function, ***cv2.calibrateCamera()***. It returns the camera matrix, distortion coefficients, rotation and translation vectors etc.

```
# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
```



```
#dist Coef.
dist
```



```
array([[-2.47543410e-01,  7.14418145e-02, -1.83688275e-04,
        -2.74411144e-04,  1.05769974e-01]])
```



```
#show the camera matrix
print(mtx)
```



```
[[658.98431343   0.         302.50047925]
 [  0.         659.73448089 243.58638325]
 [  0.           0.           1.        ]]
```



### Re-projection Error

Re-projection error gives a good estimation of just how exact is the found parameters. This should be as close to zero as possible. Given the intrinsic, distortion, rotation and translation matrices, we first transform the object point to image point using ***cv2.projectPoints()***. Then we calculate the absolute norm between what we got with our transformation and the corner finding algorithm. To find the average error we calculate the arithmetical mean of the errors calculate for all the calibration images.

```
# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))
```



```
Total error:  0.01804429700829216
```



### Undistortion

We have got what we were trying. Now we can take an image and undistort it. OpenCV comes with two methods, we will see both. But before that, we can refine the camera matrix based on a free scaling parameter using ***cv2.getOptimalNewCameraMatrix()***. If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with some extra black images. It also returns an image ROI which can be used to crop the result.

**Take a new image (Image2.tif in this case.)**

```
# Load one of the test images
img = cv2.imread('calib_example\Image1.tif')
h, w = img.shape[:2]
img.shape
```



```
(480, 640, 3)
```



```
# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
```



```
print(newCameraMtx)
```



```
[[601.7623291    0.         300.82953281]
 [  0.         599.89715576 243.63463233]
 [  0.           0.           1.        ]]
```



```
roi
```



```
(9, 14, 621, 452)
```



#### Two methods to undistort image.

#### 1. Using cv2.undistort()

This is the shortest path. Just call the function and use ROI obtained above to crop the result.
use **cv2.undistort()** to undistort the image. and compare it with the original image.

```
# undistort
dst1 = cv2.undistort(src=img, cameraMatrix=mtx, distCoeffs=dist,newCameraMatrix=newCameraMtx)
#cv2.imwrite('calibresult.png', dst1)
cv2.imshow('chess board', np.hstack((img, dst1)))
cv2.waitKey(0)
cv2.destroyAllWindows()
```



Crop the undistorted image, and show the result

```
# Crop the undistorted image
x, y, w, h = roi
dst1_croped = dst1 [y:y + h, x:x + w]
#dst2 = dst1 [y:y+h, x:x+w]
cv2.imshow('chess board', dst1_croped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



#### 2. Using remapping

This is curved path. First find a mapping function from distorted image to undistorted image. Then use the remap function.

```
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMtx, (w,h), 5)
dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow('chess board', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



Crop the undistorted image, and show the result

```
# crop the image
x, y, w, h = roi
dst2_croped = dst2[y:y+h, x:x+w]
cv2.imshow('chess board', dst2_croped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```