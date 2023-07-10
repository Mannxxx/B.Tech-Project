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



# Working Explanation of Code

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

    