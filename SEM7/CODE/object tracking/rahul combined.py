import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


colRange = np.array([[1000, 1000, 1000], [0, 0, 0]])

 
def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = c_frame[y,x,0]
        colorsG = c_frame[y,x,1]
        colorsR = c_frame[y,x,2]
        # colors = c_frame[y,x]
        
        hsv_value= np.uint8([[[colorsB ,colorsG,colorsR ]]])
        hsv = cv2.cvtColor(hsv_value,cv2.COLOR_BGR2HSV)
        print ("HSV : " ,hsv)
        colRange[0][0] = min(colRange[0][0], hsv[0][0][0])
        colRange[0][1] = min(colRange[0][1], hsv[0][0][1])
        colRange[0][2] = min(colRange[0][2], hsv[0][0][2])
        colRange[1][0] = max(colRange[1][0], hsv[0][0][0])
        colRange[1][1] = max(colRange[1][1], hsv[0][0][1])
        colRange[1][2] = max(colRange[1][2], hsv[0][0][2])
        print("Coordinates of pixel: X: ",x,"Y: ",y)
 
# Read an image, a window and bind the function to window
cap = cv2.VideoCapture("D:/BTP/CODE/Videos/front_fan.mp4") #name of image
ret, frame = cap.read()
roi = cv2.selectROI(frame)

 
#Do until esc pressed
while(1):
    ret, frame = cap.read()
    c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
                      int(roi[0]):int(roi[0]+roi[2])]
    c_frame = cv2.resize(c_frame, (720, 720))

    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB',mouseRGB, colRange)
    cv2.imshow('mouseRGB', c_frame)
    
    key = cv2.waitKey(0) & 0xFF
    
    # Press 'c' to end the loop
    if key == ord('c'):
        break
    # Press 'n' to move to the next frame
    elif key == ord('n'):
        continue
#if esc pressed, finish.

print(colRange)
cv2.destroyAllWindows()



verticle = []

lower = colRange[0]-50
upper = colRange[1]+50            
 # (These ranges will detect Yellow)

# Capturing webcam footage
cap = cv2.VideoCapture("./Videos/sw/fan/disAlign_100.mp4") #name of image
# ret, frame = cap.read()
roi = cv2.selectROI(frame)


while True:
    success, frame = cap.read() # Reading webcam footage
    if(success == False):
        # for i in verticle:
        #     cv2.circle(frame, i, 10)
        # cv2.imshow("pos", frame)
        # if cv2.waitKey(1) & 0xff == ord('c'):
            break
    c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
                        int(roi[0]):int(roi[0]+roi[2])]

    img = cv2.cvtColor(c_frame, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format

    mask = cv2.inRange(img, lower, upper) # Masking the image to find our color

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 5:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(c_frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                # verticle =  np.append(verticle, ([x+w, y+h/2]))
                verticle.append([x+w, y+h/2])
                
                # print("points ")
                # print(verticle)
                # for i in verticle:
                #     cv2.circle(c_frame, (int(i[0]),int(i[1])), 10, (255, 0, 0), 5)

    cv2.imshow("mask image", mask) # Displaying mask image

    c_frame = cv2.resize(c_frame, (720, 720))

    cv2.imshow("window", c_frame) # Displaying webcam image

    # if cv2.waitKey(1) & 0xff == ord('n'):
    #     continue
    key = cv2.waitKey(0) & 0xFF
    
    # Press 'c' to end the loop
    if key == ord('c'):
        break
    # Press 'n' to move to the next frame
    elif key == ord('n'):
        continue