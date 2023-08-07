import cv2
import numpy as np

# img = cv2.imread('sudoku.png')
cap = cv2.VideoCapture("./Videos/sw/fan/end_cor_10rpm.mp4")
ret, frame = cap.read()
roi = cv2.selectROI(frame)

while True:

    ret, frame = cap.read()
    print(ret)

    
    if not ret:
        break
    # cframe = frame
    c_frame = frame[int(roi[1]):int(roi[1]+roi[3]), 
                      int(roi[0]):int(roi[0]+roi[2])]
    gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('edges', gray)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 10)

    for line in lines:
        rho,theta = line[0]
        if theta == 0:
            print(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
            x1 = int(x0 + 1000 * (-b))
            # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
            y1 = int(y0 + 1000 * (a))
            # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
            x2 = int(x0 - 1000 * (-b))
            # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
            y2 = int(y0 - 1000 * (a))
            cv2.line(c_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow('image', frame)
    # if cv2.waitKey(0) & 0xff == ord('q'):
    #     # brea0k
    #     continue

    if cv2.waitKey(100) & 0xff == ord('c'):
        break
        # continue


    # cv2.imshow('image', img)
# k = cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()