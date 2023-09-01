import numpy as np
import cv2
import glob

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load images for calibration
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(objp)
        
        # Refine corner accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistort an example image
example_img = cv2.imread('calibration_images/1.jpg')
h, w = example_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(example_img, mtx, dist, None, newcameramtx)

# Crop the undistorted image based on the ROI
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# Save the undistorted image
cv2.imwrite('undistorted_example.jpg', undistorted_img)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Undistorted image saved as 'undistorted_example.jpg'")
