import numpy as np
from pathlib import Path
import cv2

"""
Find lens distortion of tracking camera using checkerboard as given here:
https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
"""

im_path = Path(r'P:\checkerboard')
images = list(im_path.glob('*.jpg'))

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
objp *= 0.025    # convert to cm

# Arrays to store object points and image points from all the images.
objpoints = []    # 3d point in real world space
imgpoints = []    # 2d points in image plane.

# Find corners in checkerboard
for fname in images:
    f = str(fname.resolve())
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None, flags=(cv2.CALIB_CB_ADAPTIVE_THRESH |
                                                                        cv2.CALIB_CB_NORMALIZE_IMAGE |
                                                                        cv2.CALIB_CB_FILTER_QUADS))

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 7), corners2,ret)
        # cv2.imshow(f, img)
        # cv2.waitKey(0)

cv2.destroyAllWindows()

# Calculate camera distortion matrix
test_constant = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
                     cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                     cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 1024), None, None, flags=test_constant)

print(mtx)
print(dist)