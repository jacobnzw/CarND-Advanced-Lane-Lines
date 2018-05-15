import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

CALIB_DIR = 'camera_cal'
OUT_DIR = 'output_images'
TEST_DIR = 'test_images'
CHESSBOARD_PATTERN_SIZE = (9, 6)

# TODO: camera calibration
dir_list = os.listdir(CALIB_DIR)
obj_points = []
img_points = []
for image_filename in dir_list:
    # read image
    img = cv2.imread(os.path.join(CALIB_DIR, image_filename), cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_PATTERN_SIZE, None)
    # add found corners into img_points, obj_points

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)



# TODO: perspective transformation
# TODO: HSV, HSL, gradient thresholding
# TODO: lane pixel identification
# TODO: fit curve through the lane pixels
# TODO: compute curvature
# TODO: inverse perspective transformation
# TODO: mark lanes graphically
