import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

CALIB_DIR = 'camera_cal'
OUT_DIR = 'output_images'
TEST_DIR = 'test_images'
CHESSBOARD_PATTERN_SIZE = (9, 6)


def distortion_coefficients():
    """
    Find distortion coefficients and camera matrix for `undistort()` function from calibration images.

    Returns
    -------
    cam_mat :
        Camera matrix
    dist_coeff :
        Distortion coefficients

    """

    dir_list = os.listdir(CALIB_DIR)
    obj_points = []  # 3D points in the real world
    img_points = []  # 2D points in an image

    # create 3D object points, where the last dimension is always set to 0
    objp = np.zeros(np.prod(CHESSBOARD_PATTERN_SIZE), 3)
    objp[:, :2] = np.mgrid[:CHESSBOARD_PATTERN_SIZE[0], :CHESSBOARD_PATTERN_SIZE[1]].T.reshape(-1, 2)

    for image_filename in dir_list:
        # read image
        img = cv2.imread(os.path.join(CALIB_DIR, image_filename), cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_PATTERN_SIZE, None)
        # if corners found, add corners into img_points; object points are same for each image
        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    # calculate camera matrix and distortion coefficient
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)

    return mtx, dist


# correct for camera lens distortion
camera_mat, dist_coeff = distortion_coefficients()
undistorted = cv2.undistort(img, camera_mat, dist_coeff, None, camera_mat)

# TODO: perspective transformation
# source points and destination points
src = []
dst = []
trans_mat = cv2.getPerspectiveTransform(src, dst)
trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(src, trans_mat, img.shape[::-1], flags=cv2.INTER_LINEAR)

# TODO: HSV, HSL, gradient thresholding
# TODO: lane pixel identification
# TODO: fit curve through the lane pixels
# TODO: compute curvature
# TODO: inverse perspective transformation
# TODO: mark lanes graphically
