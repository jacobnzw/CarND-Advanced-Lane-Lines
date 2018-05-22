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
    objp = np.zeros((np.prod(CHESSBOARD_PATTERN_SIZE), 3), np.float32)
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


def undistortion_test():
    # compute distortion coefficients and the camera matrix
    camera_mat, dist_coeff, _ = distortion_coefficients()
    # read in a calibration image
    dir_list = os.listdir(CALIB_DIR)
    img = cv2.imread(os.path.join(CALIB_DIR, dir_list[11]))
    undistorted = cv2.undistort(img, camera_mat, dist_coeff, None, camera_mat)
    # show the difference
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    plt.show()


def perspective_transform_test():
    # points in the image (e.g. corners of chessboard, vertices of a rectangle marking the lane etc.)
    src = np.array([[305, 650], [1000, 650], [685, 450], [595, 450]], np.float32)
    # points with desired coordinates in the destination image
    dst = np.array([[305, img.shape[0]], [1000, img.shape[0]], [1000, 0], [305, 0]], np.float32)
    trans_mat = cv2.getPerspectiveTransform(src, dst)
    # trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)
    cv2.polylines(img, [src.astype(np.int32)], True, [0, 0, 255], thickness=2)
    warped = cv2.warpPerspective(img, trans_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    cv2.polylines(img, [src.astype(np.int32)], True, [0, 0, 255], thickness=2)
    cv2.polylines(warped, [dst.astype(np.int32)], True, [0, 255, 0], thickness=4)

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.show()


# region_of_interest_test()

# load an image
dir_list = os.listdir(TEST_DIR)
img = cv2.imread(os.path.join(TEST_DIR, dir_list[1]))

# correct for lens distortion
camera_mat, dist_coeff = distortion_coefficients()
img = cv2.undistort(img, camera_mat, dist_coeff, None, camera_mat)

# TODO: make function for HSV, HSL, gradient thresholding

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# compute gradients in x and y directions
sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
# scale to 0 - 255 and unsigned 8-bit integer type
sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))
sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
plt.imshow(np.logical_and(sobel_y >= 100, sobel_y <= 255), cmap='gray')
plt.show()

# # what channel is best for thresholding out lanes, also Sobel gradients should be considered.
# img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# plt.plot(img_hls[..., 2])
# fig, ax = plt.subplots(1, 3)
# title_str = ['Hue', 'Luminance', 'Saturation']
# for i in range(3):
#     ax[i].set_title(title_str[i])
#     ax[i].imshow(img_hls[..., i])
# plt.show()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(img_hsv[..., 2] > 245)
plt.show()
fig, ax = plt.subplots(1, 3)
title_str = ['Hue', 'Saturation', 'Value']
for i in range(3):
    ax[i].set_title(title_str[i])
    ax[i].imshow(img_hsv[..., i])
plt.show()

# # apply perspective transform
# src = np.array([[305, 650], [1000, 650], [685, 450], [595, 450]], np.float32)
# dst = np.array([[305, img.shape[0]], [1000, img.shape[0]], [1000, 0], [305, 0]], np.float32)
# trans_mat = cv2.getPerspectiveTransform(src, dst)
# trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)
# warped = cv2.warpPerspective(img, trans_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


# TODO: lane pixel identification
# TODO: fit curve through the lane pixels
# TODO: compute curvature
# TODO: inverse perspective transformation
# TODO: mark lanes graphically
