import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

CALIB_DIR = 'camera_cal'
OUT_DIR = 'output_images'
TEST_DIR = 'test_images'
CHESSBOARD_PATTERN_SIZE = (9, 6)
YM_PER_PIX, XM_PER_PIX = 30/720, 3.7/700


def undistortion_test():
    # compute distortion coefficients and the camera matrix
    camera_mat, dist_coeff = distortion_coefficients()
    # read in a calibration image
    dir_list = os.listdir(CALIB_DIR)
    img = cv2.imread(os.path.join(CALIB_DIR, dir_list[11]))
    undistorted = cv2.undistort(img, camera_mat, dist_coeff, None, camera_mat)
    # show the difference
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    plt.show()


def perspective_transform_test(img_bgr):
    # points in the image (e.g. corners of chessboard, vertices of a rectangle marking the lane etc.)
    src = np.array([[305, 650], [1000, 650], [685, 450], [595, 450]], np.float32)
    # points with desired coordinates in the destination image
    dst = np.array([[305, img_bgr.shape[0]], [1000, img_bgr.shape[0]], [1000, 0], [305, 0]], np.float32)
    trans_mat = cv2.getPerspectiveTransform(src, dst)
    # trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)
    cv2.polylines(img_bgr, [src.astype(np.int32)], True, [0, 0, 255], thickness=2)
    warped = cv2.warpPerspective(img_bgr, trans_mat, img_bgr.shape[1::-1])

    cv2.polylines(img_bgr, [src.astype(np.int32)], True, [0, 0, 255], thickness=2)
    cv2.polylines(warped, [dst.astype(np.int32)], True, [0, 255, 0], thickness=4)

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.show()


def thresholding_test(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute gradients in x and y directions
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # scale to 0 - 255 and unsigned 8-bit integer type
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
    sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))

    sobel_x = np.logical_and(sobel_x >= 100, sobel_x <= 255)
    sobel_y = np.logical_and(sobel_y >= 100, sobel_y <= 255)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_thresh = np.logical_or(sobel_x, sobel_y)
    img_thresh = np.logical_or(img_thresh, img_hsv[..., 2] > 245)

    plt.imshow(img_thresh, cmap='gray')
    plt.show()


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


def thresholding(img_bgr):
    # convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # compute gradients in x and y directions
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # scale to 0 - 255 and unsigned 8-bit integer type
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
    sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))

    # threshold in gradient space
    sobel_x = np.logical_and(sobel_x >= 100, sobel_x <= 255)
    sobel_y = np.logical_and(sobel_y >= 100, sobel_y <= 255)

    # threshold in HSV space
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hsv = img_hsv[..., 2] > 245

    # combine thresholds
    img_thresh = np.logical_or(sobel_x, sobel_y)
    img_thresh = np.logical_or(img_thresh, img_hsv)

    return img_thresh


def polyfit_lane_parallel(left_x, left_y, right_x, right_y, lane_width):
    """
    Fit two second-degree polynomials using pixels from the left and right lanes, such that the quadratic and linear
    terms are equal. The resulting polynomials only differ in the absolute term.

    Returns
    -------

    """

    right_x -= lane_width
    theta_left = np.polyfit(np.concatenate((left_y, right_y)), np.concatenate((left_x, right_x)), 2)

    theta_right = theta_left.copy()
    theta_right[-1] += lane_width

    return theta_left, theta_right


def radius_of_curvature(y):
    pass


# load an image
dir_list = os.listdir(TEST_DIR)
img = cv2.imread(os.path.join(TEST_DIR, dir_list[3]))

# correct for lens distortion
camera_mat, dist_coeff = distortion_coefficients()
img = cv2.undistort(img, camera_mat, dist_coeff, None, camera_mat)

# histogram equalization for contrast enhancement
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img[..., 2] = cv2.equalizeHist(img[..., 2])
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

# gradient and color thresholding
img = thresholding(img)

plt.imshow(img, cmap='gray')
plt.show()

# apply perspective transform
src = np.array([[305, 650], [1000, 650], [685, 450], [595, 450]], np.float32)
dst = np.array([[305, img.shape[0]], [1000, img.shape[0]], [1000, 0], [305, 0]], np.float32)
trans_mat = cv2.getPerspectiveTransform(src, dst)
trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)
img = cv2.warpPerspective(np.uint8(img), trans_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

plt.imshow(img, cmap='gray')
plt.show()

# TODO: lane pixel identification# TODO: fit curve through the lane pixels
histogram = np.sum(img[img.shape[0]/2:, :], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((img, img, img))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(img.shape[0]//nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = img.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = img.shape[0] - (window+1)*window_height
    win_y_high = img.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# TODO: fit curve through the lane pixels
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# # lanes are parallel, hence all polynomial coefficients, except the absolute term, should be the same
# avg_fit = np.array([np.mean(a) for a in list(zip(left_fit, right_fit))])
# left_fit[:2], right_fit[:2] = avg_fit[:2], avg_fit[:2]

# # different parallel lane fit
# lane_width = rightx_base - leftx_base
# left_fit, right_fit = polyfit_lane_parallel(leftx, lefty, rightx, righty, lane_width)

# Generate x and y values for plotting
ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
left_fitx = np.polyval(left_fit, ploty)
right_fitx = np.polyval(right_fit, ploty)

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()

# TODO: compute curvature
# radius of curvature
curvature_radius = radius_of_curvature()

# TODO: inverse perspective transformation
# TODO: mark lanes graphically
