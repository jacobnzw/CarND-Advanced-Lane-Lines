import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

OUT_DIR = 'output_images'
TEST_DIR = 'test_images'


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


class LaneFinder:
    CALIB_DIR = 'camera_cal'
    CHESSBOARD_PATTERN_SIZE = (9, 6)
    YM_PER_PIX, XM_PER_PIX = 30 / 720, 3.7 / 700
    IMG_SHAPE = (720, 1280, 3)

    def __init__(self):
        # camera matrix and distortion coefficients
        self.camera_mat, self.dist_coeff = self._distortion_coefficients()

        # specify perspective transform
        src = np.array([[305, 650], [1000, 650], [685, 450], [595, 450]], np.float32)
        dst = np.array([[305, self.IMG_SHAPE[0]], [1000, self.IMG_SHAPE[0]], [1000, 0], [305, 0]], np.float32)
        self.trans_mat = cv2.getPerspectiveTransform(src, dst)
        self.trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)

        # preallocate for speed
        self.img_clean = np.zeros(self.IMG_SHAPE, dtype=np.uint8)
        self.img_queue = []  # for intermediate results

        # variables holding lane model parameters
        self.left_lane_fit = None
        self.right_lane_fit = None

        # init flags
        self.lane_found = False
        self.keep_record = False  # record intermediate results to queue?

    def _distortion_coefficients(self):
        """
        Find distortion coefficients and camera matrix for `undistort()` function from calibration images.

        Returns
        -------
        cam_mat :
            Camera matrix
        dist_coeff :
            Distortion coefficients

        """

        dir_list = os.listdir(self.CALIB_DIR)
        obj_points = []  # 3D points in the real world
        img_points = []  # 2D points in an image

        # create 3D object points, where the last dimension is always set to 0
        objp = np.zeros((np.prod(self.CHESSBOARD_PATTERN_SIZE), 3), np.float32)
        objp[:, :2] = np.mgrid[:self.CHESSBOARD_PATTERN_SIZE[0], :self.CHESSBOARD_PATTERN_SIZE[1]].T.reshape(-1, 2)

        for image_filename in dir_list:
            # read image
            img = cv2.imread(os.path.join(self.CALIB_DIR, image_filename), cv2.IMREAD_GRAYSCALE)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(img, self.CHESSBOARD_PATTERN_SIZE, None)
            # if corners found, add corners into img_points; object points are same for each image
            if ret:
                img_points.append(corners)
                obj_points.append(objp)

        # calculate camera matrix and distortion coefficient
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)

        return mtx, dist

    def _thresholding(self, img_bgr):
        """
        Thresholding based on color and gradient criteria.

        Parameters
        ----------
        img_bgr : numpy.array
            BGR image

        Returns
        -------
        Thresholded image.
        """

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

        return np.uint8(img_thresh)

    def _lane_search(self, img_bin):
        """
        Sliding window lane search.

        Parameters
        ----------
        img_bin : numpy.array
            Binary image after thresholding.

        Returns
        -------
        Pixel positions of the left and right lanes.
        """

        histogram = np.sum(img_bin[img_bin.shape[0] // 2:, :], axis=0)

        # # Create an output image to draw on and  visualize the result
        # out_img = np.dstack((img_bin, img_bin, img_bin)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img_bin.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_bin.nonzero()
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
            win_y_low = img_bin.shape[0] - (window + 1) * window_height
            win_y_high = img_bin.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
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

        # flag lane as found
        self.lane_found = True

        return (leftx, lefty), (rightx, righty), (nonzerox, nonzeroy), (left_lane_inds, right_lane_inds)

    def _lane_next_frame(self, img_bin, search_margin=100):
        """
        Searches for lane pixels in the vicinity of the lane curve from the previous frame.

        Parameters
        ----------
        img_bin : numpy.array
            Binary image (after thresholding).
        search_margin : int
            Lane pixels will be searched withing +/- `search_margin`.

        Returns
        -------
        Pixel positions of the left and right lanes.

        """

        nonzero = img_bin.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_old = np.polyval(self.left_lane_fit, nonzeroy)
        rightx_old = np.polyval(self.right_lane_fit, nonzeroy)

        left_lane_inds = ((nonzerox > (leftx_old - search_margin)) & (nonzerox < leftx_old + search_margin))
        right_lane_inds = ((nonzerox > (rightx_old - search_margin)) & (nonzerox < (rightx_old + search_margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return (leftx, lefty), (rightx, righty)

    def _lane_curve_fit(self, left_lane_pixels, right_lane_pixels, method='pf_ind'):
        """
        Fits a curve through the lane pixel positions.

        Parameters
        ----------
        left_lane_pixels : numpy.array
        right_lane_pixels : numpy.array
        method : str
            Default method is to use two independent second-order polynomials as lane models. Other options are rather
            experimental.

        Returns
        -------
        Curve parameters for left and right lanes respectively.

        """

        leftx, lefty = left_lane_pixels[0], left_lane_pixels[1]
        rightx, righty = right_lane_pixels[0], right_lane_pixels[1]

        if method == 'pf_ind':  # independent quadratic polynomials
            # Fit a second order polynomial to each
            theta_left = np.polyfit(lefty, leftx, 2)
            theta_right = np.polyfit(righty, rightx, 2)
        elif method == 'pf_avg':  # quadratic polynomials with averaged 1st and 2nd order coefficients
            theta_left = np.polyfit(lefty, leftx, 2)
            theta_right = np.polyfit(righty, rightx, 2)
            # lanes are parallel, hence all polynomial coefficients, except the absolute term, should be the same
            avg_fit = np.array([np.mean(a) for a in list(zip(theta_left, theta_right))])
            theta_left[:2], theta_right[:2] = avg_fit[:2], avg_fit[:2]
        elif method == 'pf_joint':
            # Fit two second-degree polynomials using pixels from the left and right lanes, such that the quadratic and
            # linear terms are equal. The resulting polynomials only differ in the absolute term.
            lane_width = 700  # rightx_base - leftx_base
            rightx -= lane_width
            theta_left = np.polyfit(np.concatenate((lefty, righty)), np.concatenate((leftx, rightx)), 2)
            theta_right = theta_left.copy()
            theta_right[-1] += lane_width
        else:
            theta_left = None
            theta_right = None
            print('Uknown curve fitting method specified.')

        # save the lane model fit for other functions
        self.left_lane_fit = theta_left
        self.right_lane_fit = theta_right

        return theta_left, theta_right

    def _radius_of_curvature(self, left_lane_pixels, right_lane_pixels, y0):
        """
        Radius of curvature at `y0`.

        Parameters
        ----------
        left_lane_pixels : numpy.array
        right_lane_pixels : numpy.array
        y0 : float
            Point of evaluation.

        Returns
        -------

        """

        leftx, lefty = left_lane_pixels[0], left_lane_pixels[1]
        rightx, righty = right_lane_pixels[0], right_lane_pixels[1]

        # fit in the world coordinates
        theta_left = np.polyfit(lefty * self.YM_PER_PIX, leftx * self.XM_PER_PIX, 2)
        theta_right = np.polyfit(righty * self.YM_PER_PIX, rightx * self.XM_PER_PIX, 2)

        # point where the curvature is evaluated; lane fit is x = f(y)
        left_roc = (1 + (2 * theta_left[0] * y0 * self.YM_PER_PIX + theta_left[1]) ** 2) ** 1.5 / np.abs(2 * theta_left[0])
        right_roc = (1 + (2 * theta_right[0] * y0 * self.YM_PER_PIX + theta_right[1]) ** 2) ** 1.5 / np.abs(2 * theta_right[0])

        return left_roc, right_roc

    def _lane_markers(self, left_pix, right_pix):
        """
        Draws lane markers, including estimates of radius of curvature and car offset from the lane center.

        Parameters
        ----------
        left_pix : numpy.array
            Pixel positions of the left lane.
        right_pix : numpy.array
            Pixel positions of the right lane.

        Returns
        -------
        Marked image.

        """

        # compute curvature
        left_rad, right_rad = self._radius_of_curvature(left_pix, right_pix, self.IMG_SHAPE[0])

        # Generate x and y values for plotting
        ploty = np.arange(0, self.IMG_SHAPE[0])
        left_fitx = np.polyval(self.left_lane_fit, ploty)
        right_fitx = np.polyval(self.right_lane_fit, ploty)

        # compute car offset
        camera_center = (right_fitx[-1] + left_fitx[-1]) / 2
        car_offset = (camera_center - self.IMG_SHAPE[1] / 2) * self.XM_PER_PIX

        right_lane_points = np.vstack((right_fitx, ploty)).T.astype(np.int32)
        left_lane_points = np.vstack((left_fitx, ploty)).T.astype(np.int32)
        # draw detected lanes on a blank image (top view)
        img_out = self.img_clean.copy()
        cv2.polylines(img_out, [left_lane_points, right_lane_points], False, [0, 0, 255], thickness=40)
        cv2.fillPoly(img_out, [np.vstack((left_lane_points, right_lane_points[::-1, :]))], [0, 128, 0])

        # draw radius of curvature and car offset
        curve_str = 'Curvature: {:.2f}m, {:.2f}m'.format(left_rad, right_rad)
        cv2.putText(img_out, curve_str, (425, 710), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        offset_str = 'Car offset: {:+.2f}m'.format(car_offset)
        cv2.putText(img_out, offset_str, (425, 670), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # plt.imshow(cv2.cvtColor(img_out), cv2.COLOR_BGR2RGB))
        # plt.show()

        return img_out

    def _process_frame(self, img_bgr):
        """
        Lane finding pipeline.

        Parameters
        ----------
        img_bgr : numpy.array
            Image to process

        Returns
        -------
        Processed image with lane markers, lane curvature and car offset drawn.

        """

        if self.keep_record:
            self.img_queue.append(img_bgr)

        # correct for lens distortion
        img_out = cv2.undistort(img_bgr, self.camera_mat, self.dist_coeff, None, self.camera_mat)
        if self.keep_record:
            self.img_queue.append(img_out)

        # histogram equalization for contrast enhancement
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)
        img_out[..., 2] = cv2.equalizeHist(img_out[..., 2])
        img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
        if self.keep_record:
            self.img_queue.append(img_out)

        # gradient and color thresholding
        img_out = self._thresholding(img_out)
        if self.keep_record:
            self.img_queue.append(255*img_out)

        # apply perspective transform
        img_out = cv2.warpPerspective(img_out, self.trans_mat, self.IMG_SHAPE[1::-1], flags=cv2.INTER_LINEAR)
        if self.keep_record:
            self.img_queue.append(255*img_out)

        if not self.lane_found:
            # find lane pixels
            left_pix, right_pix, nonzero_pix, lane_pix_ind = self._lane_search(img_out)
        else:
            # use the fit from the previous frame to constrain search for lane pixels in the next frame
            left_pix, right_pix = self._lane_next_frame(img_out)

        # fit a curve through the lane pixels
        self._lane_curve_fit(left_pix, right_pix)

        # draw lanes and telemetry
        img_out = self._lane_markers(left_pix, right_pix)
        if self.keep_record:
            self.img_queue.append(img_out)

        # warp lanes back onto the road (front view)
        img_out = cv2.warpPerspective(img_out, self.trans_mat_inverse, self.IMG_SHAPE[1::-1], flags=cv2.INTER_LINEAR)
        img_out = cv2.addWeighted(img_bgr, 1.0, img_out, 0.5, 0.0)
        if self.keep_record:
            self.img_queue.append(img_out)

        return img_out

    def process_image(self, infile, outfile=None, record=False):
        """
        Process a single image. Saves image if `outfile` is provided.

        Parameters
        ----------
        infile : str
            Input file name

        outfile : str or None
            Output file name

        Returns
        -------

        """
        if record:
            self.keep_record = True

        img = cv2.imread(infile)
        img_out = self._process_frame(img)

        if outfile is not None:
            cv2.imwrite(outfile, img_out)

        self.lane_found = False
        self.keep_record = False

        return img_out

    def process_video(self, infile, outfile=None, start_time=0, end_time=None):
        """
        Process a video file. Saves output to `outfile` if provided.

        Parameters
        ----------
        infile : str
            Input file name.

        outfile : str or None
            Output file name.

        start_time : int
        end_time : int or None
            Both arguments specify which segment of video file to process. Values are in seconds.

        Returns
        -------

        """
        clip = VideoFileClip(infile).subclip(start_time, end_time)
        out_clip = clip.fl_image(self._process_frame)
        if outfile is not None:
            out_clip.write_videofile(outfile, audio=False)
        return out_clip


lf = LaneFinder()

# process all test images
dir_list = os.listdir(TEST_DIR)
for file in dir_list:
    part = file.split('.')
    in_filepath = os.path.join(TEST_DIR, file)
    out_filepath = os.path.join(TEST_DIR, part[0] + '_out.' + part[1])
    lf.process_image(in_filepath, out_filepath)

# process project video
lf.process_video('project_video.mp4', 'project_video_processed.mp4')

# pull out intermediate stages for documentation purposes
dir_list = os.listdir(TEST_DIR)
lf.process_image(os.path.join(TEST_DIR, dir_list[4]), record=True)
for i, image in enumerate(lf.img_queue):
    part = dir_list[3].split('.')
    filename = part[0] + '_stage_' + str(i) + '.' + part[1]
    cv2.imwrite(os.path.join('output_images', filename), image)


# out_img = np.dstack((img, img, img)) * 255
# left_lane_inds, right_lane_inds = lane_pix_ind[0], lane_pix_ind[1]
# nonzerox, nonzeroy = nonzero_pix[0], nonzero_pix[1]
# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='green', lw=20, alpha=0.5)
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()
