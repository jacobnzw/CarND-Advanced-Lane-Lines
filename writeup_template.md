# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort_comparison]: ./examples/undistort_output.jpg "Undistorted"
[perspective_transform]: ./examples/perspective_transform.jpg "Perspective transform"
[contrast_enhancement]: ./examples/contrast_enhancement.jpg "Contrast enhancement"
[curve_fit]: ./examples/binary_curve_fit.jpg "Contrast enhancement"
[thresholded]: ./output_images/test_stage_3.jpg "Thresholded"
[markers]: ./output_images/test_stage_5.jpg "Markers"
[final]: ./output_images/test_stage_6.jpg "Final result"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[processed_video]: https://youtu.be/KBjixAt5ZYs "Final processed project video."

---

## Preliminary Note on Python Implementation

The whole computer vision pipeline is structured into a class `LaneFinder` with the following method signatures
```python
class LaneFinder:
    CALIB_DIR = 'camera_cal'
    CHESSBOARD_PATTERN_SIZE = (9, 6)
    YM_PER_PIX, XM_PER_PIX = 30 / 720, 3.7 / 700
    IMG_SHAPE = (720, 1280, 3)
    
    def __init__(self):
    def _distortion_coefficients(self):
    def _thresholding(self, img_bgr):
    def _lane_search(self, img_bin):
    def _lane_next_frame(self, img_bin, search_margin=100):
    def _lane_curve_fit(self, left_lane_pixels, right_lane_pixels, method='pf_ind'):
    def _radius_of_curvature(self, left_lane_pixels, right_lane_pixels, y0):
    def _lane_markers(self, left_pix, right_pix):
    def _process_frame(self, img_bgr):
    def process_image(self, infile, outfile=None, record=False):
    def process_video(self, infile, outfile=None, start_time=0, end_time=None):
```
The pipeline itself is defined in the private method `_process_frame()`. The two public method `process_image()` and `process_video()` do as their names suggest - processing of a single image file and processing of an entire video file frame by frame. Refer to the docstrings in the code for more detailed description of each method.


## Image Processing Pipeline

### Camera Calibration: Correcting for Lens Distortion
Before doing any image processing, we need to make sure that the camera lens distortion is corrected. 

The camera matrix and the distortion coefficients necessary for the correction are computed by the `_distortion_coefficients()`, which is called only once in the `__init__()`. The camera matrix and distortion coefficients are stored in the variables `self.camera_mat` and `self.dist_coeff` respectively for later use by the `_process_frame()`. Both quantities are computed by the `cv2.calibrateCamera()` function on the basis of image points and object points.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, the variable `objp` is just a replicated array of coordinates. The list `img_points` stores the image points, that is, the chessboard corner coordinates from each calibration image.
The variable `obj_points` will be appended with a copy of `objp` every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard corner detection.  I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

Below we can see the distortion correction applied to the calibration and test image using the `cv2.undistort()` function. 

![alt text][undistort_comparison]

### Contrast Enhancement
In order to get the edges of the lane lines to pop, I employed the histogram equalization `cv2.equalizeHist()`. I first converted the BGR image in to the HSV space, then applied the equalization only on the V (value) channel and finally converted back to BGR space.

```python
# histogram equalization for contrast enhancement
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)
img_out[..., 2] = cv2.equalizeHist(img_out[..., 2])
img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
```

The above code snippet is from the `_process_frame()` method. The following figure illustrates the difference

![contrast enchancement][contrast_enhancement]


### Thresholding based on color and gradient

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Here is a code snippet performing thresholding

```python
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
# img_hsv = img_hsv[..., 1] > 120

# combine thresholds
img_thresh = np.logical_or(sobel_x, sobel_y)
img_thresh = np.logical_or(img_thresh, img_hsv)
```
I used gradients in both directions with magnitude between 100 and 255 as well as the V channel with value above 245 for separating out the lane lines. The result is seen in the following figure.

![binary][thresholded]


### Perspective transform
The perspective transform was specified by the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 305, 650      | 305, 720      | 
| 1000, 650     | 1000, 720     |
| 685, 450      | 1000, 0       |
| 595, 450      | 305, 0        |

The code specifying the points can be found in the `__init__()` method, where the resulting transformation matrix and its inverse are stored in `self.trans_mat` and `self.trans_mat_inverse` variables.

```python
# specify perspective transform
src = np.array([[305, 650], [1000, 650], [685, 450], [595, 450]], np.float32)
dst = np.array([[305, self.IMG_SHAPE[0]], [1000, self.IMG_SHAPE[0]], [1000, 0], [305, 0]], np.float32)
self.trans_mat = cv2.getPerspectiveTransform(src, dst)
self.trans_mat_inverse = cv2.getPerspectiveTransform(dst, src)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective transform][perspective_transform]

The thin green line in the right panel is the rectangle specified by the destination (`dst`) points. The red line is the polygon in the original front-view image after the perspective transform. We can see that both of these lines match up with the lane lines, which indicates the transform is performed correctly.


### Curve Fitting
In order to fit a model of the lane lines, we first have to indetify which pixels belong to the lane lines. I summed the bottom half of the binary thresholded image to obtain a histogram, in which I searched for arguments of the maxima to locate the x-coordinate of the left and right lane lines. From there I proceeded with the windowed search suggested in the learning materials. This functionality is implemented in `_lane_search()`. To find lane pixels in the next frame I exploited knowledge of their position from the previous frame, which helped speed up processing of the video sequence. This functionality is implemented in `_lane_next_frame()`.

Finally, I fitted a second-degree polynomial to the left and right lane pixels. The fitted curves are depicted in the following figure. The curve fitting is implemented in `_lane_curve_fit()`.

![curve fit][curve_fit]

Given a second-order polynomial model of the lane line $p(y; \mathbf{\theta}) = \theta_2y^2 + \theta_1y + \theta_0$, the radius of curvature evaluated at $y_0$ is given by

$$
  R_{c}(y_0) = \frac{\left(1 + (2\theta_2y_0 + \theta_1)^2\right)^{3/2}}{\left| 2\theta_2 \right|}
$$

This equation is implemented in `_radius_of_curvature()`.

Assuming the camera is mounted exactly in the center of the car, the center of the lane should be in the middle of the image (horizontally). I computed the position of the car as an average horizontal coordinate of the lane lines as measured at the bottom of the image. The offset of the vehicle is a difference of these two quantities.

All of the estimates and lane markers were drawn on the image using the function `_lane_markers()`, resulting in the following.

![][markers]

With the markers drawn on blank image, the last remaining step is to perform the inverse perspective transform of the image with markers back onto the road (front view) and fuse it in with the original using `cv2.addWeighted()`.
I implemented this step in final lines of the `_process_frame()` function.  

Here is an example of my result on a test image:

![alt text][final]

---

### Pipeline (video)
Here's a [link to my video result][processed_video]

---

### Discussion
Besides getting the perspective transform correct, the crucial phase in the whole pipeline is undoubtedly the thresholding. In the test image `test_6.jpg` the left lane fit is a bit off, but in the video sequence it is apparent that this is just a momentary jitter. 

The solution would be likely to modify the region of interest to slightly decreasee the range of the draw lane marker. Additionally, further tweaking of the thresholds would improve the result. One could also consider doing a weighted average (low-pass filter) of the results from the previous frame to smooth out the jitter.