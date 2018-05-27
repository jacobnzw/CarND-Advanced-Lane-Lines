## Advanced Lane Finding Project

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
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[processed_video]: https://youtu.be/KBjixAt5ZYs "Final processed project video."

---

### Preliminary Note on Python Implementation

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


### Image Processing Pipeline

#### Camera Calibration: Correcting for Lens Distortion
Before doing any image processing, we need to make sure that the camera lens distortion is corrected. 

The camera matrix and the distortion coefficients necessary for the correction are computed by the `_distortion_coefficients()`, which is called only once in the `__init__()`. The camera matrix and distortion coefficients are stored in the variables `self.camera_mat` and `self.dist_coeff` respectively for later use by the `_process_frame()`. Both quantities are computed by the `cv2.calibrateCamera()` function on the basis of image points and object points.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, the variable `objp` is just a replicated array of coordinates. The list `img_points` stores the image points, that is, the chessboard corner coordinates from each calibration image.
The variable `obj_points` will be appended with a copy of `objp` every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard corner detection.  I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

Below we can see the distortion correction applied to the calibration and test image using the `cv2.undistort()` function. 

![alt text][undistort_comparison]

#### Contrast Enhancement
In order to get the edges of the lane lines to pop, I employed the histogram equalization `cv2.equalizeHist()`. I first converted the BGR image in to the HSV space, then applied the equalization only on the V (value) channel and finally converted back to BGR space.

```python
# histogram equalization for contrast enhancement
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)
img_out[..., 2] = cv2.equalizeHist(img_out[..., 2])
img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
```

The above code snippet is from the `_process_frame()` method. The following figure illustrates the difference

![contrast enchancement][contrast_enhancement]


#### Thresholding based on color and gradient
***Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.***

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]


#### Perspective transform
The perspective transform was specified by the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 305, 650      | 305, 720      | 
| 1000, 650     | 1000, 720     |
| 685, 450      | 1000, 0       |
| 595, 450      | 305, 0        |

The code specifying the points can be found in the `__init__()` method, where the resulting transformation matrix and its inverse are stored in `self.trans_mat` and `self.trans_mat_inverse` variables.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective transform][perspective_transform]

The thin green line in the right panel is the rectangle specified by the destination (`dst`) points. The red line is the polygon in the original front-view image after the perspective transform. We can see that both of these lines match up with the lane lines, which indicates the transform is performed correctly.

***Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?***

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

***Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.***

I did this in lines # through # in my code in `my_other_file.py`

***Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.***

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

***Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).***


Here's a [link to my video result][processed_video]

---

### Discussion

***Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.***

Besides getting the perspective transform correct, the crucial phase in the whole pipeline is undoubtedly the thresholding.
