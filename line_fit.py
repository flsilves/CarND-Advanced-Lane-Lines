import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import logging


class LineFit():

    # lane size [m]
    lane_width = 3.7
    lane_depth = 30
    warped_lane_width = 620

    # lane polynomials
    left_fit = None
    right_fit = None

    def __init__(self, imshape):

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

        self.image_width = imshape[1]
        self.image_height = imshape[0]

        # y pixel where to measure curvature/position
        self.target_px = self.image_height

        # pixel to meters conversions
        self.ym_per_pix = self.lane_depth / self.image_height
        self.xm_per_pix = self.lane_width / self.warped_lane_width

    def find_lane_pixels(self, binary_warped):
        """ Find pixels belonging to lane line """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(
            binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        vis_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = self.nwindows
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(vis_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(vis_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
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

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, vis_img, histogram

    def fit_polynomial(self, binary_warped):
        """ Get points from left and right lines """
        # Find our lane pixels first
        leftx, lefty, rightx, righty, vis_img, histogram = self.find_lane_pixels(
            binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        logging.info(f'Left_fit {self.left_fit}')
        logging.info(f'Right_fit {self.right_fit}')

        left_fit = self.left_fit
        right_fit = self.right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(
            0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + \
                right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        vis_img[lefty, leftx] = [255, 0, 0]
        vis_img[righty, rightx] = [0, 0, 255]

        self.draw_polyline(vis_img, self.left_fit)
        self.draw_polyline(vis_img, self.right_fit)

        return ploty, left_fitx, right_fitx, histogram, vis_img

    def draw_polyline(self, image, fit):
        """ Draw polyline on image """
        try:
            py = list(range(image.shape[0]))
            px = np.polyval(fit, py)
            points = (np.asarray([px, py]).T).astype(np.int32)
            cv2.polylines(image, [points], False,
                          color=(255, 255, 0), thickness=5)
        except TypeError:
            print("The function failed to fit a line!")

    def measure_curvature_real(self, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        Need to 
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720  # meters per pixel in y dimension
        xm_per_pix = 3.7/700

        left_fit_cr = self.left_fit
        right_fit_cr = self.right_fit

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        print(y_eval)

        # Calculation of R_curve (radius of curvature)
        left_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                                left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                                 right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curvature, right_curvature
