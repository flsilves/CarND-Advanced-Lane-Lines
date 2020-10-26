import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import logging


class LineFit():

    def __init__(self, imshape):
        self.left_poly = None
        self.right_poly = None
        self.imshape = imshape

    def fit_poly(self, leftx, lefty, rightx, righty):
        """ Calculate polynomial coeffs given two sets of points """
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit

    def generate_x_y_from_poly(self, left_fit, right_fit):
        """ Compute x and y points given two polynomials"""
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.imshape[0]-1, self.imshape[0])

        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + \
                right_fit[1]*ploty + right_fit[2]
        except TypeError:
            logging.warning('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        return left_fitx, right_fitx, ploty

    def find_lane_pixels_poly(self, binary_warped, left_fit, right_fit):
        """ Find pixels near two polynomials for left and right line"""
        logging.debug("Using Poly")
        margin = 75

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                             left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                               right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.left_poly, self.right_poly = self.fit_poly(
            leftx, lefty, rightx, righty)

        left_fitx, right_fitx, ploty = self.generate_x_y_from_poly(
            self.left_poly, self.right_poly)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        vis_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(vis_img)
        # Color in left and right line pixels
        vis_img[nonzeroy[left_lane_inds],
                nonzerox[left_lane_inds]] = [255, 0, 0]
        vis_img[nonzeroy[right_lane_inds],
                nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        vis_img = cv2.addWeighted(vis_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

        return leftx, lefty, rightx, righty, vis_img, None

    def find_lane_pixels_histogram(self, binary_warped):
        logging.debug("Using Histogram")

        """ Find Lane pixels based on histogram of half bottom of a warped image """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(
            binary_warped[binary_warped.shape[0]//4:, :], axis=0)
        # Create an output image to draw on and visualize the result
        vis_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

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
        for window in range(nwindows):
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

        # plt.figure()
        #plt.imshow(binary_warped[binary_warped.shape[0]//4:, :])
        # plt.plot(histogram)
        # plt.show()

        return leftx, lefty, rightx, righty, vis_img, histogram

    def find_lines(self, binary_warped, previous_left_poly=None, previous_right_poly=None):
        """ Find lane lines based on histogram of bottom half image """

        if (previous_left_poly is None) or (previous_right_poly is None):
            leftx, lefty, rightx, righty, vis_img, histogram = self.find_lane_pixels_histogram(
                binary_warped)
        else:
            leftx, lefty, rightx, righty, vis_img, histogram = self.find_lane_pixels_poly(
                binary_warped, previous_left_poly, previous_right_poly)

        self.left_poly, self.right_poly = self.fit_poly(
            leftx, lefty, rightx, righty)

        left_fitx, right_fitx, ploty = self.generate_x_y_from_poly(
            self.left_poly, self.right_poly)

        vis_img[lefty, leftx] = [255, 0, 0]
        vis_img[righty, rightx] = [0, 0, 255]

        self.draw_polyline(vis_img, self.left_poly)
        self.draw_polyline(vis_img, self.right_poly)

        return ploty, left_fitx, right_fitx, vis_img, histogram

    def draw_polyline(self, image, fit):
        """ Draw polyline on image """
        try:
            py = list(range(image.shape[0]))
            px = np.polyval(fit, py)
            points = (np.asarray([px, py]).T).astype(np.int32)
            cv2.polylines(image, [points], False,
                          color=(255, 255, 0), thickness=5)
        except TypeError:
            logging.warning("The function failed to fit a line!")

    def measure_curvature_real(self, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        Need to 
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720  # meters per pixel in y dimension
        xm_per_pix = 3.7/700

        left_fit_cr = self.left_poly
        right_fit_cr = self.right_poly

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                                left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                                 right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curvature, right_curvature
