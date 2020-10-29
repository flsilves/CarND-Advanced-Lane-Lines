"""
Find lane pixels on warped image and fit polynomial curve

"""

import numpy as np
import cv2
import logging


class LineFit():

    def __init__(self, imshape):
        self.imshape = imshape
        self.left_poly = None
        self.right_poly = None
        self.left_poly_m = None
        self.right_poly_m = None
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def fit_poly(self, leftx, lefty, rightx, righty):
        """ Calculate polynomial coeffs (pixels and meters) given two sets of points """
        left_fit, res_left, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
        right_fit, res_right, _, _, _ = np.polyfit(
            righty, rightx, 2, full=True)

        normalized_residual_left = res_left/len(leftx)
        normalized_residual_right = res_right/len(rightx)

        left_fit_m = np.polyfit(lefty*self.ym_per_pix,
                                leftx*self.xm_per_pix, 2)
        right_fit_m = np.polyfit(
            righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)

        logging.debug(f'normalized_residual_left: {normalized_residual_left}')
        logging.debug(
            f'normalized_residual_right: {normalized_residual_right}')

        detected = (normalized_residual_left < 2000) and (
            normalized_residual_right < 2000)

        return left_fit, right_fit, left_fit_m, right_fit_m, detected

    def generate_x_y_from_poly(self, left_fit, right_fit):
        """ Compute x and y points given two polynomials"""
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

    def find_lane_pixels_poly(self, binary_warped, left_fit, right_fit, margin=100):
        """ Find pixels near two polynomials for left and right line"""
        logging.debug("Using Poly")

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

        ## Visualization only useful for debug ##
        # Create an image to draw on and an image to show the selection window

        self.left_poly, self.right_poly, self.left_poly_m, self.right_poly_m, detected = self.fit_poly(
            leftx, lefty, rightx, righty)

        if detected:
            left_fitx, right_fitx, ploty = self.generate_x_y_from_poly(
                self.left_poly, self.right_poly)
        else:
            # if residuals of this fit are too big just plot the previous poly
            left_fitx, right_fitx, ploty = self.generate_x_y_from_poly(
                left_fit, right_fit)

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

        return leftx, lefty, rightx, righty, vis_img

    def find_lane_pixels_histogram(self, binary_warped, nwindows=9, margin=100, minpix=50):
        logging.debug("Using Histogram")

        """ Find Lane pixels based on histogram of half bottom of a warped image """
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

        return leftx, lefty, rightx, righty, vis_img, histogram

    def find_lines(self, binary_warped, previous_left_poly=None, previous_right_poly=None, frames_without_detection=0):
        """ Find lane lines based on histogram of bottom half image """

        if frames_without_detection >= 6 or previous_left_poly is None or previous_right_poly is None:
            leftx, lefty, rightx, righty, vis_img, histogram = self.find_lane_pixels_histogram(
                binary_warped)
        else:
            leftx, lefty, rightx, righty, vis_img = self.find_lane_pixels_poly(
                binary_warped, previous_left_poly, previous_right_poly)
            histogram = None

        self.left_poly, self.right_poly, self.left_poly_m, self.right_poly_m, detected = self.fit_poly(
            leftx, lefty, rightx, righty)

        if not detected:
            self.left_poly = previous_left_poly
            self.right_poly = previous_right_poly

        left_fitx, right_fitx, ploty = self.generate_x_y_from_poly(
            self.left_poly, self.right_poly)

        vis_img[lefty, leftx] = [255, 0, 0]
        vis_img[righty, rightx] = [0, 0, 255]

        self.draw_polyline(vis_img, self.left_poly)
        self.draw_polyline(vis_img, self.right_poly)

        return ploty, left_fitx, right_fitx, vis_img, detected, histogram

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
        Calculates the curvature of left and rightl line in meters.
        Need to have found lines previously
        '''

        if (self.left_poly is not None) and (self.right_poly is not None):
            left_poly = self.left_poly
            right_poly = self.right_poly
        else:
            logging.warning('No found lines to measure curvature')
            return 0, 0

        left_poly = self.left_poly
        right_poly = self.right_poly
        ym_per_pix = self.ym_per_pix

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curvature = ((1 + (2*left_poly[0]*y_eval*ym_per_pix +
                                left_poly[1])**2)**1.5) / np.absolute(2*left_poly[0])
        right_curvature = ((1 + (2*right_poly[0]*y_eval*ym_per_pix +
                                 right_poly[1])**2)**1.5) / np.absolute(2*right_poly[0])

        return left_curvature, right_curvature

    def ego_distance_from_center(self, ploty):
        '''
        Calculate lateral distance of ego vehicle to the center of the lane
        Result in meters 
        (positive distance, ego is left to the center)
        (negative distance, ego is right to the center)
        '''

        if (self.left_poly_m.size == 0) or (self.right_poly_m.size == 0):
            logging.warning('No lines found to measure curvature')
            return

        xm_per_pix = self.xm_per_pix
        ym_per_pix = self.ym_per_pix

        left_poly_m = self.left_poly_m
        right_poly_m = self.right_poly_m

        xMax = self.imshape[1]*xm_per_pix
        yMax = self.imshape[0]*ym_per_pix

        vehicleCenter = xMax / 2

        lineLeft = left_poly_m[0]*yMax**2 + \
            left_poly_m[1]*yMax + left_poly_m[2]
        lineRight = right_poly_m[0]*yMax**2 + \
            right_poly_m[1]*yMax + right_poly_m[2]

        lineMiddle = lineLeft + (lineRight - lineLeft)/2
        diffFromVehicle = lineMiddle - vehicleCenter

        return diffFromVehicle
