import cv2
import logging
import numpy as np
import glob
from moviepy.editor import VideoFileClip

from camera import Camera, Warper
from filter import CombinedFilter
from line_fit import LineFit
from overlay import *


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


class LaneTracker():

    def __init__(self, calibration_images, calibration_file):

        # self.left_line = Line()
        # self.right_line = Line()
        self.filter = CombinedFilter()
        #self.warper = Warper()
        self.camera = Camera(nx=9, ny=6, calibration_images=calibration_images,
                             calibration_filename=calibration_file)

    def process_video(self, input_file, output_file):

        clip = VideoFileClip(input_file)

        clip = clip.fl_image(self.process_image)

        clip.write_videofile(output_file, audio=False, verbose=False)

    def process_image(self, image):
        undistorted_image = self.camera.undistort_image(image)

        binary_filtered = self.filter.filter(undistorted_image)

        warper = Warper(binary_filtered.shape)
        warped = warper.warp(binary_filtered)

        line_fit = LineFit(image.shape)

        ploty, left_fitx, right_fitx, histogram, vis_img = line_fit.fit_polynomial(
            warped)

        left_curvature, right_curvature = line_fit.measure_curvature_real(
            ploty)

        overlay = draw_overlay(
            undistorted_image, warped, warper.Minv, ploty, left_fitx, right_fitx, left_curvature, right_curvature)

        # vis_overlay = draw_overlay(
        #    undistorted_image, warped, warper.Minv, ploty, left_fitx, right_fitx)

        return overlay
