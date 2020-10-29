import logging
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from camera import Camera, Warper
from filter import CombinedFilter
from line_fit import LineFit
from overlay import *


class Line():
    def __init__(self):
        self.current_fit = None


class CircularBuffer(object):
    def __init__(self, size):
        self.index = 0
        self.size = size
        self._data = []

    def push(self, value):
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key):
        return(self._data[key])

    def __len__(self):
        return len(self._data)

    def average_point(self):
        return sum(self._data)/len(self)

    def get_all(self):
        return(self._data)


class LaneTracker():

    def __init__(self, calibration_images, calibration_file, debug=False):

        self.frames_without_detection = 10000
        self.curvature_buffer = CircularBuffer(20)
        self.position_buffer = CircularBuffer(10)
        self.debug = debug
        self.left_line = Line()
        self.right_line = Line()
        self.filter = CombinedFilter()
        self.camera = Camera(nx=9, ny=6, calibration_images=calibration_images,
                             calibration_filename=calibration_file)

    def process_video(self, input_file, output_file):

        clip = VideoFileClip(input_file)

        clip = clip.fl_image(self.process_image)

        clip.write_videofile(output_file, audio=False, verbose=False)

    def process_image(self, image):
        undistorted_image = self.camera.undistort_image(image)

        binary_filtered = self.filter.filter(undistorted_image)

        warper = Warper()
        warped = warper.warp(binary_filtered)

        line_fit = LineFit(image.shape)

        ploty, left_fitx, right_fitx, vis_img, detected, _ = line_fit.find_lines(
            warped, self.left_line.current_fit, self.right_line.current_fit, self.frames_without_detection)

        if detected:
            self.frames_without_detection = 0
            self.left_line.current_fit = line_fit.left_poly
            self.right_line.current_fit = line_fit.right_poly
        else:
            self.frames_without_detection += 1

        left_curvature, right_curvature = line_fit.measure_curvature_real(
            ploty)

        self.curvature_buffer.push(min(left_curvature, right_curvature))

        ego_lateral_distance = line_fit.ego_distance_from_center(
            ploty)

        self.position_buffer.push(ego_lateral_distance)

        overlay = draw_overlay(
            undistorted_image, warped, warper.Minv, ploty, left_fitx, right_fitx, self.curvature_buffer.average_point(), self.position_buffer.average_point())

        if not self.debug:
            # simply write the input video with the overlay on top
            return overlay
        else:

            frames_text = f'Frames without detection: {self.frames_without_detection}'

            put_text(overlay, frames_text, ypos=150)

            # Construct a 2x2 video to visualize the pipeline stages
            warped = np.dstack((warped*255, warped*255, warped*255))

            binary_filtered = np.dstack(
                (binary_filtered*255, binary_filtered*255, binary_filtered*255))

            im_panels_top = cv2.hconcat([overlay, vis_img])
            im_panels_bottom = cv2.hconcat([binary_filtered, warped])

            im_final = cv2.vconcat([im_panels_top, im_panels_bottom])
            return im_final
