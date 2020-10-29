"""
Unit tests for line finding 
"""

import unittest
import logging
from test_utils import *
import numpy as np

TEST_OUTPUT_DIR = 'test_tracking'


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(nx=9, ny=6, calibration_images=CALIBRATION_IMAGES,
                             calibration_filename=CALIBRATION_FILE)

        self.sobel = SobelFilter(kernel_size=3)
        self.hls = HLSFilter()
        self.combined = CombinedFilter()

        self.test_images, self.filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        self.warper = Warper()
        self.lane_tracker = LaneTracker(CALIBRATION_IMAGES, CALIBRATION_FILE)

    def tearDown(self):
        return

    def test_line_finding(self):

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Finding Lanes: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            undistorted_image_copy = np.copy(undistorted_image)

            filtered = self.combined.filter(undistorted_image)

            warped = self.warper.warp(filtered)

            self.warper.draw_src(undistorted_image)
            self.warper.draw_dst(undistorted_image)

            line_fit = LineFit(
                undistorted_image.shape)

            ploty, left_fitx, right_fitx, vis_img, _, histogram = line_fit.find_lines(
                warped)

            left_curvature_m, right_curvature_m = line_fit.measure_curvature_real(
                ploty)

            ego_lateral_distance = line_fit.ego_distance_from_center(ploty)

            overlay = draw_overlay(
                undistorted_image_copy, warped, self.warper.Minv, ploty, left_fitx, right_fitx, min(left_curvature_m, right_curvature_m), ego_lateral_distance)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_hist.png'
            plot_histogram(
                warped[warped.shape[0]//2:, :], histogram, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_filtered.png'
            save_before_and_after_image(
                undistorted_image, filtered, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_warp.png'
            save_before_and_after_image(
                undistorted_image, warped, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_poly.png'
            save_before_and_after_image(
                undistorted_image, vis_img, filename)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_overlay.png'
            save_before_and_after_image(
                undistorted_image, overlay, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)

    unittest.main()
