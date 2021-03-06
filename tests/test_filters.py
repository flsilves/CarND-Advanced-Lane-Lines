"""
Unit tests for filter module
"""

import unittest
import logging
from test_utils import *
import numpy as np


TEST_OUTPUT_DIR = 'test_filters'


class FilterTests(unittest.TestCase):

    def setUp(self):
        self.camera = Camera(nx=9, ny=6, calibration_images=CALIBRATION_IMAGES,
                             calibration_filename=CALIBRATION_FILE)

        self.sobel = SobelFilter(kernel_size=3)
        self.hls = HLSFilter()
        self.combined = CombinedFilter()
        self.test_images, self.filenames = get_images_from_dir(ROAD_IMAGES_DIR)

    def tearDown(self):
        return

    def test_sobel_y(self):
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Sobel_y: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary, scaled, sobel = self.sobel.filter_y(gray)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_y.png'
            save_before_and_after_image(
                test_image, binary, filename, 'gray')

    def test_sobel_x(self):
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Sobel_x: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary, scaled, sobel = self.sobel.filter_x(gray)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_x.png'
            save_before_and_after_image(
                test_image, binary, filename, 'gray')

    def test_sobel_dir(self):
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Sobel dir mag: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary_x, scaled_x, sobel_x = self.sobel.filter_x(gray)
            binary_y, scaled_y, sobel_y = self.sobel.filter_y(gray)

            sdir_binary, sobel_dir = self.sobel.filter_dir(sobel_x, sobel_y)
            smag_binary, smag_scaled = self.sobel.filter_mag(sobel_x, sobel_y)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_dir.png'
            save_before_and_after_image(
                test_image, sdir_binary, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_mag.png'
            save_before_and_after_image(
                test_image, smag_binary, filename, 'gray')

    def test_sobel_xy_mag(self):
        logging.info('Applying sobel dir on road images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Sobel xy|mag: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            sx_binary, sx_scaled, sobel_x = self.sobel.filter_x(gray)
            sy_binary, sy_scaled, sobel_y = self.sobel.filter_y(gray)
            smag_binary, smag_scaled = self.sobel.filter_mag(sobel_x, sobel_y)
            sdir_binary, sobel_dir = self.sobel.filter_dir(sobel_x, sobel_y)

            sobel_xy_binary = Transform.binary_and(sx_binary, sy_binary)
            sobel_md_binary = Transform.binary_and(smag_binary, sdir_binary)
            sobel_all_binary = Transform.binary_or(
                sobel_xy_binary, sobel_md_binary)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_xy.png'
            save_before_and_after_image(
                test_image, sobel_xy_binary, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_md.png'
            save_before_and_after_image(
                test_image, sobel_md_binary, filename, 'gray')

    def test_sobel(self):
        logging.info('Apply sobel and hls filter')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Sobel_all: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            binary = self.sobel.filter(undistorted_image)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel_final.png'
            save_before_and_after_image(
                test_image, binary, filename, 'gray')

    def test_s_filter(self):
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'S_filter: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            s_binary, s_channel = self.hls.filter(undistorted_image)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_hls.png'
            save_before_and_after_image(
                test_image, s_binary, filename, 'gray')

    def test_combined(self):
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Combined: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            #binary = self.combined.filter(undistorted_image)
            s_binary, s_channel = self.hls.filter(undistorted_image)
            binary_sobel = self.sobel.filter(undistorted_image)
            binary_final = Transform.binary_or(binary_sobel, s_binary)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_hls.png'
            save_before_and_after_image(
                test_image, s_binary, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_sobel.png'
            save_before_and_after_image(
                test_image, binary_sobel, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_final.png'
            save_before_and_after_image(
                test_image, binary_final, filename, 'gray')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)

    unittest.main()
