"""
Unit tests for image thresholding
"""

import unittest
import logging
from test_utils import *
import numpy as np


TEST_OUTPUT_DIR = 'test_threshold_images'


class ImageThresholdTest(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(nx=9, ny=6, calibration_images=CALIBRATION_IMAGES,
                             calibration_filename=CALIBRATION_FILE)

        self.sobel = SobelFilter(kernel_size=3)
        self.hls = HLSFilter()

    def tearDown(self):
        return

    def xtest_sobel_y(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(test_images):
            logging.info(f'-> {filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary, scaled, sobel = self.sobel.filter_y(gray)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_sobel_y.png'
            save_before_and_after_image(
                test_image, binary, filename)

    def xtest_sobel_x(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(test_images):
            logging.info(f'-> {filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary, scaled, sobel = self.sobel.filter_x(gray)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_sobel_x.png'
            save_before_and_after_image(
                test_image, binary, filename)

    def xtest_sobel_dir(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(test_images):
            logging.info(f'-> {filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary_x, scaled_x, sobel_x = self.sobel.filter_x(gray)
            binary_y, scaled_y, sobel_y = self.sobel.filter_y(gray)

            sdir_binary, sobel_dir = self.sobel.filter_dir(sobel_x, sobel_y)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_sobel_dir.png'
            save_before_and_after_image(
                test_image, sdir_binary, filename)

    def xtest_sobel_xy_mag_all(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying sobel dir on road images')

        for idx, test_image in enumerate(test_images):
            logging.info(f'-> {filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            sx_binary, sx_scaled, sobel_x = self.sobel.filter_x(gray)
            sy_binary, sy_scaled, sobel_y = self.sobel.filter_y(gray)
            smag_binary, smag_scaled = self.sobel.filter_mag(sobel_x, sobel_y)
            sdir_binary, sobel_dir = self.sobel.filter_dir(sobel_x, sobel_y)

            sobel_xy_binary = Transform.binary_and(sx_binary, sy_binary)
            sobel_md_binary = Transform.binary_and(smag_binary, sdir_binary)
            sobel_all_binary = Transform.binary_or(
                sobel_xy_binary, sobel_md_binary)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_sobel_xy.png'
            save_before_and_after_image(
                test_image, sobel_xy_binary, filename)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_sobel_md.png'
            save_before_and_after_image(
                test_image, sobel_md_binary, filename)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_sobel_all.png'
            save_before_and_after_image(
                test_image, sobel_md_binary, filename)

    def xtest_all(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Apply sobel and hls filter')

        for idx, test_image in enumerate(test_images):
            logging.info(f'-> {filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            s_binary, s_channel = self.hls.filter_s(undistorted_image)

            gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

            sobel_all_binary = self.sobel.filter_all(gray)

            result = Transform.binary_or(sobel_all_binary, s_binary)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_filter.png'
            save_before_and_after_image(
                test_image, result, filename)

    def test_s_filter(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying sobel_y on road images')

        for idx, test_image in enumerate(test_images):
            logging.info(f'-> {filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            s_binary, s_channel = self.hls.filter_s(undistorted_image)

            shape = s_binary.shape

            half = shape[0]//2

            test_image = test_image[half:, :]
            s_binary = s_binary[half:, :]

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_hls.png'
            save_before_and_after_image(
                test_image, s_binary, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)

    unittest.main()
