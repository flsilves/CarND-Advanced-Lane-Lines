"""
Unit tests for camera calibration
"""

import unittest
import logging
import sys
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from test_utils import *


TEST_OUTPUT_DIR = 'test_threshold_images'


class CameraCalibrationTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        calibration_images = glob.glob(
            f'{CALIBRATION_IMAGES_DIR}/calibration*.jpg')
        self.camera = Camera(nx=9, ny=6, calibration_images=calibration_images,
                             calibration_filename=CALIBRATION_FILE)

    def tearDown(self):
        return

    def xtest_threshold_road_images(self):
        test_images = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying threshold on road images')

        for idx, test_image in enumerate(test_images):
            logging.debug('Image %d', idx)

            undistorted_image = self.camera.undistort_image(test_image)

            edge_detector = EdgeDetector()
            edges = edge_detector.detect(undistorted_image)
            filename = f'{TEST_OUTPUT_DIR}/threshold_{str(idx)}_undistorted.png'
            save_before_and_after_image(
                test_image, edges, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
    else:
        files = glob.glob(f'{TEST_OUTPUT_DIR}/*.png')
        logging.info('Deleting %d images from previous run', len(files))
        for f in files:
            os.remove(f)

    unittest.main()
