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
from utils import *


sys.path.append("..")  # nopep8
from camera import Camera  # nopep8
from threshold import *  # nopep8

CALIBRATION_IMAGES_DIR = "../camera_cal/"
ROAD_IMAGES_DIR = "../test_images/"

TEST_OUTPUT_DIR = "test_threshold_images"


class ThresholdImageTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        calibration_images = glob.glob('../camera_cal/calibration*.jpg')
        self.camera = Camera(nx=9, ny=6, calibration_images=calibration_images,
                             calibration_filename='../calibration.pickle')

    def tearDown(self):
        return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
    else:
        files = glob.glob(f'{TEST_OUTPUT_DIR}/*.png')
        logging.info("Deleting %d images from previous run", len(files))
        for f in files:
            os.remove(f)

    unittest.main()
