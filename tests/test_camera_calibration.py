"""
Unit tests for camera calibration
"""

import unittest
import logging
from test_utils import *


TEST_OUTPUT_DIR = 'test_camera_calibration_output'


class CameraCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(nx=9, ny=6, calibration_images=CALIBRATION_IMAGES,
                             calibration_filename=CALIBRATION_FILE)

    def tearDown(self):
        return

    def test_undistort_road_images(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Undistorting road test images')

        self.undistort_and_save(test_images, filenames)

    def test_undistort_calibration_images(self):
        test_images, filenames = get_images_from_dir(CALIBRATION_IMAGES_DIR)
        logging.info('Undistorting calibration images')

        self.undistort_and_save(test_images, filenames)

    def undistort_and_save(self, images, filenames):
        for idx, test_image in enumerate(images):
            logging.debug('Image %d', idx)
            undistorted_image = self.camera.undistort_image(test_image)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_undistort.png'

            save_before_and_after_image(
                test_image, undistorted_image, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)
    unittest.main()
