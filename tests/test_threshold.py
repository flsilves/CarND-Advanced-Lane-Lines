"""
Unit tests for image thresholding
"""

import unittest
import logging
from test_utils import *


TEST_OUTPUT_DIR = 'test_threshold_images'


class ImageThresholdTest(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(nx=9, ny=6, calibration_images=CALIBRATION_IMAGES,
                             calibration_filename=CALIBRATION_FILE)

    def tearDown(self):
        return

    def test_threshold_images(self):
        test_images = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying threshold on road images')

        for idx, test_image in enumerate(test_images):
            logging.debug('Image %d', idx)

            undistorted_image = self.camera.undistort_image(test_image)

            #edge_detector = EdgeDetector()
            #edges = edge_detector.detect(undistorted_image)
            filename = f'{TEST_OUTPUT_DIR}/threshold_{str(idx)}.png'
            save_before_and_after_image(
                test_image, test_image, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)

    unittest.main()
