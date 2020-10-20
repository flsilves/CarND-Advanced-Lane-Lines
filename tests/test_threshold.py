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

    def tearDown(self):
        return

    def test_threshold_images(self):
        test_images, filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info('Applying threshold on road images')

        for idx, test_image in enumerate(test_images):
            logging.info('Image %d', idx)

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            binary, scaled, sobel = self.sobel.filter_y(gray)

            mean = np.mean(sobel, axis=(0, 1))

            mean2 = np.mean(scaled, axis=(0, 1))

            logging.info('max %s', np.max(scaled))
            logging.info('min %s', np.min(scaled))

            np.savetxt('filename.txt', scaled, fmt="%d")

            logging.info('median %s', mean)
            logging.info('median2 %s', mean2)

            # print(binary)

            logging.info('binary gray %s', binary.shape)
            logging.info('scaled gray %s', gray.shape)

            filename = f'{TEST_OUTPUT_DIR}/{filenames[idx]}_threshold.png'
            save_before_and_after_image(
                test_image, binary, filename)

            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)

    unittest.main()
