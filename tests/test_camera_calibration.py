"""
Unit tests for camera calibration
"""

import unittest
import sys
import glob
import os
import cv2
import matplotlib.pyplot as plt
import logging


sys.path.append("..")  # nopep8
from camera import Camera  # nopep8

CALIBRATION_IMAGES_DIR = "../camera_cal/"

ROAD_IMAGES_DIR = "../test_images/"
TEST_OUTPUT_DIR = "test_camera_calibration_output"


def get_images_from_dir(path):
    image_list = []
    for filename in glob.glob(f'{path}/*.jpg'):
        jpg_image = cv2.imread(filename)
        image_list.append(jpg_image)

    for filename in glob.glob(f'{path}/*.png'):
        png_image = cv2.imread(filename)
        image_list.append(png_image)

    return image_list


def save_before_and_after_image(before_img, after_img, save_file):

    before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
    after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)

    image_dpi = 72

    width_inches = int(2*before_img.shape[1]) / image_dpi
    height_inches = int(2*before_img.shape[0]) / image_dpi

    logging.debug("width %f height:%f", width_inches, height_inches)
    figsize = (width_inches, height_inches)

    figure, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, dpi=image_dpi, frameon=False)

    ax1.imshow(before_img)
    ax1.axis('off')
    ax1.set_title("original", fontsize=25)
    ax2.axis('off')

    ax2.imshow(after_img)
    ax2.set_title("undistorted", fontsize=25)

    figure.savefig(save_file, dpi='figure',
                   bbox_inches='tight')

    plt.close('all')


class CameraCalibrationTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        calibration_images = glob.glob('../camera_cal/calibration*.jpg')
        self.camera = Camera(nx=9, ny=6, calibration_images=calibration_images,
                             calibration_filename='../calibration.pickle')

    def tearDown(self):
        return

    def test_undistort_road_images(self):
        test_images = get_images_from_dir(ROAD_IMAGES_DIR)
        logging.info("Undistorting test images")

        for idx, test_image in enumerate(test_images):
            logging.debug("Image %d", idx)
            undistorted_image = self.camera.undistort_image(test_image)

            filename = f"{TEST_OUTPUT_DIR}/road_{str(idx)}_undistorted.png"
            save_before_and_after_image(
                test_image, undistorted_image, filename)

    def test_undistort_calibration_images(self):
        test_images = get_images_from_dir(CALIBRATION_IMAGES_DIR)
        logging.info("Undistorting calibration images")

        for idx, test_image in enumerate(test_images):
            logging.debug("Image %d", idx)
            undistorted_image = self.camera.undistort_image(test_image)

            filename = f"{TEST_OUTPUT_DIR}/calibration{str(idx)}_undistorted.png"
            save_before_and_after_image(
                test_image, undistorted_image, filename)


if __name__ == '__main__':
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
    else:
        files = glob.glob(f'{TEST_OUTPUT_DIR}/*.png')
        logging.info("Deleting %d images from previous run", len(files))
        for f in files:
            os.remove(f)

    unittest.main()
