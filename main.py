""" Main module """

import logging
import glob
from camera import Camera


def main():
    """ Main """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    calibration_images = glob.glob('camera_cal/calibration*.jpg')

    camera = Camera(nx=9, ny=6, calibration_images=calibration_images,
                    calibration_filename='calibration.pickle')

    camera.show_undistorted_images()


if __name__ == "__main__":
    main()
