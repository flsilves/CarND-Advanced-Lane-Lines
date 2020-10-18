import pickle
import logging
import os
import math
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import re


class CalibrationFile:
    def __init__(self, filename):
        self.filename = filename

    def exists(self):
        return os.path.isfile(self.filename)

    def save(self, data):
        pickle.dump(data, open(self.filename, "w+b"))

    def load(self):
        return pickle.load(open(self.filename, "r+b"))


class CameraModel:
    """Computes objpoints, imgpoints pair based on chessboard images for calibration"""

    objpoints = []  # 3d points in real world space.
    imgpoints = []  # 2d points in image plane.

    def __init__(self):
        self.nx = 9
        self.ny = 6
        self.calibration_images = glob.glob("camera_cal/calibration*.jpg")
        self.calibration_file = CalibrationFile("camera_cal/calibration.p")
        self.image_shape = self.get_shape(self.calibration_images[0])

        self.corner_images = []
        self.image_names = []

        logging.info('Camera image shape x:%d y:%d',
                     self.image_shape[0], self.image_shape[1])

    def get_shape(self, img):
        shape = cv2.imread(img).shape
        return (shape[1], shape[0])

    def save_calibration_file(self, ret, mtx, dist, rvecs, tvecs):
        """ Save calibration file """
        data = {'ret': ret, 'mtx': mtx, 'dist': dist,
                'rvecs': rvecs, 'tvecs': tvecs}
        self.calibration_file.save(data)

    def load_calibration_file(self):
        """ Load calibration file """
        data = self.calibration_file.load()
        return data['ret'], data['mtx'], data['dist'], data['rvecs'], data['tvecs']

    def calibrate(self):
        """ Get all imgpoints"""

        if self.calibration_file.exists():
            logging.info('Loading calibration file: "%s"',
                         self.calibration_file.filename)
            ret, mtx, dist, rvecs, tvecs = self.load_calibration_file()

        else:
            logging.info('No calibration file available, calibrating...')

            objp = np.zeros((self.nx * self.ny, 3), np.float32)
            objp[:, :2] = np.mgrid[0: self.nx, 0: self.ny].T.reshape(-1, 2)

            for filename in self.calibration_images:
                logging.info('Finding corners in: %s', filename)
                self.find_corners(filename, objp)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.image_shape, None, None)

            self.save_calibration_file(ret, mtx, dist, rvecs, tvecs)

    def find_corners(self, filename, objp):
        """ Obtain objpoint and imgpoints from a single chessboard image """

        bgr_image = cv2.imread(filename)

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        corners_found, corners = cv2.findChessboardCorners(
            gray_image, (self.nx, self.ny), None)

        chessboard_corners = cv2.drawChessboardCorners(
            bgr_image, (self.nx, self.ny), corners, corners_found)

        self.corner_images.append(chessboard_corners)
        self.image_names.append(os.path.basename(filename))

        if corners_found:
            self.objpoints.append(objp)
            self.imgpoints.append(corners)
        else:
            logging.warning(
                "Unable to find corners in: %s", filename
            )

    def show_calibration_images(self, cols=4, rows=5):
        """ Display calibration images in a subplot grid """

        figure, axes = plt.subplots(
            rows, cols, figsize=(15, 10), frameon=False)

        for index, sub in enumerate(axes.flat):
            sub.axis('off')
            sub.set_title(self.image_names[index], fontsize=9)
            sub.imshow(self.corner_images[index])

        plt.tight_layout()
        plt.show()

        figure.savefig('chessboard_identified_corners.png', dpi='figure')


def main():
    logging.basicConfig(level=logging.INFO)

    c = CameraModel()
    c.calibrate()
    # c.show_calibration_images()
    # ProcessProjectVideo(subclip_seconds=None)


if __name__ == "__main__":
    # cProfile.run('main')
    main()
