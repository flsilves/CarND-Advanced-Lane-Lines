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
        data = pickle.load(open(self.filename, "r+b"))
        return data


class CameraModel:
    """Computes objpoints, imgpoints pair based on chessboard images for calibration"""

    objpoints = []  # 3d points in real world space.
    imgpoints = []  # 2d points in image plane.

    cal_w = None
    cal_h = None
    mtx = None
    dist = None

    def __init__(self):
        self.nx = 9
        self.ny = 6
        self.calibration_images = glob.glob("camera_cal/calibration*.jpg")
        self.corner_images = []
        self.calibration_file = CalibrationFile("camera_cal/calibration.p")

    def save_calibration_file(self):
        data = dict()
        data["objpoints"] = self.objpoints
        data["imgpoints"] = self.imgpoints
        self.calibration_file.save(data)

    def load_calibration_file(self):
        "Load calibration file if it exists"
        if self.calibration_file.exists():
            data = self.calibration_file.load()
            self.objpoints = data["objpoints"]
            self.imgpoints = data["imgpoints"]
            return True
        else:
            return False

    def calibrate(self):
        """ Get all imgpoints"""
        # if self.load_calibration_file():
        #    return

        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0: self.nx, 0: self.ny].T.reshape(-1, 2)

        for filename in self.calibration_images:
            self.calibrate_single(filename, objp)

        self.save_calibration_file()

    def calibrate_single(self, filename, objp):
        """ Obtain objpoint and imgpoints from a single chessboard image """
        logging.info('file: %s', filename)

        bgr_image = cv2.imread(filename)

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        corners_found, corners = cv2.findChessboardCorners(
            gray_image, (self.nx, self.ny), None)

        chessboard_corners = cv2.drawChessboardCorners(
            bgr_image, (self.nx, self.ny), corners, corners_found)

        self.corner_images.append(chessboard_corners)

        if corners_found:
            self.objpoints.append(objp)
            self.imgpoints.append(corners)
        else:
            logging.warning(
                "Unable to find corners in: %s", filename
            )


def main():

    logging.basicConfig(level=logging.DEBUG)

    c = CameraModel()
    c.calibrate()
    # ProcessProjectVideo(subclip_seconds=None)


if __name__ == "__main__":
    # cProfile.run('main')
    main()
