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
        self.filename = "camera_cal/" + filename

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
    images = []  # images from which these points where computed.

    cal_w = None
    cal_h = None
    mtx = None
    dist = None

    def __init__(self):
        self.nx = 9
        self.ny = 6
        self.target_images = glob.glob("camera_cal/calibration*.jpg")

        self.calibration_file = CalibrationFile("calibration.p")

    def save(self):
        data = dict()
        data["objpoints"] = self.objpoints
        data["imgpoints"] = self.imgpoints
        data["images"] = self.images
        self.calibration_file.save(data)

    def load_calibration_file(self):
        if self.calibration_file.exists():
            data = self.calibration_file.load()
            self.objpoints = data["objpoints"]
            self.imgpoints = data["imgpoints"]
            self.images = data["images"]
            return True
        else:
            return False

    def calibrate(self):
        """ Get all imgpoints"""
        if self.load_calibration_file():
            return

        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0: self.nx, 0: self.ny].T.reshape(-1, 2)

        for image in self.target_images:
            self.calibrate_single(image, objp)

        self.save()

    def calibrate_single(self, filename, objp):
        """ Obtain objpoint and imgpoints from a single chessboard image """
        logging.info('file: %s', filename)

        img = cv2.imread(filename)

        corners_found, corners = cv2.findChessboardCorners(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self.nx, self.ny), None)

        if corners_found:
            self.images.append(filename)
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
