"""
Camera package for camera calibration and undistorting camera images

"""

import pickle
import logging
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class WarpMachine:
    h = 720
    left = 210
    right = 1110
    top = 460
    top_left = 580
    top_right = 705
    dst_l = 0
    dst_r = 960

    def __init__(self, imshape):

        self.imshape = imshape

        self.src = self.get_src_region(self.imshape)
        self.dst = self.get_dst_region(self.imshape)
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_LINEAR)

    def get_src_region(self, imshape, height_factor=.6, top_left_factor=.45, right_top_factor=.55):
        left_bottom = [0, imshape[0]]
        left_top = [top_left_factor*imshape[1], height_factor*imshape[0]]
        right_top = [right_top_factor*imshape[1], height_factor*imshape[0]]
        right_bottom = [imshape[1], imshape[0]]

        vertices = np.array(
            [[left_bottom, left_top, right_top, right_bottom]], np.float32)
        return vertices

    def get_dst_region(self, imshape, height_factor=.6, top_left_factor=.45, right_top_factor=.55):
        print(imshape[0], imshape[1])
        left_bottom = [280, imshape[0]]
        left_top = [280, 0]
        right_top = [imshape[1] - 280, 0]
        right_bottom = [imshape[1] - 280, imshape[0]]

        vertices = np.array(
            [[left_bottom, left_top, right_top, right_bottom]], np.float32)
        return vertices

    def unwarp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.Minv, img_size, flags=cv2.INTER_LINEAR)

    def draw_src(self, image):
        cv2.polylines(image, [np.int32(self.src)], 1, (0, 255, 0), thickness=2)

    def draw_dst(self, image):
        cv2.polylines(image, [np.int32(self.dst)], 1, (255, 0, 0), thickness=2)


class PickleFile:
    """ Utility class to load/save pickle file"""

    def __init__(self, filename):
        self.filename = filename

    def exists(self):
        return os.path.isfile(self.filename)

    def save(self, data):
        with open(self.filename, 'wb') as file_handle:
            pickle.dump(obj=data, file=file_handle)

    def load(self):
        with open(self.filename, 'rb') as file_handle:
            return pickle.load(file=file_handle)


class Camera:
    """Computes objpoints, imgpoints pair based on chessboard images for calibration"""

    def __init__(self, nx, ny, calibration_images, calibration_filename):
        logging.info('Initializing camera model...')
        self.calibration_images = calibration_images
        self.calibration_file = PickleFile(calibration_filename)
        self.image_shape = self.get_shape(self.calibration_images[0])
        self.corner_images = []
        self.undistorted_images = []
        self.image_names = []
        self.calibration = {}
        self.nx = nx
        self.ny = ny
        logging.info('Camera image shape x:%d y:%d',
                     self.image_shape[0], self.image_shape[1])

        self.get_calibration_filenames_()
        self.calibrate()

    def get_shape(self, image_filename):
        """ Get shape of camera images """
        shape = cv2.imread(image_filename).shape
        return (shape[1], shape[0])

    def get_calibration_filenames_(self):
        """ Get basename of all calibration images, just to display in plot's title """
        for filename in self.calibration_images:
            self.image_names.append(os.path.basename(filename))

    def save_calibration_file(self):
        """ Save calibration file """
        self.calibration_file.save(self.calibration)

    def load_calibration_file(self):
        """ Load calibration file """
        self.calibration = self.calibration_file.load()

    def calibrate(self):
        """ Calculate calibration values based on chessboard images or load values from file """

        if self.calibration_file.exists():
            logging.info('Loading calibration file: "%s"',
                         self.calibration_file.filename)

            self.load_calibration_file()

        else:
            logging.info('No calibration file available, calibrating...')

            objp = np.zeros((self.nx * self.ny, 3), np.float32)
            objp[:, :2] = np.mgrid[0: self.nx, 0: self.ny].T.reshape(-1, 2)

            objpoints = []
            imgpoints = []

            for filename in self.calibration_images:
                logging.info('Finding corners in: "%s"', filename)
                imgp = self.find_corners(filename)

                if imgp is not None:
                    objpoints.append(objp)
                    imgpoints.append(imgp)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, self.image_shape, None, None)

            self.calibration = {'ret': ret, 'mtx': mtx, 'dist': dist,
                                'rvecs': rvecs, 'tvecs': tvecs}

            self.save_calibration_file()

    def find_corners(self, filename):
        """ Get identified corner points from a single chessboard image """

        bgr_image = cv2.imread(filename)

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        retval, corners = cv2.findChessboardCorners(
            gray_image, (self.nx, self.ny), None)

        chessboard_corners = cv2.drawChessboardCorners(
            bgr_image, (self.nx, self.ny), corners, retval)

        self.corner_images.append(chessboard_corners)

        if retval:
            return corners
        else:
            logging.warning(
                "Unable to find corners in: %s", filename
            )
            return

    def show_corner_images(self, save_file='chessboard_identified_corners.png'):
        """ Show calibration images with identified corners """
        logging.info("Plotting calibration images with corners")
        self.show_images_(self.corner_images, self.image_names, save_file)

    def show_undistorted_images(self, save_file='chessboard_undistorted.png'):
        """ Show calibration images after undistorting """
        logging.info("Plotting undistorted images")
        self.undistort_calibration_images_()
        self.show_images_(self.undistorted_images,
                          self.image_names, save_file)

    def show_images_(self, images, titles, save_file, rows=5, cols=4):
        """ Display calibration images in 5x4 subplot """
        figure, axes = plt.subplots(
            rows, cols, figsize=(15, 10), frameon=False)

        logging.info("Displaying %d images", len(images))
        for index, sub in enumerate(axes.flat):
            sub.axis('off')
            sub.set_title(titles[index], fontsize=9)
            sub.imshow(images[index])

        plt.tight_layout()
        plt.show()

        figure.savefig(save_file, dpi='figure')

    def undistort_image(self, img):
        """ Returns undistorted image """
        return cv2.undistort(img, self.calibration['mtx'], self.calibration['dist'])

    def undistort_calibration_images_(self):
        """ Undistort calibration files """
        for calibration_image in self.calibration_images:
            bgr_image = cv2.imread(calibration_image)
            self.undistorted_images.append(self.undistort_image(bgr_image))
