import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO move transform to gray to the inside of this class


class Transform(object):
    def to_binary(image, thresholds):
        """ """
        binary_image = np.zeros_like(image)
        binary_image[(image >= thresholds[0]) & (image <= thresholds[1])] = 1
        return binary_image

    def scale(image, bits=8):
        """ Resize values of image to a 0-255 scale"""
        max_out = 2**bits - 1
        print(max_out)
        image = np.absolute(image)
        scaled_image = (max_out * image) / np.max(image)
        scaled_image = np.uint8(scaled_image)
        return scaled_image

    def binary_and(binary_1, binary_2):
        binary = np.zeros_like(binary_1)
        binary[(binary_1 == 1) & (binary_2 == 1)] = 1
        return binary

    def binary_or(binary_1, binary_2):
        binary = np.zeros_like(binary_1)
        binary[(binary_1 == 1) | (binary_2 == 1)] = 1
        return binary

    def deg_to_rad(theta_deg, delta_deg):
        theta = theta_deg * np.pi / 180.0
        delta = delta_deg * np.pi / 180.0
        return (theta - delta, theta + delta)


class HLSFilter:

    # TODO adaptative threshold based on values of current image, it should all be relative
    def __init__(self):
        pass

    def filter_s(self, image, thresholds=[150, 255]):
        """ Transform RBG image to HLS, return binary image based on Saturation channel """
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        shape = s_channel.shape

        half = shape[0]//2

        thresholds[0] = 5*np.median(s_channel[half:, :])
        #thresholds[0] = 2*np.median(s_channel)

        s_binary = Transform.to_binary(s_channel, thresholds)
        return s_binary, s_channel


class SobelFilter:
    """ Apply sobel filter on gray images"""
    # TODO adaptative threshold based on values of current image, it should all be relative

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def filter_x(self, gray, thresholds=[50, 255]):
        """ Filter by sobel x component """
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        scaled = Transform.scale(sobel, bits=8)

        half = scaled.shape[0]//2
        thresholds[0] = 4*np.median(scaled[half:, :])
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    def filter_y(self, gray, thresholds=[50, 255]):
        """ Filter by sobel y component """
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        scaled = Transform.scale(sobel, bits=8)
        half = scaled.shape[0]//2
        thresholds[0] = 4*np.median(scaled[half:, :])
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    # TODO explore giving a component more impact than other
    def filter_mag(self, sx, sy, thresholds=[50, 255]):
        """ Filter based on combined sobel x and y  """
        sobel_magnitude = np.sqrt(sx ** 2 + sy ** 2)
        scaled = Transform.scale(sobel_magnitude, bits=8)
        half = scaled.shape[0]//2
        thresholds[0] = 2*np.median(scaled[half:, :])
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled

    # TODO check the filtering by direction #
    def filter_dir(self, sx, sy, thresholds=[0.7, 1.3]):
        sobel = np.arctan2(np.absolute(sy), np.absolute(sx))
        binary = Transform.to_binary(sobel, thresholds)
        return binary, sobel

    def filter_all(self, gray):
        # threshoold should be relative based on the average of the gray image, check my previous project
        sx_binary, sx_scaled, sobel_x = self.filter_x(gray)
        sy_binary, sy_scaled, sobel_y = self.filter_y(gray)
        smag_binary, smag_scaled = self.filter_mag(sobel_x, sobel_y)
        sdir_binary, sobel_dir = self.filter_dir(sobel_x, sobel_y)

        # TODO Probably tune this
        sobel_xy_binary = Transform.binary_and(sx_binary, sy_binary)
        sobel_md_binary = Transform.binary_and(smag_binary, sdir_binary)
        sobel_all_binary = Transform.binary_or(
            sobel_xy_binary, sobel_md_binary)

        return sobel_all_binary
