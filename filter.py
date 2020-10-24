import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Transform(object):
    """ Util class for binary images """
    def to_binary(image, thresholds):
        """ """
        binary_image = np.zeros_like(image)
        binary_image[(image >= thresholds[0]) & (image <= thresholds[1])] = 1
        return binary_image

    def scale(image, bits=8):
        """ Resize values of image to a 0-255 scale"""
        max_out = 2**bits - 1
        image = np.absolute(image)
        scaled_image = (max_out * image) / np.max(image)
        scaled_image = np.uint8(scaled_image)
        return scaled_image

    def binary_and(left, right):
        """ Return AND of two binary images """
        binary = np.zeros_like(left)
        binary[(left == 1) & (right == 1)] = 1
        return binary

    def binary_or(left, right):
        """ Return OR of two binary images """
        binary = np.zeros_like(left)
        binary[(left == 1) | (right == 1)] = 1
        return binary


class CombinedFilter(object):
    """ Combine HLS and Sobel filters """

    def __init__(self, kernel_size=3):
        self.hls_filter = HLSFilter()
        self.sobel_filter = SobelFilter(kernel_size)

    def filter(self, image):
        s_binary, s_channel = self.hls_filter.filter(image)
        binary_sobel = self.sobel_filter.filter(image)
        return Transform.binary_or(binary_sobel, s_binary)


class HLSFilter:
    """ Saturation filter in HLS space """

    def __init__(self):
        pass

    def filter(self, image, thresholds=[150, 255]):
        """ Transform BGR image to HLS, return binary image based on Saturation channel """
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]

        shape = s_channel.shape

        half = shape[0]//2

        thresholds[0] = 3*np.median(s_channel[half:, :])

        s_binary = Transform.to_binary(s_channel, thresholds)
        return s_binary, s_channel


class SobelFilter:
    """ Sobel Filter:  (X & Y) | (DIR & MAG) """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def filter_x(self, gray, thresholds=[50, 255]):
        """ Filter gray image by sobel x component """
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        scaled = Transform.scale(sobel, bits=8)

        half = scaled.shape[0]//2
        thresholds[0] = 4*np.median(scaled[half:, :])
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    def filter_y(self, gray, thresholds=[50, 255]):
        """ Filter gray image by sobel y component """
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        scaled = Transform.scale(sobel, bits=8)
        half = scaled.shape[0]//2
        thresholds[0] = 4*np.median(scaled[half:, :])
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    def filter_mag(self, sx, sy, thresholds=[50, 255]):
        """ Filter by magnitude given sobel x and y  """
        sobel_magnitude = np.sqrt(sx ** 2 + sy ** 2)
        scaled = Transform.scale(sobel_magnitude, bits=8)
        half = scaled.shape[0]//2
        thresholds[0] = 2*np.median(scaled[half:, :])
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled

    def filter_dir(self, sx, sy, thresholds=[0.7, 1.3]):
        """ Filter gray image by direction (rad) """
        sobel = np.arctan2(np.absolute(sy), np.absolute(sx))
        binary = Transform.to_binary(sobel, thresholds)
        return binary, sobel

    def filter(self, image):
        """ Complete sobel filter, input: BGR image """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sx_binary, sx_scaled, sobel_x = self.filter_x(gray)
        sy_binary, sy_scaled, sobel_y = self.filter_y(gray)
        smag_binary, smag_scaled = self.filter_mag(sobel_x, sobel_y)
        sdir_binary, sobel_dir = self.filter_dir(sobel_x, sobel_y)

        sobel_xy_binary = Transform.binary_and(sx_binary, sy_binary)
        sobel_md_binary = Transform.binary_and(smag_binary, sdir_binary)
        sobel_all_binary = Transform.binary_or(
            sobel_xy_binary, sobel_md_binary)

        return sobel_all_binary
