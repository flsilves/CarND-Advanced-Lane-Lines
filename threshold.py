import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Transform(object):
    def to_binary(image, thresholds):
        """ """
        binary_image = np.zeros_like(image)
        binary_image[(image >= thresholds[0]) & (image <= thresholds[1])] = 1
        return binary_image

    def to_8_bits(image):
        """ Resize values of image to a 0-255 scale"""
        bits = 8
        max_out = bits**2
        image = np.absolute(image)
        scaled_image = np.uint8(max_out * image / np.max(image))
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

    def binary_s_filter(self, image, thresholds=(150, 255)):
        """ Transform RBG image to HLS, return binary image based on Saturation channel """
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = Transform.to_binary(s_channel, thresholds)
        return s_binary, s_channel


class SobelFilter:
    """ Apply sobel filter on gray images"""
    ## TODO adaptative threshold based on values of current image, it should all be relative

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def filter_x(self, gray, thresholds=(50, 255)):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        scaled = Transform.to_8_bits(sobel)
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    def filter_y(self, gray, thresholds=(50, 255)):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        scaled = Transform.to_8_bits(sobel)
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    def filter_mag(self, sx, sy, thresholds=(50, 255)):
        sobel = np.sqrt(sx ** 2 + sy ** 2)
        scaled = Transform.to_8_bits(sobel)
        binary = Transform.to_binary(scaled, thresholds)
        return binary, scaled, sobel

    def filter_dir(self, sx, sy, thresholds=(60, 20)):
        rad_threshold = Transform.deg_to_rad(
            thresholds[0], thresholds[1])  # TODO check this
        absx = np.absolute(sx)
        absy = np.absolute(sy)
        sobel = np.arctan2(absy, absx)
        binary = Transform.to_binary(sobel, rad_threshold)
        return binary, sobel

   def binary_all_filter(self, gray)
        sx_binary, sx_scaled, sobel_x = self.sobel.filter_x(gray)
        sy_binary, sy_scaled, sobel_y = self.sobel.filter_y(gray)
        smag_binary, smag_scaled, sobel_mag = self.sobel.filter_mag(sobel_x, sobel_y)
        sdir_binary, sobel_dir = self.sobel.filter_dir(sobel_x, sobel_y)

        sobel_xy_binary = Transform.binary_and(sx_binary, sy_binary)
        sobel_md_binary = Transform.binary_and(smag_binary, sdir_binary)
        sobel_all_binary = Transform.binary_or(sobel_xy_binary, sobel_md_binary)

        return sobel_all_binary     
