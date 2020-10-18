import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Transform(object):
    def to_binary(image, threshold):
        binary = np.zeros_like(image)
        binary[(image >= threshold[0]) & (image <= threshold[1])] = 1
        return binary

    def to_8_bits(image):
        img_abs = np.absolute(image)
        scaled = np.uint8(255 * img_abs / np.max(img_abs))
        return scaled

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
    def __init__(self):
        pass

    def filter_s(self, image, threshold=(150, 255)):
        """Convert to HLS color space and separate the S channel"""
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = Transform.to_binary(s_channel, threshold)
        return s_binary, s_channel


class SobelFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def filter_x(self, gray, threshold=(50, 255)):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        scaled = Transform.to_8_bits(sobel)
        binary = Transform.to_binary(scaled, threshold)
        return binary, scaled, sobel

    def filter_y(self, gray, threshold=(50, 255)):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        scaled = Transform.to_8_bits(sobel)
        binary = Transform.to_binary(scaled, threshold)
        return binary, scaled, sobel

    def filter_mag(self, sx, sy, threshold=(50, 255)):
        sobel = np.sqrt(sx ** 2 + sy ** 2)
        scaled = Transform.to_8_bits(sobel)
        binary = Transform.to_binary(scaled, threshold)
        return binary, scaled, sobel

    def filter_dir(self, sx, sy, threshold=(60, 20)):
        rad_threshold = Transform.deg_to_rad(threshold[0], threshold[1])
        absx = np.absolute(sx)
        absy = np.absolute(sy)
        sobel = np.arctan2(absy, absx)
        binary = Transform.to_binary(sobel, rad_threshold)
        return binary, sobel


class EdgeDetector(object):

    image = None
    result = None
    s_binary = None
    sobel_all_binary = None

    def __init__(self):
        self.sobel = SobelFilter(kernel_size=13)
        self.hls = HLSFilter()

    def detect(self, image):
        # gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # HLS
        s_binary, s_channel = self.hls.filter_s(image)

        # Sobel
        sx_binary, sx_scaled, sobel_x = self.sobel.filter_x(gray)
        sy_binary, sy_scaled, sobel_y = self.sobel.filter_y(gray)
        smag_binary, smag_scaled, sobel_mag = self.sobel.filter_mag(
            sobel_x, sobel_y)
        sdir_binary, sobel_dir = self.sobel.filter_dir(sobel_x, sobel_y)

        # combined
        sobel_xy_binary = Transform.binary_and(sx_binary, sy_binary)
        sobel_md_binary = Transform.binary_and(smag_binary, sdir_binary)
        sobel_all_binary = Transform.binary_or(
            sobel_xy_binary, sobel_md_binary)
        result = Transform.binary_or(sobel_all_binary, s_binary)

        # keep results for visualization
        self.image = image
        self.s_binary = cv2.cvtColor(s_binary, cv2.COLOR_GRAY2RGB)
        self.sobel_all_binary = cv2.cvtColor(
            sobel_all_binary, cv2.COLOR_GRAY2RGB)
        self.result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return result

    def build_result_vis(self):
        """image + s_binary + sobel_all_binary + output"""

        # scale binary images
        max_value = np.max(self.image)
        self.s_binary = self.s_binary * max_value
        self.sobel_all_binary = self.sobel_all_binary * max_value
        self.result = self.result * max_value

        def put_text(image, text, color=(0, 0, 255)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (10, 100), font, 3, color, 5, cv2.LINE_AA)

        # Put labels
        put_text(self.result, "Detection")
        put_text(self.s_binary, "HLS: S Channel Binary")
        put_text(self.sobel_all_binary, "Sobel Binary")

        # Build 2x2 image grid
        ca = np.concatenate((self.image, self.result), axis=1)
        cb = np.concatenate((self.s_binary, self.sobel_all_binary), axis=1)
        vis = np.concatenate((ca, cb), axis=0)

        return vis

    def display_result(self, fname, ax):
        vis = self.build_result_vis()

        # save
        #save_image(vis, fname, "edge_detection_", "_result")

        # show
        ax.imshow(vis)
        ax.axis("off")

    def display_hls(self):
        """image + s_channel + s_binary"""
        pass

    def display_sobel(self):
        """image + sx/thresh + sy/thresh + smag/thresh + sdir/thresh + XY + MD + ALL"""
        pass

    def display_pipeline(self, fname, ax):
        self.display_result(fname, ax)
        self.display_sobel()
        self.display_hls()
