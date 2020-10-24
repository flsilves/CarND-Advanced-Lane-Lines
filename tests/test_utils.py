"""
Free-hand functions for unit tests
"""

import unittest
import glob
import logging
import sys
import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


sys.path.append('..')  # nopep8
from camera import Camera, WarpMachine
from filter import HLSFilter, SobelFilter, CombinedFilter, Transform
from lane_finder import *


CALIBRATION_FILE = '../calibration.pickle'
CALIBRATION_IMAGES_DIR = '../camera_cal/'
ROAD_IMAGES_DIR = '../test_images/'

CALIBRATION_IMAGES = glob.glob(
    f'{CALIBRATION_IMAGES_DIR}/calibration*.jpg')


def remove_old_files(target_directory):
    """ Delete files from previous test run """

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        files = glob.glob(f'{target_directory}/*.png')
        logging.info('Deleting %d images from previous run', len(files))
        for f in files:
            os.remove(f)


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def get_images_from_dir(path):
    image_list = []
    filenames = []

    for filename in glob.glob(f'{path}/*.jpg'):
        jpg_image = cv2.imread(filename)
        jpg_image = cv2.cvtColor(jpg_image, cv2.COLOR_BGR2RGB)
        image_list.append(jpg_image)
        filenames.append(Path(filename).stem)

    for filename in glob.glob(f'{path}/*.png'):
        png_image = cv2.imread(filename)
        png_image = cv2.cvtColor(png_image, cv2.COLOR_BGR2RGB)
        image_list.append(png_image)
        filenames.append(Path(filename).stem)

    return image_list, filenames


def save_before_and_after_image(before_img, after_img, save_file, cmap='viridis'):

    image_dpi = 72

    width_inches = int(2*before_img.shape[1]) / image_dpi
    height_inches = int(2*before_img.shape[0]) / image_dpi

    logging.debug('width %f height:%f', width_inches, height_inches)
    figsize = (width_inches, height_inches)

    figure, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, dpi=image_dpi, frameon=False)

    figure.tight_layout()
    ax1.imshow(before_img)
    ax1.axis('off')
    ax1.set_title('original', fontsize=25)
    ax2.axis('off')

    ax2.imshow(after_img, cmap=cmap)
    ax2.set_title('undistorted', fontsize=25)

    figure.savefig(save_file, dpi='figure',
                   bbox_inches='tight')

    plt.close('all')
