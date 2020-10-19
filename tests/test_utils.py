import unittest
import glob
import logging
import matplotlib.pyplot as plt
import sys
import cv2


sys.path.append('..')  # nopep8
from camera import Camera  # nopep8
from threshold import *  # nopep8

CALIBRATION_FILE = '../calibration.pickle'
CALIBRATION_IMAGES_DIR = '../camera_cal/'
ROAD_IMAGES_DIR = '../test_images/'


def get_images_from_dir(path):
    image_list = []
    for filename in glob.glob(f'{path}/*.jpg'):
        jpg_image = cv2.imread(filename)
        jpg_image = cv2.cvtColor(jpg_image, cv2.COLOR_BGR2RGB)
        image_list.append(jpg_image)

    for filename in glob.glob(f'{path}/*.png'):
        png_image = cv2.imread(filename)
        png_image = cv2.cvtColor(png_image, cv2.COLOR_BGR2RGB)
        image_list.append(png_image)

    return image_list


def save_before_and_after_image(before_img, after_img, save_file):

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

    ax2.imshow(after_img)
    ax2.set_title('undistorted', fontsize=25)

    figure.savefig(save_file, dpi='figure',
                   bbox_inches='tight')

    plt.close('all')
