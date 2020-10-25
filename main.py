""" Main module """

import logging
import glob
from camera import Camera
from lane_tracker import LaneTracker

CALIBRATION_FILE = 'calibration.pickle'
CALIBRATION_IMAGES = glob.glob(
    'camera_cal/calibration*.jpg')


def main():
    """ Main """
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    input_file = "project_video.mp4"
    output_file = "project_video_output.mp4"

    ProcessProjectVideo(input_file, output_file)


def ProcessProjectVideo(input_file, output_file):
    logging.info(f'Processing video: {input_file} -> {output_file}')

    tracker = LaneTracker(CALIBRATION_IMAGES, CALIBRATION_FILE)
    tracker.process_video(input_file, output_file)


if __name__ == "__main__":
    main()
