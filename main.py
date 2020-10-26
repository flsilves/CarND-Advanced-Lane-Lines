""" Main module """

import logging
import glob
from camera import Camera
from lane_tracker import LaneTracker

CALIBRATION_FILE = 'calibration.pickle'
CALIBRATION_IMAGES = glob.glob(
    'camera_cal/calibration*.jpg')


PROJECT_VIDEO = 'project_video'
CHALLENGE_VIDEO = 'challenge_video'
HARDER_VIDEO = 'harder_challenge_video'


def main():
    """ Main """
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    selected_video = PROJECT_VIDEO

    input_file = f'{selected_video}.mp4'
    output_file = f'{selected_video}_output.mp4'

    ProcessProjectVideo(input_file, output_file)


def ProcessProjectVideo(input_file, output_file):
    logging.info(f'Processing video: {input_file} -> {output_file}')

    tracker = LaneTracker(CALIBRATION_IMAGES, CALIBRATION_FILE)
    tracker.process_video(input_file, output_file)


if __name__ == "__main__":
    main()
