""" Main module """

import logging
from camera import Camera


def main():
    """ Main """
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    input_file = "project_video.mp4"
    output_file = "project_video_output.mp4"

    ProcessProjectVideo(input_file, output_file)


def ProcessProjectVideo(input_file, output_file):
    logging.info(f'Processing video: {input_file} -> {output_file}')

    tracker = LaneTracker()
    tracker.process_video(input_file, output_file)


if __name__ == "__main__":
    main()
