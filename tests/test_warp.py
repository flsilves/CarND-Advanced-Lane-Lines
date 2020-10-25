"""
Unit tests for horizon detection and warp image
"""

import unittest
import logging
from test_utils import *
import numpy as np

TEST_OUTPUT_DIR = 'test_warp'


class WarpTest(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(nx=9, ny=6, calibration_images=CALIBRATION_IMAGES,
                             calibration_filename=CALIBRATION_FILE)

        self.sobel = SobelFilter(kernel_size=3)
        self.hls = HLSFilter()
        self.combined = CombinedFilter()

        self.test_images, self.filenames = get_images_from_dir(ROAD_IMAGES_DIR)
        self.warper = WarpMachine(self.test_images[0].shape)

    def tearDown(self):
        return

    def xtest_horizon_detection(self):
        logging.info('Horizon detection')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'-> {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            logging.info('shape %s', undistorted_image.shape)

            gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

            blur_gray = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

            threshold_gray = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

            # print(x1, x2, y1, y2)

            # result = cv2.line(undistorted_image, (x1, x2),
            #                  (y1, y2), (0, 255, 0), 9)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_horizon.png'
            save_before_and_after_image(
                undistorted_image, threshold_gray, filename, 'gray')

    def test_warp(self):
        logging.info('Warp images')

        for idx, test_image in enumerate(self.test_images):
            logging.info(f'Warp Images: {self.filenames[idx]}')

            undistorted_image = self.camera.undistort_image(test_image)

            filtered = self.combined.filter(undistorted_image)

            # vertices = self.warper.get_region_of_interest_vertices(
            #    filtered.shape, 0.55)

            warped = self.warper.warp(filtered)

            # final = cv2.polylines(
            #    undistorted_image, [vertices], True, (0, 0, 255), 3)

            self.warper.draw_src(undistorted_image)
            self.warper.draw_dst(undistorted_image)

            warped_poly, histogram, ploty, left_fitx, right_fitx = fit_polynomial(
                warped)
            logging.info(type(histogram))

            bottom_half = warped[warped.shape[0]//2:, :]

            warp_zero = np.zeros_like(warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(
                color_warp, self.warper.Minv, (test_image.shape[1], test_image.shape[0]))
            # Combine the result with the original image
            result = cv2.addWeighted(test_image, 1, newwarp, 0.3, 0)

            #pos_str = "Left" if pos < 0 else "Right"
            crl_text = "Radius of curvature (left) = %.1f km" % (1000 / 1000)
            crr_text = "Radius of curvature (right) = %.1f km" % (1000 / 1000)
            # cr_text = "Radius of curvature (avg) = %.1f km" % (
            #    (left_cr + right_cr) / 2000)
            # pos_text = "Vehicle is %.1f m %s from the lane center" % (
            #    np.abs(pos), pos_str)

            def put_text(image, text, color=(255, 255, 255), ypos=100):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, text, (100, ypos),
                            font, 1, color, thickness=2)

            put_text(result, crl_text, ypos=50)
            put_text(result, crr_text, ypos=100)
            #put_text(vis_overlay, cr_text, ypos=150)
            #put_text(vis_overlay, pos_text, ypos=200)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_hist.png'
            plot_histogram(
                bottom_half, histogram, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_filtered.png'
            save_before_and_after_image(
                undistorted_image, filtered, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_warp.png'
            save_before_and_after_image(
                undistorted_image, warped, filename, 'gray')

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_poly.png'
            save_before_and_after_image(
                undistorted_image, warped_poly, filename)

            filename = f'{TEST_OUTPUT_DIR}/{self.filenames[idx]}_overlay.png'
            save_before_and_after_image(
                undistorted_image, result, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    remove_old_files(TEST_OUTPUT_DIR)

    unittest.main()
