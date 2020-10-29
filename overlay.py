""" 
Draw overlay on top of image: with detected lane, curvature and position

"""

import cv2
import logging
import numpy as np


def put_text(image, text, color=(255, 255, 255), ypos=100):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (100, ypos),
                font, 1, color, thickness=2)


def draw_overlay(original_image, warped, Minv, ploty, left_fitx, right_fitx, curvature_meters, ego_lateral_distance):
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
        color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    curvature_text = "Curvature radius: %.2f km" % (curvature_meters / 1000)

    if ego_lateral_distance >= 0.01:
        distance_text = "Vehicle is %.2fm to the left" % (
            abs(ego_lateral_distance))

    elif ego_lateral_distance <= -0.01:
        distance_text = "Vehicle is %.2fm to the right" % (
            abs(ego_lateral_distance))
    else:
        distance_text = "Vehicle is centered"

    put_text(result, curvature_text, ypos=50)
    put_text(result, distance_text, ypos=100)

    return result
