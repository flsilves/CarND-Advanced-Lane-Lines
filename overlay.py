import cv2
import logging
import numpy as np


def put_text(image, text, color=(255, 255, 255), ypos=100):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (100, ypos),
                font, 1, color, thickness=2)


def draw_overlay(original_image, warped, Minv, ploty, left_fitx, right_fitx, left_curvature, right_curvature):
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

    # pos_str = "Left" if pos < 0 else "Right"
    crl_text = "Radius of curvature (left) = %.1f km" % (left_curvature / 1000)
    crr_text = "Radius of curvature (right) = %.1f km" % (
        right_curvature / 1000)

    put_text(result, crl_text, ypos=50)
    put_text(result, crr_text, ypos=100)

    return result
