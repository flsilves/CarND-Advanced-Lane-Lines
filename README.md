# **Advanced Lane Finding** 


[//]: # (Image References)

[image1]: ./writeup_images/calibration3_undistort.png "chessboard_undistort"
[image2]: ./writeup_images/test4_undistort.png "road_undistort"  
[image3]: ./writeup_images/test5_sobel_final.png "filter_sobel"
[image4]: ./writeup_images/test5_hls.png "filter_hls"
[image5]: ./writeup_images/test5_final.png "filter_final"
[image6]: ./writeup_images/straight_lines1_warp.png "warp"
[image7]: ./writeup_images/test4_poly.png "histogram_poly"
[image8]: ./writeup_images/test4_hist.png "histogram"  
[image9]: ./writeup_images/test6_overlay.png "final_overlay"



## Camera Calibration

The camera calibration was made using the chessboard images along with the opencv functions presented in the course material.   
  
`cv2.findChessboardCorners()` failed to find corners in the following images: 

```
Unable to find corners in: ../camera_cal/calibration1.jpg
Unable to find corners in: ../camera_cal/calibration4.jpg
Unable to find corners in: ../camera_cal/calibration5.jpg
```  

`calibration4.jpg` has one missing corner, the remaining ones probably have the chessboard overly cropped.  

#### Undistortion examples:

![alt text][image1]

![alt text][image2]


## Pipeline (images)



### Filter to binary  

`filter.py` is the module used to filter the images in order to identify lane-lines.

The filter used is a combination of gradient (using sobel) and color (using saturation channel).
  
Sobel filter: `(X & Y) | (DIR & MAG)` Pretty much what was covered on the course. The thresholds are based on the mean values of each sobel component.
  
Saturation filter: `s_channel > low_threshold ` The threshold is simply the mean of the saturation values of the half bottom of the image (where the road is usually located) multiplied by constant. This helps to adapt to different frames with different exposure levels.   

##### - Sobel filter:  
![alt text][image3]

##### - Saturation filter:  
![alt text][image4]

##### - Combined filter:  
![alt text][image5]

---

### Perspective Transform  
  
Class `Warper` is used to perform the perspective transform, the image with straight lines was used to define a source polygon (green in the image below) that would match a straight line from a birds-view perspective.  
  
![alt text][image6]   

Note: The polygon is hardcoded, so it doesn't adapt to road slope, that's very noticeable on the challenge/harder videos and a limitation. 

### Finding Lane-lines  
  
`line_fit.py`is used to identify the pixels belonging to the left and right lane, if there's no previous detection a sliding windows approach is used with an histogram to idenfity the first windows.
  
![alt text][image7]
![alt text][image8]

If there's previous detections the area around the last lane-lines are used to search for the lane-line's pixels.

### Curvature and position  
  
`measure_curvature_real()` is used to calculate the curvature of each lane-line, based on the calculated polynomials

`ego_distance_from_center()` calculates the distance from the center of the image to the center of the lane (at the very bottom of the image)  
  
The polynomials are adjusted to meters from pixels based on the ratios provided in the course.

The values displayed on the video are averaged out accross several frames to avoid sudden jumps.

### Lane identification  
    
Final result with the lane space fitted in the original image:
  
  ![alt text][image9]

## Pipeline (video)


Aditional logic to validate the detection of lines was added, based on the normalized residuals of the polynomial fit.  
  
If the algorithm fails to detect lines for several frames the `sliding_window` approach is used instead. If there's a successfull detection then the area around the last detection is used.   

#### Links to the result videos:
  
[project\_video\_output.mp4](https://github.com/flsilves/CarND-Advanced-Lane-Lines/blob/master/linked_videos/project_video_output.mp4)


[project\_video\_output\_debug.m4v](https://github.com/flsilves/CarND-Advanced-Lane-Lines/blob/master/linked_videos/project_video_output_debug.m4v) (2x2 panel with the different stages of the pipeline)


## Discussion (video)
  
#### Shortcomings:  
    
  - The warped process is hardcoded, sometimes when the car hits a bump or the road slope increases/descreases the lines don't appear straight anymore. Also sometimes the lines of interest are out of the warped region.  

  - Detection of the lines could be done in separate for the left and right, currently if the fitting fails for one it counts as an invalid detection.  
   
  - A better technique for evaluating the detection of lines could be used, instead of the residuals of the polyfit. Similiarity with the previous detected lines could be an alternative.  
  
  - Dashed lines are sometimes a problem because the fitting is strongly affected by single pixels on the extremes of the image when there's no lane markings on it.  
  
  - Averaging of the detected lines could be done to smooth some jumps in the video, however averaging 2nd degree polynomials is not trivial.   
  
  - The algorithm struggles when there's multiple lines close to the center, for example road boundaries, separation walls, and so on. More robust logic should be used to identify the car's lane. 

