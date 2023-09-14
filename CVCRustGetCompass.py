import cv2
import numpy as np 
import mss
import CVCRustCompass as cpLines
import win32gui, win32con, win32api, win32ui
import tkinter as tk
import serial
import time
import math
import seaborn as sns
import matplotlib.pyplot as plt
from random import *
import multiprocessing
from multiprocessing import Process, Queue
import pandas as pd

def screenshot(monitor):
    with mss.mss() as sct:

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        return img

def find_furthest_black_pixel(img):
    # Set index for furthest
    black_px_index = 0
    # Loop through pixels in segment
    for px in range(img.shape[1]):
        # If pixel is black, set it as current furthest
        if(img[0][px] == 0):
            black_px_index = px
    # Return the furthest black pixel in segment provided
    return black_px_index

def find_furthest_black_pixel_from_left(img):
    h, w = img.shape
    w -= 1
    # Set index for furthest
    black_px_index = 0
    # Loop through pixels in segment
    for px in range(w, w - 100, -1):
        # If pixel is black, set it as current furthest
        if(img[0][px] != 0):
            return px
    # Return the furthest black pixel in segment provided
    return black_px_index

def find_closest_white_pixel(img, side):
    # Width and Height
    h, w = img.shape
    w -= 1
    # Storage index
    
    # Iterate from left to right
    if(side == 0):
        for px in range(w):
            if(img[0][px] != 0): # If the current pixel is white
                return px # Return index
        return w # If no white pixels found from 0 to w, return w

    # Else iterate from right to left
    for px in range(w, 0, -1):
        if(img[0][px] != 0): # If the current pixel is white
            return px # Return index
    return 0 # If no white pixels found from w to 0, return 0

def find_furthest_white_pixel(img, side):
    # Width and Height
    h, w = img.shape
    w -= 1
    # Storage index
    white_px_index = 0
    
    # Loop through from the left
    if(img[0][0] == 0):
        return -1
    if(side == 0):
        for px in range(w):
            if(img[0][px] != 0):
                white_px_index = px
            else:
                return white_px_index
        return white_px_index
    
    # Loop through from the right
    if(img[0][w] == 0):
        return -1
    for px in range(w, 0, -1):
        if(img[0][px] != 0):
            white_px_index = px
        else:
            return white_px_index
    return white_px_index

def image_end_correction(lines_img, gap_corrected, xyxy_):
    corrected = gap_corrected.copy()
    h, w = corrected.shape
    # If there is only one line, and it doesn't start at 0
    if(len(xyxy_) == 2 and xyxy_[0] != 0):
        # If the last pixel is not black
        if(corrected[0][0]) != 0:
            x = find_furthest_white_pixel(corrected, 0) # Find from the left over
            if(x < w-2):
                xyxy_ = []
                xyxy_.append(0) 
                xyxy_.append(x) 
                cv2.line(lines_img, (0, 0), (x, 0), (0,0,255), 5)

    # If there is only one line, and it starts at 0
    elif(len(xyxy_) == 2 and xyxy_[0] == 0):
        # If the first pixel is not black
        if(corrected[0][w-1]) != 0:
            x = find_furthest_white_pixel(corrected, 1) # Find from the right over
            if(x > 1):
                xyxy_ = []
                xyxy_.append(w)
                xyxy_.append(x)
                cv2.line(lines_img, (w, 0), (x, 0), (0,0,255), 5)
    # If there are more than 2 points, (ie 2 lines), fix noise right
    elif(len(xyxy_) > 2 and xyxy_[0] == 0):
        if(xyxy_[3] != w):
            xyxy_ = [0, w]
            #xyxy_[3] = w 
    # If there are more than 2 points, (ie 2 lines), fix noise left
    elif(len(xyxy_) > 2 and xyxy_[3] >= w-1):
        if(xyxy_[0] != 0):
            xyxy_[0] = 0
            xyxy_[3] = w
    else:
        return (lines_img, xyxy_) # Return unchanged
    xyxy_.sort() # Sort
    return (lines_img, xyxy_) # Return corrected ends


# Correct for small holes that would otherwise deam a segment not a conjoint line
def image_gap_correction(img_o):
    # Get image shape
    img_ltr = img_o.copy()
    h, w = img_ltr.shape
    # Set index
    x = 0
    # Loop through the pixels width(as all vertical pixels are duplicates)
    while(x < w):
        # Set width of area search
        w_area = 72

        min_w_area = 10
        # Minimum percentage of black pixels in segment
        seg_min_percentage = .5

        # Set counters
        black_pxs = 0
        white_pxs = 0

        # Check if pixel is black
        if(img_ltr[0][x] == 0):
            # Copy image
            segment_img = img_ltr.copy()

            # If x is 0, then the first pixel is black, find the furthest black pixel within range
            if(x == 0):
                w_area = find_furthest_black_pixel(img_ltr.copy()[:, x:(x + w_area)])
            if(w_area < min_w_area):
                w_area = min_w_area
            # Using furthest black pixel, check if the segment should be converted to all black
            segment_img = segment_img[:, x:(x + w_area)]

            # Loop through segment pixels
            black_pxs = np.sum(segment_img == 0)
            white_pxs = np.sum(segment_img == 255)

            # Check if there are black pixels, then check if there is more black pixels than white pixels
            if(black_pxs != 0 and (black_pxs / (black_pxs + white_pxs) > seg_min_percentage)):
                # More black pixels, set entire segment to black
                img_ltr[0:h, x:x + w_area] = 0
            else:
                # More white pixels, set entire segment to white
                img_ltr[0:h, x:x + w_area] = 255

            x += w_area # Increment by total area just converted
        else:
            x += 1 # Increment by 1 px
        # Do final check if pixel at w is black is just noise
    if(img_ltr[0][w-1] == 0):
        segment_img = img_ltr.copy()
        closest_white = find_furthest_black_pixel_from_left(img_ltr) + 1 # Get index of where cloest white pixel from left is
        if((w-1) - closest_white <= 10):
            img_ltr[0:h, closest_white:w] = 255

    # Now right to left
    img_rtl = img_o.copy()
    h, w = img_rtl.shape
    # Set index
    x = w - 1
    # Loop through the pixels width(as all vertical pixels are duplicates)
    while(x >= 0):
        # Set width of area search
        w_area = 72
        min_w_area = 10
        # Minimum percentage of black pixels in segment
        seg_min_percentage = .5
        # Set counters
        black_pxs = 0
        white_pxs = 0
        # Check if pixel is black
        if(img_rtl[0][x] == 0):
            # Copy image
            segment_img = img_rtl.copy()

            # If x is w-1, then the first pixel is black, find the furthest black pixel within range
            if(x == w-1):
                w_area = find_furthest_black_pixel_from_left(img_rtl.copy()[:, x-w_area:x])
            if(w_area < min_w_area):
                w_area = min_w_area
            # Using furthest black pixel, check if the segment should be converted to all black
            segment_img = segment_img[:, x-w_area:x]
            # Loop through segment pixels
            black_pxs = np.sum(segment_img == 0)
            white_pxs = np.sum(segment_img == 255)

            # Check if there are black pixels, then check if there is more black pixels than white pixels
            if(black_pxs != 0 and (black_pxs / (black_pxs + white_pxs) > seg_min_percentage)):
                # More black pixels, set entire segment to black
                img_rtl[0:h, x  - w_area:x + 1] = 0
            else:
                # More white pixels, set entire segment to white
                img_rtl[0:h, x - w_area:x + 1] = 255

            x -= w_area # Increment by total area just converted
        else:
            x -= 1 # Increment by 1 px

    if(img_rtl[0][w-1] == 0):
        segment_img = img_rtl.copy()
        closest_white = find_furthest_black_pixel_from_left(img_rtl) + 1 # Get index of where cloest white pixel from left is
        if((w-1) - closest_white <= 10):
            img_rtl[0:h, closest_white:w] = 255

    return (img_ltr,img_rtl)# Return corrected image

def compute_lines(img):
    xyxy = []
    l_index = find_furthest_white_pixel(img, 0) # Left  |0 -> l_index|
    #If l_index returns -1, then find where the line end point |___*-------------|
    if(l_index == -1):
        l_index = find_closest_white_pixel(img, 0)
    r_index = find_furthest_white_pixel(img, 1) # Right |r_index <- w|
    #If r_index returns -1, then find where the line end point |-------------*___|
    if(r_index == -1):
        r_index = find_closest_white_pixel(img, 1)
    if(l_index == r_index):
        xyxy.append(0)
        xyxy.append(l_index)
    elif(l_index > r_index):
        xyxy.append(0)
        xyxy.append(img.shape[1])
    elif(l_index != 0 and r_index != (img.shape[1]) - 1):
        xyxy.append(0)
        xyxy.append(l_index)
        xyxy.append(r_index)
        xyxy.append(img.shape[1])
    elif(l_index != 0 and r_index == (img.shape[1] - 1)):
        xyxy.append(l_index)
        xyxy.append(img.shape[1])
    elif(l_index == 0 and r_index == (img.shape[1] - 1)):
        xyxy.append(0)
        xyxy.append(img.shape[1])
    return xyxy

def combine_lines(deltas_left, deltas_right, deltas_center):
    xyxy_all = []
    xyxy_l, curr_num_lines_l = deltas_left, len(deltas_left) // 2
    xyxy_r, curr_num_lines_r = deltas_right, len(deltas_right) // 2
    xyxy_c, curr_num_lines_c = deltas_center, len(deltas_center) // 2

    # Middle gap correction
    if(len(xyxy_c) == 4):
        for i in range(1, len(xyxy_c) - 1, 2):
            point_1, point_2 = xyxy_c[i], xyxy_c[i+1]
            if(point_2 >= 99):
                xyxy_c[i] -= (72 - (xyxy_c[i + 1] - xyxy_c[i]))

    for i in range(curr_num_lines_l * 2):
        xyxy_l[i]
        xyxy_all.append(xyxy_l[i])
    for i in range(curr_num_lines_r * 2):
        xyxy_r[i] += 1525 + 200
        xyxy_all.append(xyxy_r[i])
    for i in range(curr_num_lines_c * 2):
        xyxy_c[i] += 1525
        xyxy_all.append(xyxy_c[i])
    xyxy_all.sort()
    remove_list = []
    for i in range(1, len(xyxy_all) - 1, 2):
        point_1, point_2 = xyxy_all[i], xyxy_all[i+1]
        if(point_1 == point_2):
            remove_list.append(point_1)
            remove_list.append(point_2)
    res = [i for i in xyxy_all if i not in remove_list]

    points = []
    for i in range(1, len(res) - 1, 2):
        point_1, point_2 = res[i], res[i+1]
        points.append((point_1 + point_2) / 2)
    return res, points

# Find closest value in array
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]   

def get_point_changes(current_points, previous_points):
    current_num_points = len(current_points)
    previous_num_points = len(previous_points)
    delta_num_points = (current_num_points - previous_num_points)

    # If previous is empty, or there is no points
    if(current_num_points == 0 or previous_num_points == 0):
        return 0
    # If there is less points than before
    if(delta_num_points < 0):
        prev_point = closest(previous_points, current_points[0])
        diff = current_points[0] - prev_point
        return diff * -1
    # If there is more points than before
    if(delta_num_points > 0):
        curr_point = closest(current_points, previous_points[0])
        diff = curr_point - previous_points[0]
        return diff * -1
    diff = []
    for i in range(current_num_points):
        diff.append(current_points[i] - previous_points[i])
    return np.mean(diff) * -1
    
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # For finding avg runtime # 
    frame_count = 1
    time_taken = 0

    # Set lower and upper bounds for what color to extract
    lower_bound = np.array([0,0,0])
    upper_bound = np.array([0,0,240])

    # Vars to control image size ratio
    screen_width = 1920
    total_width = 690   # The entire compass image width
    total_height = 1    #                                                                 OFFSET      LEFT       CENTER      RIGHT      OFFSET
    offset = 20         # Offset that is "chopped" from left and right of entire image | <--15px--> <--340px--> <--10px--> <--340px--> <--15px--> |
    center_split_width = 40
    split_width = int(((total_width - (offset * 2)) / 2) - (center_split_width / 2))
    overlap = 0         # Number of pixels the center overlaps with right and left splits

    # Store previous x coordinates
    prev_xyxy_all = []
    prev_points = []
    total_diff = 0
    time_before_next = time.perf_counter()
    
    # Main loop
    left = int(screen_width/2) - int(split_width + offset + center_split_width/2)
    top = 8
    width = total_width
    height = total_height
    monitor = {'top': top, 'left': left, 'width': width, 'height': height}
    while True:
        #Timer for performance
        timer_start = time.perf_counter()

        # Get image of entire compass
        image_compass = screenshot(monitor) # Center dot is 8px wide

        # Percent to scale
        scale_percent = 500 # percent of original size

        # Total compass w/h
        width_compass = int(image_compass.shape[1] * scale_percent / 100) 
        height_compass = int(image_compass.shape[0] * scale_percent / 100)

        # Dimensions
        dim_compass = (width_compass, height_compass)

        # Resize compass
        frame_compass = cv2.cvtColor(image_compass, cv2.COLOR_RGB2BGR)
        frame_compass = cv2.resize(image_compass, dim_compass, interpolation = cv2.INTER_AREA)

        # Converts compass from BGR to HSV
        hsv_compass = cv2.cvtColor(frame_compass, cv2.COLOR_BGR2HSV)

        # Here we are defining range of color in HSV
        mask_compass = cv2.inRange(hsv_compass, lower_bound, upper_bound)

        # Get bitwise and frame - mask
        res_compass = cv2.bitwise_and(frame_compass, frame_compass, mask=mask_compass)

        # Convert compass to grayscale
        gray_compass = cv2.cvtColor(res_compass,cv2.COLOR_BGR2GRAY)
        
        # Remove/Inverse image
        ret_compass, thresh_compass = cv2.threshold(gray_compass,127,255,cv2.THRESH_BINARY)
        thresh_compass = cv2.bitwise_not(thresh_compass)

        # Correct image for gaps that should be included, like South on compass
        ltr, rtl = image_gap_correction(thresh_compass)
                
        # Get center images as two seperate halfs
        center_left = ltr[0:total_height * 5, ((split_width + offset - overlap) * 5):(split_width + offset + int(center_split_width/2)) * 5]
        center_right = rtl[0:total_height * 5, (split_width + offset + int(center_split_width/2)) * 5:(split_width + offset + center_split_width + overlap) * 5]

        # Do a second pass on the center images to remove noise
        ltr_center_left, rtl_center_left = image_gap_correction(center_left)
        ltr_center_right, rtl_center_right = image_gap_correction(center_right)

        # Concat both center images back to 1 image
        thresh_center = cv2.hconcat([ltr_center_left, ltr_center_right])

        # Split compass
        thresh_left = ltr.copy()[0:total_height * 5, (offset * 5):(split_width + offset + overlap) * 5]
        thresh_right = rtl.copy()[0:total_height * 5, (split_width + offset + center_split_width - overlap) * 5:(total_width - offset) * 5]

        xyxy_l = compute_lines(thresh_left)
        xyxy_r = compute_lines(thresh_right)
        xyxy_c = compute_lines(thresh_center)

        xyxy_all, xyxy_points = combine_lines(xyxy_l, xyxy_r, xyxy_c)

        diff = get_point_changes(xyxy_points, prev_points)
        total_diff += diff

        # Get current value in degrees
        current_diff = (total_diff/5) / 8.44
        current_angle = current_diff % 360

        prev_points = xyxy_points

        #If key press down arrow
        if(win32api.GetAsyncKeyState(0x28) < 0):
            # Busy while loop
            while(win32api.GetAsyncKeyState(0x28) < 0):
                pass
            prev_xyxy_all = xyxy_all
            total_diff = 0

        print("LOOP: ", time.perf_counter() - timer_start, "ms")

class Compass:
    def __init__(self):
        # Vars to control image size ratio
        self.screen_width = 1920
        self.total_width = 690   # The entire compass image width
        self.total_height = 1                                                                   #   OFFSET      LEFT       CENTER      RIGHT      OFFSET
        self.offset = 20         # Offset that is "chopped" from left and right of entire image | <--20px--> <--305px--> <--40px--> <--305px--> <--20px--> |
        self.center_split_width = 40
        self.split_width = int(((self.total_width - (self.offset * 2)) / 2) - (self.center_split_width / 2))
        self.overlap = 0         # Number of pixels the center overlaps with right and left splits
        self.scale_percent = 500 # Percent to upscale screenshots

        # Vars to store previous frame data
        self.prev_xyxy_all = []
        self.prev_points = []
        self.total_diff = 0
        
        # Vars for image color
        self.lower_bound = np.array([0,0,0])
        self.upper_bound = np.array([0,0,240])

    def screenshot(self, left, top, width, height):
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {'top': top, 'left': left, 'width': width, 'height': height}
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))
            return img

    def resize_screenshot(self, img):
        # Total compass w/h
        width_compass = int(img.shape[1] * self.scale_percent / 100) 
        height_compass = int(img.shape[0] * self.scale_percent / 100)

        # Dimensions to scale to
        dim_compass = (width_compass, height_compass)

        # Resize img
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(img, dim_compass, interpolation = cv2.INTER_AREA)
        return frame

    def find_furthest_black_pixel(self, img):
        # Set index for furthest
        black_px_index = 0
        # Loop through pixels in segment
        for px in range(img.shape[1]):
            # If pixel is black, set it as current furthest
            if(img[0][px] == 0):
                black_px_index = px
        # Return the furthest black pixel in segment provided
        return black_px_index

    def find_furthest_black_pixel_from_left(self, img):
        h, w = img.shape
        w -= 1
        # Set index for furthest
        black_px_index = 0
        # Loop through pixels in segment
        for px in range(w, w - 100, -1):
            # If pixel is black, set it as current furthest
            if(img[0][px] != 0):
                return px
        # Return the furthest black pixel in segment provided
        return black_px_index

    def find_closest_white_pixel(self, img, side):
        # Width and Height
        h, w = img.shape
        w -= 1
        # Storage index
        
        # Iterate from left to right
        if(side == 0):
            for px in range(w):
                if(img[0][px] != 0): # If the current pixel is white
                    return px # Return index
            return w # If no white pixels found from 0 to w, return w

        # Else iterate from right to left
        for px in range(w, 0, -1):
            if(img[0][px] != 0): # If the current pixel is white
                return px # Return index
        return 0 # If no white pixels found from w to 0, return 0

    def find_furthest_white_pixel(self, img, side):
        # Width and Height
        h, w = img.shape
        w -= 1
        # Storage index
        white_px_index = 0
        
        # Loop through from the left
        if(img[0][0] == 0):
            return -1
        if(side == 0):
            for px in range(w):
                if(img[0][px] != 0):
                    white_px_index = px
                else:
                    return white_px_index
            return white_px_index
        
        # Loop through from the right
        if(img[0][w] == 0):
            return -1
        for px in range(w, 0, -1):
            if(img[0][px] != 0):
                white_px_index = px
            else:
                return white_px_index
        return white_px_index

    def image_end_correction(self, lines_img, gap_corrected, xyxy_):
        corrected = gap_corrected.copy()
        h, w = corrected.shape
        # If there is only one line, and it doesn't start at 0
        if(len(xyxy_) == 2 and xyxy_[0] != 0):
            # If the last pixel is not black
            if(corrected[0][0]) != 0:
                x = self.find_furthest_white_pixel(corrected, 0) # Find from the left over
                if(x < w-2):
                    xyxy_ = []
                    xyxy_.append(0) 
                    xyxy_.append(x) 

        # If there is only one line, and it starts at 0
        elif(len(xyxy_) == 2 and xyxy_[0] == 0):
            # If the first pixel is not black
            if(corrected[0][w-1]) != 0:
                x = self.find_furthest_white_pixel(corrected, 1) # Find from the right over
                if(x > 1):
                    xyxy_ = []
                    xyxy_.append(w)
                    xyxy_.append(x)
        # If there are more than 2 points, (ie 2 lines), fix noise right
        elif(len(xyxy_) > 2 and xyxy_[0] == 0):
            if(xyxy_[3] != w):
                xyxy_ = [0, w]
        # If there are more than 2 points, (ie 2 lines), fix noise left
        elif(len(xyxy_) > 2 and xyxy_[3] >= w-1):
            if(xyxy_[0] != 0):
                xyxy_[0] = 0
                xyxy_[3] = w
        else:
            return (lines_img, xyxy_) # Return unchanged
        xyxy_.sort() # Sort
        return (lines_img, xyxy_) # Return corrected ends

    # Correct for small holes that would otherwise deam a segment not a conjoint line
    def image_gap_correction(self, img_o):
        # Get image shape
        img_ltr = img_o.copy()
        h, w = img_ltr.shape
        # Set index
        x = 0
        # Loop through the pixels width(as all vertical pixels are duplicates)
        while(x < w):
            # Set width of area search
            w_area = 72

            min_w_area = 10
            # Minimum percentage of black pixels in segment
            seg_min_percentage = .5

            # Set counters
            black_pxs = 0
            white_pxs = 0

            # Check if pixel is black
            if(img_ltr[0][x] == 0):
                # Copy image
                segment_img = img_ltr.copy()

                # If x is 0, then the first pixel is black, find the furthest black pixel within range
                if(x == 0):
                    w_area = self.find_furthest_black_pixel(img_ltr.copy()[:, x:(x + w_area)])
                if(w_area < min_w_area):
                    w_area = min_w_area
                # Using furthest black pixel, check if the segment should be converted to all black
                segment_img = segment_img[:, x:(x + w_area)]

                # Loop through segment pixels
                black_pxs = np.sum(segment_img == 0)
                white_pxs = np.sum(segment_img == 255)

                # Check if there are black pixels, then check if there is more black pixels than white pixels
                if(black_pxs != 0 and (black_pxs / (black_pxs + white_pxs) > seg_min_percentage)):
                    # More black pixels, set entire segment to black
                    img_ltr[0:h, x:x + w_area] = 0
                else:
                    # More white pixels, set entire segment to white
                    img_ltr[0:h, x:x + w_area] = 255

                x += w_area # Increment by total area just converted
            else:
                x += 1 # Increment by 1 px
        # Do final check if pixel at w is black is just noise
        if(img_ltr[0][w-1] == 0):
            segment_img = img_ltr.copy()
            closest_white = self.find_furthest_black_pixel_from_left(img_ltr) + 1 # Get index of where cloest white pixel from left is
            if((w-1) - closest_white <= 10):
                img_ltr[0:h, closest_white:w] = 255

        # Now right to left
        img_rtl = img_o.copy()
        h, w = img_rtl.shape
        # Set index
        x = w - 1
        # Loop through the pixels width(as all vertical pixels are duplicates)
        while(x >= 0):
            # Set width of area search
            w_area = 72
            min_w_area = 10
            # Minimum percentage of black pixels in segment
            seg_min_percentage = .5
            # Set counters
            black_pxs = 0
            white_pxs = 0
            # Check if pixel is black
            if(img_rtl[0][x] == 0):
                # Copy image
                segment_img = img_rtl.copy()

                # If x is w-1, then the first pixel is black, find the furthest black pixel within range
                if(x == w-1):
                    w_area = self.find_furthest_black_pixel_from_left(img_rtl.copy()[:, x-w_area:x])
                if(w_area < min_w_area):
                    w_area = min_w_area
                # Using furthest black pixel, check if the segment should be converted to all black
                segment_img = segment_img[:, x-w_area:x]
                # Loop through segment pixels
                black_pxs = np.sum(segment_img == 0)
                white_pxs = np.sum(segment_img == 255)

                # Check if there are black pixels, then check if there is more black pixels than white pixels
                if(black_pxs != 0 and (black_pxs / (black_pxs + white_pxs) > seg_min_percentage)):
                    # More black pixels, set entire segment to black
                    img_rtl[0:h, x  - w_area:x + 1] = 0
                else:
                    # More white pixels, set entire segment to white
                    img_rtl[0:h, x - w_area:x + 1] = 255

                x -= w_area # Increment by total area just converted
            else:
                x -= 1 # Increment by 1 px

        if(img_rtl[0][w-1] == 0):
            segment_img = img_rtl.copy()
            closest_white = self.find_furthest_black_pixel_from_left(img_rtl) + 1 # Get index of where cloest white pixel from left is
            if((w-1) - closest_white <= 10):
                img_rtl[0:h, closest_white:w] = 255

        return (img_ltr,img_rtl)# Return corrected image

    def compute_lines(self, img):
        xyxy = []
        l_index = self.find_furthest_white_pixel(img, 0) # Left  |0 -> l_index|
        #If l_index returns -1, then find where the line end point |___*-------------|
        if(l_index == -1):
            l_index = self.find_closest_white_pixel(img, 0)
        r_index = self.find_furthest_white_pixel(img, 1) # Right |r_index <- w|
        #If r_index returns -1, then find where the line end point |-------------*___|
        if(r_index == -1):
            r_index = self.find_closest_white_pixel(img, 1)
        if(l_index == r_index):
            xyxy.append(0)
            xyxy.append(l_index)
        elif(l_index > r_index):
            xyxy.append(0)
            xyxy.append(img.shape[1])
        elif(l_index != 0 and r_index != (img.shape[1]) - 1):
            xyxy.append(0)
            xyxy.append(l_index)
            xyxy.append(r_index)
            xyxy.append(img.shape[1])
        elif(l_index != 0 and r_index == (img.shape[1] - 1)):
            xyxy.append(l_index)
            xyxy.append(img.shape[1])
        elif(l_index == 0 and r_index == (img.shape[1] - 1)):
            xyxy.append(0)
            xyxy.append(img.shape[1])
        return xyxy

    def combine_lines(self, deltas_left, deltas_right, deltas_center):
        xyxy_all = []
        xyxy_l, curr_num_lines_l = deltas_left, len(deltas_left) // 2
        xyxy_r, curr_num_lines_r = deltas_right, len(deltas_right) // 2
        xyxy_c, curr_num_lines_c = deltas_center, len(deltas_center) // 2

        # Middle gap correction
        if(len(xyxy_c) == 4):
            for i in range(1, len(xyxy_c) - 1, 2):
                point_1, point_2 = xyxy_c[i], xyxy_c[i+1]
                if(point_2 >= 99):
                    xyxy_c[i] -= (72 - (xyxy_c[i + 1] - xyxy_c[i]))
        for i in range(curr_num_lines_l * 2):
            xyxy_l[i]
            xyxy_all.append(xyxy_l[i])
        for i in range(curr_num_lines_r * 2):
            xyxy_r[i] += 1525 + 200
            xyxy_all.append(xyxy_r[i])
        for i in range(curr_num_lines_c * 2):
            xyxy_c[i] += 1525
            xyxy_all.append(xyxy_c[i])
        xyxy_all.sort()
        remove_list = []
        for i in range(1, len(xyxy_all) - 1, 2):
            point_1, point_2 = xyxy_all[i], xyxy_all[i+1]
            if(point_1 == point_2):
                remove_list.append(point_1)
                remove_list.append(point_2)
        res = [i for i in xyxy_all if i not in remove_list]

        points = []
        for i in range(1, len(res) - 1, 2):
            point_1, point_2 = res[i], res[i+1]
            points.append((point_1 + point_2) / 2)
        return res, points

        # Find closest value in array
    def closest(self, lst, K):
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]   

    def get_point_changes(self, current_points, previous_points):
        current_num_points = len(current_points)
        previous_num_points = len(previous_points)
        delta_num_points = (current_num_points - previous_num_points)

        # If previous is empty, or there is no points
        if(current_num_points == 0 or previous_num_points == 0):
            return 0
        # If there is less points than before
        if(delta_num_points < 0):
            prev_point = self.closest(previous_points, current_points[0])
            diff = current_points[0] - prev_point
            return diff * -1
        # If there is more points than before
        if(delta_num_points > 0):
            curr_point = self.closest(current_points, previous_points[0])
            diff = curr_point - previous_points[0]
            return diff * -1
        diff = []
        for i in range(current_num_points):
            diff.append(current_points[i] - previous_points[i])
        return np.mean(diff) * -1

    def set_previous(self, x, xyxy_all):
        self.prev_xyxy_all = xyxy_all
        self.total_diff = x

    def image_correction(self):
        # Get compass screenshot
        image_compass = self.screenshot(int(self.screen_width/2) - int(self.split_width + self.offset + self.center_split_width/2), 8, self.total_width, self.total_height)
        frame_compass = self.resize_screenshot(image_compass)

        # Converts compass from BGR to HSV
        hsv_compass = cv2.cvtColor(frame_compass, cv2.COLOR_BGR2HSV)

        # Here we are defining range of color in HSV
        mask_compass = cv2.inRange(hsv_compass, self.lower_bound, self.upper_bound)

        # Get bitwise and frame - mask
        res_compass = cv2.bitwise_and(frame_compass, frame_compass, mask=mask_compass)

        # Convert compass to grayscale
        gray_compass = cv2.cvtColor(res_compass,cv2.COLOR_BGR2GRAY)
        
        # Remove/Inverse image
        ret_compass, thresh_compass = cv2.threshold(gray_compass,127,255,cv2.THRESH_BINARY)
        thresh_compass = cv2.bitwise_not(thresh_compass)

        # Correct image for gaps that should be included, like South on compass
        ltr, rtl = self.image_gap_correction(thresh_compass)
                
        # Get center images as two seperate halfs
        center_left = ltr[0:self.total_height * 5, ((self.split_width + self.offset - self.overlap) * 5):(self.split_width + self.offset + int(self.center_split_width/2)) * 5]
        center_right = rtl[0:self.total_height * 5, (self.split_width + self.offset + int(self.center_split_width/2)) * 5:(self.split_width + self.offset + self.center_split_width + self.overlap) * 5]

        # Do a second pass on the center images to remove noise
        ltr_center_left, rtl_center_left = self.image_gap_correction(center_left)
        ltr_center_right, rtl_center_right = self.image_gap_correction(center_right)

        # Concat both center images back to 1 image
        thresh_center = cv2.hconcat([ltr_center_left, ltr_center_right])

        # Split compass
        thresh_left = ltr.copy()[0:self.total_height * 5, (self.offset * 5):(self.split_width + self.offset + self.overlap) * 5]
        thresh_right = rtl.copy()[0:self.total_height * 5, (self.split_width + self.offset + self.center_split_width - self.overlap) * 5:(self.total_width - self.offset) * 5]

        xyxy_l = self.compute_lines(thresh_left)
        xyxy_r = self.compute_lines(thresh_right)
        xyxy_c = self.compute_lines(thresh_center)

        xyxy_all, xyxy_points = self.combine_lines(xyxy_l, xyxy_r, xyxy_c)

        diff = get_point_changes(xyxy_points, self.prev_points)
        self.total_diff += diff

        # Get current value in degrees
        current_diff = (self.total_diff/5) / 8.44
        current_angle = current_diff % 360

        self.prev_points = xyxy_points

        return current_diff, xyxy_all

