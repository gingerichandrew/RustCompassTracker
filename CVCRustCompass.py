import math
import poly_point_isect as bot
import cv2 as cv
import numpy as np
from random import *

def find_horz_lines(img):
    lines_x = []

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 30
    high_threshold = 60
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 10  # distance resolution in pixels of the Hough grid
    theta = np.pi / 2  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (int((x1 + x2)/2), 0), (int((x1 + x2)/2), 44), (128,128,128), 1)
                lines_x.append((x1 + x2)/2)
                #print(np.mean(lines_x))
    except:
        pass

    return line_image

img_merged_lines = 0
def get_lines(lines_in):
    try:
        if cv.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]
    except:
        pass

def process_lines(thresh, res, threshhold1_, threshhold2_, max_line_gap, blank, w_name):
    img = res
    #edges = cv.Canny(thresh, threshold1=50, threshold2=200, apertureSize = 3)
    lines = cv.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=max_line_gap)
    # l[0] - line; l[1] - angle
    try:
        for line in get_lines(lines):
            leftx, boty, rightx, topy = line
            cv.line(img, (leftx, boty), (rightx,topy), (0,0,255), 6) 
    except:
        pass
    # merge lines
        
    # prepare
    _lines = []
    try:
        for _line in get_lines(lines):
            _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
    except:
        pass
    # sort
    try:
        _lines_x = []
        _lines_y = []
        for line_i in _lines:
            orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
            if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
                _lines_y.append(line_i)
            else:
                _lines_x.append(line_i)
                
        _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
        _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])
            
        merged_lines_x = merge_lines_pipeline_2(_lines_x)
        merged_lines_y = merge_lines_pipeline_2(_lines_y)
        
        merged_lines_all = []
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)
        #print("process groups lines", len(_lines), len(merged_lines_all))
    except:
        pass

    img_merged_lines = np.copy(img) * 0  # creating a blank to draw lines on

    points_x = []
    for line in merged_lines_all:
        thresh_y = abs(line[0][1] - line[1][1])
        if(thresh_y < 10):
            points_x.append(line[0][0])
            points_x.append(line[1][0])
            cv.line(img_merged_lines, (line[0][0], 0), (line[1][0], 0), (0,0,255), 5)
    return(img_merged_lines, points_x)

def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 10
    min_angle_to_merge = 20
    
    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break
            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)
        
    
    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))
    
    return super_lines_final

def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]
    
    line_i = lines[0]
    
    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
        
    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        #sort by y
        points = sorted(points, key=lambda point: point[1])
        
        if use_log:
            print("use y")
    else:
        points = sorted(points, key=lambda point: point[0])
        
        if use_log:
            print("use x")
    
    return [points[0], points[len(points)-1]]

def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])
    
    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False
    
def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude
 

def DistancePointLine(px, py, x1, y1, x2, y2):
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    
    return min(dist1,dist2,dist3,dist4)

class HoughBundler:
    '''Clasterize and merge each cluster of cv.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 10
        min_angle_to_merge = 20
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv.HoughLinesP()
        for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # if vertical
                if 45 < orientation < 135:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        img_merged_lines = np.copy(img) * 0  # creating a blank to draw lines on
        points_x = []
        for line in merged_lines_all:
            thresh_y = abs(line[0][1] - line[1][1])
            if(thresh_y < 10):
                points_x.append(line[0][0])
                points_x.append(line[1][0])
                cv.line(img_merged_lines, (line[0][0], 0), (line[1][0], 0), (0,0,255), 5)
        return img_merged_lines, points_x


    #img_merged_lines = np.copy(img) * 0  # creating a blank to draw lines on

    #points_x = []
    #for line in merged_lines_all:
        #thresh_y = abs(line[0][1] - line[1][1])
        #if(thresh_y < 10):
            #points_x.append(line[0][0])
            #points_x.append(line[1][0])
            #cv.line(img_merged_lines, (line[0][0], 0), (line[1][0], 0), (0,0,255), 5)
    #return(img_merged_lines, points_x)