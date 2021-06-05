# -*- coding: utf-8 -*-
import cv2 as cv
import math
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
def roi_mask(img, corner_points):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, corner_points, 255)
    masked_img = cv.bitwise_and(img, mask)
    return masked_img
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv.HoughLinesP(img, rho, theta, threshold,minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines
def myhough(img):
    thetas = np.deg2rad(np.arange(-90.0, 90.0,1)).astype(int)
    cost = np.cos(thetas)
    sint = np.sin(thetas)
    row,column = np.nonzero(img)
    rhoss = []
    for i in range(len(thetas)):
        r = row[i]
        c = column[i]
        rho = r*cost[i]+c*sint[i]
        rho = rho.astype(int)
        rhoss.append(rho)
    return rhoss,thetas
'''
 gaussian 引數
'''
blur_ksize = 199
'''
 canny 檢測高低閥值
'''
canny_lth = 43
canny_hth = 122
'''
 hough 引數
'''
rho = 10
theta = np.pi / 180
threshold = 39
min_line_len = 130
max_line_gap = 29
def process_an_image(img):
    '''
    一、將照片灰化，將照片做 gaussian 濾波，對照片做 canny 檢測
    '''
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur_gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    edges = cv.Canny(blur_gray, canny_lth, canny_hth)
    # [[[0 540], [460 325], [520 325], [960 540]]]
    points = np.array([[(0, edges.shape[0]), (460, 325), (520, 325), (edges.shape[1],edges.shape[0])]])
    roi_edges = roi_mask(edges, points)
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines = hough_lines(roi_edges, rho, theta,threshold, min_line_len, max_line_gap)
    r,h = myhough(edges)
    arr = []
    global arra
    arra = []
    for z,w in zip(r,h):
        a = np.cos(w)
        b = np.sin(w)
        x0 = z * a
        y0 = z * b
        x1 = int(x0 + 1000 * (-b))
        if(x1<0):
            x1=abs(x1)
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (b))
        if(x2<0):
            x2 = abs(x2)
        y2 = int(y0 - 1000 * (a))
        if(y2<0):
            y2 = abs(y2)
        s = ([x1,y1,x2,y2])
        arr.append(s)
        arra = np.array(arr)
    leftpart,rightpart = [],[]
    for a,b,c,d in arra:
        slope = (d-b)/(c-a)
        if slope <0:
            leftpart.append(arra)
        else:
            rightpart.append(arra)
    if(len(leftpart)<=0 or len(rightpart)<=0):
        return
    clean_lines(leftpart,0.1)
    clean_lines(rightpart,0.1)
    left_points = [(x1, y1) for arra in leftpart for x1, y1, x2, y2 in arra]
    left_points = left_points + [(x2, y2) for arra in leftpart for x1, y1, x2, y2 in arra]
    right_points = [(x1, y1) for arra in rightpart for x1, y1, x2, y2 in arra]
    right_points = right_points + [(x2, y2) for arra in rightpart for x1, y1, x2, y2 in arra]
    left_results = least_squares_fit(left_points, 325, img.shape[0])
    right_results = least_squares_fit(right_points, 325, img.shape[0])
    vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])
    cv.fillPoly(img, vtxs, (0, 255, 0))
    draw_lanes(drawing, lines)
    result = cv.addWeighted(img, 0.9, drawing, 0.2, 0)
    return result
def draw_lanes(img, lines, color=[0,255, 0]):
    left_lines, right_lines = [],[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return
    '''
    清理異常數據
    '''
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    left_results = least_squares_fit(left_points, 325, img.shape[0])
    right_results = least_squares_fit(right_points, 325, img.shape[0])
    vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])
    '''
    填充車道區域
    '''
    cv.fillPoly(img, vtxs, (0,0,255))
'''
迭代計算斜率平均值，清理差異較大的數據
'''
def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break
'''
使用 least square 來 fit
'''
def least_squares_fit(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    '''
    polyfit() 第三個引數為 fit 多項式的階數，所以一代表線性
    '''
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    return [(xmin, ymin), (xmax, ymax)]
if __name__ == "__main__":
    img = cv.imread('2.jpg')
    img = cv.resize(img,(1000,500))
    result = process_an_image(img)
    cv.imshow("", np.hstack((img, result)))
    cv.waitKey(0)
