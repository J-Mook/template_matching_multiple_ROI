import os
import cv2, math
import numpy as np
import pandas as pd
import pyautogui

def LaneFiltering(lines):
    
    if lines is None:
        return None
    
    Lanes = []
    
    for i in range(0,len(lines)):
        l = lines[i][0]
        x1,y1,x2,y2 = l[0],l[1],l[2],l[3]
        if x1 == x2:
            Lanes.append([x1,y1,x2,y2])
            continue
        angle = math.atan((y2-y1)/(x2-x1))*180/math.pi
        if abs(angle) > 10:
            Lanes.append([x1,y1,x2,y2])
    
    return Lanes

def SearchLines(img):
    if img is None:
        return None
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)
    edges = cv2.Canny(gray,40, 100)
    
    cv2.imshow('edges',edges)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80, lines=None, minLineLength=20, maxLineGap=20)
    
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (255,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow('HoughLinesP',edges)
    
    return lines
    

def LaneRecognition():
	while True:
		screen = pyautogui.screenshot(region=(10, 50, 853, 480))
		img = np.array(screen)
		# src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

		lines = SearchLines(img)
		Lane = LaneFiltering(lines)
			
		if Lane is not None:
			for i in range(0, len(Lane)):
				cv2.line(img, (Lane[i][0], Lane[i][1]), (Lane[i][2], Lane[i][3]), (0,255,0), 3, cv2.LINE_AA)

		cv2.imshow('Lane',img)

		if cv2.waitKey(1) == 27:
			break
	cv2.destroyAllWindows()

LaneRecognition()