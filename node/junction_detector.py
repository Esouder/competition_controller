#import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np

class JunctionDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_pavement_width=None

    def detect_junction(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)
        sum_x = 0
        pixel_count = 1
        height = frame_threshold.shape[0]
        width = frame_threshold.shape[1]
        pavement_width = 0
        for x in range(0,width-1):
            if frame_threshold[height-1][x]:
                pavement_width+=1
        #print(pavement_width)