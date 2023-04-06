#import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np

class PedestrianDetector:
    def __init__(self):
        self.bridge = CvBridge()
    def detectCrosswalk(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        MINIMUM_VALID_POINTS = 1500
        hsv_lower_bounds = np.array([0, 234, 212])
        hsv_upper_bounds = np.array([0, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bounds, hsv_upper_bounds)
        hsv_passed= cv2.bitwise_and(frame,frame, mask= mask)
        greyscale_image=cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)
        height, width = greyscale_image.shape
        sum = 0
        for y in range(height-20,height-1):
            for x in range(0,width-1):
                if greyscale_image[y][x]:
                    sum += 1
        cv2.putText(frame, f"count:{sum}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.imshow("Crosswalk", greyscale_image)
        cv2.waitKey(3)

        return sum > MINIMUM_VALID_POINTS



