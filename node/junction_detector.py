import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class JunctionDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_pavement_width=None
        self.ped_annotated_pub = rospy.Publisher("/pedestrian_detector/image_annotated", Image, queue_size=1)


    def detect_width(self, data):
        detection_offset = 150
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        frame_out = frame.copy()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        height = frame_threshold.shape[0]
        width = frame_threshold.shape[1]
        pavement_width = 0
        first_pavement = width
        last_pavement = 0
        for x in range(0,width-1):
            if frame_threshold[height-detection_offset][x]:
                if x < first_pavement:
                    first_pavement = x
                if x > last_pavement:
                    last_pavement = x
                pavement_width+=1
        print(pavement_width)
        cv2.line(frame_out, (first_pavement, height-detection_offset), (last_pavement, height-detection_offset), (255, 255, 0), 2)
        cv2.putText(frame_out, f"Width: {pavement_width}", (first_pavement, height - detection_offset), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)

        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))


        return pavement_width