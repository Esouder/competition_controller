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
        # print(f"pavement_width: {pavement_width}")
        cv2.line(frame_out, (first_pavement, height-detection_offset), (last_pavement, height-detection_offset), (255, 255, 0), 2)
        cv2.putText(frame_out, f"Width: {pavement_width}", (first_pavement, height - detection_offset), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)

        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        return pavement_width
    
    def detect_truck(self, data):
        kernel = np.ones((5,5), np.uint8)
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv_lower_bounds = np.array([0, 0, 45])
        hsv_upper_bounds = np.array([0, 255, 56])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bounds, hsv_upper_bounds)
        hsv_passed = cv2.bitwise_and(frame,frame, mask= mask)
        greyscale_image = cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(greyscale_image, 25, 255, cv2.THRESH_BINARY)[1]
        eroded = cv2.erode(threshold_image, kernel, iterations = 1)
        dilated = cv2.dilate(eroded, kernel, iterations = 4)
        contours_thisframe, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        best_contour = None
        for contour in contours_thisframe:
            area = cv2.contourArea(contour)
            if area > largest_area:
                best_contour = contour
                largest_area = area
        x, y, w, h = cv2.boundingRect(best_contour)
        # if(not out_frame == None):
        #     cv2.rectangle(out_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        local_out_frame = frame
        cv2.rectangle(local_out_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(local_out_frame, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),2)
        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(local_out_frame, "bgr8"))
        # cv2.imshow("PANTS",local_out_frame)
        # cv2.waitKey(3)
        return x+w/2, y+h/2