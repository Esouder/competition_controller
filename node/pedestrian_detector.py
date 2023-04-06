#import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
import numpy as np

class PedestrianDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.ped_annotated_pub = rospy.Publisher("/pedestrian_detector/image_annotated", Image, queue_size=1)

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
        # cv2.imshow("Crosswalk", greyscale_image)
        # cv2.waitKey(3)

        return sum > MINIMUM_VALID_POINTS
    
    def detectPants(self, data):
        kernel = np.ones((5,5), np.uint8)
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv_lower_bounds = np.array([100, 61, 0])
        hsv_upper_bounds = np.array([179, 255, 79])
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
        cv2.rectangle(local_out_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(local_out_frame, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(local_out_frame, "bgr8"))
        # cv2.imshow("PANTS",local_out_frame)
        # cv2.waitKey(3)
        return x+w/2, y+h/2


