import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class JunctionDetector:
    ''' 
        Junction Detector
        
        Class of detectors used to detect important objects and environment 
        elements while at or approaching a junction between the outer and inner 
        loops of track
    '''
    def __init__(self):
        # Init a image bridge
        self.bridge = CvBridge()
        # No pavement width has been seen before
        self.last_pavement_width=None
        # (misnamed) publisher for debug images from this state
        self.ped_annotated_pub = rospy.Publisher("/pedestrian_detector/image_annotated", Image, queue_size=1)


    def detect_width(self, data):
        '''
            Detect the width of pavement in front of the robot
        '''

        # No. Of pixels above the bottom of the image to find the width
        detection_offset = 150

        # Prepare the image and debug image
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        frame_out = frame.copy()

        # Convert to HSV 
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Select only the 'blue' channel, if this were an RGB representation
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0

        # Convert to monochrome and thrshold
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        # Find height and width of frame
        height = frame_threshold.shape[0]
        width = frame_threshold.shape[1]

        # Detected width of pavement
        pavement_width = 0

        # First detected pavement pixel and last detected pavement pixel
        # since we scan from left to right, we bein with the first being on the 
        # rightmost edge (room to move left) and the last being on the leftmost
        # edge (room to move right)
        first_pavement = width
        last_pavement = 0

        # Scan through a row of pixels and detect the pavement width
        for x in range(0,width-1):
            if frame_threshold[height-detection_offset][x]:
                if x < first_pavement:
                    first_pavement = x
                if x > last_pavement:
                    last_pavement = x
                pavement_width+=1

        # Place a line and print the width on the debug image
        cv2.line(frame_out, (first_pavement, height-detection_offset), (last_pavement, height-detection_offset), (255, 255, 0), 2)
        cv2.putText(frame_out, f"Width: {pavement_width}", (first_pavement, height - detection_offset), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)

        # Publish the debug image
        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        return pavement_width
    
    def detect_truck(self, data):
        ''' 
            Detect the location of the inner-ring truck in the frame
        '''
        
        # Get the frame and debug frame as a cv2 comparable image
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        local_out_frame = frame

        # Perform HSV thresholding to isolate the pixels on the truck
        hsv_lower_bounds = np.array([0, 0, 45])
        hsv_upper_bounds = np.array([0, 255, 56])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bounds, hsv_upper_bounds)
        hsv_passed = cv2.bitwise_and(frame,frame, mask= mask)

        # Convert to greyscale and threshold
        greyscale_image = cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(greyscale_image, 25, 255, cv2.THRESH_BINARY)[1]

        # Erode and heavily dilate to merge the many truck pixels into one big 
        # blob
        kernel = np.ones((5,5), np.uint8)
        eroded = cv2.erode(threshold_image, kernel, iterations = 1)
        dilated = cv2.dilate(eroded, kernel, iterations = 4)

        # Find the contour around each blob in the image
        contours_thisframe, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find the best contour based on largest area
        largest_area = 0
        best_contour = None
        for contour in contours_thisframe:
            area = cv2.contourArea(contour)
            if area > largest_area:
                best_contour = contour
                largest_area = area

        # Determine the position and size of the best contour
        x, y, w, h = cv2.boundingRect(best_contour)

        # Place a rectange and print the location on the debug frame
        cv2.rectangle(local_out_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(local_out_frame, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),2)
        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(local_out_frame, "bgr8"))

        # Return the centroid of the truck
        return x+w/2, y+h/2