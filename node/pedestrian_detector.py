import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
import numpy as np

class PedestrianDetector:
    ''' 
        Pedestrian Detector
        
        Class of detectors used to detect important objects and environment 
        elements while at or approaching a crosswalk on the outer loop of track
    '''

    def __init__(self):
        # Start the image bridge and the debug publisher
        self.bridge = CvBridge()
        self.ped_annotated_pub = rospy.Publisher("/pedestrian_detector/image_annotated", Image, queue_size=1)

    def detectCrosswalk(self, data):
        ''' Detect a Crosswalk in a frame'''

        # Minimum number of crosswalk-looking pixels in a frame to meet the 
        # threshold of a crossalk being detected
        MINIMUM_VALID_POINTS = 1500

        # Convert to a cv2 compatible image
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Perform HSB thresholding for the distinctive crosswalk colour
        hsv_lower_bounds = np.array([0, 234, 212])
        hsv_upper_bounds = np.array([0, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bounds, hsv_upper_bounds)
        hsv_passed= cv2.bitwise_and(frame,frame, mask= mask)

        # Convert to greyscale
        greyscale_image=cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)

        # determine dimenstions of image
        height, width = greyscale_image.shape

        # Total up the number of pixels that look like part of a crosswalk in
        # the last 100 lines of the image
        sum = 0
        for y in range(height-100,height-1):
            for x in range(0,width-1):
                if greyscale_image[y][x]:
                    sum += 1

        # Return if there are more than enough valid points
        return sum > MINIMUM_VALID_POINTS
    
    def detectPants(self, data):
        ''' 
            Detect the location of a pedestraian based of his distinctive blue
            jeans. This is also vital to winning a culture victory
        '''

        # Perform HSV thresholding to find pants
        kernel = np.ones((5,5), np.uint8)
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv_lower_bounds = np.array([100, 61, 0])
        hsv_upper_bounds = np.array([179, 255, 79])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bounds, hsv_upper_bounds)
        hsv_passed = cv2.bitwise_and(frame,frame, mask= mask)

        # Greyscale and threshold
        greyscale_image = cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(greyscale_image, 25, 255, cv2.THRESH_BINARY)[1]

        # Erode and dilate to reduce noise and form pants pixels into a
        # contigous blob
        eroded = cv2.erode(threshold_image, kernel, iterations = 1)
        dilated = cv2.dilate(eroded, kernel, iterations = 4)

        # Find the contours of all the blocks of pixels and select the largest 
        # one as the 'best'
        contours_thisframe, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        best_contour = None
        for contour in contours_thisframe:
            area = cv2.contourArea(contour)
            if area > largest_area:
                best_contour = contour
                largest_area = area

        # Get position and size of the bounding box around the best contour
        x, y, w, h = cv2.boundingRect(best_contour)

        # Place a bounding box around the pants on the debug frame and publish
        # it
        local_out_frame = frame
        cv2.rectangle(local_out_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(local_out_frame, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        self.ped_annotated_pub.publish(self.bridge.cv2_to_imgmsg(local_out_frame, "bgr8"))

        # Return the centroid of the pants
        return x+w/2, y+h/2


