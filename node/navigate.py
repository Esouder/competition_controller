#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from cv_bridge import CvBridge

NAVIGATION_SETPOINT = 0.80

class Navigator():
    def __init__(self):
        self.current_state = 'noState'
        self.move = Twist()
        self.move_publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.annotated_feed_pub = rospy.Publisher("/competition_controller/image_annotated", Image, queue_size=1)
        self.bridge = CvBridge()

    # def get_adaptive_speed_factor(self, error, threshold, max_factor):
    #     if(abs(error)>threshold):
    #         asf = 1
    #     else:
    #         # Scales between 1 and max_factor
    #         asf = (((threshold**2)-(error**2))**2)/(threshold**4) * max_factor + 1 
    #         print(asf)

    #     return asf
    
    def navigate_pave(self, frame) -> None:
        '''Navigation algorithm based on pavement'''
        frame_out = frame.copy()
        kP = 0.02
        kD = 0.001
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((9,9),np.uint8)
        frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 1)

        frame_sobel = cv2.Sobel(frame_threshold, cv2.CV_64F, 1, 1, ksize=3)
        frame_corrected = frame_sobel.astype(np.uint8)
        frame_cropped = frame_corrected[400:-1][0:-1]
        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 150, minLineLength=250, maxLineGap=250)
        height = frame.shape[0]
        width = frame.shape[1]
        frame_lines = np.zeros((height-400,width,3), np.uint8)
        detection_area_top = 200
        detection_area_bottom = 100
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.line(frame_out, (x1, y1+400), (x2, y2+400), (0, 0, 255), 2)
            frame_lines_grey = cv2.cvtColor(frame_lines, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("OBSERVE HYPNOTOAD", frame_lines_grey)
            # cv2.waitKey(3)
            sum_x = 0
            pixel_count = 1
            height = frame_lines_grey.shape[0]
            width = frame_lines_grey.shape[1]

            for y in range(height-detection_area_top,height-detection_area_bottom):
              for x in range(int(width/2),width-1):
                 if frame_lines_grey[y][x]:
                      sum_x += x
                      pixel_count+=1
            x_avg = int(sum_x / pixel_count)
        except TypeError:
            x_avg = 0

        # Place a line at the average x position on the image
        cv2.line(frame_out, (x_avg, height-detection_area_top+400), (x_avg, height-detection_area_bottom+400), (0, 255, 0), thickness=10)
        # cv2.imshow("XAVG", frame_out)
        # cv2.waitKey(3)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        error = width*NAVIGATION_SETPOINT - x_avg
        self.move.linear.x = 0.4
        #derivative = prev_error-error
        #cv2.putText(frame_out, f"error:{error} | derivative: {derivative}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        if(x_avg<=1):
            self.move.angular.z = 0
        else:
            self.move.angular.z = error*kP #+ derivative*kD
        
        prev_error = error

    def navigate_pave_left(self, frame) -> None:
        '''Navigation algorithm based on pavement'''
        frame_out = frame.copy()
        kP = 0.01
        kD = 0.001
        height = frame.shape[0]
        width = frame.shape[1]
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((21,21),np.uint8)
        frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 2)

        frame_sobel = cv2.Sobel(frame_threshold, cv2.CV_64F, 1, 1, ksize=3)
        frame_corrected = frame_sobel.astype(np.uint8)
        frame_cropped = frame_corrected[400:-1, 0:int(width/2)]
        #cv2.imshow("CROPPED", frame_cropped)
        #cv2.waitKey(3)

        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 50, minLineLength=75, maxLineGap=250)
        
        frame_threshold_out = frame_threshold.copy()
        cv2.rectangle(frame_threshold_out, (0,height), (int(width/2),400), (255,0,0), 2)
        #cv2.imshow("OBSERVE", frame_threshold_out)
        #cv2.waitKey(3)
        frame_lines = np.zeros((height-400,width,3), np.uint8)
        detection_area_top = 300
        detection_area_bottom = 0
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.line(frame_out, (x1, y1+400), (x2, y2+400), (0, 0, 255), 2)
            frame_lines_grey = cv2.cvtColor(frame_lines, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("OBSERVE HYPNOTOAD", frame_lines_grey)
            #cv2.waitKey(3)
            sum_x = 0
            pixel_count = 1
            height = frame_lines_grey.shape[0]
            width = frame_lines_grey.shape[1]

            for y in range(height-detection_area_top,height-detection_area_bottom):
              for x in range(0,int(width/2)):
                 if frame_lines_grey[y][x]:
                      sum_x += x
                      pixel_count+=1
            x_avg = int(sum_x / pixel_count)
        except TypeError:
            x_avg = 0

        # Place a line at the average x position on the image
        cv2.line(frame_out, (x_avg, height-detection_area_top+400), (x_avg, height-detection_area_bottom+400), (0, 255, 0), thickness=10)
        # cv2.imshow("XAVG", frame_out)
        # cv2.waitKey(3)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        error = width*0.2 - x_avg
        self.move.linear.x = 0.2
        #derivative = prev_error-error
        #cv2.putText(frame_out, f"error:{error} | derivative: {derivative}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        if(x_avg<=1):
            self.move.angular.z = 0.4
        else:
            self.move.angular.z = error*kP #+ derivative*kD
        
        prev_error = error

    def navigate_pave_inside(self, frame) -> None:
        '''Navigation algorithm based on pavement'''
        frame_out = frame.copy()
        kP = 0.0125
        kD = 0.001
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((9,9),np.uint8)
        frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 3)

        frame_sobel = cv2.Sobel(frame_threshold, cv2.CV_64F, 1, 1, ksize=3)
        frame_corrected = frame_sobel.astype(np.uint8)
        frame_cropped = frame_corrected[400:-1, 640:-1]
        #cv2.imshow("the marge of lake labarge", frame_cropped)
        #cv2.waitKey(3)
        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 50, minLineLength=75, maxLineGap=250)
        height = frame.shape[0]
        width = frame.shape[1]
        frame_lines = np.zeros((height-400,width,3), np.uint8)
        detection_area_top = 200
        detection_area_bottom = 100
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_lines, (x1+640, y1), (x2+640, y2), (255, 255, 255), 2)
                    cv2.line(frame_out, (x1+640, y1+400), (x2+640, y2+400), (0, 0, 255), 2)
            frame_lines_grey = cv2.cvtColor(frame_lines, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("OBSERVE HYPNOTOAD", frame_lines_grey)
            # cv2.waitKey(3)
            sum_x = 0
            pixel_count = 1
            height = frame_lines_grey.shape[0]
            width = frame_lines_grey.shape[1]

            for y in range(height-detection_area_top,height-detection_area_bottom):
              for x in range(int(width/2),width-1):
                 if frame_lines_grey[y][x]:
                      sum_x += x
                      pixel_count+=1
            x_avg = int(sum_x / pixel_count)
        except TypeError:
            x_avg = 0

        # Place a line at the average x position on the image
        cv2.line(frame_out, (x_avg, height-detection_area_top+400), (x_avg, height-detection_area_bottom+400), (0, 255, 0), thickness=10)
        # cv2.imshow("XAVG", frame_out)
        # cv2.waitKey(3)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        error = width*NAVIGATION_SETPOINT - x_avg
        self.move.linear.x = 0.3
        #derivative = prev_error-error
        #cv2.putText(frame_out, f"error:{error} | derivative: {derivative}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        if(x_avg<=1):
            self.move.angular.z = 0
        else:
            self.move.angular.z = error*kP #+ derivative*kD
        
        prev_error = error


    def navigate_pre_grass(self, frame) -> None:
        '''short turn to get onto the grass'''
        self.move.angular.z = 1.5
        self.move.linear.x = 0


    def navigate_grass(self, frame) -> None:
        '''Navigation algorithm based on grass'''

        frame_out = frame

        kP = 0.01
        kD = 0.001
        lower = np.array([28, 41, 106])
        upper = np.array([37, 79, 255])
        bright = 1.36
        unbrightened = frame
        cols, rows, _ = unbrightened.shape
        brightness = np.sum(unbrightened) / (255 * cols * rows)
        ratio = brightness / bright
        if ratio >= 1:
          img = unbrightened
        else:
            # Otherwise, adjust brightness to get the target brightness
            img = cv2.convertScaleAbs(unbrightened, alpha = 1 / ratio, beta = 0)
        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        hsv_passed= cv2.bitwise_and(img,img, mask= mask)

        greyscale_image=cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)

        navigation_start = 500
        navigation_end = 780
        width, height = greyscale_image.shape

        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(greyscale_image, kernel, iterations = 1)
        dilated = cv2.dilate(eroded, kernel, iterations = 2)
        dilroded_image = dilated

        sobeled_image = cv2.Sobel(dilroded_image, cv2.CV_64F, 0, 1, ksize=3)

        frame_corrected = sobeled_image.astype(np.uint8)
        frame_cropped = frame_corrected[400:-1, 640:-1]

        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 40, minLineLength=25, maxLineGap=500)

        navigation_area_image = sobeled_image[0:-1][navigation_start:navigation_end]

        out_height = frame_out.shape[0]
        out_width = frame_out.shape[1]


        # cv2.imshow("OBSERVE HYPNOTOAD", frame_cropped)
        # cv2.waitKey(3)
        

        sum_x = 0
        pixel_count = 1
        height = navigation_area_image.shape[0]
        width = navigation_area_image.shape[1]
        for y in range(0,height-1):
            for x in range(int(width*0.5),width-1):
                if navigation_area_image[y][x]:
                    sum_x += x
                    pixel_count+=1

        x_avg = int(sum_x / pixel_count)

        # Place a line at the average x position on the image
        frame_out = frame.copy()
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_out, (x1+640, y1+400), (x2+640, y2+400), (0, 0, 255), 2)
        except TypeError:
            print("bad")
        cv2.line(frame_out, (x_avg, frame_out.shape[0]-200), (x_avg, frame_out.shape[0]-100), (0, 255, 0), thickness=10)
        # cv2.imshow("XAVG", frame_out)
        # cv2.waitKey(3)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        error = width*NAVIGATION_SETPOINT - x_avg
        #cv2.line(frame_out, (x_avg, out_height-100), (x_avg, out_height-1), (0, 255, 0), thickness=10)
        #cv2.line(frame_out, (0, navigation_start), (out_width-1, navigation_start), (0, 0, 255), thickness=10)
        #cv2.line(frame_out, (0, navigation_end), (out_width-1, navigation_end), (0, 0, 255), thickness=10)
        #cv2.line(frame_out, (int(width/2), 0), (int(width/2), out_height-1), (255, 0, 0), thickness=10)
        cv2.rectangle(frame_out, (int(out_width*0.5),navigation_start), (out_width-1, navigation_end), (255,0,0), 5)
        cv2.putText(frame_out, f"error:{error}", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)


        if(x_avg<=1):
            self.move.angular.z = 0
        else:
            self.move.angular.z = error*kP

        self.move.linear.x = 0.2

        # cv2.imshow("DEBUG", frame_out)

    def navigate_pre_pave_inside(self, frame):
        self.move.angular.z = 0.25
        self.move.linear.x = 0.2
    
    def navigate_startup(self, frame) -> None:
        '''Navigate during initial startup'''
        self.move.angular.z = 0.5
        self.move.linear.x = 0.15

    def navigate_crosswalk_traverse(self, frame) ->None:
        '''Navigate during crosswalk traversal'''
        self.move.angular.z = 0
        self.move.linear.x = 0.5
    
    def navigate_stopped(self, frame) -> None:
        '''Navigate, but, like, don't'''
        self.move.angular.z = 0
        self.move.linear.x = 0

    def navigate_stop_turn_left(self, frame) -> None:
        '''Turn left'''
        self.move.angular.z = 0.1
        self.move.linear.x = -0.01
    
    def navigate_stop_turn_right(self, frame) -> None:
        '''Turn right'''
        self.move.angular.z = -0.1
        self.move.linear.x = -0.01

    def navigate(self, data):
        '''Run a navigation step based on a single frame'''
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if self.current_state == "StartupTurnAndDrive":
            self.navigate_startup(frame)
        elif self.current_state == "PaveNavigate":
            self.navigate_pave(frame)
        elif self.current_state == "PaveNavigateLeft":
            self.navigate_pave_left(frame)
        elif self.current_state == "PrePaveNavigateInside":
            self.navigate_pre_pave_inside(frame)
        elif self.current_state == "PaveNavigateInside":
            self.navigate_pave_inside(frame)
        elif self.current_state == "JunctionWait":
            self.navigate_stopped(frame)
        elif self.current_state == "PreGrassNavigate":
            self.navigate_pre_grass(frame)
        elif self.current_state == "GrassNavigate":
            self.navigate_grass(frame)
        elif self.current_state == "StopTurnLeft":
            self.navigate_stop_turn_left(frame)
        elif self.current_state == "StopTurnRight":
            self.navigate_stop_turn_right(frame)
        elif self.current_state == "CrosswalkWait":
            self.navigate_stopped(frame)
        elif self.current_state == "CrosswalkTraverse":
            self.navigate_crosswalk_traverse(frame)
        elif self.current_state == "Finished":
            self.navigate_stopped(frame)
        else:
            pass
        
        self.move_publisher.publish(self.move)
    
    def update_state(self, state):
        '''Update the state'''
        self.current_state = state.data

### Node ###

def navigate_node(initiator_msg):
    navigator = Navigator()
    camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, navigator.navigate)
    state_sub = rospy.Subscriber("/competition_controller/state", String, navigator.update_state)
    rospy.spin()

if __name__ == '__main__':
    # What an ugly piece of boilerplate. This is why people hate you, Python
    # Wait until the competition controller tells us to start
    rospy.init_node('navigate')
    start_subscriber = rospy.Subscriber("/competition_controller/start", String, navigate_node)
    print("----------------------------------------")
    print("Waiting for competition controller to start")
    print("----------------------------------------")
    rospy.spin()
    # navigate_node()