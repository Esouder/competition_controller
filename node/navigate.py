#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from cv_bridge import CvBridge

# Set point for right-side navigation
NAVIGATION_SETPOINT = 0.80

class Navigator():
    '''
        Navigator
        
        The Navigator class handles all of the navigation for the robot. It is
        the only class that can issue cmd_vel messages. Navigation mode is
        selected based on the state reported by the state machine and can either 
        be pre-progammed constant motions or based on the information in camera
        frames.
    '''
    def __init__(self):
        ''' Create a Navigator instance'''

        # Initialize the current state to no state
        self.current_state = 'noState'

        # Initialize the move object to be published and the publisher
        self.move = Twist()
        self.move_publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        # Initialize the debug frame publisher
        self.annotated_feed_pub = rospy.Publisher("/competition_controller/image_annotated", Image, queue_size=1)

        # Initialize the image bridge
        self.bridge = CvBridge()
    
    def navigate_pave(self, frame) -> None:
        '''
            Navigate Pave

            Navigation algorithm for first pavement sections of the outer track. 
            Works by tracking the outside (right) edge of the track.    
        '''
        # Initialize the debug frame
        frame_out = frame.copy()

        # Set proportionality constant
        kP = 0.02

        # Convert the image to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Isolate the 'blue' channel, if the image were BGR
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0

        # Convert to greyscale and threshold
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        # Perform dilation to remove gaps in the pavement caused by the 1m grid
        # squares
        kernel = np.ones((9,9),np.uint8)
        frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 1)

        # Perform sobel edge detection to find the edges of tha pavement
        frame_sobel = cv2.Sobel(frame_threshold, cv2.CV_64F, 1, 1, ksize=3)

        # Convert to uint8
        frame_corrected = frame_sobel.astype(np.uint8)
        
        # Crop out the top 400 pixels of the image to focus only on the areas
        # with track
        frame_cropped = frame_corrected[400:-1][0:-1]

        # Find probabilistic hough lines based off the edges
        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 150, minLineLength=250, maxLineGap=250)

        # Create a blank image in the shape of the cropped image
        height = frame.shape[0]
        width = frame.shape[1]
        frame_lines = np.zeros((height-400,width,3), np.uint8)

        # Define the area to detect lines in when finding the average position
        # of the lines.
        # Values are in pixels from the bottom of the frame
        detection_area_top = 200
        detection_area_bottom = 100

        try:
            # Draw lines on the blank image based on the hough lines detected earlier
            # Also draw them on the debug image
            # In a try/catch block in case there are no lines this frame
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.line(frame_out, (x1, y1+400), (x2, y2+400), (0, 0, 255), 2)
            
            # Convert to greyscale
            frame_lines_grey = cv2.cvtColor(frame_lines, cv2.COLOR_BGR2GRAY)

            # Get the average location of the pixels in the right half of the 
            # detection area of the hough lines image.
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

        # Place a line at the average x position on the debug frame and publish
        # it
        cv2.line(frame_out, (x_avg, height-detection_area_top+400), (x_avg, height-detection_area_bottom+400), (0, 255, 0), thickness=10)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        # Calculate the error based on the setpoint 
        error = width*NAVIGATION_SETPOINT - x_avg

        # Constant linear rate
        self.move.linear.x = 0.35

        # Set rotation based on error and proportionality constant, unless the 
        # x_avg is 1, meaning we didn't detect a single pixel and we should go 
        # straight.
        if(x_avg<=1):
            self.move.angular.z = 0
        else:
            self.move.angular.z = error*kP #+ derivative*kD

    def navigate_pave_left(self, frame) -> None:
        '''
            Navigate Pave Left

            Navigation algorithm for final pavement sections of the outer track,
            after the grass section. Designed to turn into the junction to the 
            inner track. Works by tracking the inside (left) edge of the track.
        '''

        # Create a debug frame
        frame_out = frame.copy()
        
        # Set proportionality constant
        kP = 0.01

        # Get frame height
        height = frame.shape[0]
        width = frame.shape[1]

        # Convert the image to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Isolate the 'blue' channel, if the image were BGR
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0

        # Convert to greyscale and threshold
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        # Perform dilation to remove gaps in the pavement caused by the 1m grid
        # squares
        kernel = np.ones((21,21),np.uint8)
        frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 2)

        # Perform sobel edge detection to find the edges of tha pavement
        frame_sobel = cv2.Sobel(frame_threshold, cv2.CV_64F, 1, 1, ksize=3)

        # Convert to uint8
        frame_corrected = frame_sobel.astype(np.uint8)

        # Crop out the top 400 pixels of the image to focus only on the areas
        # with track
        frame_cropped = frame_corrected[400:-1, 0:int(width/2)]

        # Find probabilistic hough lines based off the edges
        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 50, minLineLength=75, maxLineGap=250)
        
        # Create a debug frame based on the thresholded image and draw a
        # rectangle showing the area searched to generate lines on it
        frame_threshold_out = frame_threshold.copy()
        cv2.rectangle(frame_threshold_out, (0,height), (int(width/2),400), (255,0,0), 2)

        # Create a blank image in the shape of the cropped image
        frame_lines = np.zeros((height-400,width,3), np.uint8)

        # Define the area to detect lines in when finding the average position
        # of the lines.
        # Values are in pixels from the bottom of the frame
        detection_area_top = 300
        detection_area_bottom = 0

        try:
            # Draw lines on the blank image based on the hough lines detected earlier
            # Also draw them on the debug image
            # In a try/catch block in case there are no lines this frame
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.line(frame_out, (x1, y1+400), (x2, y2+400), (0, 0, 255), 2)

            # Convert to greyscale
            frame_lines_grey = cv2.cvtColor(frame_lines, cv2.COLOR_BGR2GRAY)

            # Get the average location of the pixels in the left half of the 
            # detection area of the hough lines image.
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
            print("TypeError at navigate_pave_left")
            x_avg = 0


        # Place a line at the average x position on the debug frame and publish
        # it
        cv2.line(frame_out, (x_avg, height-detection_area_top+400), (x_avg, height-detection_area_bottom+400), (0, 255, 0), thickness=10)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        # Calculate the error based on a setpoint of 0.2 of the width of the 
        # frame
        error = width*0.2 - x_avg

        # Constant lienar rate
        self.move.linear.x = 0.2

        # Set rotation based on error and proportionality constant, unless the 
        # x_avg is 1, meaning we didn't detect a single pixel and we should 
        # default to turning slight left.
        if(x_avg<=1):
            print("TURNING LEFT since no line detected")
            self.move.angular.z = 0.4
        else:
            self.move.angular.z = error*kP
        
    def navigate_pave_inside(self, frame) -> None:
        '''
            Navigate Pave Inside

            Navigation algorithm for the inner pavement track, after the grass 
            section. Works by tracking the inside (right) edge of the track.
        '''

        # Create a debug frame
        frame_out = frame.copy()

        # Set the proportionality constant
        kP = 0.0125

        # Convert the image to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Isolate the 'blue' channel, if the image were BGR
        frame_blues = frame_hsv
        frame_blues[:, :, 1] = 0
        frame_blues[:, :, 2] = 0

        # Convert to greyscale and threshold
        frame_grey = cv2.cvtColor(frame_blues, cv2.COLOR_BGR2GRAY)
        _, frame_threshold = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY_INV)

        # Perform dilation to remove gaps in the pavement caused by the 1m grid
        # squares
        kernel = np.ones((9,9),np.uint8)
        frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 3)

        # Perform sobel edge detection to find the edges of tha pavement
        frame_sobel = cv2.Sobel(frame_threshold, cv2.CV_64F, 1, 1, ksize=3)

        # Convert to uint8
        frame_corrected = frame_sobel.astype(np.uint8)

        # Crop out the top 400 pixels of the image to focus only on the areas
        # with track
        frame_cropped = frame_corrected[400:-1, 640:-1]

        # Find probabilistic hough lines based off the edges
        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 50, minLineLength=75, maxLineGap=250)

        # Create a debug frame based on the thresholded image and draw a
        # rectangle showing the area searched to generate lines on it
        height = frame.shape[0]
        width = frame.shape[1]

        # Create a blank image in the shape of the cropped image
        frame_lines = np.zeros((height-400,width,3), np.uint8)

        # Define the area to detect lines in when finding the average position
        # of the lines.
        # Values are in pixels from the bottom of the frame
        detection_area_top = 200
        detection_area_bottom = 100
        try:
            # Draw lines on the blank image based on the hough lines detected earlier
            # Also draw them on the debug image
            # In a try/catch block in case there are no lines this frame
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_lines, (x1+640, y1), (x2+640, y2), (255, 255, 255), 2)
                    cv2.line(frame_out, (x1+640, y1+400), (x2+640, y2+400), (0, 0, 255), 2)

            # Convert to greyscale
            frame_lines_grey = cv2.cvtColor(frame_lines, cv2.COLOR_BGR2GRAY)

            # Get the average location of the pixels in the right half of the 
            # detection area of the hough lines image.
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
        except TypeError as e:
            print("TypeError at navigate_pave_inside")
            print(e)
            x_avg = 0

        # Place a line at the average x position on the debug frame and publish
        # it
        cv2.line(frame_out, (x_avg, height-detection_area_top+400), (x_avg, height-detection_area_bottom+400), (0, 255, 0), thickness=10)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        # Calculate the error based on the default setpoint
        error = width*NAVIGATION_SETPOINT - x_avg

        # Constant lienar rate
        self.move.linear.x = 0.3

        # Set rotation based on error and proportionality constant, unless the 
        # x_avg is 1, meaning we didn't detect a single pixel and we should 
        # default to turning slight left.
        if(x_avg<=1):
            self.move.angular.z = 0
        else:
            self.move.angular.z = error*kP

    def navigate_pre_grass(self, frame) -> None:
        ''' 
            Navigate Pre Grass

            Pre-programmed constant speed and angular rate turn during entry 
            onto the grass.
        '''

        self.move.angular.z = 1.5
        self.move.linear.x = 0


    def navigate_grass(self, frame) -> None:
        '''
            Navigate Grass
            
            Navigation algorithm for navigation on the grass/hill. Works by 
            tracking the ouside (right) edge of the track
        '''

        # Create a debug frame
        frame_out = frame

        # Set the proportionality constant
        kP = 0.01

        # Perform HSV thresholding and brightess compensation
        # Brightness compenstion brings a frame below a set brightness to up to 
        # that level. Also get the shape of the image.
        lower = np.array([28, 41, 106])
        upper = np.array([37, 79, 255])
        bright = 1.36
        unbrightened = frame
        cols, rows, _ = unbrightened.shape
        brightness = np.sum(unbrightened) / (255 * cols * rows)
        ratio = brightness / bright
        if ratio >= 1:
            # The image is brighter than the target brightness
            img = unbrightened
        else:
            # Adjust brightness to get the target brightness
            img = cv2.convertScaleAbs(unbrightened, alpha = 1 / ratio, beta = 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        hsv_passed= cv2.bitwise_and(img,img, mask= mask)

        # Convert to greyscale
        greyscale_image=cv2.cvtColor(hsv_passed, cv2.COLOR_BGR2GRAY)

        # Define the vertical portion of the image to search for the painted 
        # lines om
        navigation_start = 500
        navigation_end = 780

        # Get width and height
        width, height = greyscale_image.shape

        # Erode and dilate the image to remove noise
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(greyscale_image, kernel, iterations = 1)
        dilated = cv2.dilate(eroded, kernel, iterations = 2)
        dilroded_image = dilated

        # Perform sobel edge detection
        sobeled_image = cv2.Sobel(dilroded_image, cv2.CV_64F, 0, 1, ksize=3)

        # Experemental: perform hough lines detection (not used for navigation)
        frame_corrected = sobeled_image.astype(np.uint8)
        frame_cropped = frame_corrected[400:-1, 640:-1]
        lines = cv2.HoughLinesP(frame_cropped, 1, np.pi/180, 40, minLineLength=25, maxLineGap=500)

        # crop the image to the area used for navigation
        navigation_area_image = sobeled_image[0:-1][navigation_start:navigation_end]

        # Ger the dimentions of the debug image
        out_height = frame_out.shape[0]
        out_width = frame_out.shape[1]
        
        # Find the average x position of the painted lines in the right half of 
        # the cropped image
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

        # Create a debug image
        frame_out = frame.copy()
        try:
            # Place experimental lines on the output image
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame_out, (x1+640, y1+400), (x2+640, y2+400), (0, 0, 255), 2)
        except TypeError:
            print("No Grass Lines Found")
        
        # Place line on the average x position in the debug frame and publish the debug frame
        cv2.line(frame_out, (x_avg, frame_out.shape[0]-200), (x_avg, frame_out.shape[0]-100), (0, 255, 0), thickness=10)
        self.annotated_feed_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, "bgr8"))

        # Determine the rror based off the standard setpoint
        error = width*NAVIGATION_SETPOINT - x_avg

        # Set rotation based on error and proportionality constant, unless the 
        # x_avg is 1, meaning we didn't detect a single pixel and we should not
        # turn.
        if(x_avg<=1):
            self.move.angular.z = 0
        else:
            self.move.angular.z = error*kP

        # Constant linear rate
        self.move.linear.x = 0.25

    def navigate_pre_pave_inside(self, frame):
        ''' 
            Navigate Pre Pave Inside

            Pre-programmed constant speed and angular rate turn during entry  
            into inside loop.
        '''
        self.move.angular.z = 0.25
        self.move.linear.x = 0.15
    
    def navigate_startup_straight(self, frame) -> None:
        '''
            Navigate Startup Straight
        
            Pre-programmed constant speed (zero angular rate) during initial 
            entry into outside loop.
        '''
        self.move.angular.z = 0.1
        self.move.linear.x = 0.15
    
    def navigate_startup_turn(self, frame) -> None:
        '''
            Navigate Startup Turn
        
            Pre-programmed constant speed and angular rate turn during entry  
            into outside loop.
        '''
        self.move.angular.z = 1.0
        self.move.linear.x = 0.05

    def navigate_crosswalk_traverse(self, frame) ->None:
        '''
            Navigate Crosswalk Traverse

            Drive straight and fast through the crosswalk
        '''
        self.move.angular.z = 0
        self.move.linear.x = 0.5
    
    def navigate_stopped(self, frame) -> None:
        '''
            Navigate Stopped
            Navigate, but, like, don't
        '''
        self.move.angular.z = 0
        self.move.linear.x = 0

    def navigate_stop_turn_left(self, frame) -> None:
        '''Turn left'''
        self.move.angular.z = 0.1
        self.move.linear.x = -0.005
    
    def navigate_stop_turn_right(self, frame) -> None:
        '''Turn right'''
        self.move.angular.z = -0.1
        self.move.linear.x = -0.005

    def navigate(self, data):
        '''
            Run a navigation step based on a single frame. Selects a navigation
            algorithm based on the current state and calculates the move using
            it. Should be called on each frame.
        '''

        # Create a cv2 compatable image
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Select and run a navigation algorithm based on the current state
        if self.current_state == "StartupStraight":
            self.navigate_startup_straight(frame)
        elif self.current_state == "StartupTurnAndDrive":
            self.navigate_startup_turn(frame)
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
        
        # Publish the move
        self.move_publisher.publish(self.move)
    

    def update_state(self, state):
        '''
            Callback function to update the navigator's internal storage of the
            navigation state whenever we recieve a new message from the state
            machine node
        '''
        self.current_state = state.data

### Node ###

def navigate_node(initiator_msg):
    '''
        Create the ros node to run the navigator
    '''
    
    # Initialize the navigator
    navigator = Navigator()

    # Create the camera subscriber and assign the navigator callback
    camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, navigator.navigate)

    # Create the state subscriber and assign the update state callback
    state_sub = rospy.Subscriber("/competition_controller/state", String, navigator.update_state)

    # Ensure the node does not die
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