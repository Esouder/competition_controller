#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class Navigator():
    def __init__(self):
        self.current_state = 'noState'
        self.move = Twist()
        self.move_publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    
    def navigate_pave(self, frame) -> None:
        '''Navigation algorithm based on pavement'''

    def navigate_grass(self, frame) -> None:
        '''Navigation algorithm based on grass'''
    
    def navigate_startup(self, frame) -> None:
        '''Navigate during initial startup'''
        self.move.angular.z = 0.5
        self.move.linear.x = 0.15
    
    def navigate_stopped(self, frame) -> None:
        '''Navigate, but, like, don't'''
        self.move.angular.z = 0
        self.move.linear.x = 0

    def navigate(self, frame):
        '''Run a navigation step based on a single frame'''
        if self.current_state == "StartupTurnAndDrive":
            self.navigate_startup(frame)
        elif self.current_state == "PaveNavigate":
            self.navigate_pave(frame)
        elif self.current_state == "GrassNavigate":
            self.navigate_grass(frame)
        elif self.current_state == "Finished":
            self.navigate_stopped(frame)
        else:
            pass
        
        self.move_publisher.publish(self.move)
    
    def update_state(self, state):
        '''Update the state'''
        self.current_state = state.data

### Node ###

def navigate_node():
    rospy.init_node('topic_subscriber')
    navigator = Navigator()
    camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, navigator.navigate)
    state_sub = rospy.Subscriber("/state", String, navigator.update_state)
    rospy.spin()

if __name__ == '__main__':
    # What an ugly piece of boilerplate. This is why people hate you, Python.
    navigate_node()