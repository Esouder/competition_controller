#! /usr/bin/env python3
from __future__ import annotations
import rospy
from enum import Enum
from abc import ABC, abstractmethod
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
import time
from cv_bridge import CvBridge

from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

from surface_detector import SurfaceDetector
from pedestrian_detector import PedestrianDetector
from junction_detector import JunctionDetector

# eww globals

surface_detector = None
pedestrian_detector = PedestrianDetector()
junction_detector = JunctionDetector()
competition_start_time = None

# Constants
MAX_COMPETITION_TIME = 3.75*60 # Automatically send end message after this time

### Abstract State ###

class AbstractState(ABC):

    def __init__(self):
        print(f"Entered new state: {self.get_state_name()}")

    @abstractmethod
    def get_state_name(self) -> str:
        '''Get the name for the state'''
        pass
    @abstractmethod
    def evaluate_transition(self, data) -> AbstractState:
        '''Fetch the next state'''
        pass

### Actual States ###
    
class State_StartupTurnAndDrive(AbstractState):
    def __init__(self):
        super().__init__()
        time.sleep(1)
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 3.0
        self.stateEntryAction()
    def stateEntryAction(self):
        global competition_start_time
        start_publisher = rospy.Publisher("/license_plate", String, queue_size=1)
        time.sleep(1)
        competition_start_time = rospy.get_time()
        competition_begin_message = str("Team4,multi21,0,AA00")
        start_publisher.publish(competition_begin_message)
        print("Began Timer")
    def get_state_name(self) -> str:
        return "StartupTurnAndDrive"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigate()
        else:
            return self
    
class State_PaveNavigate(AbstractState):
    def __init__(self):
        super().__init__()
        self.previous_transitions = 0
    def get_state_name(self) -> str:
        return "PaveNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        if(pedestrian_detector.detectCrosswalk(data)):
            return State_CrosswalkWait()
        current_surface = surface_detector.poll(data)
        if current_surface == SurfaceDetector.RoadSurface.GRASS:
            if(self.previous_transitions > 9):
                return State_PreGrassNavigate()
            else:
                self.previous_transitions +=1
                return self
        else:
            self.previous_transitions = 0
            return self
        
class State_PaveNavigateLeft(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.widening_threshold = 1000
        self.narrowest_point = self.widening_threshold
        self.post_narrow_counter = 0
        self.post_narrow_counter_trigger = 17
        self.detection_start_time = 3
        self.has_widened = False
        self.finished_widening = False
        self.has_passed_time_debug = False
    def get_state_name(self) -> str:
        return "PaveNavigateLeft"
    def evaluate_transition(self, data) -> AbstractState:
        width = junction_detector.detect_width(data)
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished()
        
        if rospy.get_time() > self.__state_entry_time + self.detection_start_time:
            if not self.has_passed_time_debug:
                print("beginning detection")
                self.has_passed_time_debug = True
            if width > self.widening_threshold:
                self.has_widened = True
            if width < self.narrowest_point and self.has_widened:
                self.finished_widening = True
                self.narrowest_point = width
                self.post_narrow_counter = 0
            if width > self.narrowest_point and self.finished_widening:
                if self.post_narrow_counter < self.post_narrow_counter_trigger:
                    self.post_narrow_counter += 1
                else:
                    return State_JunctionWait()
        
        # if self.positive_function_secondary_count == 0:
        #     if(junction_detector.detect_junction(data)):
        #         self.positive_junction_count += 1
        #         if(self.positive_junction_count >= 15):
        #             self.positive_function_secondary_count += 1
        #     else:
        #         self.positive_junction_count = 0
        #     return self
        # else:
        #     if(junction_detector.detect_junction(data)):
        #         self.positive_junction_count += 1
        #         if(self.positive_junction_count >= 3):
        #             return State_JunctionWait()
        #         else:
        #             return self
        #     else:
        #         self.positive_junction_count = 0
        #        return self
        return self
    
class State_PrePaveNavigateInside(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1
    def get_state_name(self) -> str:
        return "PrePaveNavigateInside"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigateInside()
        else:
            return self
                
class State_PaveNavigateInside(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "PaveNavigateInside"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        else:
            return self
        
class State_JunctionWait(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "JunctionWait"
    def evaluate_transition(self, data) -> AbstractState:
        pos_x, pos_y = junction_detector.detect_truck(data)
        print(f"x: {pos_x}, y: {pos_y}")
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished()
        elif 0 < pos_x < 400 and 0< pos_y < 700:
            return State_PrePaveNavigateInside()
        else:
            return self
        

class State_PreGrassNavigate(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1
    def get_state_name(self) -> str:
        return "PreGrassNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_GrassNavigate()
        else:
            return self

class State_GrassNavigate(AbstractState):
    def __init__(self):
        super().__init__()
        self.previous_transitions = 0
    def get_state_name(self) -> str:
        return "GrassNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        current_surface = surface_detector.poll(data)
        if current_surface == SurfaceDetector.RoadSurface.PAVEMENT:
            if(self.previous_transitions > 5):
                return State_PaveNavigateLeft()
            else:
                self.previous_transitions += 1
                return self
        else:
            self.previous_transitions = 0
            return self

class State_CrosswalkWait(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 0.5
        self.pedestrianState = "waiting"
        self.middle = 650
        self.cross_half_width = 200
    def get_state_name(self) -> str:
        return "CrosswalkWait"
    def evaluate_transition(self, data) -> AbstractState:
        # Pedestrian ~400 left and ~900 right
        pedestrain_location = pedestrian_detector.detectPants(data)
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        # if the pedestrian is in the middle of the crosswalk
        elif(self.middle - self.cross_half_width < pedestrain_location[0] < self.middle +self.cross_half_width):
            self.pedestrianState = "crossing"
            return self
        elif(self.pedestrianState == "crossing" and (self.middle - self.cross_half_width > pedestrain_location[0] or pedestrain_location[0] > self.middle +self.cross_half_width)):
            return State_CrosswalkTraverse()
        else:
            return self
    
class State_CrosswalkTraverse(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1
    def get_state_name(self) -> str:
        return "CrosswalkTraverse"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigate()
        else:
            return self
        
class State_StopTurnLeft(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1.0
    def get_state_name(self) -> str:
        return "StopTurnLeft"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_StopTurnRight()
        else:
            return self

class State_StopTurnRight(AbstractState):
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 4.0
    def get_state_name(self) -> str:
        return "StopTurnRight"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigate()
        else:
            return self
   
class State_StopSend(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "StopSend"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        else:
            return self
    
class State_Finished(AbstractState):
    def __init__(self, called_by = "Unknown"):
        super().__init__()
        print("Called by: " + called_by)
        self.stateEntryAction()
    def stateEntryAction(self):
        start_publisher = rospy.Publisher("/license_plate", String, queue_size=1)
        time.sleep(1)
        competition_end_message = str("Team4,multi21,-1,AA00")
        start_publisher.publish(competition_end_message)
        print("Ended Timer")
    def get_state_name(self) -> str:
        return "Finished"
    def evaluate_transition(self, data) -> AbstractState:
        return self
    
class State_Error(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "Error"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        else:
            return self

### State Machine ###
class StateMachine:
    def __init__(self):
        self.current_state = State_StartupTurnAndDrive()
        self.pub = rospy.Publisher('/competition_controller/state', String, queue_size=1)
        self.plate_stop_sub = rospy.Subscriber("/plate_reader/requested_driving_state", String, self.plate_stop_callback)
        self.stop_states_lookup = {"StopTurnLeft": State_StopTurnLeft, "StopTurnRight": State_StopTurnRight}
    
    def update_state(self, data):
        self.current_state = self.current_state.evaluate_transition(data)
        self.pub.publish(self.current_state.get_state_name())

    def get_current_state(self):
        return self.current_state

    def get_current_state_name(self):
        return self.current_state.get_state_name()
    
    def plate_stop_callback(self, data):
        if(data.data in ["StopTurnRight", "StopTurnLeft"]):
            # If the command is to stop and we already are in a stop state, do nothing
            if self.current_state.get_state_name() in ["StopTurnRight", "StopTurnLeft"]:
                pass
            # If the command is to stop and we are not in a stop state, 
            # save the current state (so we can go back to it) and go to the stop state
            else:
                self.prev_state_before_stop = self.current_state
                # go to the state we are commanded to go to
                self.current_state = self.stop_states_lookup[data.data]()
                self.pub.publish(self.current_state.get_state_name())
        elif(data.data == "Drive"):
            # If the command is to drive and we are in a stop state, go back to the previous state
            if self.current_state.get_state_name() in ["StopTurnRight", "StopTurnLeft"]:
                self.current_state = self.prev_state_before_stop
                self.pub.publish(self.current_state.get_state_name())
        else:
            print("Error: Invalid driving state requested")
    



### Node ###

def state_machine_node(initiator_msg):
    global surface_detector
    print(rospy.get_time())
    surface_detector = SurfaceDetector()
    state_machine = StateMachine()
    sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, state_machine.update_state)
    rospy.spin()

if __name__ == '__main__':
    # What an ugly piece of boilerplate. This is why people hate you, Python.
    # Wait until we get the start message before proceeding
    rospy.init_node('state_machine_node')
    start_subscriber = rospy.Subscriber("/competition_controller/start", String, state_machine_node)
    print("----------------------------------------")
    print("Waiting for competition controller to start")
    print("----------------------------------------")
    rospy.spin()
    # state_machine_node()