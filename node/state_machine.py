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

from tensorflow.keras import models
from tensorflow.keras import optimizers

from surface_detector import SurfaceDetector

# eww globals

surface_detector = None

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
        self.__target_time_in_state = 3
        self.stateEntryAction()
    def stateEntryAction(self):
        start_publisher = rospy.Publisher("/license_plate", String, queue_size=1)
        time.sleep(1)
        competition_begin_message = str("Team4,multi21,0,AA00")
        start_publisher.publish(competition_begin_message)
        print("Began Timer")
    def get_state_name(self) -> str:
        return "StartupTurnAndDrive"
    def evaluate_transition(self, data) -> AbstractState:
        if(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_Finished()
        else:
            return self
    
class State_PaveNavigate(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "PaveNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        current_surface = surface_detector.poll(data)
        if current_surface == SurfaceDetector.RoadSurface.GRASS:
            return State_GrassNavigate()
        else:
            return self

class State_GrassNavigate(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "GrassNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        current_surface = surface_detector.poll(data)
        if current_surface == SurfaceDetector.RoadSurface.PAVEMENT:
            return State_PaveNavigate()
        else:
            return self

class State_CrosswalkWait(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "CrosswalkWait"
    def evaluate_transition(self, data) -> AbstractState:
        return self
    
class State_CrosswalkTraverse(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "CrosswalkTraverse"
    def evaluateTransition(self, data) -> AbstractState:
        return self

class State_TransToInnerLoop(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self) -> str:
        return "TransToInnerLoop"
    def evaluate_transition(self, data) -> AbstractState:
        return self
   
class State_StopSend(AbstractState):
    def __init__(self):
        super().__init__()
    def get_state_name(self, data) -> str:
        return "StopSend"
    def evaluate_transition(self) -> AbstractState:
        return self
    
class State_Finished(AbstractState):
    def __init__(self):
        super().__init__()
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
        return self

### State Machine ###
class StateMachine:
    def __init__(self):
        self.current_state = State_StartupTurnAndDrive()
        self.pub = rospy.Publisher('/state', String, queue_size=1)
    
    def update_state(self, data):
        self.current_state = self.current_state.evaluate_transition(data)
        self.pub.publish(self.current_state.get_state_name())

    def get_current_state(self):
        return self.current_state

    def get_current_state_name(self):
        return self.current_state.get_state_name()
    



### Node ###

def state_machine_node():
    global surface_detector
    rospy.init_node('topic_subscriber')
    print(rospy.get_time())
    state_machine = StateMachine()
    surface_detector = SurfaceDetector()
    sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, state_machine.update_state)
    rospy.spin()

if __name__ == '__main__':
    # What an ugly piece of boilerplate. This is why people hate you, Python.
    state_machine_node()