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

TEAM_ID = "Team4'"
TEAM_PW = "OvenPanicTruths"

########## Globals ##########
# (ew)

# The surface detector cannot be initialized until after the node has started
# as it references a ros environment variable with system path information
# which is only accessable after the node has started, hence why it is 'None'.
surface_detector = None

# Initialize a Pedestrian Detector object used by the state machine
pedestrian_detector = PedestrianDetector()

# Initialize a Junction Detector object used by the state machine
junction_detector = JunctionDetector()

# Global variable to hold the time the competition begins at. Filled during
# the initialization of the first state
competition_start_time = None

# Constants
MAX_COMPETITION_TIME = 3.75*60 # Automatically send end message after this time

######### Abstract State #########

class AbstractState(ABC):
    '''
        Abstract Base State Classs
        
        Provides a common base type for driving states and 
    '''
    def __init__(self):
        '''
            Abstract Base State initialization
        
            logs to stdout the name of the state on state entry
        '''
        print(f"Entered new state: {self.get_state_name()}")

    @abstractmethod
    def get_state_name(self) -> str:
        '''
            Get the name for the state

            This state name is published to a topic by the state machine to
            encode state information to other nodes.
        '''
        pass
    @abstractmethod
    def evaluate_transition(self, data) -> AbstractState:
        '''
            Determine the next state

            Determines and returns the next appropriate state for the navigation
            system based on the allowable state transitions and information 
            avaliable about the vehicle
        '''
        pass

########## Actual States ##########
class State_StartupStraight(AbstractState):
    '''
        Startup Straight

        First state on initialization. 
        
        Preprogrammed drive straight for 1.2s
    '''
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 2.2 # 1.2s + 1 second of sleep time
        self.stateEntryAction()
    def stateEntryAction(self):
        '''
            Actions run on first-time initialization of the state
        '''

        global competition_start_time

        # Init the publisher used to start the timer
        start_publisher = rospy.Publisher("/license_plate", String, queue_size=1)

        # Give the publisher time to start and register
        time.sleep(1)

        # Publish the message to start the timer and record the time it's sent at
        competition_start_time = rospy.get_time()
        competition_begin_message = str(f"{TEAM_ID},{TEAM_PW},0,AA00")
        start_publisher.publish(competition_begin_message)
        print("Began Timer") # log this action to stdout

    def get_state_name(self) -> str:
        return "StartupStraight"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Go to StartupTurnAndDrive if we pass the target time in state
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_StartupTurnAndDrive()
        
        # Otherwise stay in this state
        else:
            return self
    
class State_StartupTurnAndDrive(AbstractState):
    '''
        Startup Turn And Drive

        Pre-programmed drive and turn onto the path to align correctly
        
    '''
    def __init__(self):
        super().__init__()
        time.sleep(1)
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1.0
    def get_state_name(self) -> str:
        return "StartupTurnAndDrive"
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Go to PaveNavigate if we pass the target time in state
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigate()
        
        # Otherwise stay in this state
        else:
            return self
    
class State_PaveNavigate(AbstractState):
    '''
        Pave Navigate

        Main navigation state for driving on pavement on the first/largest
        portion of the outside loop
    '''

    def __init__(self):
        super().__init__()
        self.previous_transitions = 0
    def get_state_name(self) -> str:
        return "PaveNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Go to Crosswalk Wait if we see a crosswalk
        if(pedestrian_detector.detectCrosswalk(data)):
            return State_CrosswalkWait()
        
        # Determine the current surface
        current_surface = surface_detector.poll(data)

        # Transition to Pre Grass Navigate if we detect we're on grass for at
        # least 10 frames
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
    '''
        Pave Navigate Left

        Navigation state used to turn into the inner loop, after exiting the
        grass. It tracks the left side of the track.
        
    '''
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()

        ### Internal variables used to track information for junction detection

        # Threshold to determine if we've passed the 'wide point' of the track
        self.widening_threshold = 1000

        # Narrowest point we've encountered
        self.narrowest_point = self.widening_threshold

        # Number of frames since we've seen the narrowest point
        self.post_narrow_counter = 0

        # Number of frames neccesary beyond the narrowest point to proceed
        self.post_narrow_counter_trigger = 17

        # Seconds after state entry to begin looking for the junction
        self.detection_start_time = 3

        # If we have passed the widening point
        self.has_widened = False

        # If we have completed the widening phase
        self.finished_widening = False

        # flag so we only print that detection has begun once
        self.has_passed_time_debug = False
        
    def get_state_name(self) -> str:
        return "PaveNavigateLeft"
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished()
        
        # Determine the width of the pavement
        width = junction_detector.detect_width(data)

        # If we've passed the detection start time, start the algorithm to 
        # determine when to stop at the junction:

        #   print 'beginning detection'  the first time we pass the entry time
        #   
        #   The first time the width exceeds the widening threshold, mark that 
        #   we have begun the 'widening stage' (self.has_widened)
        #
        #   Once we have begun that stage, check if the curent width is less
        #   than the smallest width encountered so far
        #
        #       If it is smaller, record it as the new narrowest width, mark
        #       that we have completed the widening portion 
        #       (self.finished_widening), and reset the post narrow counter
        #       
        #       If it's larger, count frames until we reach 
        #       self.post_narrow_counter_trigger frames of consecutive widening
        #       and transition to the Junction Wait state.
        #
        # The purpose of this is to detect when we start going around the corner
        # and the track appears to widen as we enter the junction, and then stop
        # once we have passed the narrowest point. The requirement for several 
        # consecutive frames of widening after passing the narrowest point 
        # reduces the likleyhood of false positives and better aligns us for the
        # next state.
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
        
        # Default return condition
        return self
    
class State_PrePaveNavigateInside(AbstractState):
    ''' 
        Pre Pave Navigate Inside
        
        A pre-programmed drive and turn that takes place before the execution of 
        the pave navigate inside state, in order to better align the robot
    '''
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1.0 

    def get_state_name(self) -> str:
        return "PrePaveNavigateInside"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Otherwise go to PaveNavigateInside once we reach the target time in state
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigateInside()
        else:
            return self
                
class State_PaveNavigateInside(AbstractState):
    '''
        Pave Navigate Inside
        
        Pavement Navigation state for the inner loop of the track
    '''
    def __init__(self):
        super().__init__()

    def get_state_name(self) -> str:
        return "PaveNavigateInside"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        # This is the final state of the competition sequence, so there's no
        # other exit conditions
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        else:
            return self
        
class State_JunctionWait(AbstractState):
    ''' 
        Junction Wait
        
        State to wait at the junction between outer & inner loops until the 
        truck passes
    '''

    def __init__(self):
        super().__init__()

    def get_state_name(self) -> str:
        return "JunctionWait"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''
        
        # Determine the position of the truck on screen
        pos_x, pos_y = junction_detector.detect_truck(data)

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished()
        
        # Move to the PrePaveNavigateInside state when the truck is safely past 
        # the robot
        elif 0 < pos_x < 400 and 0< pos_y < 700:
            return State_PrePaveNavigateInside()
        else:
            return self
        

class State_PreGrassNavigate(AbstractState):
    '''
        Pre Grass Navigate
        
        Pre-programmed maneuver in order to better align the robot for grass
        driving
    '''
    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1
    
    def get_state_name(self) -> str:
        return "PreGrassNavigate"
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Go to Grass Navigate once passed the target time in-state
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
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Determine the current surface, and move to the Pavement Navigate Left
        # State if we detect more than a certain number of pavement frames in a 
        # row
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
    '''
        Crosswalk Wait
        
        Wait at the crosswalk for the pedestrian to cross
    '''
    def __init__(self):
        super().__init__()

        # Pedestrian detector state
        #   - 'waiting' for pedestrian to cross, or
        #   - pedestrian currently 'crossing'
        #initialized to 'waiting'
        self.pedestrianState = "waiting"

        # Middle of the crosswalk (px)
        self.middle = 650
        
        # Width of half the crosswalk
        self.cross_half_width = 200

    def get_state_name(self) -> str:
        return "CrosswalkWait"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Get the location of the pedestrian('s pants)
        # Pedestrian ~400 left and ~900 right
        pedestrain_location = pedestrian_detector.detectPants(data)

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        # If the pedestrian is in the crosswalk, go to the 
        # 'crossing' state
        elif(self.middle - self.cross_half_width < pedestrain_location[0] < self.middle +self.cross_half_width):
            self.pedestrianState = "crossing"
            return self
        
        # If the pedestrian was previously in the crosswalk and now isn't, move 
        # to the CrosswalkTraverse state
        elif(self.pedestrianState == "crossing" and (self.middle - self.cross_half_width > pedestrain_location[0] or pedestrain_location[0] > self.middle +self.cross_half_width)):
            return State_CrosswalkTraverse()
        
        else:
            return self
    
class State_CrosswalkTraverse(AbstractState):
    '''
        Crosswalk Traverse
        
        Pre-programmed quick movement across the crosswalk
    '''

    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1
    def get_state_name(self) -> str:
        return "CrosswalkTraverse"
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Otherwise go to Pave Navigate once we exceed the target time in state
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigate()
        
        else:
            return self
        
class State_StopTurnLeft(AbstractState):
    '''
        Stop Turn Left

        Stop and turn left as commanded by the plate detector

        Transitions into and out of this state are usually commanded by the 
        plate detector node.
    '''

    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 1.0
    
    def get_state_name(self) -> str:
        return "StopTurnLeft"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Go to Stop Turn right if passed the max time in state
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_StopTurnRight()
        
        # Otherwise stay in state
        else:
            return self

class State_StopTurnRight(AbstractState):
    '''
        Stop Turn Left

        Stop and turn left as commanded by the plate detector

        Transitions into and out of this state are usually commanded by the 
        plate detector node.
    '''

    def __init__(self):
        super().__init__()
        self.__state_entry_time = rospy.get_time()
        self.__target_time_in_state = 4.0

    def get_state_name(self) -> str:
        return "StopTurnRight"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        
        # Go to Pave Navigate if we exceed the time in state
        elif(rospy.get_time() >=  self.__target_time_in_state + self.__state_entry_time):
            return State_PaveNavigate()
        
        # Otherwise stay in this state
        else:
            return self
   
class State_StopSend(AbstractState):
    '''
        Stop Send

        Transitions into and out of this state are usually commanded by the 
        plate detector node.
    '''

    def __init__(self):
        super().__init__()

    def get_state_name(self) -> str:
        return "StopSend"
    
    def evaluate_transition(self, data) -> AbstractState:
        '''Determine the next state'''

        # Go to Finished if we pass the max competition time
        if(rospy.get_time() > competition_start_time + MAX_COMPETITION_TIME):
            return State_Finished(self.get_state_name())
        else:
            return self
    
class State_Finished(AbstractState):
    '''
        Finished
        
        Stops moving when the time is up. Publishes the message to end the timer
    '''

    def __init__(self, called_by = "Unknown"):
        super().__init__()
        print("Finished State Called by: " + called_by)
        self.stateEntryAction()
    
    def stateEntryAction(self):
        # Initialize the publisher, wait for it to start, and then send a message
        # to end the timer, logging that action to stdout
        start_publisher = rospy.Publisher("/license_plate", String, queue_size=1)
        time.sleep(1)
        competition_end_message = str("Team4,multi21,-1,AA00")
        start_publisher.publish(competition_end_message)
        print("Ended Timer")
    
    def get_state_name(self) -> str:
        return "Finished"
    
    def evaluate_transition(self, data) -> AbstractState:
        return self

########## State Machine ##########
class StateMachine:
    '''
        State Machine class that manages the state of the robot navigation
    '''
    def __init__(self):
        ''' 
            Initialize the state of the robot navigation to be in Startup Straight
        '''
        
        # Start in Startup Straight
        self.current_state = State_StartupStraight()
        
        # Init the state publisher that will communicate the navigation state to
        # other nodes
        self.pub = rospy.Publisher('/competition_controller/state', String, queue_size=1)

        # Init the plate detector-commanded stop subscriber to listen for when 
        # the plate detector needs us to stop and get a better view of a plate
        self.plate_stop_sub = rospy.Subscriber("/plate_reader/requested_driving_state", String, self.plate_stop_callback)

        self.stop_states_lookup = {"StopTurnLeft": State_StopTurnLeft, "StopTurnRight": State_StopTurnRight}
    
    def update_state(self, data):
        ''' 
            Update and publish the state by calling the current state's 
            evaluate_transition() method
        '''

        # Determine the next state based on the evaluate_transition method of
        # the current state
        self.current_state = self.current_state.evaluate_transition(data)

        # Publish the new state
        self.pub.publish(self.current_state.get_state_name())

    def get_current_state(self):
        ''' Get the current state of the state machine'''
        return self.current_state

    def get_current_state_name(self):
        ''' Get the name of the current state of the state machine'''
        return self.current_state.get_state_name()
    
    def plate_stop_callback(self, data):
        ''' 
            Callback for when we receive a message over the plate_stop_sub 
            subscriber
        '''
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
        elif(data.data == "Finished"):
            # Sent if we have read all the plates
            self.current_state = State_Finished("Plate Reader")
        else:
            print("Error: Invalid driving state requested")
    



########## Node #############

def state_machine_node(initiator_msg):
    ''' 
        Start up the state machine node and don't let it end, calling 
        'update state' on each new frame
    '''
    global surface_detector
    print(f"Initialized at time: {rospy.get_time()}")
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