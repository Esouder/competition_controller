from tensorflow.keras import models
from tensorflow.keras import optimizers

import rospy
import cv2
from cv_bridge import CvBridge
from enum import Enum
import numpy as np


class SurfaceDetector:
    '''
        Surface Detector

        Detects if the current surface is pavement or grass
    '''

    class RoadSurface(Enum):
        PAVEMENT = 0
        GRASS = 1
    
    def __init__(self):
        '''
            Init the Surface Detector. NOTE: must be called by an INITIALIZED 
            ros node
        '''

        model_location = rospy.get_param('~model_location')
        self.model = models.load_model(model_location, compile=False)
        LEARNING_RATE = 1e-4
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE), metrics=['acc'])
        self.bridge = CvBridge()
        self.current_surface = self.RoadSurface.PAVEMENT # always start on pavement

    def poll(self, data):
        ''' 
            Run the surface detector on a given frame
        '''
        # Convert to a cv2 friendly format
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Resize and crop the frame
        small_frame = cv2.resize(frame, (int(frame.shape[1]*0.025),int(frame.shape[0]*0.05)))
        cropped_small_frame = small_frame[18:-1]

        # Determine the probability of grass and output the most likely surface
        grass_prob = self.model(np.array([cropped_small_frame]))[0][0]
        current_surface = self.RoadSurface.GRASS if grass_prob > 0.5 else self.RoadSurface.PAVEMENT

        return current_surface