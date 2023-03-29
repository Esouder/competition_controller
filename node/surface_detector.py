from tensorflow.keras import models
from tensorflow.keras import optimizers

import rospy
import cv2
from cv_bridge import CvBridge
from enum import Enum
import numpy as np


class SurfaceDetector:
    class RoadSurface(Enum):
        PAVEMENT = 0
        GRASS = 1
    
    def __init__(self):
        model_location = rospy.get_param('~model_location')
        self.model = models.load_model(model_location, compile=False)
        LEARNING_RATE = 1e-4
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE), metrics=['acc'])
        self.bridge = CvBridge()
        self.current_surface = self.RoadSurface.PAVEMENT # always start on pavement

    def poll(self, data, debug=False):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if debug:
            frame_out = frame

        small_frame = cv2.resize(frame, (int(frame.shape[1]*0.025),int(frame.shape[0]*0.05)))
        cropped_small_frame = small_frame[18:-1]
        grass_prob = self.model.predict(np.array([cropped_small_frame]))[0][0]
        current_surface = self.RoadSurface.GRASS if grass_prob > 0.5 else self.RoadSurface.PAVEMENT

        if debug:
            cv2.putText(frame_out, f"it's {'GRASS' if current_surface == self.RoadSurface.GRASS else 'PAVEMENT'} ({grass_prob*100 if grass_prob > 0.5 else (1-grass_prob)*100}%)", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
            return current_surface, frame_out
        else:
            return current_surface