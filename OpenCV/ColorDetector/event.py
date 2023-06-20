import numpy as np
from droneSimulation.sensors.utils import getTrackBars
import cv2
from time import sleep


class Event:
    EV_TAKEOFF = -1
    EV_NOEVENT = 0
    EV_RIGHT = 1
    EV_FRONT = 2
    EV_FLIP = 3
    EV_LAND = 4

    def __init__(self, camera, trigger: dict) -> None:
        self.current_event = self.EV_TAKEOFF
        self.camera = camera
        self.trigger = trigger

    def read(self):
        # detect the test mask
        lower, upper = getTrackBars("TrackBars")
        teste, _ = self.camera.detectHSVColor((lower, upper))
        
        img = self.camera.frame
        teste = cv2.bitwise_and(img, img, mask=teste)
        cv2.imshow("teste", teste)

        # detect the all colors
        detect = self.camera.detectColors(self.trigger)

        # self.camera.show()

        # returns the event of the color if only one color is identified
        if len(detect) == 1:
            match detect[0]:
                case "Yellow":
                    self.current_event = self.EV_FRONT
                case "Green":
                    self.current_event = self.EV_RIGHT
                case "Red":
                    self.current_event = self.EV_LAND
        else:
            self.current_event = self.EV_NOEVENT

        return self.current_event
