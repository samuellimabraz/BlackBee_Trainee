import sys
sys.path.insert(0, r'D:\Samuel\UNIFEI\Black_Bee\BlackBee-Trainee')

import cv2
import time
import numpy as np

from droneSimulation.Drone import Drone
from event import Event
from state_machine import StateMachine
from droneSimulation.sensors.utils import createTrackBars

hsv_colors = {
    "Yellow": np.array(([15, 210, 90], [30, 255, 255])),
    "Red": np.array(([170, 105, 125], [179, 255, 255])),
    "Green": np.array(([35, 150, 50], [75, 255, 255])),
}


def main():
    drone = Drone()
    camera = drone.sensors["camera"]

    event = Event(camera=camera, trigger=hsv_colors)
    state_machine = StateMachine(drone, event)

    createTrackBars("TrackBars")

    while True:
        start = time.time()

        success = camera.read()

        if not success:
            continue

        # event.read()
        state_machine.loop()

        camera.show()

        res = 0.05 - (time.time() - start)
        if res > 0:
            time.sleep(res)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    drone.sensors["camera"].cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
