from event import Event
import numpy as np
from time import sleep

class StateMachine:
    # states of drone
    STATE_OFF = -1
    STATE_LANDED = 0
    STATE_FLYING = 1

    # static variables to monitor event changes
    previous_event = None
    counter = 0

    def __init__(self, drone, event) -> None:
        self.drone = drone
        self.event = event
        self.current_state = self.STATE_OFF

    def loop(self):
        event = self.event.read()
        print("Evento detectado: ", event, "\n")

        if (self.previous_event is not None) and (self.previous_event == event):
            self.counter += 1
        else:
            self.counter = 0
        
        self.previous_event = event

        if self.counter == 60:
            self.counter = 0
            if (self.current_state == self.STATE_OFF) and (event == Event.EV_NOEVENT):
                self.drone.take_off()
                self.current_state = self.STATE_FLYING
            elif self.current_state == self.STATE_FLYING:
                match event:
                    case Event.EV_FRONT:
                        self.drone.move_horizontal(2.0)
                    case Event.EV_RIGHT:
                        self.drone.move_lateral(0.2)
                    case Event.EV_FLIP:
                        self.drone.front_flip()
                    case Event.EV_LAND:
                        self.drone.land()
                        self.current_state = self.STATE_LANDED
                    
