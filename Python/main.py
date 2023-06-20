import Drone
import Ambiente


drone = Drone.Drone()

mission = [
    [drone.take_off, Ambiente.altura_max / 2.0],
    [drone.parked],
    [drone.move_horizontal, 2.0],
    [drone.move_lateral, 0.2],
    [drone.front_flip, 360],
]

drone.set_mission(mission=mission)


for i in range(len(mission)):
    drone.next_action()
    Ambiente.simula_ambiente(drone, 3)