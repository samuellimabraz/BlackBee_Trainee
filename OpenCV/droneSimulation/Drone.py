from time import sleep
from decimal import Decimal
import numpy as np

from .sensors import *


class Drone:
    def __init__(self, mission=None):
        self.position = {"x": Decimal(0), "y": Decimal(-1.5), "z": Decimal(0)}
        self.angle = 0
        self.sensors = {
            "lidar_lateral_1": Lidar(
                self.position, self.angle, ("x", 0), Decimal(0.15)
            ),
            "lidar_lateral_2": Lidar(
                self.position, self.angle, ("x", -1), Decimal(-0.15)
            ),
            "lidar_baixo": Lidar(self.position, self.angle, ("y", -1), Decimal(0.0)),
            "camera": Camera(0, 0.25),
        }

    def set_sensor(self, name, sensor: Sensor):
        self.sensors.update({name: sensor})

    def log_position(self):
        print(f"Drone position: ", end="")
        for eixo, value in self.position.items():
            print(f"{eixo} = {value:.4f}, ", end="")
        print(f"\u03b8 = {self.angle}°\n")

    def log_data_sensors(self):
        for id, sensor in self.sensors.items():
            print(f"{id}: {sensor.get_data()}")
        print()

    def log_data_sensor(self, lidar):
        print(
            f"{lidar}: val = {self.sensors[lidar].get_data():.4f}, \u03b8 = {self.sensors[lidar].angle}°"
        )

    def take_off(self):
        self.log_position()
        print("Take off")
        print("Drone subindo...")

        dist = Decimal(Ambiente.limits["y"][0])
        err = Decimal(0.003)
        step_max = Decimal(0.5)
        target = Decimal(dist - err)

        while self.sensors["lidar_baixo"].get_data() < target:
            self.log_data_sensor("lidar_baixo")
            data = self.sensors["lidar_baixo"].get_data()
            step = Decimal(step_max * (1 - (data / dist)))
            sleep(abs(float(step + Decimal(0.18))))
            self.position["y"] += step

        self.log_data_sensor("lidar_baixo")

        print("Drone atingiu a altura correta\n")

    def parked(self):
        self.log_position()
        print("Drone parado...")
        mean = np.mean(self.sensors["camera"].get_data())

        while not (mean > 3.0 and mean < 4.0):
            print("Identificando sinal...")
            sleep(0.6)
            mean = np.mean(self.sensors["camera"].get_data())

        print("Sinal identificado, iniciando o próximo estado")

    def land(self):
        self.log_position()
        print("Land")

        dist = Decimal(Ambiente.limits["y"][1])
        err = Decimal(0.007)
        target = Decimal(self.sensors["lidar_baixo"].get_data() + dist + err)
        step_max = Decimal(0.5)
        step_min = Decimal(0.05)
        step = Decimal(0)

        print(f"Target: {target}")

        while self.sensors["lidar_baixo"].get_data() > (target):
            self.log_data_sensor("lidar_baixo")

            data = self.sensors["lidar_baixo"].get_data()

            step = (Decimal(step_max * (data / dist))) if step > step_min else step_min
            sleep(abs(float(step + Decimal(0.18))))
            self.position["y"] -= step

        self.log_data_sensor("lidar_baixo")

    def move_vertical(self, dist):
        pass

    def move_lateral(self, dist):
        self.log_position()
        print(f"Move lateral {dist}m")

        if dist < 0:
            sin = Decimal(-1)
            print("Drone indo para esquerda...\n")
        else:
            sin = Decimal(1)
            print("Drone indo para direita...\n")

        dist = Decimal(dist)
        err = Decimal(0.002) * sin
        step_max = Decimal(0.05) * sin
        step_min = Decimal(0.006) * sin
        step = Decimal(0)

        target_1 = (
            Decimal(self.sensors["lidar_lateral_1"].get_data() - dist - err) * sin
        )
        target_2 = (
            Decimal(self.sensors["lidar_lateral_2"].get_data() + dist - err) * sin
        )

        while (
            self.sensors["lidar_lateral_1"].get_data() * sin > target_1
            and self.sensors["lidar_lateral_2"].get_data() * sin < target_2
        ):
            self.log_data_sensor("lidar_lateral_1")
            self.log_data_sensor("lidar_lateral_2")
            print()
            data = self.sensors["lidar_lateral_1"].get_data() * sin
            step = (
                (Decimal(step_max * ((data - target_1) / dist)) * sin)
                if step > step_min
                else step_min
            )
            sleep(abs(float(step + Decimal(0.5))))
            self.position["x"] += step

        print(f"Drone moveu {dist:.4f}m lateralmente\n")

    def move_horizontal(self, dist):
        self.log_position()
        print(f"Move horizontal {dist:.4f}m")

        if dist < 0:
            sin = Decimal(-1)
            print("Drone indo para tras...")
        else:
            sin = Decimal(1)
            print("Drone indo para frente...")

        dist = Decimal(dist)
        err = Decimal(0.005) * sin
        bias = Decimal(0.04) * sin
        step_max = Decimal(0.55) * sin
        step_min = Decimal(0.04) * sin
        step = Decimal(0)

        target = Decimal(self.position["z"] + dist - err) * sin

        while self.position["z"] * sin < target:
            step = (
                (Decimal(step_max * ((target - self.position["z"] * sin) / dist)) * sin)
                if step > step_min
                else step_min
            )

            sleep(abs(float(step + Decimal(0.15))))
            self.position["z"] += step
            print(f"Avanço de {self.position['z']:.4f}")

        print(f"Drone moveu {dist:.4f}m na horizontal\n")

    def front_flip(self, ang=360):
        print("Flip")

        print("Centralizando drone:\n")
        self.move_lateral(self.position["x"] * -1)
        self.move_horizontal(self.position["z"] * -1)

        self.log_position()

        while self.angle < ang:
            self.log_data_sensor("lidar_baixo")
            sleep(0.5)
            self.angle += 10
            self.sensors["lidar_baixo"].angle += 10

        self.log_data_sensor("lidar_baixo")
        print("Drone fez um flip frontal")

    def corrige(self, variacao):
        if self.state == "flying":
            print("Corrigindo posição do drone...")

            self.position["x"] -= variacao
            self.position["y"] -= variacao
            self.position["z"] -= variacao

            sleep(1.5)

            print("Posição do drone corrigida\n")
        else:
            print("Drone precisa decolar primeiro")
