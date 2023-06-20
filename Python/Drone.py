from time import sleep
from decimal import Decimal
import numpy as np

from Ambiente import altura_max, largura_max, comprimento_max, limits
from Sensor import Sensor, Lidar, Camera


class Drone:
    global limits

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
            "camera": Camera(self.position, self.angle, 5),
        }

        # ["take_off", "move_forward", "flip", "land", "landed"]
        self.state = "landed"
        self.state_machine = mission
        self.current_state_index = -1

    def set_mission(self, mission: list):
        self.state_machine = mission

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

    def take_off(self, dist):
        if self.state == "landed":
            self.log_position()
            print("Take off")
            print("Drone subindo...")

            dist = Decimal(altura_max / 2.0)
            err = Decimal(0.003)
            step_max = Decimal(0.5)
            target = Decimal(dist - err)

            while self.sensors["lidar_baixo"].get_data() < target:
                self.log_data_sensor("lidar_baixo")
                data = self.sensors["lidar_baixo"].get_data()
                step = Decimal(step_max * (1 - (data / dist)))
                sleep(abs(float(step)))
                self.position["y"] += step

            self.log_data_sensor("lidar_baixo")

            self.state = "flying"
            print("Drone atingiu a altura correta\n")
        else:
            print("Drone já está voando\n")

    def parked(self):
        if self.state == "flying":
            self.log_position()
            print("Drone parado...")
            mean = np.mean(self.sensors["camera"].get_data())

            while not (mean > 3.0 and mean < 4.0):
                print("Identificando sinal...")
                sleep(0.6)
                mean = np.mean(self.sensors["camera"].get_data())

            print("Sinal identificado, iniciando o próximo estado")
        else:
            print("Drone já está voando\n")

    def land(self):
        if self.state == "flying":
            self.log_position()
            print("Land")

            err = Decimal(0.007)
            step_max = Decimal(0.5)

            while self.sensors["lidar_baixo"].get_data() > (Decimal(0.0) + err):
                self.log_data_sensor("lidar_baixo")

                data = self.sensors["lidar_baixo"].get_data()

                step = Decimal(step_max * (1 - (data / Decimal(altura_max))))
                sleep(abs(float(step)))
                self.position["y"] -= step

            self.log_data_sensor("lidar_baixo")

            self.state = "landed"
            print("Drone pousou")
        else:
            print("Drone já está pousado")

    def move_vertical(self, dist):
        pass

    def move_lateral(self, dist):
        if self.state == "flying":
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
                sleep(abs(float(step + Decimal(0.2))))
                self.position["x"] += step

            print(f"Drone moveu {dist:.4f}m lateralmente\n")
        else:
            print("Drone precisa decolar primeiro")

    def move_horizontal(self, dist):
        if self.state == "flying":
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
                    (
                        Decimal(step_max * ((target - self.position["z"] * sin) / dist))
                        * sin
                    )
                    if step > step_min
                    else step_min
                )

                sleep(abs(float(step)))
                self.position["z"] += step
                print(f"Avanço de {self.position['z']:.4f}")

            print(f"Drone moveu {dist:.4f}m na horizontal\n")
        else:
            print("Drone precisa decolar primeiro")

    def front_flip(self, ang):
        if self.state == "flying":
            print("Flip")

            print("Centralizando drone:\n")
            self.move_lateral(self.position["x"] * -1)
            self.move_horizontal(self.position["z"] * -1)

            self.log_position()

            while self.angle < ang:
                self.log_data_sensor("lidar_baixo")
                sleep(0.08)
                self.angle += 10
                self.sensors["lidar_baixo"].angle += 10

            self.log_data_sensor("lidar_baixo")
            print("Drone fez um flip frontal")
        else:
            print("Drone precisa decolar primeiro")

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

    def next_action(self):
        if self.current_state_index == len(self.state_machine):
            self.current_state_index = -1
            print("Missão Concluida")

        func_state = self.state_machine[self.current_state_index + 1][0]

        print(f"Estado: {func_state.__name__}")

        if len(self.state_machine[self.current_state_index + 1]) > 1:
            param = self.state_machine[self.current_state_index + 1][1]
            func_state(param)
        else:
            func_state()

        self.log_position()
        self.log_data_sensors()
        self.current_state_index += 1
