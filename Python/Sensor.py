from decimal import Decimal
from math import sin, radians
import numpy as np

from Ambiente import limits


class Sensor:
    def __init__(self, pos: dict, angulo) -> None:
        if pos != None:
            self.position = pos
        else:
            self.position = {"x": 0, "y": 0, "z": 0}

        self.angle = angulo

    def set_position(self, x=0, y=0, z=0, pos: dict = None):
        if pos:
            self.position = pos
        else:
            self.position["x"] = x
            self.position["y"] = y
            self.position["z"] = z


class Lidar(Sensor):
    def __init__(self, pos: dict, angulo, eixo, dif: Decimal) -> None:
        super().__init__(pos=pos, angulo=angulo)
        self.eixo = eixo
        self.dif_ref = dif

    def get_data(self):
        angle = self.angle
        """ # reduzir o ângulo
        angle = self.angle % 360.0
        # forçar a ser o resto positivo, para que 0 <= angulo < 360
        angle = (angle + 360) % 360
        # forçar na classe de resíduo de valor absoluto mínimo, para que -180 < angulo <= 180
        if angle > 180:
            angle -= 360
        self.angle = angle """

        if angle in [0, 180, 360]:
            distance = abs(
                limits[self.eixo[0]][
                    self.eixo[1]
                ]  # Limite postivo ou negativo do eixo,
                - (self.position[self.eixo[0]])  # Coordenada do sensor no eixo
                - self.dif_ref  # Diferença de referncial a coordenada do drone
            )
        else:
            distance = abs(limits["z"][0] * Decimal(sin(radians(angle))))

        return Decimal(distance)


class Camera(Sensor):
    def __init__(self, pos: dict, angulo, size) -> None:
        super().__init__(pos, angulo)
        self.size = size
        self.img = np.array([[0 for i in range(size)] for j in range(size)])

    def get_data(self):
        self.img = np.random.uniform(low=0, high=9, size=(self.size, self.size))
        return self.img
