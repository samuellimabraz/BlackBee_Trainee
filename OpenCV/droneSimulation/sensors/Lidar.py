from .Sensor import Sensor
from .Ambiente import Ambiente
from math import sin ,radians
from decimal import Decimal

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
                Ambiente.limits[self.eixo[0]][
                    self.eixo[1]
                ]  # Limite postivo ou negativo do eixo,
                - (self.position[self.eixo[0]])  # Coordenada do sensor no eixo
                - self.dif_ref  # Diferença de referncial a coordenada do drone
            )
        else:
            distance = abs(Ambiente.limits["z"][0] * Decimal(sin(radians(angle))))

        return Decimal(distance)