
class Sensor:
    def __init__(self, pos: dict=None, angulo=0) -> None:
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
