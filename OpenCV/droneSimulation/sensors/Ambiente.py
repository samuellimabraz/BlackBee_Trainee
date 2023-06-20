from decimal import Decimal
from time import sleep, time
from random import uniform
import numpy as np

class Ambiente:
    altura_max = 3.0
    largura_max = 1.0
    comprimento_max = 23.0
    limits = {
        "x": [Decimal(largura_max / 2.0), Decimal(-largura_max / 2.0)],
        "y": [Decimal(altura_max / 2.0), Decimal(-altura_max / 2.0)],
        "z": [Decimal(comprimento_max / 2.0), Decimal(-comprimento_max / 2.0)],
    }

def simula_ambiente(drone, tempo_estabiliza):
    print("Simulação de ambiente...")
    start_time = time()
    while True:
        mean = np.mean(drone.sensors["camera"].get_data())
        sensor_1 = drone.sensors["lidar_lateral_1"].get_data()
        sensor_2 = drone.sensors["lidar_lateral_2"].get_data()
        sensor_3 = drone.sensors["lidar_baixo"].get_data()

        variacao = Decimal(uniform(-0.03, 0.03))
        drone.position["x"] += variacao
        drone.position["y"] += variacao
        drone.position["z"] += variacao

        sensor_1 = abs(sensor_1 - drone.sensors["lidar_lateral_1"].get_data())
        sensor_2 = abs(sensor_2 - drone.sensors["lidar_lateral_2"].get_data())
        sensor_3 = abs(sensor_3 - drone.sensors["lidar_baixo"].get_data())

        erro = Decimal(0.025)

        if (sensor_1 > erro) or (sensor_2 > erro) or (sensor_3 > erro) or (mean > 7.0):
            print("Interferência detectada")
            drone.corrige(variacao)
            start_time = time()

        current_time = time()
        elapsed_time = current_time - start_time

        minutos = int(elapsed_time // 60)
        segundos = int(elapsed_time % 60)

        print(f"Tempo decorrido: {minutos:02d}:{segundos:02d}", end="\r")
        sleep(1)

        if elapsed_time >= tempo_estabiliza:
            break
    print(f"Drone estabilizado por {tempo_estabiliza}s\n")
