#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty, UInt8, Int16
from geometry_msgs.msg import Twist

from threading import Timer

from BebopModule import Bebop

class Controller:
    def __init__(self) -> None:
        self.drone = Bebop()

        self.takeoff_pub = rospy.Publisher("/bebop/takeoff", Empty, queue_size=1)
        self.land_pub = rospy.Publisher("/bebop/land", Empty, queue_size=1)
        self.vel_pub = rospy.Publisher("/bebop/cmd_vel", Twist, queue_size=1)
        self.flip_pub = rospy.Publisher("/bebop/flip", UInt8, queue_size=1)

        self.current_hand_event = -2
        self.timer = None

    def face_follower(self, event):
        if event.data != 0:
            rospy.loginfo(f"Face event: {event.data}")

    def hand_callback(self, event):
        #rospy.loginfo(f"Hand event: {event.data}")
        if self.current_hand_event != event.data:
            # Novo evento detectado, cancela temporizador anterior (se houver)
            if self.timer is not None:
                self.timer.cancel()

            if event.data != -1:
                self.timer = Timer(2.0, self.perform_hand_action)
                self.timer.start()

        self.current_hand_event = event.data

    def perform_hand_action(self):
        # Executa a ação com base no evento atual
        rospy.loginfo(f"Hand event: {self.current_hand_event}")
        if (self.current_hand_event == 0) and (not self.drone.flying):
            self.drone.takeoff()
            rospy.sleep(3)
        elif self.current_hand_event == 1:
            self.drone.move_right()
        elif self.current_hand_event == 2:
            self.drone.move_left()
        elif self.current_hand_event == 3:
            self.drone.move_front()
        elif self.current_hand_event == 4:
            self.drone.flip()
        elif self.current_hand_event == 5:
            self.drone.land()
        elif self.current_hand_event == 6:
            self.drone.wave()

    def start(self):
        rospy.init_node("controller_node", anonymous=True)
        rospy.Subscriber("hand_event", Int16, self.hand_callback)
        rospy.Subscriber("face_event", UInt8, self.face_follower)
        rospy.spin()


def main():
    control = Controller()
    control.start()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

# # Parâmetros do controle PID
# Kp_linear = 0.00004  # Constante proporcional do controle PID para linear.x
# Kd_linear = 0.01  # Constante derivativa do controle PID para linear.x
# Kp_angular = 0.01  # Constante proporcional do controle PID para angular.z
# Kd_angular = 0.02  # Constante derivativa do controle PID para angular.z

# target_area = 30000  # Área desejada para a pessoa
# area_tolerance = 4000  # Tolerância para considerar movimento linear
# target_center = 320  # Coordenada x do centro desejado
# center_tolerance = 50  # Tolerância para considerar a pessoa centrada

# prev_error_area = 0
# prev_error_yaw = 0


# def control(img, imgOut):
#     area, center = detectPeople(img, imgOut)
#     event = detectHand(img, imgOut)

#     if center:
#         # Controle PID para movimento linear (linear.x)
#         error_area = target_area - area

#         control_linear = Kp_linear * error_area + Kd_linear * (
#             error_area - prev_error_area
#         )

#         # Controle PID para Yaw (angular.z)

#         error_yaw = center[0] - img.shape[1] // 2

#         control_angular = Kp_angular * error_yaw + Kd_angular * (
#             error_yaw - prev_error_yaw
#         )

#         # Ajustar os valores de controle dentro dos limites dos tópicos
#         control_linear = max(min(control_linear, 1.0), -1.0)
#         control_angular = max(min(control_angular, 1.0), -1.0)

#         # Atualizar os erros anteriores
#         prev_error_area = error_area
#         prev_error_yaw = error_yaw

#         if abs(error_area) > area_tolerance:
#             # err > 0 : forward; err < 0 : backward
#             vel.linear.x = control_linear

#         if abs(error_yaw) > center_tolerance:
#             vel.angular.z = control_angular

#     cmd_pub.publish(vel)
