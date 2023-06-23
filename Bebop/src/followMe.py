#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

from HandDetector import detectHand
from PoseDetector import detectPeople

# Parâmetros do controle PID
Kp_linear = 0.005  # Constante proporcional do controle PID para linear.x
Kd_linear = 0.01  # Constante derivativa do controle PID para linear.x
Kp_angular = 0.01  # Constante proporcional do controle PID para angular.z
Kd_angular = 0.02  # Constante derivativa do controle PID para angular.z

target_area = 30000  # Área desejada para a pessoa
area_tolerance = 4000  # Tolerância para considerar movimento linear
target_center = 320  # Coordenada x do centro desejado
center_tolerance = 50  # Tolerância para considerar a pessoa centrada

prev_error_area = 0
prev_error_yaw = 0

cmd_pub = rospy.Publisher("/bebop/cmd_vel", Twist, queue_size=1)
vel = Twist()

def control(img, imgOut):
    area, center = detectPeople(img, imgOut)
    event = detectHand(img, imgOut)

    if center:
        # Controle PID para movimento linear (linear.x)
        error_area = target_area - area

        control_linear = Kp_linear * error_area + Kd_linear * (
            error_area - prev_error_area
        )

        # Controle PID para Yaw (angular.z)

        error_yaw = center[0] - img.shape[1] // 2
        
        control_angular = Kp_angular * error_yaw + Kd_angular * (
            error_yaw - prev_error_yaw
        )

        # Ajustar os valores de controle dentro dos limites dos tópicos
        control_linear = max(min(control_linear, 1.0), -1.0)
        control_angular = max(min(control_angular, 1.0), -1.0)

        # Atualizar os erros anteriores
        prev_error_area = error_area
        prev_error_yaw = error_yaw

        if abs(error_area) > area_tolerance:
            # err > 0 : forward; err < 0 : backward
            vel.linear.x = control_linear

        if abs(error_yaw) > center_tolerance:
            vel.angular.z = control_angular

    cmd_pub.publish(vel)


def callback(img):
    try:
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    imgShow = cv_image.copy()

    control(cv_image, imgShow)

    cv2.imshow("Image", imgShow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        rospy.signal_shutdown("Image shutdown")
        cv2.destroyAllWindows()


def main():
    rospy.init_node("follow_me", anonymous=True)

    takeoff_pub = rospy.Publisher("/bebop/takeoff", Empty, queue_size=10)

    rospy.sleep(2)
    takeoff_pub.publish()
    rospy.sleep(3)

    image_sub = rospy.Subscribe("/bebop/image_raw", Image, callback)

    rospy.spin()


if __name__ == "__main__":
    main()
