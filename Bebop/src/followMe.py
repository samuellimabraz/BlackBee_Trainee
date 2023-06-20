#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import time
from detector import detectPeople, detectHand

# Variáveis de controle
desired_area = 50000  # Área média desejada para manter a distância padrão
k_p = 0.0005  # Ganho proporcional para o controle linear.x
k_p_yaw = 0.008  # Ganho proporcional para o controle angular.z
k_d_yaw = 0.01  # Ganho derivativo para o controle angular.z
prev_error_yaw = 0  # Erro anterior para o controle derivativo angular.z

def control(img, imgOut):
    global desired_area, k_p, k_p_yaw, k_d_yaw, prev_error_yaw

    area, center = detectPeople(img, imgOut)
    event = detectHand(img, imgOut)

    vel_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=1)
    move = Twist()

    if center is not None:
        # Controle proporcional para a direção linear.x
        error_area = (area - desired_area)
        toleranciaArea = 3500
        if abs(error_area) > toleranciaArea:
            # O movimento será oposto a diferença da área esperada
            # err > 0 : backward; err < 0 : forward
            move.linear.x = -(k_p * error_area)
        
        # Controle PD para a direção angular.z (yaw)
        error_yaw = center[0] - img.shape[1] // 2  # Erro de desvio do centro da câmera
        toleranciaYaw = 0.5
        if abs(error_yaw) > toleranciaYaw:
            derivative_yaw = error_yaw - prev_error_yaw
            prev_error_yaw = error_yaw
            move.angular.z = k_p_yaw * error_yaw + k_d_yaw * derivative_yaw

def callback(img):
    try:
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    imgShow = cv_image.copy()
    control(img=cv_image, imgShow=imgShow)

    cv2.imshow("Image", imgShow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("Image shutdown")
        cv2.destroyAllWindows()

if __name__ == '__main__':

    rospy.init_node('follow_me', anonymous=True)

    takeoff_pub = rospy.Publisher('/bebop/takeoff', Empty, queue_size=10)

    takeoff_pub.publish()
    time.sleep(5)

    image_sub = rospy.Subscribe('/bebop/image_raw', Image, callback)
    
    rospy.spin()