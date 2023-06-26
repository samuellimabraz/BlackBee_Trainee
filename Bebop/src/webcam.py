#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def webcam_publisher():
    rospy.init_node("webcam_publisher", anonymous=True)

    pub = rospy.Publisher("webcam_image", Image, queue_size=10)
    # rate = rospy.Rate(1000)

    cap = cv2.VideoCapture(0)

    print("Webcam inicada")
    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("Janela fechada")
            break

        # Converte a imagem para o formato sensor_msgs/Image
        img_msg = CvBridge().cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(img_msg)

        # rate.sleep()

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass
