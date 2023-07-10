#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Webcam:
    def __init__(self) -> None:
        rospy.init_node('webcam_publisher', anonymous=True)

        # Cria um objeto de publicação para publicar a imagem
        self.image_pub = rospy.Publisher('webcam_image', Image, queue_size=10)

        self.bridge = CvBridge()

    def publisher(self):
        cap = cv2.VideoCapture(0)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if not ret:
                continue
            try:
                # Converte o quadro em um objeto Image do ROS
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.image_pub.publish(ros_image)
            except CvBridgeError as e:
                print(e)


        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        webcam = Webcam()
        webcam.publisher()
    except rospy.ROSInterruptException:
        pass
