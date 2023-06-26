import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


import numpy as np

from utils import stackImages


class ImageDisplay:
    def __init__(self):
        # Inicialize as variáveis de instância para armazenar as mensagens recebidas
        self.pose_image = np.zeros((480, 640, 3))
        self.hand_image = np.zeros((480, 640, 3))

class ImageCombiner:
    def __init__(self):
        # Inicialize as variáveis de instância para armazenar as mensagens recebidas
        self.pose_image = None
        self.hand_image = None

    def pose_callback(self, img):
        # Atualize a imagem da pose
        try:
            self.pose_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
            rospy.loginfo(f"Pose image ok {self.pose_image.shape}")
        except CvBridgeError as e:
            self.pose_image = np.zeros((480, 640, 3))
            print(e)

        # Chame a função de processamento
        self.process_images()

    def hand_callback(self, img):
        # Atualize a imagem da mão
        try:
            self.hand_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
            rospy.loginfo(f"Hand image ok {self.hand_image.shape}")
        except CvBridgeError as e:
            self.hand_image = np.zeros((480, 640, 3))
            print(e)

        # Chame a função de processamento
        self.process_images()

    def process_images(self):

        final_img = stackImages(0.7, [self.pose_image, self.hand_image])
        cv2.imshow("Final Detect", final_img)

    def run(self):
        rospy.init_node("display_node")

        cv2.namedWindow("Final Detect")

        # Inscreva-se nos tópicos de imagem
        rospy.Subscriber("pose_image", Image, self.pose_callback)
        rospy.Subscriber("hand_image", Image, self.hand_callback)

        rospy.spin()


if __name__ == "__main__":
    wow = ImageDisplay()
    wow.run()

