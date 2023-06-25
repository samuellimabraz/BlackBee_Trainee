import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageCombiner:
    def __init__(self):
        # Inicialize as variáveis de instância para armazenar as mensagens recebidas
        self.pose_image = None
        self.hand_image = None

        # Inscreva-se nos tópicos de imagem
        rospy.Subscriber('pose_image', Image, self.pose_callback)
        rospy.Subscriber('hand_image', Image, self.hand_callback)

    def pose_callback(self, img):
        # Atualize a imagem da pose
        try:
            self.pose_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Chame a função de processamento
        self.process_images()

    def hand_callback(self, img):
        # Atualize a imagem da mão
        try:
            self.hand_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Chame a função de processamento
        self.process_images()

    def process_images(self):
        # Verifique se ambas as imagens foram recebidas
        if self.pose_image is not None and self.hand_image is not None:
            final_img = cv2.hconcat([self.pose_image, self.hand_image])
            cv2.imshow("Final Detect", final_img)

if __name__ == '__main__':
    rospy.init_node('image_combiner')
    wow = ImageCombiner()
    rospy.spin()
