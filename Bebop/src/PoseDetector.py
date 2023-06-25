import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import mediapipe as mp
from utils import findArea, estimateDistance


class PoseDetector:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8, min_tracking_confidence=0.8
        )

        self.pose_img_pub = rospy.Publisher("pose_image", Image, queue_size=1)

    def detect(self, img):
        """
        Realiza a detecção da pessoa,
        retornando a aréa formada pelos landmarks do tronco,
        além de seu ponto central
        """
        # Leitura do tópico de imagem no formato cv2
        try:
            img = CvBridge().imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Para melhorar o desempenho, opcional
        img.flags.writeable = False
        # Processa a imagem com o modelo do mediapipe
        results = self.pose.process(img)

        area, center = 0, 0

        if results.pose_landmarks:
            img.flags.writeable = True

            # Desenho dos pose landmarks no frame
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # Lista dos landmarks, com seus valores cx, cy e cz
            lmList = []
            for lm in results.pose_landmarks.landmark:
                h, w, c = img.shape
                cx, cy, cz = abs(int(lm.x * w)), abs(int(lm.y * h)), (lm.z * 1000)
                lmList.append(([cx, cy, cz]))

            # Cálculo e exibição da área do contorno formado pelos pontos
            # Ombro, quadril e joelho: [11, 23, 25, 26, 24, 12]
            area = findArea(lmList, [11, 23, 24, 12], img)

            # Calculo médio da distância dos landmarks até a camera
            dist = estimateDistance(lmList, [11, 12, 23, 24])

            cv2.putText(
                img,
                f"Area: {area:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"Dist: {dist:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        img_msg = CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
        self.pose_img_pub.publish(img_msg)

        return area, center


if __name__ == "__main__":
    rospy.init_node("pose_detector")

    penguin = PoseDetector()
    rospy.Subscriber("webcam_image", Image, penguin.detect)
    # rospy.Subscriber("/bebop/image_raw", Image, penguin.pose_detector)

    rospy.spin()
