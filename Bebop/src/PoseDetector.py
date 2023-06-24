import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import mediapipe as mp
from utils import findArea, estimateDistance

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

pose_pub = rospy.Publisher("pose_detect")

def pose_detector(img):
    """
    Realiza a detecção da pessoa,
    retornando a aréa formada pelos landmarks do tronco,
    além de seu ponto central
    """

    # Leitura do tópico de imagem
    try:
        img = CvBridge().imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    imgOut = img.copy()

    img.flags.writeable = False
    # Processa a imagem com o modelo do mediapipe
    results = pose.process(img)

    area, center = 0, 0

    if results.pose_landmarks:
        # Para melhorar o desempenho, opcional
        img.flags.writeable = True

        # Desenho dos pose landmarks no frame
        mp_drawing.draw_landmarks(
            imgOut,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        # Lista dos landmarks, com seus valores cx, cy e cz
        lmList = []
        for lm in results.pose_landmarks.landmark:
            h, w, c = img.shape
            cx, cy, cz = abs(int(lm.x * w)), abs(int(lm.y * h)), (lm.z * 1000)
            lmList.append(([cx, cy, cz]))

        # Cálculo e exibição da área do contorno formado pelos pontos
        # Ombro, quadril e joelho: [11, 23, 25, 26, 24, 12]
        area = findArea(lmList, [11, 23, 24, 12], imgOut)

        # Calculo médio da distância dos landmarks até a camera
        dist = estimateDistance(lmList, [11, 12, 23, 24])

        cv2.putText(
            imgOut,
            f"Area: {area:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            imgOut,
            f"Dist: {dist:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

    return area, center


if __name__ == "__main__":
    rospy.init_node("pose_detector")
    rospy.Subscriber("/bebop/image_raw", Image, pose_detector)
    rospy.spin()


# ------- Código utilizando a classe do cvzone ----------#

# from cvzone.PoseModule import PoseDetector

# PeopleDetector = PoseDetector(detectionCon=0.8)

# def detectPeople(img, imgOut):
#     """
#     Realiza a detecção pelo PoseDetector e retorna a área da bounding box
#     assim como seu ponto central
#     """

#     imgOut = PeopleDetector.findPose(img, True)
#     _, bboxInfo = PeopleDetector.findPosition(img, bboxWithHands=False)

#     area = 0
#     center = 0
#     if bboxInfo:
#         center = bboxInfo["center"]

#         # Calcula a area da bounding box
#         width = bboxInfo["bbox"][2] - bboxInfo["bbox"][0]
#         height = bboxInfo["bbox"][3] - bboxInfo["bbox"][1]
#         area = abs(width * height)

#         print(f"c: {center}, imgc: {img.shape[1]//2}")

#         cv2.circle(imgOut, center, 5, (255, 0, 255), cv2.FILLED)
#         cv2.putText(
#             imgOut, str(area), (200, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
#         )

#     return area, center
