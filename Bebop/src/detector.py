#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import cv2

from cvzone.FaceDetectionModule import FaceDetector

import mediapipe as mp

import numpy as np

from utils import *

class Detector:
    def __init__(self) -> None:
        self.handDetector = MyHandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.7)
        self.faceDetector = FaceDetector(minDetectionCon=0.8)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8, min_tracking_confidence=0.8
        )

        self.hand_detect_img = np.zeros((480, 640, 3))
        self.pose_detect_img = np.zeros((480, 640, 3))

        self.image_topic = rospy.get_param("~image_topic", "webcam_image")

    
    def hand_detect(self, img):
        """
        Realiza a detecção da mão em uma área próximo ao rosto,
        retornando o evento interpretado pelo gesto identificado
        """
        if self.image_topic == "bebop/image_raw":
            try:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)
        else:
            cv_img = img.copy()

        # Detecta o rosto na imagem
        img, bboxs = self.faceDetector.findFaces(cv_img)

        event = "None"

        # Se há rosto, cria uma area de reconhecimento de gestos
        if bboxs:
            # Cria área de reconhecimento
            xb, yb, wb, hb = bboxs[0]["bbox"]
            w = abs(wb + 40)
            h = abs(hb + 60)
            x = abs(xb - w - 40)
            y = abs(yb)
            drawRectangleEdges(cv_img, x, y, w, h, 20)

            detect = cv_img[y : (y + h), x : (x + w)]

            # Detecta a mão e identifica o gesto pela posição dos dedos
            hands, detect = self.handDetector.findHands(detect)

            if hands:
                hand = hands[0]
                if hand["type"] == "Right":
                    # Detecta os dedos levantados ou não
                    fingers = self.handDetector.fingersUp(hand)

                    # Cria os eventos para cada gesto
                    if fingers == [0, 1, 0, 0, 0]:
                        event = "UP"
                    elif fingers == [0, 1, 1, 0, 0]:
                        event = "DOWN"
                    elif fingers == [1, 1, 1, 1, 1]:
                        event = "WAIT"
                    elif fingers == [1, 0, 0, 0, 0]:
                        event = "LEFT"
                    elif fingers == [0, 0, 0, 0, 1]:
                        event = "RIGHT"
                    elif fingers == [0, 0, 0, 0, 0]:
                        event = "FRONT"
                    elif fingers == [0, 1, 0, 0, 1]:
                        event = "FLIP"

                    cv2.putText(
                        cv_img,
                        event,
                        (x, y),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 255),
                        2,
                    )

        #rospy.loginfo(f"Event: {event}")

        self.hand_detect_img = cv_img
    
    def people_detect(self, img):

        if self.image_topic == "bebop/image_raw":
            try:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)
        else:
            cv_img = img.copy()

        cv_img = cropImage(cv_img, 0.0, 0.47)
        cv_img = np.ascontiguousarray(cv_img)

        # Para melhorar o desempenho, opcional
        cv_img.flags.writeable = False
        # Processa a imagem com o modelo do mediapipe
        results = self.pose.process(cv_img)

        area, center, dist = 0, 0, 0

        if results.pose_landmarks:
            cv_img.flags.writeable = True

            # Desenho dos pose landmarks no frame
            self.mp_drawing.draw_landmarks(
                cv_img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # Lista dos landmarks, com seus valores cx, cy e cz
            lmList = []
            for lm in results.pose_landmarks.landmark:
                h, w, _ = cv_img.shape
                cx, cy, cz = abs(int(lm.x * w)), abs(int(lm.y * h)), (lm.z * 1000)
                lmList.append(([cx, cy, cz]))

            # Cálculo e exibição da área do contorno formado pelos pontos
            # Ombro, quadril e joelho: [11, 23, 25, 26, 24, 12]
            area = findArea(lmList, [11, 23, 24, 12], cv_img)

            # Calculo médio da distância dos landmarks até a camera
            dist = estimateDistance(lmList, [11, 12, 23, 24])

            cv2.putText(
                cv_img,
                f"Area: {area:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                cv_img,
                f"Dist: {dist:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        #rospy.loginfo(f"Area: {area}, dist: {dist}")

        self.pose_detect_img = cv_img


    def run(self):
        rospy.init_node("detector_node", anonymous=True)

        cap = cv2.VideoCapture(0)

        print("Webcam inicada")

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if not ret:
                continue
            
            self.hand_detect(frame)
            self.people_detect(frame)

            out = stackImages(1, [self.hand_detect_img, self.pose_detect_img])
            cv2.imshow("Image", out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                rospy.signal_shutdown("Janela fechada")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    opa = Detector()
    opa.run()