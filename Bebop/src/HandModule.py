#!/usr/bin/env python

import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from utils import drawRectangleEdges


class MyHandDetector(HandDetector, FaceDetector):
    fingers_gestures = {
        "TAKEOFF": [0, 1, 0, 0, 0],
        "WAIT": [1, 1, 1, 1, 1],
        "RIGHT": [0, 0, 0, 0, 1],
        "LEFT": [1, 0, 0, 0, 0],
        "FRONT": [0, 0, 0, 0, 0],
        "FLIP": [0, 1, 0, 0, 1],
        "LAND": [0, 1, 1, 0, 0],
    }

    def __init__(
        self,
        mode=False,
        maxHands=2,
        minHandDetectionCon=0.5,
        minTrackCon=0.5,
        minFaceDetectionCon=0.5,
    ):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param minHandDetectionCon: Minimum Detection Confidence Threshold for Hand
        :param minTrackCon: Minimum Tracking Confidence Threshold
        :param minFaceDetectionCon: Minimum Detection Confidence Threshold for Face
        """
        HandDetector.__init__(self, mode, maxHands, minHandDetectionCon, minTrackCon)
        FaceDetector.__init__(self, minFaceDetectionCon)

    def gestureRecognizer(self, img):
        """
        Realiza a detecção da mão em uma área próximo ao rosto,
        retornando o evento interpretado pelo gesto identificado
        """

        # Detecta o rosto na imagem
        img, bboxs = self.findFaces(img)

        event = "None"

        # Se há rosto, cria uma area de reconhecimento de gestos
        if bboxs:
            # Cria área de reconhecimento
            xb, yb, wb, hb = bboxs[0]["bbox"]
            w = abs(wb + 40)
            h = abs(hb + 60)
            x = abs(xb - w - 40)
            y = abs(yb)
            drawRectangleEdges(img, x, y, w, h, 20)

            detect = img[y : (y + h), x : (x + w)]

            # Detecta a mão e identifica o gesto pela posição dos dedos
            hands = self.findHands(detect, False)

            if hands:
                hand = hands[0]

                # Detecta os dedos levantados ou não
                fingers = self.fingersUp(hand)

                # Compara com os gestos declarados e gera o evento
                for ev, gesture in self.fingers_gestures.items():
                    if fingers == gesture:
                        event = ev
                        break

                cv2.putText(
                    img,
                    event,
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    2,
                )

        return event

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    landmark_drawing_spec = DrawingSpec(
                        color=(255, 0, 106), thickness=2, circle_radius=2
                    )
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec,
                    )
        if draw:
            return allHands, img
        else:
            return allHands


def main():
    cap = cv2.VideoCapture(0)

    detector = MyHandDetector(
        mode=False,
        maxHands=1,
        minHandDetectionCon=0.8,
        minTrackCon=0.7,
        minFaceDetectionCon=0.8,
    )

    while True:
        success, frame = cap.read()

        detector.gestureRecognizer(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
