#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from utils import drawRectangleEdges


class MyHandDetector(HandDetector):
    def __init__(self) -> None:
        super().__init__(maxHands=1, detectionCon=0.8, minTrackCon=0.7)
        self.facesDetector = FaceDetector(minDetectionCon=0.8)

        self.hand_img_pub = rospy.Publisher("hand_image", Image, queue_size=1)
        self.bridge = CvBridge()
        # Obtenha o valor do parâmetro "image_topic"
        # bebop/image_raw ou webcam_image
        self.image_topic = rospy.get_param("~image_topic", "webcam_image")

    def detect(self, img):
        """
        Realiza a detecção da mão em uma área próximo ao rosto,
        retornando o evento interpretado pelo gesto identificado
        """
        # Leitura do tópico de imagem
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Detecta o rosto na imagem
        cv_img, bboxs = self.facesDetector.findFaces(cv_img)

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
            hands, detect = self.findHands(detect)

            if hands:
                hand = hands[0]
                if hand["type"] == "Right":
                    # Detecta os dedos levantados ou não
                    fingers = self.fingersUp(hand)

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

        rospy.loginfo(f"Event: {event}")

        img_msg = CvBridge().cv2_to_imgmsg(cv_img, encoding="bgr8")
        self.hand_img_pub.publish(img_msg)

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
                    cv2.rectangle(
                        img,
                        (bbox[0] - 20, bbox[1] - 20),
                        (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                        (0, 255, 0),
                        2,
                    )
        if draw:
            return allHands, img
        else:
            return allHands

    def run(self):
        rospy.init_node("hand_detector_node")

        rospy.Subscriber(self.image_topic, Image, self.detect)

        rospy.spin()


if __name__ == "__main__":
    cafe = MyHandDetector()
    cafe.run()
