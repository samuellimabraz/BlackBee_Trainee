import cv2
import mediapipe as mp

from cvzone.FaceDetectionModule import FaceDetector
from utils import MyHandDetector, drawRectangleEdges

handsDetector = MyHandDetector(detectionCon=0.8, maxHands=1)
facesDetector = FaceDetector(minDetectionCon=0.8)

def detectHand(img, imgOut):
    """
    Realiza a detecção da mão em uma área próximo ao rosto,
    retornando o evento interpretado pelo gesto identificado
    """

    # Detecta o rosto na imagem
    img, bboxs = facesDetector.findFaces(img)

    event = "None"

    # Se há rosto, cria uma area de reconhecimento de gestos
    if bboxs:
        # Cria área de reconhecimento
        xb, yb, wb, hb = bboxs[0]["bbox"]
        w = abs(wb + 40)
        h = abs(hb + 60)
        x = abs(xb - w - 40)
        y = abs(yb)
        drawRectangleEdges(imgOut, x, y, w, h, 20)

        detect = img[y : (y + h), x : (x + w)]

        # Detecta a mão e identifica o gesto pela posição dos dedos
        hands, detect = handsDetector.findHands(detect)

        if hands:
            hand = hands[0]
            if hand["type"] == "Right":
                # Detecta os dedos levantados ou não
                fingers = handsDetector.fingersUp(hand)

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
                    imgOut, event, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
                )

    return event
