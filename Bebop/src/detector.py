import cv2
from cvzone.PoseModule import PoseDetector
from cvzone.FaceDetectionModule import FaceDetector
from utils import MyHandDetector, drawRectangleEdges

def detectPeople(img, imgOut):
    """
    Realiza a detecção pelo PoseDetector e retorna a área da bounding box
    assim como seu ponto central
    """
    detector = PoseDetector()
    img = detector.findPose(img)
    _, bboxInfo = detector.findPosition(img, bboxWithHands=False)

    area, center = None
    if bboxInfo:
        center = bboxInfo["center"]
        
        # Calcula a area da bounding box
        width = bboxInfo["bbox"][2] - bboxInfo["bbox"][0]
        height = bboxInfo["bbox"][3] - bboxInfo["bbox"][1]
        area = abs(width * height)

        cv2.circle(imgOut, center, 5, (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOut, str(area), (200 , 200), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)

    return area, center

def detectHand(img, imgOut):
    """
    Realiza a detecção da mão em uma área próximo ao rosto,
    retornando o evento interpretado pelo gesto identificado
    """
    handsDetector = MyHandDetector(detectionCon=0.8, maxHands=1)
    facesDetector = FaceDetector(minDetectionCon=0.8)

    # Detecta o rosto na imagem
    faces, bboxs = facesDetector.findFaces(img)

    event = "None"

    # Se há rosto, cria uma area de reconhecimento de gestos
    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        # faceCenter = bboxs[0]["center"]

        # Cria área de reconhecimento
        xb, yb, wb, hb = bboxs[0]["bbox"]
        
        w = abs(wb + 40)
        h = abs(hb + 60)
        x = abs(xb - w - 40)
        y = abs(yb)
        drawRectangleEdges(imgOut, x, y, w, h, 20)
        
        detect = img[y:(y+h), x:(x+w)]
        
        center = bboxs[0]["center"]
        # cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)

        # Detecta a mão e identifica o gesto pela posição dos dedos
        hands, detect = handsDetector.findHands(detect)

        if hands:
            hand = hands[0]
            if hand["type"] == "Right":
                # lmList = hand["lmList"] # List of 21 Landmarks points
                # bbox = hand["bbox"] # Bounding Box info x,y,w,h
                # centerPoint = hand["center"]  # center of the hand cx,cy
                # handType = hand["type"] # Hand Type Left or Right

                # Detecta os dedos levantados ou não
                fingers = handsDetector.fingersUp(hand)

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
        
                cv2.putText(imgOut, event, (x , y), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)
    return event     