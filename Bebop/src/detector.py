import cv2
import mediapipe as mp

# from cvzone.PoseModule import PoseDetector
from cvzone.FaceDetectionModule import FaceDetector
from utils import MyHandDetector, drawRectangleEdges, findArea, estimateDistance

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8
)

# PeopleDetector = PoseDetector(detectionCon=0.8)
handsDetector = MyHandDetector(detectionCon=0.8, maxHands=1)
facesDetector = FaceDetector(minDetectionCon=0.8)


def detectPeople(img, imgOut):
    img.flags.writeable = False

    results = pose.process(img)

    area, center = 0, 0
    if results.pose_landmarks:
        img.flags.writeable = True

        # Desenho dos pose landmarks no frame
        mp_drawing.draw_landmarks(
            imgOut,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        lmList = []
        for lm in results.pose_landmarks.landmark:
            h, w, c = img.shape
            cx, cy, cz = abs(int(lm.x * w)), abs(int(lm.y * h)), (lm.z * 1000)
            lmList.append(([cx, cy, cz]))

        # Cálculo e exibição da área do contorno formado pelos pontos
        area = findArea(lmList, imgOut)

        dist = estimateDistance(lmList)

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
        # bboxInfo - "id","bbox","score","center"
        # faceCenter = bboxs[0]["center"]

        # Cria área de reconhecimento
        xb, yb, wb, hb = bboxs[0]["bbox"]

        w = abs(wb + 40)
        h = abs(hb + 60)
        x = abs(xb - w - 40)
        y = abs(yb)
        drawRectangleEdges(imgOut, x, y, w, h, 20)

        detect = img[y : (y + h), x : (x + w)]

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

                cv2.putText(
                    imgOut, event, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
                )
    return event
