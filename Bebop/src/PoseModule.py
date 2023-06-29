import cv2
from cvzone.PoseModule import PoseDetector

import numpy as np


class MyPoseDetector(PoseDetector):
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        super().__init__(mode, smooth, detectionCon, trackCon)

    def findArea(self, img, draw=False):
        """
        Realiza a detecção da pessoa e calcula a area de seu tronco
        """

        # Processa a imagem e encontra os landmarks
        self.findPose(img, False)
        
        # Gera lista dos landmarks com suas coordenadas e a bounding box
        lmList, bboxInfo = self.findPosition(img, draw=False, bboxWithHands=False)

        area =  0
        if lmList:
            # Cálculo e exibição da área do contorno formado pelos pontos
            # Ombro, quadril e joelho: [11, 23, 25, 26, 24, 12]
            marks = [11, 23, 24, 12]

            points = np.array(lmList)[marks, 1:3].astype(int)

            if draw:
                cv2.fillPoly(img, [points], (0, 255, 0))

            area = cv2.contourArea(points)

            #center = bboxInfo["center"]
            # cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            
            cv2.putText(
                img,
                f"Area: {area:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        return area


def main():
    cap = cv2.VideoCapture(0)
    detector = MyPoseDetector(detectionCon=0.7, trackCon=0.7)

    while True:
        success, frame = cap.read()

        detector.findArea(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# # Calcula a média das distâncias cz dos landmarks
# def estimateDistance(lmList, marks):
#     cz = []
#     for i in range(len(marks)):
#         cz.append(lmList[marks[i]][2])

#     return np.mean(cz)
