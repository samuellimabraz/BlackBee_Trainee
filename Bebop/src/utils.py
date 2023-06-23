import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from mediapipe.python.solutions.drawing_utils import DrawingSpec

class MyHandDetector(HandDetector):
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
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
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
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

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
                    landmark_drawing_spec = DrawingSpec(color=(255, 0, 106), thickness=2, circle_radius=2)
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS, landmark_drawing_spec)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (0, 255, 0), 2)
                    # """ cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                    #             2, (255, 0, 255), 2) """
        if draw:
            return allHands, img
        else:
            return allHands

def findArea(lmList, draw=None):
    marks = [11, 23, 25, 26, 24, 12]

    points = np.array(lmList)[marks, :2].astype(int)
    
    if (draw is not None) and (points.size > 0):
        cv2.fillPoly(draw, [points], (0, 255, 0))

    return cv2.contourArea(points)

def estimateDistance(lmList):
    marks = [11, 12, 23, 24]
    cz = []
    for i in range(len(marks)):
        cz.append(lmList[marks[i]][2])

    return np.mean(cz)

def drawRectangleEdges(img, x, y, w, h, r):
    cv2.line(img, (x, y), (x+r, y), (0, 255, 0), 2)
    cv2.line(img, (x, y), (x, y+r), (0, 255, 0), 2)

    cv2.line(img, (x+w, y), (x+w-r, y), (0, 255, 0), 2)
    cv2.line(img, (x+w, y), (x+w, y+r), (0, 255, 0), 2)

    cv2.line(img, (x, y+h), (x, y+h-r), (0, 255, 0), 2)
    cv2.line(img, (x, y+h), (x+r, y+h), (0, 255, 0), 2)

    cv2.line(img, (x+w, y+h), (x+w-r, y+h), (0, 255, 0), 2)
    cv2.line(img, (x+w, y+h), (x+w, y+h-r), (0, 255, 0), 2)