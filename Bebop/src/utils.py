#!/usr/bin/env pythonss

import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
from mediapipe.python.solutions.drawing_utils import DrawingSpec


def cropImage(img, hcrop, wcrop):
    height = (
        int(img.shape[0] * hcrop / 2),
        int(img.shape[0] - (img.shape[0] * hcrop / 2)),
    )
    width = (
        int(img.shape[1] * wcrop / 2),
        int(img.shape[1] - (img.shape[1] * wcrop / 2)),
    )

    return img[height[0] : height[1], width[0] : width[1]]


# Calcula a area fromada pela sequencia de pontos
def findArea(lmList, marks, draw=None):
    points = np.array(lmList)[marks, :2].astype(int)

    if (draw is not None) and (points.size > 0):
        cv2.fillPoly(draw, [points], (0, 255, 0))

    return cv2.contourArea(points)


# Calcula a média das distâncias cz dos landmarks
def estimateDistance(lmList, marks):
    cz = []
    for i in range(len(marks)):
        cz.append(lmList[marks[i]][2])

    return np.mean(cz)


def drawRectangleEdges(img, x, y, w, h, r):
    cv2.line(img, (x, y), (x + r, y), (0, 255, 0), 2)
    cv2.line(img, (x, y), (x, y + r), (0, 255, 0), 2)

    cv2.line(img, (x + w, y), (x + w - r, y), (0, 255, 0), 2)
    cv2.line(img, (x + w, y), (x + w, y + r), (0, 255, 0), 2)

    cv2.line(img, (x, y + h), (x, y + h - r), (0, 255, 0), 2)
    cv2.line(img, (x, y + h), (x + r, y + h), (0, 255, 0), 2)

    cv2.line(img, (x + w, y + h), (x + w - r, y + h), (0, 255, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w, y + h - r), (0, 255, 0), 2)


def stackImages(scale, imgArray):
    """
    By: Murtaza Hassan
    Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
    Website: https://www.computervision.zone/
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale
                    )
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x],
                    (imgArray[0].shape[1], imgArray[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

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
