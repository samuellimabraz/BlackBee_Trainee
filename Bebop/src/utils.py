import cv2
import numpy as np


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
