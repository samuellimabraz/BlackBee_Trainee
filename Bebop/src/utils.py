#!/usr/bin/env python

import cv2
import numpy as np


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
