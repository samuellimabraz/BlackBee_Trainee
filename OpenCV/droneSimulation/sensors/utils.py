import cv2
import numpy as np
from math import ceil


def cropImage(img, crop):
    altura = (
        int(img.shape[0] * crop / 2),
        int(img.shape[0] - (img.shape[0] * crop / 2)),
    )
    largura = (
        int(img.shape[1] * crop / 2),
        int(img.shape[1] - (img.shape[1] * crop / 2)),
    )
    img = img[altura[0] : altura[1], largura[0] : largura[1]]
    return img


def equalize_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def uniformImageList(images, max):
    # Create a list uniformly arranged,
    # with a maximum number of frames per row.
    # Fill any unfilled frames with a black image.

    max_elements = max

    imgBlank = np.zeros_like(images[0])

    num_sublists = ceil(len(images) / max_elements)
    sublists = [
        images[i * max_elements : (i + 1) * max_elements] for i in range(num_sublists)
    ]
    for sub in sublists:
        while len(sub) < max_elements:
            sub.append(imgBlank)

    return sublists


def createTrackBars(name):
    cv2.namedWindow(name)
    cv2.resizeWindow(name, 640, 240)
    cv2.createTrackbar("Hue Min", name, 15, 179, lambda a: None)
    cv2.createTrackbar("Hue Max", name, 30, 179, lambda a: None)
    cv2.createTrackbar("Sat Min", name, 143, 255, lambda a: None)
    cv2.createTrackbar("Sat Max", name, 255, 255, lambda a: None)
    cv2.createTrackbar("Val Min", name, 80, 255, lambda a: None)
    cv2.createTrackbar("Val Max", name, 185, 255, lambda a: None)


def getTrackBars(name):
    h_min = cv2.getTrackbarPos("Hue Min", name)
    h_max = cv2.getTrackbarPos("Hue Max", name)
    s_min = cv2.getTrackbarPos("Sat Min", name)
    s_max = cv2.getTrackbarPos("Sat Max", name)
    v_min = cv2.getTrackbarPos("Val Min", name)
    v_max = cv2.getTrackbarPos("Val Max", name)
    return np.asarray([h_min, s_min, v_min]), np.asarray([h_max, s_max, v_max])


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
