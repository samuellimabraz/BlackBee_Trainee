import cv2
import numpy as np
#from cvzone.HandTrackingModule import HandDetector
from .Sensor import Sensor
from .utils import stackImages, cropImage, uniformImageList

threshold = 10000


class Camera(Sensor):
    def __init__(self, channel, crop) -> None:
        super().__init__()

        self.cap = cv2.VideoCapture(channel)
        self.frame = np.zeros((480, 640, 3))
        self.framesShow = None
        self.crop = crop

    def read(self):
        success, self.frame = self.cap.read()
        self.framesShow = [self.frame]
        if success:
            self.frame = cropImage(self.frame, self.crop)
            # self.frame = equalize_brightness(self.frame)

        return success

    def show(self):
        # Create a single window with all frames
        framesStack = uniformImageList(self.framesShow, 3)
        out = stackImages(0.7, (framesStack))
        cv2.imshow("Stacked Frames", out)

    def detectHSVColor(self, color, draw=False):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # Generate a binary mask using the specified color range in HSV
        mask = cv2.inRange(hsv, color[0], color[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))

        if draw:
            img = self.frame.copy()
            img = cv2.bitwise_and(img, img, mask=mask)
            self.framesShow.append(img)

        # Calculates the total pixels other than black
        pix = np.sum(mask[:, :] > 0)
        # Set a flag to indicate if the color is detected based on the pixel count
        flag = True if pix >= threshold else False

        return mask, flag

    def detectColors(self, colors, drawContour=True, drawMasks=True):
        frameContours = self.frame.copy()
        masks = []
        detect = []

        # Detect each HSV color from the color dict,
        # obtain its mask. If the color is identified, its contour is drawn.
        for name, value in colors.items():
            if drawMasks:
                mask, flag = self.detectHSVColor(value, True)
            else:
                mask, flag = self.detectHSVColor(value)
            
            if flag:
                detect.append(name)
                if drawContour:
                    self.getCountours(img=mask, out=frameContours, msg=name)
                    cv2.imshow("mask", frameContours)


        self.framesShow.insert(0, frameContours)

        # Returns the list with detected colors
        return detect

    def getCountours(self, img, out, msg):
        # Apply a treatment to the image to enhance contours
        blur = cv2.GaussianBlur(img, (7, 7), 1)
        edges = cv2.Canny(blur, 50, 50)
        edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        _, img = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draws the contour and a rectangle based on the largest contour found
        if len(contours) > 0:
            # Find the largest contour (based on area)
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(out, [max_contour], -1, (255, 0, 0), 2)

            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Write the message in the center of the rectangle
            cv2.putText(
                out,
                msg,
                (x + (w // 2) - 20, y + (h // 2)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                3,
            )
    
    def detectHand(self):
        pass
        #detector = HandDetector(detectionCon=0.8, maxHands=1)
