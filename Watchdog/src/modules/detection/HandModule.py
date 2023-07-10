#!/usr/bin/env python

import cv2
import numpy as np
import time

import cvzone
from cvzone.HandTrackingModule import HandDetector

import mediapipe as mp


class MyHandDetector(HandDetector):
    """
    Hand Detector using the cvzone, mediapipe HandTracking
    Detect hand and recognize finger gestures and waving gesture
    The detection can be in an area close to the face, with de bbox info
    """

    fingers_gestures = [
        [0, 1, 0, 0, 0],  # "TAKEOFF"
        [0, 0, 0, 0, 1],  # "RIGHT"
        [1, 0, 0, 0, 0],  # "LEFT"
        [0, 0, 0, 0, 0],  # "FRONT"
        [0, 1, 0, 0, 1],  # "FLIP"
        [0, 1, 1, 0, 0],  # "LAND"
    ]

    def __init__(self, mode=False, maxHands=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param minHandDetectionCon: Minimum Detection Confidence Threshold for Hand
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        super().__init__(mode, maxHands, minDetectionCon, minTrackCon)

        self.hand_previous_pos_x = 0
        self.recognized_lateral_moviment = False
        self.right_moviment = self.left_moviment = 0
        self.start_wave_time = time.time()
        self.limit_wave_time = 0.8

    def gestureRecognizer(self, imgDetect, imgDraw, face_bbox: list = []):
        """
        Performs hand detection and
        return the event interpreted by the identified gesture
        :param imgDetect: Image for detection
        :param imgDraw: Image for drawing 
        :param face_bbox: Bbox for generate detection area close to the face
        """

        # Area for detection, if the face info is provided
        if len(face_bbox):
            x, y, w, h = (
                abs(face_bbox[0] - 220),
                abs(face_bbox[1]),
                abs(face_bbox[2] + 80),
                abs(face_bbox[3] + 70),
            )
            imgDetect = imgDetect[y : (y + h), x : (x + w)]
            cvzone.cornerRect(imgDraw, [x, y, w, h])

        event = -1

        # Detect hand and identify the gesture by finger positions
        hands = self.findHands(imgDetect, False)

        if hands:
            hand = hands[0]

            # Detect raised fingers, binary list
            fingers = self.fingersUp(hand)

            # Compare with declared fingers gestures and generate the event
            for id, gesture in enumerate(self.fingers_gestures):
                if fingers == gesture:
                    event = id
                    break

            # Recognize wave motion
            # The movement consists of shifting the center of the hand
            # twice to each side, left and right, with all fingers raised,
            # within a time limit interval
            if fingers == [1, 1, 1, 1, 1]:
                hand_x_center = hand["center"][0]  # cx of the hand

                if hand_x_center < self.hand_previous_pos_x - 20:
                    self.left_moviment += 1
                elif hand_x_center > self.hand_previous_pos_x + 20:
                    self.right_moviment += 1

                if (self.right_moviment == 1 or self.left_moviment == 1) and (
                    not self.recognized_lateral_moviment
                ):
                    self.start_wave_time = time.time()
                    self.recognized_lateral_moviment = True

                if (self.right_moviment > 1 and self.left_moviment > 1) and (
                    time.time() - self.start_wave_time <= self.limit_wave_time
                ):
                    print(f"Aceno")
                    event = 6
                    self.recognized_lateral_moviment = False
                    self.right_moviment = self.left_moviment = 0

                self.hand_previous_pos_x = hand_x_center
            else:
                self.recognized_lateral_moviment = False
                self.right_moviment = self.left_moviment = 0

        cv2.putText(
            imgDraw,
            str(f"Hand:{event}"),
            (imgDraw.shape[1] - 150, imgDraw.shape[0] - 20),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 255, 0),
            2,
        )

        return event


def main():
    cap = cv2.VideoCapture(0)

    detector = MyHandDetector(
        mode=False, maxHands=1, minDetectionCon=0.8, minTrackCon=0.7
    )

    while True:
        success, frame = cap.read()

        detector.gestureRecognizer(frame, frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
