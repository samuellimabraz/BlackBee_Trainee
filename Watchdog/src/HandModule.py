#!/usr/bin/env python

import cv2
import numpy as np
import time

from cvzone.HandTrackingModule import HandDetector


class MyHandDetector(HandDetector):
    fingers_gestures = [
        [0, 1, 0, 0, 0],  # "TAKEOFF"
        [0, 0, 0, 0, 1],  # "RIGHT"
        [1, 0, 0, 0, 0],  # "LEFT"
        [0, 0, 0, 0, 0],  # "FRONT"
        [0, 1, 0, 0, 1],  # "FLIP"
        [0, 1, 1, 0, 0],  # "LAND"
    ]

    def __init__(
        self,
        mode=False,
        maxHands=2,
        minDetectionCon=0.5,
        minTrackCon=0.5,
    ):
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
        self.limit_wave_time = 1.0

    def gestureRecognizer(self, img, textPoint=(20, 20)):
        """
        Performs hand detection and
        return the event interpreted by the identified gesture
        :param img: Image for detection
        :param textPoint: point (x, y) for writing the event
        """

        event = -1

        # Detect hand and identify the gesture by finger positions
        hands = self.findHands(img, False)

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

                if hand_x_center < self.hand_previous_pos_x - 24:
                    self.left_moviment += 1
                elif hand_x_center > self.hand_previous_pos_x + 24:
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
                self.start_wave_time = time.time()
                self.recognized_lateral_moviment = False
                self.right_moviment = self.left_moviment = 0

        cv2.putText(img, str(event), textPoint, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return event


def main():
    cap = cv2.VideoCapture(0)

    detector = MyHandDetector(
        mode=False,
        maxHands=1,
        minDetectionCon=0.8,
        minTrackCon=0.7,
    )

    while True:
        _, frame = cap.read()

        detector.gestureRecognizer(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
