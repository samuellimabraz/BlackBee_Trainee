#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16, UInt8
from cv_bridge import CvBridge, CvBridgeError

import cv2
import cvzone

from modules.detection.DetectorModule import Detector
from modules.detection.FaceDetectorModule import FaceDetector
from modules.detection.HandModule import MyHandDetector

from modules.utils import cropImage


class GestureDetector(Detector, MyHandDetector, FaceDetector):
    """
    Performs face detection with the lightweight FaceDetector model, 
    creating a detection area close to the face for gesture detection
    """
    def __init__(
        self,
        node,
        mode,
        maxHands,
        handDetectionCon,
        minHandTrackCon,
        face_model_selection,
        faceDetectionCon,
    ):
        Detector.__init__(self, node_name=node)
        MyHandDetector.__init__(self, mode, maxHands, handDetectionCon, minHandTrackCon)
        FaceDetector.__init__(self, face_model_selection, faceDetectionCon)

        # Variables and Publisher for ROS topics
        self.gesture_event = Int16()
        self.gesture_event_pub = rospy.Publisher("hand_event", Int16, queue_size=1)

    def detect(self, img):
        img = super().detect(img)

        # Fit image, Bebop: (480, 856, 3), Webcam: (480, 640, 3)
        # img = cropImage(img, 0.0, 0.252)
        img = cv2.resize(img, (640, 480))
        imgaux = img.copy()

        # Face detection, return the depth dist, and moviment event
        img, bbox, self.face_depth, self.face_event = self.detect_face(
            img=img, focus_length=640, dead_zone=100, draw=False
        )

        # Hand detection, recognize gestures in an area close to the face
        self.gesture_event = self.gestureRecognizer(
            imgDetect=imgaux, imgDraw=img, face_bbox=bbox
        )

        cv2.imshow("Hand Detect", img)
        self.gesture_event_pub.publish(self.gesture_event)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("Janela fechada")
            return


def main():
    penguin = GestureDetector(
        node="hand_gesture_node",
        mode=False,
        maxHands=1,
        handDetectionCon=0.7,
        minHandTrackCon=0.7,
        face_model_selection=0,
        faceDetectionCon=0.75,
    )
    penguin.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


