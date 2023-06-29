#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16
from cv_bridge import CvBridge, CvBridgeError

import cv2
import cvzone

from HandModule import MyHandDetector
from FaceModule import FaceDetector

import numpy as np

from Utils import *


class Detector(MyHandDetector, FaceDetector):
    def __init__(
        self,
        mode,
        maxHands,
        minHandDetectionCon,
        minHandTrackCon,
        maxFaces,
        refine_landmarks,
        minFaceDetectionCon,
        minFaceTrackCon,
        focusLength,
    ):
        MyHandDetector.__init__(
            self,
            mode,
            maxHands,
            minHandDetectionCon,
            minHandTrackCon,
            minFaceDetectionCon,
        )
        FaceDetector.__init__(
            self,
            maxFaces,
            refine_landmarks,
            minFaceDetectionCon,
            minFaceTrackCon,
            focus_length=focusLength,
        )

        self.bridge = CvBridge()

        self.gesture_event = Int16()
        self.gesture_event_pub = rospy.Publisher("hands_action", Int16, queue_size=1)

        self.face_depth = Int16()
        self.depth_pub = rospy.Publisher("face_depth", Int16, queue_size=1)

    def detect(self, img):
        if self.image_topic == "bebop":
            try:
                img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)

        # Bebop: (480, 856, 3), Webcam: (480, 640, 3)
        img = cropImage(img, 0.15, 0.3)

        img, bbox, dist = self.detect_face(img, False)
        self.face_depth = dist
        self.depth_pub.publish(self.face_depth)

        if dist > 0:
            # Area de reconhecimento para as mãos
            x, y, w, h = bbox[0] - 220, bbox[1], bbox[2] + 80, bbox[3] + 70
            imgDetect = img[y : (y + h), x : (x + w)]

            cvzone.cornerRect(img, [x, y, w, h])

            self.gesture_event = self.gestureRecognizer(imgDetect, (20, 20))
            self.gesture_event_pub.publish(self.gesture_event)

        cv2.imshow("Detect", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("Janela fechada")
            return

    def bebop_run(self):
        rospy.Subscriber("bebop/image_raw", Image, self.detect, queue_size=1)
        rospy.spin()

    def webcam_run(self):
        cap = cv2.VideoCapture(0)

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if not ret:
                continue

            self.detect(frame)

        cap.release()

    def run(self):
        rospy.init_node("detector_node", anonymous=True)

        self.image_topic = rospy.get_param("~image_topic", "webcam_image")
        print(f"Captura: {self.image_topic}")

        if self.image_topic == "bebop":
            self.bebop_run()
        elif self.image_topic == "webcam":
            self.webcam_run()


def main():
    opa = Detector(
        mode=False,
        maxHands=1,
        minHandDetectionCon=0.85,
        minHandTrackCon=0.8,
        maxFaces=1,
        refine_landmarks=True,
        minFaceDetectionCon=0.8,
        minFaceTrackCon=0.8,
        focusLength=640,
    )
    opa.run()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
