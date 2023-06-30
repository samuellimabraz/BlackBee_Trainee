#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16, UInt8
from cv_bridge import CvBridge, CvBridgeError

import cv2
import cvzone

from HandModule import MyHandDetector
from FaceModule import FaceDetector

import numpy as np

from utils import *


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
        self.gesture_event_pub = rospy.Publisher("hand_event", Int16, queue_size=1)

        self.face_depth = Int16(0)

        self.depth_pub = rospy.Publisher("face_depth", Int16, queue_size=1)

        self.face_event = UInt8()
        self.face_event_pub = rospy.Publisher("face_event", UInt8, queue_size=1)

    def detect(self, img):
        if self.image_topic == "bebop":
            try:
                img = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)

        # Bebop: (480, 856, 3), Webcam: (480, 640, 3)
        #img = cropImage(img, 0.0, 0.252)
        img = cv2.resize(img, (640, 480))

        # Face detection, return the depth dist, and movient event
        img, bbox, self.face_depth, self.face_event = self.detect_face(img, False)

        self.depth_pub.publish(self.face_depth)
        self.face_event_pub.publish(self.face_event)

        #Hand detection, recognize gestures in an area close to the face
        if bbox:
            #Area for detection
            x, y, w, h = (
                abs(bbox[0] - 220),
                abs(bbox[1]),
                abs(bbox[2] + 80),
                abs(bbox[3] + 70),
            )
            imgDetect = img[y : (y + h), x : (x + w)]

            #cvzone.cornerRect(img, [x, y, w, h])

            self.gesture_event = self.gestureRecognizer(imgDetect, (20, 20))
            self.gesture_event_pub.publish(self.gesture_event)
    

        cv2.imshow("Detect", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("Janela fechada")
            return

    def bebop_run(self):
        rospy.Subscriber("bebop/image_raw/compressed", CompressedImage, self.detect, queue_size=10)
        rospy.spin()

    def webcam_run(self):
        cap = cv2.VideoCapture(0)

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if not ret:
                continue

            self.detect(frame)

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        rospy.init_node("detector_node", anonymous=True)

        self.image_topic = rospy.get_param("~image_topic", "webcam")
        print(f"Captura: {self.image_topic}")

        if self.image_topic == "bebop":
            self.bebop_run()
        elif self.image_topic == "webcam":
            self.webcam_run()


def main():
    opa = Detector(
        mode=False,
        maxHands=1,
        minHandDetectionCon=0.7,
        minHandTrackCon=0.6,
        maxFaces=1,
        refine_landmarks=True,
        minFaceDetectionCon=0.6,
        minFaceTrackCon=0.6,
        focusLength=640,
    )
    opa.run()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
