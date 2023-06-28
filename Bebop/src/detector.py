#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import cv2

from HandModule import MyHandDetector
from PoseModule import MyPoseDetector

import numpy as np

from utils import *


class Detector(MyHandDetector, MyPoseDetector):
    def __init__(
        self,
        mode=False,
        maxHands=2,
        minHandDetectionCon=0.5,
        minHandTrackCon=0.5,
        minFaceDetectionCon=0.5,
        minPoseDetectionCon=0.5,
        minPoseTrackCon=0.5,
    ):
        MyHandDetector.__init__(
            self,
            mode,
            maxHands,
            minHandDetectionCon,
            minHandTrackCon,
            minFaceDetectionCon,
        )
        MyPoseDetector.__init__(self, mode, True, minPoseDetectionCon, minPoseTrackCon)

        rospy.init_node("detector_node", anonymous=True)
        self.image_topic = rospy.get_param("~image_topic", "webcam_image")

        self.hand_detect_img = np.zeros((480, 640, 3))
        self.pose_detect_img = np.zeros((480, 640, 3))

        self.bridge = CvBridge()

    def detect(self, img):
        if self.image_topic == "bebop":
            try:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)
        else:
            cv_img = img.copy()

        self.gestureRecognizer(cv_img)
        self.findArea(cv_img)

        out = stackImages(1, [self.hand_detect_img, self.pose_detect_img])
        cv2.imshow("Image", out)
        cv2.waitKey(1)

    def bebop_run(self):
        rospy.Subscriber("bebop/image_raw/compressed", CompressedImage, self.detect)
        rospy.spin()

    def webcam_run(self):
        cap = cv2.VideoCapture(0)

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if not ret:
                continue

            self.detect(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                rospy.signal_shutdown("Janela fechada")
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        print(f"Captura: {self.image_topic}")

        if self.image_topic == "bebop":
            self.bebop_run()
        elif self.image_topic == "webcam":
            self.webcam_run()


if __name__ == "__main__":
    opa = Detector()
    opa.run()
