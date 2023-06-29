#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import cv2
import cvzone

from HandModule import MyHandDetector
from PoseModule import MyPoseDetector


import numpy as np

from utils import *


class Detector(MyHandDetector, MyPoseDetector):
    def __init__(
        self,
        mode,
        maxHands,
        minHandDetectionCon,
        minHandTrackCon,
        minFaceDetectionCon,
        minPoseDetectionCon,
        minPoseTrackCon,
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

        self.bridge = CvBridge()

    def detect(self, img):
        if self.image_topic == "bebop":
            try:
                img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)

        # Bebop: (480, 856, 3)
        img = cropImage(img, 0.15, 0.3)

        self.findArea(img)
        self.gestureRecognizer(img)

        #out = cvzone.stackImages([self.hand_detect_img, self.pose_detect_img], 2, 1)
        
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
        minFaceDetectionCon=0.72,
        minPoseDetectionCon=0.8,
        minPoseTrackCon=0.8
    )
    opa.run()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass