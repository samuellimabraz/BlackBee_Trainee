#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16, UInt8
from cv_bridge import CvBridge, CvBridgeError

import cv2
import cvzone

from modules.detection.DetectorModule import Detector
from modules.detection.FaceMeshModule import FaceMeshDetector


class DepthDetector(Detector, FaceMeshDetector):
    """
    Perform face detection with FaceMesh model
    and publish estimated distance 
    and control events to center the face with the image
    """
    def __init__(
        self,
        node_name: str,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        Detector.__init__(self, node_name)
        FaceMeshDetector.__init__(
            self,
            max_num_faces,
            refine_landmarks,
            min_detection_confidence,
            min_tracking_confidence,
        )

        self.face_depth = Int16(0)
        self.depth_pub = rospy.Publisher("face_depth", Int16, queue_size=1)

        self.face_event = UInt8()
        self.face_event_pub = rospy.Publisher("face_event", UInt8, queue_size=1)

    def detect(self, img):
        img = super().detect(img)

        # Fit image, Bebop: (480, 856, 3), Webcam: (480, 640, 3)
        # img = cropImage(img, 0.0, 0.252)
        img = cv2.resize(img, (640, 480))

        # Face detection, return the depth dist, and moviment event
        img, bbox, self.face_depth, self.face_event = self.detect_face(
            img=img, focus_length=640, dead_zone=100, draw=False
        )


        self.depth_pub.publish(self.face_depth)
        if self.face_event:
            self.face_event_pub.publish(self.face_event)

        cv2.imshow("Depth Detect", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("Janela fechada")
            return


def main():
    jacu = DepthDetector(
        node_name="depth_detection_node",
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.75,
    )
    rospy.sleep(7)

    jacu.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
