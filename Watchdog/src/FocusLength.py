#!/usr/bin/env python

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


detector = FaceMeshDetector(maxFaces=1, minDetectionCon=0.5, minTrackCon=0.6)


def findFocus(img):
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)

        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # real distance between the reference object

        # # Finding the Focal Length
        # d = 100  # real distance to the camera
        # f = (w * d) / W
        # print(f)

        # Finding distance
        f = 650
        d = (W * f) / w
        print(d)

        cvzone.putTextRect(
            img, f"Depth: {int(d)}cm", (face[10][0] - 100, face[10][1] - 50), scale=2
        )


def webcam_main():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        findFocus(img)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def callback(img):
    try:
        img = CvBridge().imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    findFocus(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

def bebop_main():
    rospy.Subscriber("bebop/image_raw", Image, callback, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    # rospy.init_node("focus_node", anonymous=True)
    # bebop_main()
    webcam_main()
