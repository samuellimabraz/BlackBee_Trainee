import cv2
import cvzone
import mediapipe as mp
import math


# This model is lighter than FaceMesh, but with less precision
class FaceDetector:
    """
    Face detector using the FaceDetector model by mediapipe
    Detect face and stipule a distance of the camera with focus distance
    and the face width real size
    """

    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_face(self, img, focus_length, dead_zone, draw=False):
        ih, iw, _ = img.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        img.flags.writeable = True
        iag = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dist = None
        bbox = []
        event = 0

        if results.detections:
            detection = results.detections[0]

            bboxC = detection.location_data.relative_bounding_box
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Calculate the center of the face
            # cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

            # # Calculate the distance using the bounding box width
            # w = bbox[2]
            # W = 12  # Real measure, width face

            # dist = int((W * focus_length) / w)

            # cvzone.putTextRect(
            #     img,
            #     f"Depth: {dist}cm",
            #     (0, 29),
            #     scale=2,
            # )

            # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if draw:
                self.mp_drawing.draw_detection(img, detection)

        return img, bbox, dist, event
