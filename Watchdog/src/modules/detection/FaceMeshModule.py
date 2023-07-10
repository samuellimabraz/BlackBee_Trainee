#!/usr/bin/env python

import cv2
import cvzone
import mediapipe as mp
import math


# This model generate 470 3D landmarks, is best for precision and control
class FaceMeshDetector:
    """
    Face detector using the FaceMesh model by mediapipe
    Detect face and stipule a distance of the camera with focus distance
    and the eyes real distance
    """

    def __init__(
        self,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_face(self, img, focus_length, dead_zone, draw=False):
        ih, iw, _ = img.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dist = 0
        bbox = []
        event = 0
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark

            # Calculete de distance of the people,
            # with the distance of the eyes landmarks
            x1, y1 = (face[145].x * iw, face[145].y * ih)
            x2, y2 = (face[374].x * iw, face[374].y * ih)

            w = math.hypot(x2 - x1, y2 - y1)

            W = 6.3  # Real measure, distance of the eyes

            # Finding distance to the camera, f = 640
            # focus distance is a property of the camera
            dist = int((W * focus_length) / w)

            cvzone.putTextRect(
                img,
                f"Depth: {dist}cm",
                (0, 29),
                scale=2,
            )

            # Creates a mesh and checks if the central point, the nose's landmark,
            # is within the central zone.
            # Generating events so that the camera
            # is centered with the face
            cx, cy = int(face[1].x * iw), int(face[1].y * ih)

            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if cx < iw // 2 - dead_zone:
                event = 1
                cv2.putText(
                    img,
                    " GO LEFT ",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                # cv2.rectangle(
                #     img,
                #     (0, int(ih / 2 - deadZone)),
                #     (int(iw / 2) - deadZone, int(ih / 2) + deadZone),
                #     (0, 0, 255),
                #     cv2.FILLED,
                # )
            elif cx > int(iw / 2) + dead_zone:
                event = 2
                cv2.putText(
                    img,
                    " GO RIGHT ",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                # cv2.rectangle(
                #     img,
                #     (int(iw / 2 + deadZone), int(ih / 2 - deadZone)),
                #     (iw, int(ih / 2) + deadZone),
                #     (0, 0, 255),
                #     cv2.FILLED,
                # )
            elif cy < int(ih / 2) - dead_zone:
                event = 3
                cv2.putText(
                    img,
                    " GO UP ",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                # cv2.rectangle(
                #     img,
                #     (int(iw / 2 - deadZone), 0),
                #     (int(iw / 2 + deadZone), int(ih / 2) - deadZone),
                #     (0, 0, 255),
                #     cv2.FILLED,
                # )
            elif cy > int(ih / 2) + dead_zone:
                event = 4
                cv2.putText(
                    img,
                    " GO DOWN ",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                # cv2.rectangle(
                #     img,
                #     (int(iw / 2 - deadZone), int(ih / 2) + deadZone),
                #     (int(iw / 2 + deadZone), ih),
                #     (0, 0, 255),
                #     cv2.FILLED,
                # )

            cv2.line(
                img,
                (int(iw / 2), int(ih / 2)),
                (cx, cy),
                (0, 0, 255),
                3,
            )

            cv2.line(
                img,
                (int(iw / 2) - dead_zone, 0),
                (int(iw / 2) - dead_zone, ih),
                (255, 255, 0),
                3,
            )
            cv2.line(
                img,
                (int(iw / 2) + dead_zone, 0),
                (int(iw / 2) + dead_zone, ih),
                (255, 255, 0),
                3,
            )
            cv2.line(
                img,
                (0, int(ih / 2) - dead_zone),
                (iw, int(ih / 2) - dead_zone),
                (255, 255, 0),
                3,
            )
            cv2.line(
                img,
                (0, int(ih / 2) + dead_zone),
                (iw, int(ih / 2) + dead_zone),
                (255, 255, 0),
                3,
            )

            # Find the indices of the peripheral points for the boundig box
            # xmin, xmax = int(min(face, key=lambda l: l.x).x * iw), int(
            #     max(face, key=lambda l: l.x).x * iw
            # )
            # ymin, ymax = int(min(face, key=lambda l: l.y).y * ih), int(
            #     max(face, key=lambda l: l.y).y * ih
            # )
            # bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            if draw:
                self.draw_landmarks(img, results.multi_face_landmarks[0])

        return img, bbox, dist, event

    def draw_landmarks(self, img, face_landmarks):
        # Draw all 470 ladmarks and connections
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
