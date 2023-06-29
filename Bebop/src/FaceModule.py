#!/usr/bin/env python

import cv2
import cvzone
import mediapipe as mp
import math


# Face detector using the FaceMesh model by mediapipe
# Detect face and stipule a distance of the camera with focus distance
class FaceDetector:
    def __init__(
        self,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        focus_length=1000,
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

        self.focus_lenght = focus_length

    def detect_face(self, img, draw=False):
        ih, iw, _ = img.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dist = -1
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark

            x1, y1 = (face[145].x * iw, face[145].y * ih)
            x2, y2 = (face[374].x * iw, face[374].y * ih)

            w = math.hypot(x2 - x1, y2 - y1)

            W = 6.3  # Real measure, distance of the eyes

            # # Finding the Focal Length
            # d = 50
            # f = (w*d)/W
            # print(f)

            # Finding distance to the camera, f = 640
            # focus distance is a property of the camera
            dist = int((W * self.focus_lenght) / w)

            cvzone.putTextRect(
                img,
                f"Depth: {dist}cm",
                (int(face[10].x * iw) - 100, int(face[10].y * ih) - 50),
                scale=2,
            )

            # Find the indices of the peripheral points for the boundig box
            xmin, xmax = int(min(face, key=lambda l: l.x).x * iw), int(
                max(face, key=lambda l: l.x).x * iw
            )
            ymin, ymax = int(min(face, key=lambda l: l.y).y * ih), int(
                max(face, key=lambda l: l.y).y * ih
            )
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            if draw:
                self.draw_landmarks(img, results.multi_face_landmarks[0])

        return img, bbox, dist

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


def main():
    detector = FaceDetector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    )

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        frame, _ = detector.detect_face(frame, True)

        cv2.imshow("MediaPipe Face Mesh", frame)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
