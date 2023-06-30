#!/usr/bin/env python

import cv2
import cvzone
import mediapipe as mp
import math

frameHeight = 480
frameWidth = 640
deadZone = 100

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
        bbox = []
        event = 0
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark

            x1, y1 = (face[145].x * iw, face[145].y * ih)
            x2, y2 = (face[374].x * iw, face[374].y * ih)

            w = math.hypot(x2 - x1, y2 - y1)

            W = 6.3  # Real measure, distance of the eyes

            # Finding distance to the camera, f = 640
            # focus distance is a property of the camera
            dist = int((W * self.focus_lenght) / w)

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

            if (cx < frameWidth//2 - deadZone):
                event = 1
                cv2.putText(img, " GO LEFT " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                cv2.rectangle(img,(0,int(frameHeight/2-deadZone)),(int(frameWidth/2)-deadZone,int(frameHeight/2)+deadZone),(0,0,255),cv2.FILLED)
            elif (cx > int(frameWidth / 2) + deadZone):
                event = 2
                cv2.putText(img, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                cv2.rectangle(img,(int(frameWidth/2+deadZone),int(frameHeight/2-deadZone)),(frameWidth,int(frameHeight/2)+deadZone),(0,0,255),cv2.FILLED)
            elif (cy < int(frameHeight / 2) - deadZone):
                event = 3
                cv2.putText(img, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                cv2.rectangle(img,(int(frameWidth/2-deadZone),0),(int(frameWidth/2+deadZone),int(frameHeight/2)-deadZone),(0,0,255),cv2.FILLED)
            elif (cy > int(frameHeight / 2) + deadZone):
                event = 4
                cv2.putText(img, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
                cv2.rectangle(img,(int(frameWidth/2-deadZone),int(frameHeight/2)+deadZone),(int(frameWidth/2+deadZone),frameHeight),(0,0,255),cv2.FILLED)

            cv2.line(img, (int(frameWidth/2),int(frameHeight/2)), (cx,cy),(0, 0, 255), 3)
            
            cv2.line(img,(int(frameWidth/2)-deadZone,0),(int(frameWidth/2)-deadZone,frameHeight),(255,255,0),3)
            cv2.line(img,(int(frameWidth/2)+deadZone,0),(int(frameWidth/2)+deadZone,frameHeight),(255,255,0),3)
            cv2.line(img, (0,int(frameHeight / 2) - deadZone), (frameWidth,int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
            cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)
            

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

from HandModule import MyHandDetector

def main():
    facedetector = FaceDetector(
        max_num_faces=1, 
        refine_landmarks=False, 
        min_detection_confidence=0.8, 
        min_tracking_confidence=0.8,
        focus_length=640
    )
    handdetector = MyHandDetector(
        False,
        1,
        0.70,
        0.7
    )

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        frame, bbox, dist, event = facedetector.detect_face(frame, False)
        
        # if dist > 0:
        #     # Area de reconhecimento para as mãos
        #     x, y, w, h = abs(bbox[0] - 220), abs(bbox[1]), abs(bbox[2] + 80), abs(bbox[3] + 70)
        #     imgDetect = frame[y : (y + h), x: (x + w)]
        #     cvzone.cornerRect(frame, [x, y, w, h])
        #     event = handdetector.gestureRecognizer(imgDetect, (20, 20))
        
        cv2.imshow('MediaPipe Face Mesh', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
