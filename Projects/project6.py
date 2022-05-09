import cv2 as cv
import math
import numpy as np
import mediapipe as mp

"""
Project 6 - Auto Face Detection and Face Mesh
"""
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles


def detect_face():
    frameWidth, frameHeight = 504, 437

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.7, model_selection=0) as face_detection:

        while True:
            success, img = vid.read()
            if not success:
                print('Cannot read frame(s). Exiting program...')
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = face_detection.process(img)

            # Draw the face detection annotations on the image.
            img.flags.writeable = True
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(img, detection)

            # Flip the image horizontally for a selfie-view display.
            cv.imshow('MediaPipe Face Detection', cv.flip(img, 1))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv.destroyAllWindows()


def detect_face_mesh():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    frameWidth, frameHeight = 504, 437

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            success, img = vid.read()
            if not success:
                print('Cannot read frame(s). Exiting program...')
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = face_mesh.process(img)

            # Draw the face mesh annotations on the image.
            img.flags.writeable = True
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

            # Flip the image horizontally for a selfie-view display.
            cv.imshow('MediaPipe Face Mesh', cv.flip(img, 1))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv.destroyAllWindows()
