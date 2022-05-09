import cv2 as cv
import mediapipe as mp

"""
Project 7 - Objectron (3D Object Detection)
"""

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron


def detect_object():
    frameWidth, frameHeight = 504, 437

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=5,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.99,
                                model_name='Shoe') as objectron:
        while True:
            success, img = vid.read()
            if not success:
                print('Cannot read frame(s). Exiting program...')
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = objectron.process(img)

            # Draw the face detection annotations on the image.
            img.flags.writeable = True
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(
                        img, detected_object.landmark_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(img, detected_object.rotation, detected_object.translation)

            # Flip the image horizontally for a selfie-view display.
            cv.imshow('MediaPipe Objectron', cv.flip(img, 1))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv.destroyAllWindows()
