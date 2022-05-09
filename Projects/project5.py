import cv2 as cv
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

"""
Project 5 - Hand Gesture to Control Volume
"""


def control_vol_with_hand():
    frameWidth, frameHeight = 504, 437

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    volMin, volMax = volume.GetVolumeRange()[:2]

    while True:
        success, img = vid.read()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        images = []

        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id, image in enumerate(handlandmark.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(image.x*w), int(image.y*h)
                    images.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        if images:
            x1, y1 = images[4][1], images[4][2]
            x2, y2 = images[8][1], images[8][2]

            length = hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [15, 220], [volMin, volMax])
            print(vol, length)

            # When sound is too loud, turn the line to red
            if length > 200:
                cv.circle(img, (x1, y1), 4, (0, 0, 255), cv.FILLED)
                cv.circle(img, (x2, y2), 4, (0, 0, 255), cv.FILLED)
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            else:
                cv.circle(img, (x1, y1), 4, (255, 0, 0), cv.FILLED)
                cv.circle(img, (x2, y2), 4, (255, 0, 0), cv.FILLED)
                cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            volume.SetMasterVolumeLevel(vol, None)

        cv.imshow('Volume Control', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
