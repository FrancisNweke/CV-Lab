import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
Project 4 - Fire Detection
"""


def detect_fire():
    frameWidth, frameHeight = 510, 390

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    while True:
        success, img = vid.read()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        cv.imshow('Fire Detection', img)

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(imgRGB)
        plt.show()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
