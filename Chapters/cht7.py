import cv2 as cv
import numpy as np
from Chapters.cht6 import stack_images

# Chapter 7 - Color Detection
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources\\lamborghini.jpg'


def nothing(x):
    pass


# Detect color in a picture
def detect_color_in_photo():
    # Create window
    cv.namedWindow('TrackBars')
    cv.resizeWindow('TrackBars', 640, 240)

    # Create track bars for hue, saturation and value
    cv.createTrackbar('Hue Min', 'TrackBars', 0, 179, nothing)
    cv.createTrackbar('Hue Max', 'TrackBars', 179, 179, nothing)
    cv.createTrackbar('Saturation Min', 'TrackBars', 0, 255, nothing)
    cv.createTrackbar('Saturation Max', 'TrackBars', 214, 255, nothing)
    cv.createTrackbar('Value Min', 'TrackBars', 56, 255, nothing)
    cv.createTrackbar('Value Max', 'TrackBars', 229, 255, nothing)

    while True:
        img = cv.imread(data_directory)
        imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        h_min = cv.getTrackbarPos('Hue Min', 'TrackBars')
        h_max = cv.getTrackbarPos('Hue Max', 'TrackBars')
        s_min = cv.getTrackbarPos('Saturation Min', 'TrackBars')
        s_max = cv.getTrackbarPos('Saturation Max', 'TrackBars')
        v_min = cv.getTrackbarPos('Value Min', 'TrackBars')
        v_max = cv.getTrackbarPos('Value Max', 'TrackBars')

        # Define lower and upper bounds
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv.inRange(imgHSV, lower, upper)

        imgResult = cv.bitwise_and(img, img, mask=mask)

        # Stack images
        imgStacked = stack_images(0.7, ([img, imgHSV], [mask, imgResult]))

        # cv.imshow('Mask', mask)
        # cv.imshow('Color Detection', imgResult)
        cv.imshow('Color Detection', imgStacked)

        cv.waitKey(0)
        cv.destroyAllWindows()


# Detect color in a video
def detect_color_in_video():
    frameWidth, frameHeight = 420, 380

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    # Create window
    cv.namedWindow('HSV')
    cv.resizeWindow('HSV', 640, 240)

    # Create track bars for hue, saturation and value
    cv.createTrackbar('Hue Min', 'HSV', 0, 179, nothing)
    cv.createTrackbar('Saturation Min', 'HSV', 179, 179, nothing)
    cv.createTrackbar('Value Min', 'HSV', 0, 255, nothing)
    cv.createTrackbar('Hue Max', 'HSV', 255, 255, nothing)
    cv.createTrackbar('Saturation Max', 'HSV', 255, 255, nothing)
    cv.createTrackbar('Value Max', 'HSV', 255, 255, nothing)

    while True:
        success, img = vid.read()
        imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        h_min = cv.getTrackbarPos('Hue Min', 'HSV')
        h_max = cv.getTrackbarPos('Hue Max', 'HSV')
        s_min = cv.getTrackbarPos('Saturation Min', 'HSV')
        s_max = cv.getTrackbarPos('Saturation Max', 'HSV')
        v_min = cv.getTrackbarPos('Value Min', 'HSV')
        v_max = cv.getTrackbarPos('Value Max', 'HSV')

        # Define lower and upper bounds
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv.inRange(imgHSV, lower, upper)

        imgResult = cv.bitwise_and(img, img, mask=mask)

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        # Stack images
        imgStacked = stack_images(0.5, ([img, mask, imgResult]))

        cv.imshow('Color Picker', imgStacked)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
