import cv2 as cv
import numpy as np

"""
Project 1 - Virtual Paint
"""

myColors = [[57, 76, 0, 100, 255, 255],  # G
            [90, 48, 0, 118, 255, 255],
            [0, 113, 100, 255, 255, 241]]  # B

colorValues = [[0, 255, 0],  # BGR
               [255, 0, 0],
               [0, 0, 255]]
"""
myColors = [[5,107,0,19,255,255],
            [133,56,0,159,156,255],
            [57,76,0,100,255,255],
            [90,48,0,118,255,255]]
colorValues = [[51,153,255],          ## BGR
                 [255,0,255],
                 [0,255,0],
                 [255,0,0]]
"""
myPoints = []  # [x, y, colorId]


def virtual_paint():
    frameWidth, frameHeight = 420, 380

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)
    vid.set(10, 150)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    while True:
        success, img = vid.read()
        imgResult = img.copy()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        newDataPoints = find_color(img, myColors, colorValues, imgResult)
        if len(newDataPoints) != 0:
            for point in newDataPoints:
                myPoints.append(point)

        if len(myPoints) != 0:
            draw_on_canvas(myPoints, colorValues, imgResult)

        cv.imshow('Virtual Paint', imgResult)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()


def find_color(imgCO, colors, colorValue, imgRe):
    imgHSV = cv.cvtColor(imgCO, cv.COLOR_BGR2HSV)
    count = 0
    newPoints = []

    for color in colors:
        # Define lower and upper bounds
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv.inRange(imgHSV, lower, upper)

        # Get contours
        x, y = get_contours(mask)
        cv.circle(imgRe, (x, y), 10, colorValue[count], cv.FILLED)

        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1

    return newPoints


def get_contours(imgCON):
    im2, contours, hierarchy = cv.findContours(imgCON, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            # cv.drawContours(img, contour, -1, (255, 0, 0), 2)
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv.boundingRect(approx)
    return x + w // 2, y


def draw_on_canvas(dataPoints, myColorValues, imgRst):
    for pt in dataPoints:
        cv.circle(imgRst, (pt[0], pt[1]), 10, myColorValues[pt[2]], cv.FILLED)
