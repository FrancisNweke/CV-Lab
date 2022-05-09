import cv2 as cv
import numpy as np
from Chapters.cht6 import stack_images

"""
Project 2 - Document Scanner
"""

imgWidth, imgHeight = 620, 490


def preprocess_image(img):
    kernel = np.ones((5, 5), np.uint8)

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 200, 200)
    imgDilated = cv.dilate(imgCanny, kernel, iterations=2)
    imgEroded = cv.erode(imgDilated, kernel, iterations=1)

    return imgEroded


def get_contours(img, imgResult):
    im2, contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 5000:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv.drawContours(imgResult, biggest, -1, (255, 0, 0), 15)

    return biggest


def reorder(dataPoints):
    points = dataPoints.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)

    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints


def get_wrap(img, biggest):
    biggest = reorder(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv.getPerspectiveTransform(pt1, pt2)
    wrpImage = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))

    imgCropped = wrpImage[20:wrpImage.shape[0] - 20, 20:wrpImage.shape[1] - 20]
    imgCropped = cv.resize(imgCropped, (imgWidth, imgHeight))

    return imgCropped


def scan_doc():

    vid = cv.VideoCapture(0)
    vid.set(3, imgWidth)
    vid.set(4, imgHeight)
    vid.set(10, 150)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    while True:
        success, img = vid.read()

        if not success:
            print('Cannot read frame(s). Exiting program...')

        cv.resize(img, (imgWidth, imgHeight))
        imgResult = img.copy()
        imgThreshold = preprocess_image(img)

        biggestContour = get_contours(imgThreshold, imgResult)
        imgWarped = get_wrap(img, biggestContour)

        if biggestContour.size != 0:
            imgArray = ([img, imgThreshold],
                        [imgResult, imgWarped])
        else:
            imgArray = ([img, imgThreshold],
                        [img, img])

        stackedImages = stack_images(0.6, imgArray)
        cv.imshow('Document Scanner', stackedImages)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
