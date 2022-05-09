import cv2 as cv
import numpy as np
from Chapters.cht6 import stack_images

# Chapter 8 - Contour and Shape Detection
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources\\shapes01.png'


def detect_contour_shape():
    img = cv.imread(data_directory)
    imgContour = img.copy()

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1.7)
    imgCanny = cv.Canny(imgGray, 70, 70)

    get_contour(imgCanny, imgContour)

    imgBlank = np.zeros_like(img)
    imgStack = stack_images(0.5, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))

    cv.imshow('Contour and Shape', imgStack)

    cv.waitKey(0)


def get_contour(imgCanny, imgContour):
    im2, contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        print(area)
        if area > 500:
            cv.drawContours(imgContour, contour, -1, (255, 0, 0), 2)
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            objCor = len(approx)
            x, y, w, h = cv.boundingRect(approx)

            objectType = 'None'
            if objCor == 3:
                objectType = 'Triangle'
            elif objCor == 4:
                aspectRatio = w / float(h)
                if 0.95 < aspectRatio < 1.05:
                    objectType = 'Square'
                else:
                    objectType = 'Rectangle'
            elif objCor > 4:
                objectType = 'Circle'
            else:
                objectType = 'Unknown'

            cv.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                       (0, 0, 0), 2)
