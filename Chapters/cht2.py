import cv2 as cv
import sys
import numpy as np

# Chapter 2 - Basic Functions
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources'


def basic_img_func(img_file):
    img = cv.imread(data_directory + f'\\{img_file}')
    kernel = np.ones((5, 5), np.uint8)

    if img is None:
        sys.exit(f'Could not find file.')

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(img, (7, 7), 0)
    imgCanny = cv.Canny(imgGray, 150, 200)
    imgDilated = cv.dilate(imgCanny, kernel, iterations=1)
    imgEroded = cv.erode(imgDilated, kernel, iterations=1)

    cv.imshow('Original Image', img)
    cv.imshow('Gray Image', imgGray)
    cv.imshow('Blurred Image', imgBlur)
    cv.imshow('Canny Image', imgCanny)
    cv.imshow('Dilated Image', imgDilated)
    cv.imshow('Eroded Image', imgEroded)

    cv.waitKey(0)
