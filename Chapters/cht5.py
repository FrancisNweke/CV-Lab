import cv2 as cv
import numpy as np

# Chapter 5 - Warp Perspective
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources'


def wrp(img_file):
    img = cv.imread(data_directory + f'\\{img_file}')

    width, height = 250, 350
    pt1 = np.float32([[414, 4], [684, 160], [184, 419], [438, 581]])
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pt1, pt2)
    wrpImage = cv.warpPerspective(img, matrix, (width, height))

    cv.imshow('Original Image', img)
    cv.imshow('Warp Perspective', wrpImage)

    cv.waitKey(0)
