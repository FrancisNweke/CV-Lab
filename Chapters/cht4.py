import cv2 as cv
import numpy as np


# Chapter 4 - Shapes and Text
def draw_write_text():
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    cv.line(img, (0, 0), (img.shape[1], img.shape[0]), color=(0, 255, 0), thickness=3)
    cv.rectangle(img, (50, 50), (310, 310), color=(255, 124, 20), thickness=2)
    cv.circle(img, (390, 140), 63, color=(120, 99, 255), thickness=cv.FILLED)
    cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, color=(255, 230, 244), thickness=2)
    cv.putText(img, 'HELLO', (10, 500), fontFace=cv.FONT_ITALIC, fontScale=4, color=(120, 124, 110), thickness=2, lineType=cv.LINE_AA)

    cv.imshow('Line, Shapes and Text', img)
    cv.waitKey(0)
