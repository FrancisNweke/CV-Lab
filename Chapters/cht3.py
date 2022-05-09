import cv2 as cv
import sys

# Chapter 3 - Resizing and Cropping
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources'


def resize_crop(img_file):
    img = cv.imread(data_directory + f'\\{img_file}')

    if img is None:
        sys.exit(f'Could not find file.')

    # (300 - width/column, 250 - height/row)
    img_resize = cv.resize(img, (300, 250))
    print(img.shape)
    print(img_resize.shape)

    img_cropped = img[0:200, 75:280]

    cv.imshow('Original Image', img)
    cv.imshow('Resized Image', img_resize)
    cv.imshow('Cropped Image', img_cropped)
    cv.waitKey(0)
