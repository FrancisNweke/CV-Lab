import cv2 as cv
import sys

"""
Chapter 1 - Read Image and Video
"""
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources'


# Read an image
def read_image(image_file):
    img = cv.imread(data_directory + f'\\{image_file}')

    if img is None:
        sys.exit(f'Could not find file: {image_file}.')

    cv.imshow(f'{image_file}', img)
    cv.waitKey(0)


# Read a Video
def read_video(video_file):
    vid = cv.VideoCapture(data_directory + f"\\{video_file}")

    while vid.isOpened():
        success, img = vid.read()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        cv.imshow(f'Playing {video_file}', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()


# Read a frames from Webcam
def capture_video():
    vid = cv.VideoCapture(0)
    vid.set(3, 640)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    while True:
        success, img = vid.read()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        cv.imshow('Webcam', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()
