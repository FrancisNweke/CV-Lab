import cv2 as cv
import mediapipe as mp
import numpy as np

"""
Project 8 - Selfie Segmentation
"""
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources\\colorpic9.jpg'
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def segment_selfie():
    BG_COLOR = (192, 192, 192)  # gray
    frameWidth, frameHeight = 640, 480 # 504, 437

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:

        bg_image = None

        while True:
            success, img = vid.read()
            if not success:
                print('Cannot read frame(s). Exiting program...')
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = selfie_segmentation.process(img)

            # Draw the face detection annotations on the image.
            img.flags.writeable = True
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g.,
            bg = cv.imread(data_directory)
            # (width/column, height/row)
            bg_image = cv.resize(bg, (frameWidth, frameHeight))

            #   b) Blur the input image by applying image filtering, e.g.,
            img = cv.GaussianBlur(img, (5, 5), 1, borderType=cv.BORDER_WRAP)

            if bg_image is None:
                bg_image = np.zeros(img.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR

            output_image = np.where(condition, img, bg_image)

            # Flip the image horizontally for a selfie-view display.
            cv.imshow('Selfie Segmentation', output_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv.destroyAllWindows()