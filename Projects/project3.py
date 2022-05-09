import cv2 as cv

"""
Project 3 - Russian Vehicle Plate Detector
"""

data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources'

# Load face cascade
plateCascadeXml = data_directory + '\\haarcascades\\haarcascade_russian_plate_number.xml'
plateCascade = cv.CascadeClassifier(plateCascadeXml)


def detect_number_plate():
    global imgRoi
    count = 0
    frameWidth, frameHeight = 510, 390

    vid = cv.VideoCapture(0)
    vid.set(3, frameWidth)
    vid.set(4, frameHeight)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    while True:
        success, img = vid.read()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Detect number plate
        numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

        for (x, y, w, h) in numberPlates:
            area = w * h
            if area > 500:
                cv.rectangle(img, (x, y), (x + w, y + h), (129, 255, 180), 2)
                cv.putText(img, 'Number Plate', (x, y - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (129, 255, 180), 1)
                imgRoi = img[y:y + h, x:x + w]
                cv.imshow('Russian Plate Detection', imgRoi)

        cv.imshow('Russian Plate Detection', img)

        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite(f'{data_directory}\\PlateNumber'+str(count)+'.jpg', imgRoi)
            cv.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv.FILLED)
            cv.putText(img, 'Image Saved', (150, 265), cv.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
            cv.imshow('Result', img)
            cv.waitKey(500)
            count += 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
