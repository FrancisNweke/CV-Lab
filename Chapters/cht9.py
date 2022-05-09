import cv2 as cv

# Chapter 9 - Face Detection
data_directory = 'E:\\Development Projects\\AI Data\\OpenCV_Resources'

# Load face cascade
faceCascadeXml = data_directory + '\\haarcascades\\haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(faceCascadeXml)

# Load smile cascade
smileCascadeXml = data_directory + '\\haarcascades\\haarcascade_smile.xml'
smileCascade = cv.CascadeClassifier(smileCascadeXml)

# Load eye cascade
eyeCascadeXml = data_directory + '\\haarcascades\\haarcascade_eye.xml'
eyeCascade = cv.CascadeClassifier(eyeCascadeXml)


# Detect face(s), smile and eyes in a photo
def face_detection(img_file):
    img = cv.imread(data_directory + f'\\{img_file}')

    # Resize image by reducing the dimensions
    imgResize = cv.resize(img, (985, 590))
    imgGray = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)

    # Detect face
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(imgResize, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roiGray = imgGray[y:y + h, x:x + w]
        roiImg = imgResize[y:y + h, x:x + w]

        # Detect smile
        smiles = smileCascade.detectMultiScale(roiGray, 1.1, 16)

        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roiImg, (sx, sy), (sx + sw, sy + sh), (220, 200, 220), 2)

        # Detect eye
        eyes = eyeCascade.detectMultiScale(roiGray, 1.2, 3)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roiImg, (ex, ey), (ex + ew, ey + eh), (200, 0, 190), 2)

    cv.imshow('Face Detection', imgResize)

    cv.waitKey(0)


# Detect faces, smile and eyes in a video
def detect_faces_in_video():
    vid = cv.VideoCapture(0)
    vid.set(3, 640)

    if not vid.isOpened():
        print('Cannot open camera.')
        exit()

    while True:
        success, img = vid.read()
        if not success:
            print('Cannot read frame(s). Exiting program...')

        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Detect face
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (129, 255, 180), 2)
            cv.putText(img, 'Face', (x, y - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (129, 255, 180), 1)
            roiGray = imgGray[y:y + h, x:x + w]
            roiImg = img[y:y + h, x:x + w]

            # Detect smile
            smiles = smileCascade.detectMultiScale(roiGray, 1.8, 20)

            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(roiImg, (sx, sy), (sx + sw, sy + sh), (220, 200, 220), 2)
                cv.putText(roiImg, 'Smile', (sx, sy - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (220, 200, 220), 1)

            # Detect eye
            eyes = eyeCascade.detectMultiScale(roiGray, 1.1, 10)

            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roiImg, (ex, ey), (ex + ew, ey + eh), (200, 0, 190), 2)
                cv.putText(roiImg, 'Eye', (ex, ey - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 0, 190), 1)

        cv.imshow('Face, Smile and Eye Detection', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
