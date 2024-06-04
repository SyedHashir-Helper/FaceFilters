import cv2
import numpy as np
import mediapipe as mp

lipsIndices = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 
37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 
312, 13, 82, 81, 80, 191

]

leftEyeIndices = [
    33, 133, 160, 158, 159, 157, 173, 153, 144, 163, 7, 246, 161, 160, 159, 
    158, 157, 173, 153, 144, 163, 7, 246
]
def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 400,240)
cv2.createTrackbar("Blue", "BGR", 0,255, empty)
cv2.createTrackbar("Green", "BGR", 0,255, empty)
cv2.createTrackbar("Red", "BGR", 0,255, empty)

def createBoundingBox(img, points, scale=5, masked=False, crop=True):
    if masked:
        # mask = cv2.fillPoly(mask, [points], (255,255,255))
        mask = np.zeros_like(img)
        hull = cv2.convexHull(points)
        cv2.fillPoly(mask, [hull], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)

    if crop:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0,0), None, scale, scale)
        return imgCrop
    else:
        return mask
def getPoints(points, indices):
    indPoints = []
    for i in range(len(points)):
        if i in indices:
            indPoints.append(points[i])
    return indPoints

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


while True:

    img = cv2.imread('image.jpeg')
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []
            for idx, lm in enumerate(face_landmarks.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_points.append((x, y))

                # Draw landmarks on the image
                if idx in lipsIndices:
                    pass
                    # cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
                    # cv2.putText(img, str(idx), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,255),1)



    faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    # leftEye = createBoundingBox(img, np.array(getPoints(landmark_points, leftEyeIndices)))
    # cv2.imshow("LeftEye: ", leftEye)
    lips = createBoundingBox(img, np.array(getPoints(landmark_points, lipsIndices)), masked=True, crop=False)

    imgColorLips = np.zeros_like(lips)

    b = cv2.getTrackbarPos("Blue", "BGR")
    g = cv2.getTrackbarPos("Green", "BGR")
    r = cv2.getTrackbarPos("Red", "BGR")
    imgColorLips[:] = b,g,r
    imgColorLips = cv2.bitwise_and(lips, imgColorLips)

    imgColorLips = cv2.GaussianBlur(imgColorLips, (7,7), 10)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
    imgColorLips = cv2.addWeighted(imgGray, 1, imgColorLips, 0.4,0)

    cv2.imshow("BGR", imgColorLips)
    cv2.imshow("Lips: Mask ", lips)
    cv2.imshow('My Img', img)
    cv2.waitKey(1)

