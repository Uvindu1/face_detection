import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('Small Talk.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resulits = faceDetection.process(imgRGB)

    if resulits.detections:
        for id, detection in enumerate(resulits.detections):
            #mpDraw.draw_detection(img, detection)#meken andama land marks nuth pennum kararanoo
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic =img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)#landmark nethuwa kotuwa adagaththe meken

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow('Dnce', img)
    cv2.waitKey(20)