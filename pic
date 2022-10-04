import cv2
import mediapipe as mp

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection( min_detection_confidence=0.75)

vid = cv2.VideoCapture(0)

while True:
    success , img = vid.read()
    img_rgb = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB )
    results = faceDetection.process(img_rgb)

    if results.detections:                                        # inside results . detection is a class. same as scorin DT.
        for id,detections in enumerate(results.detections):
            mpDraw.draw_detection(img , detections)


    cv2.imshow('Image', img)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break