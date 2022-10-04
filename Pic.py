import cv2
import mediapipe as mp

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.75)

img = cv2.imread('RDJ_7.jpg')

img_rgb = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB )
results = faceDetection.process(img_rgb)

mpDraw.draw_detection(img , results.detections.detections)



cv2.imshow('Image', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destrotAllWindows()