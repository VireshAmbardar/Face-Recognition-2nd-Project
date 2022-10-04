import cv2
import mediapipe as mp

cap = cv2.VideoCapture("Videos/Steve Carell Never Rewatches Himself In  The Office.mp4")

mpFaceDetection = mp.solutions.face_detection                                        # media pipe had face deetction in it. helps to find faces nd face points like eyes,nose ,ears,mouth
faceDetection = mpFaceDetection.FaceDetection( min_detection_confidence=0.5)       # takes a default value min_detection confidence

while True:
    success , img = cap.read()     #succes = true/False , img is matrix #BGR

    img_rgb = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB )
    results = faceDetection.process(img_rgb)

    if results.detections:                                        # inside results . detection is a class. same as scorin DT.
        for id,detections in enumerate(results.detections):       # each face has its own id once it recognized and score will tell us how much its sure about it.
            print(id,detections)
            print(detections.location_data.relative_bounding_box)
            ih ,iw ,ic = img.shape
            bboxC = detections.location_data.relative_bounding_box  #C -cordinates
            bbox = int(bboxC.xmin *iw) , int(bboxC.ymin *ih) , int(bboxC.width *iw) , int(bboxC.height *ih)     # now bounding bax will have minx value ,miny value,height and width

            # making boundary with these cordinates
            cv2.rectangle(img , bbox , (0,255,0) , 1)
            cv2.putText(img, f'Score:{int(detections.score[0]*100)}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 255, 0), 2)

            #Fancy items
            #left Top
            cv2.line(img , (int(bboxC.xmin *iw) , int(bboxC.ymin *ih)) , (int(bboxC.xmin *iw +50) , int(bboxC.ymin *ih)) ,(0,255,0)  , 5)
            cv2.line(img, (int(bboxC.xmin * iw), int(bboxC.ymin * ih + 50)), (int(bboxC.xmin * iw ), int(bboxC.ymin * ih)), (0, 255, 0), 5)

            #right top
            cv2.line(img, ( int(bboxC.xmin * iw + int(bboxC.width *iw) ), int(bboxC.ymin * ih) ),( int(bboxC.xmin * iw + int(bboxC.width *iw) ), int(bboxC.ymin * ih)+50 ), (0, 255, 0), 5)
            cv2.line(img, ( int(bboxC.xmin * iw + int(bboxC.width *iw)-50 ), int(bboxC.ymin * ih) ),( int(bboxC.xmin * iw + int(bboxC.width *iw) ), int(bboxC.ymin * ih)), (0, 255, 0), 5)

            # left bottom
            cv2.line(img, (  int(bboxC.xmin * iw)         ,   int(bboxC.ymin*ih + int(bboxC.height *ih)) ),
                          (  int(bboxC.xmin * iw + 50 )   ,   int(bboxC.ymin * ih + int(bboxC.height *ih)) ),    (0, 255, 0), 5)

            cv2.line(img, (int(bboxC.xmin * iw), int(bboxC.ymin * ih + int(bboxC.height * ih))),
                          (int(bboxC.xmin * iw), int(bboxC.ymin * ih + int(bboxC.height * ih) -50)), (0, 255, 0), 5)

            #right Bottom
            cv2.line(img, ( int(bboxC.xmin * iw + int(bboxC.width *iw) ),  int(bboxC.ymin*ih + int(bboxC.height *ih))) ,
                       ( int(bboxC.xmin * iw + int(bboxC.width *iw)-50 ),  int(bboxC.ymin*ih + int(bboxC.height *ih))) , (0, 255, 0),5 )

            cv2.line(img, (int(bboxC.xmin * iw + int(bboxC.width * iw)), int(bboxC.ymin * ih + int(bboxC.height * ih))),
                          (int(bboxC.xmin * iw + int(bboxC.width * iw)), int(bboxC.ymin * ih + int(bboxC.height * ih)-50)),(0, 255, 0), 5)

    cv2.imshow('Image', img)

    if cv2.waitKey(15) & 0xFF == ord('q'):  # if letter d is pressed then break out of loop and stop playing
        break
    