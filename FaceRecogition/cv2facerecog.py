from turtle import circle, color
import cv2
from cv2 import KeyPoint
import mediapipe as mp

cap=cv2.VideoCapture(0)
face_detection=mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=1)    #.FaceDectection(model_selection=1, min_detection_confidence=0.5)
drawing=mp.solutions.drawing_utils
while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=face_detection.process(frame1)
    #faces=output.multi_face_landmarks
    if output.detections:
        for face_no, face in enumerate(output.detections):
            drawing.draw_detection(frame, face,drawing.DrawingSpec(color=(255,0,0),thickness=4,circle_radius=2))
            #drawing.draw_axis(frame,1,1)
    cv2.imshow("video",frame)
    cv2.waitKey(1)