from pickle import NONE
import cv2
import mediapipe as mp
from threading import *
cap=cv2.VideoCapture(0)
face_mesh=mp.solutions.face_mesh    #.FaceDectection(model_selection=1, min_detection_confidence=0.5)
drawing=mp.solutions.drawing_utils
drawing_styles=mp.solutions.drawing_styles
while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=face_mesh.FaceMesh(refine_landmarks=True).process(frame1)
    #faces=output.multi_face_landmarks
    if output.multi_face_landmarks:
        for detection in output.multi_face_landmarks:
            
            
            #drawing.draw_landmarks(image=frame,
            #landmark_list=detection,
            #connections=face_mesh.FACEMESH_IRISES,
            #landmark_drawing_spec=None,#drawing.DrawingSpec(color=(0,255,0)),
            #connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())


            #drawing.draw_landmarks(image=frame,
            #landmark_list=detection,
            #connections=face_mesh.FACEMESH_CONTOURS,
            #landmark_drawing_spec=None,#drawing.DrawingSpec(color=(0,0,255)),
            #connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

            drawing.draw_landmarks(image=frame,
            landmark_list=detection,
            connections=face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0,255,0)),
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    cv2.putText(frame,"FaceMesh",(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)

    cv2.imshow("video",frame)
    cv2.waitKey(1)