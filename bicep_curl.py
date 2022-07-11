from cv2 import FONT_HERSHEY_COMPLEX, LINE_AA
import numpy as np
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def angle_calculator(a,b,c):

    a =np.array(a)
    b =np.array(b)
    c =np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs((radians*180)/np.pi)

    if angle> 180:
        angle = angle -180
    
    return angle


counter = 0
stage =None

cap = cv2.VideoCapture(0)
#setting up mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        
        #Extracting landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Getting coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            #calculte angle
            angle =angle_calculator(shoulder,elbow,wrist)
            
            #Visualize
            cv2.putText(image,str(angle),
                       tuple(np.multiply(elbow,[640,480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),1,cv2.LINE_AA)
            #curl counter
            if angle >160:
                stage = "down"
            if angle < 40 and stage == "down":
                stage = "up"
                counter +=1
                print(counter)
            
        except:
            pass
        
        # Rep data
        cv2.putText(image,"REPS",(20,60),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(110,60),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        
        #render detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(5,0,255),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(2,56,45),thickness=2,circle_radius=2))

        cv2.imshow("output",image)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

cap.release()
cv2.destroyAllWindows()