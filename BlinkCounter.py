import cv2
import numpy as np 
import imutils
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

def eye_aspect_ration(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])

    ear= (A+B)/(C * 2.0)
    return ear

def main():
    EAR_THRESH=0.22
    EAR_CONSE_FRAME=2
    COUNTER=0
    TOTAL=0
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    lStart,lEnd=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    rStart,rEnd=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()
        frame=imutils.resize(frame,width=500)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        rects=detector(gray,1)

        for rect in rects:
            shape=predictor(gray,rect)
            shape=face_utils.shape_to_np(shape)
            leftEye=shape[lStart:lEnd]
            rightEye=shape[rStart:rEnd]
            
            leftEAR=eye_aspect_ration(leftEye)
            rightEAR=eye_aspect_ration(rightEye)
            ear=(leftEAR+rightEAR)/2.0

            leftEyeHull=cv2.convexHull(leftEye)
            rightEyeHull=cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            print(ear)

            if ear<EAR_THRESH:
                COUNTER+=1
            else:
                if COUNTER>=EAR_CONSE_FRAME:
                    TOTAL+=1
                COUNTER=0
        
        cv2.resize(frame,(600,300))
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)
        #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Output",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cv2.destroyAllWindows()
    cap.release()



main()

