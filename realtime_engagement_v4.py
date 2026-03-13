import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load model
model = joblib.load("engagement_model.pkl")

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]

LEFT_BROW=[70,63,105]
RIGHT_BROW=[336,296,334]

MOUTH=[13,14]

labels={
0:"Attentive",
1:"Attentive_No_Understanding",
2:"Disattentive"
}

# buffer for smoothing
prediction_buffer = deque(maxlen=10)

# keep last stable prediction
stable_prediction = None


def eye_aspect_ratio(points):
    A=np.linalg.norm(points[1]-points[5])
    B=np.linalg.norm(points[2]-points[4])
    C=np.linalg.norm(points[0]-points[3])
    return (A+B)/(2*C)


def eyebrow_angle(p1,p2,p3):
    v1=p1-p2
    v2=p3-p2
    cosang=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.degrees(np.arccos(cosang))


cap=cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:

    while True:

        ret,frame=cap.read()
        if not ret:
            break

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(rgb)

        display_text="Analyzing face..."

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                pts=np.array([[lm.x,lm.y,lm.z] for lm in face_landmarks.landmark])

                left_eye=eye_aspect_ratio(pts[LEFT_EYE])
                right_eye=eye_aspect_ratio(pts[RIGHT_EYE])
                eye_ratio=(left_eye+right_eye)/2

                brow_left=eyebrow_angle(pts[LEFT_BROW[0]],pts[LEFT_BROW[1]],pts[LEFT_BROW[2]])
                brow_right=eyebrow_angle(pts[RIGHT_BROW[0]],pts[RIGHT_BROW[1]],pts[RIGHT_BROW[2]])
                brow_angle=(brow_left+brow_right)/2

                mouth_open=np.linalg.norm(pts[MOUTH[0]]-pts[MOUTH[1]])

                chin=pts[152]
                forehead=pts[10]

                head_pitch=chin[1]-forehead[1]
                head_yaw=pts[234][0]-pts[454][0]

                features=np.array([
                    eye_ratio,
                    brow_angle,
                    mouth_open,
                    head_pitch,
                    head_yaw
                ]).reshape(1,-1)

                pred=model.predict(features)[0]

                # add prediction
                prediction_buffer.append(pred)

                # only predict after enough frames
                if len(prediction_buffer) == prediction_buffer.maxlen:

                    final_pred=max(set(prediction_buffer),
                                   key=prediction_buffer.count)

                    stable_prediction=labels[final_pred]

        # display prediction
        if stable_prediction is not None:
            display_text=stable_prediction

        cv2.putText(frame,
                    display_text,
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Engagement Detection",frame)

        if cv2.waitKey(1)==27:
            break

cap.release()
cv2.destroyAllWindows()
