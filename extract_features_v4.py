import os
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

dataset_path = "dataset"

labels = {
    "attentive":0,
    "attentive_no_understanding":1,
    "disattentive":2
}

X=[]
y=[]

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]

LEFT_BROW=[70,63,105]
RIGHT_BROW=[336,296,334]

MOUTH=[13,14]


def eye_aspect_ratio(points):
    A=np.linalg.norm(points[1]-points[5])
    B=np.linalg.norm(points[2]-points[4])
    C=np.linalg.norm(points[0]-points[3])
    ear=(A+B)/(2.0*C)
    return ear


def eyebrow_angle(p1,p2,p3):
    v1=p1-p2
    v2=p3-p2
    cosang=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    angle=np.arccos(cosang)
    return np.degrees(angle)


def extract_features(img):

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:

        results=face_mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None

        landmarks=results.multi_face_landmarks[0]

        pts=np.array([[lm.x,lm.y,lm.z] for lm in landmarks.landmark])

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

        feature_vector=[
            eye_ratio,
            brow_angle,
            mouth_open,
            head_pitch,
            head_yaw
        ]

        return np.array(feature_vector)


print("Dataset folders detected:", os.listdir(dataset_path))


for class_name in os.listdir(dataset_path):

    # Remove hidden spaces/newline
    clean_name = class_name.strip()

    if clean_name not in labels:
        print("Skipping unknown folder:", class_name)
        continue

    path=os.path.join(dataset_path,class_name)

    for img_name in os.listdir(path):

        img_path=os.path.join(path,img_name)

        img=cv2.imread(img_path)

        if img is None:
            continue

        feat=extract_features(img)

        if feat is not None:

            X.append(feat)
            y.append(labels[clean_name])


X=np.array(X)
y=np.array(y)

np.save("X.npy",X)
np.save("y.npy",y)

print("Feature extraction completed")
print("Total samples:",len(X))
