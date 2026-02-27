import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

DATASET_PATH = "dataset"
OUTPUT_PATH = "landmark_features_v3.csv"

INNER_LEFT_BROW = 70
INNER_RIGHT_BROW = 300
OUTER_LEFT_BROW = 105
OUTER_RIGHT_BROW = 334

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

NOSE = 1
CHIN = 152

data = []

def get_point(landmarks, idx, w, h):
    x = landmarks.landmark[idx].x * w
    y = landmarks.landmark[idx].y * h
    return np.array([x, y])

for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape

        inner_left = get_point(landmarks, INNER_LEFT_BROW, w, h)
        inner_right = get_point(landmarks, INNER_RIGHT_BROW, w, h)
        outer_left = get_point(landmarks, OUTER_LEFT_BROW, w, h)
        outer_right = get_point(landmarks, OUTER_RIGHT_BROW, w, h)

        left_eye_top = get_point(landmarks, LEFT_EYE_TOP, w, h)
        left_eye_bottom = get_point(landmarks, LEFT_EYE_BOTTOM, w, h)
        right_eye_top = get_point(landmarks, RIGHT_EYE_TOP, w, h)
        right_eye_bottom = get_point(landmarks, RIGHT_EYE_BOTTOM, w, h)

        nose = get_point(landmarks, NOSE, w, h)
        chin = get_point(landmarks, CHIN, w, h)

        face_height = abs(nose[1] - chin[1]) + 1e-6

        features = [
            (inner_left[1] - left_eye_top[1]) / face_height,
            (inner_right[1] - right_eye_top[1]) / face_height,
            (outer_left[1] - inner_left[1]) / face_height,
            (outer_right[1] - inner_right[1]) / face_height,
            abs(left_eye_top[1] - left_eye_bottom[1]) / face_height,
            abs(right_eye_top[1] - right_eye_bottom[1]) / face_height
        ]

        data.append(features + [label])

df = pd.DataFrame(data, columns=[
    "inner_left_norm",
    "inner_right_norm",
    "tilt_left_norm",
    "tilt_right_norm",
    "left_ear_norm",
    "right_ear_norm",
    "label"
])

df.to_csv(OUTPUT_PATH, index=False)
print("V3 normalized feature extraction complete.")
