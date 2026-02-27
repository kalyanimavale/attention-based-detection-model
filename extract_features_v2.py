import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

DATASET_PATH = "dataset"
output_file = "landmark_features_v2.csv"

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

data = []

def get_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        x = landmarks.landmark[idx].x * w
        y = landmarks.landmark[idx].y * h
        pts.append((x, y))
    return np.array(pts)

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

        left_eyebrow = get_points(landmarks, LEFT_EYEBROW, w, h)
        right_eyebrow = get_points(landmarks, RIGHT_EYEBROW, w, h)
        left_eye = get_points(landmarks, LEFT_EYE, w, h)
        right_eye = get_points(landmarks, RIGHT_EYE, w, h)

        left_eye_height = np.max(left_eye[:,1]) - np.min(left_eye[:,1])
        right_eye_height = np.max(right_eye[:,1]) - np.min(right_eye[:,1])

        if left_eye_height == 0 or right_eye_height == 0:
            continue

        # Normalized eyebrow distance
        left_dist = (np.mean(left_eyebrow[:,1]) - np.mean(left_eye[:,1])) / left_eye_height
        right_dist = (np.mean(right_eyebrow[:,1]) - np.mean(right_eye[:,1])) / right_eye_height

        # Curvature
        left_curve = np.polyfit(left_eyebrow[:,0], left_eyebrow[:,1], 2)[0]
        right_curve = np.polyfit(right_eyebrow[:,0], right_eyebrow[:,1], 2)[0]

        # Eye Aspect Ratio (EAR)
        left_ear = left_eye_height / (np.max(left_eye[:,0]) - np.min(left_eye[:,0]) + 1e-6)
        right_ear = right_eye_height / (np.max(right_eye[:,0]) - np.min(right_eye[:,0]) + 1e-6)

        features = [
            left_dist,
            right_dist,
            left_curve,
            right_curve,
            left_ear,
            right_ear
        ]

        data.append(features + [label])

df = pd.DataFrame(data, columns=[
    "left_dist",
    "right_dist",
    "left_curve",
    "right_curve",
    "left_ear",
    "right_ear",
    "label"
])

df.to_csv(output_file, index=False)
print("Final feature extraction complete.")
