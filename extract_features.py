import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

DATASET_PATH = "model_dataset"
output_file = "landmark_features.csv"

# Eyebrow and eye landmark indices (MediaPipe)
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

data = []

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

        def get_points(indices):
            pts = []
            for idx in indices:
                x = landmarks.landmark[idx].x * w
                y = landmarks.landmark[idx].y * h
                pts.append((x, y))
            return np.array(pts)

        left_eyebrow = get_points(LEFT_EYEBROW)
        right_eyebrow = get_points(RIGHT_EYEBROW)
        left_eye = get_points(LEFT_EYE)
        right_eye = get_points(RIGHT_EYE)

        # Feature 1: eyebrow-eye vertical distance
        left_dist = np.mean(left_eyebrow[:,1]) - np.mean(left_eye[:,1])
        right_dist = np.mean(right_eyebrow[:,1]) - np.mean(right_eye[:,1])

        # Feature 2: eyebrow curvature (approx slope)
        left_slope = np.polyfit(left_eyebrow[:,0], left_eyebrow[:,1], 1)[0]
        right_slope = np.polyfit(right_eyebrow[:,0], right_eyebrow[:,1], 1)[0]

        # Feature vector
        features = [
            left_dist,
            right_dist,
            left_slope,
            right_slope
        ]

        data.append(features + [label])

df = pd.DataFrame(data, columns=[
    "left_dist", "right_dist",
    "left_slope", "right_slope",
    "label"
])

df.to_csv(output_file, index=False)
print("Feature extraction complete.")
