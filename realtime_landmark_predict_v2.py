import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

model = joblib.load("landmark_model_v2.pkl")
scaler = joblib.load("scaler_v2.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

baseline_samples = []
baseline_duration = 3
start_time = time.time()
baseline_ready = False

def get_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        x = landmarks.landmark[idx].x * w
        y = landmarks.landmark[idx].y * h
        pts.append((x, y))
    return np.array(pts)

print("Hold neutral face for 3 seconds...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        left_eyebrow = get_points(landmarks, LEFT_EYEBROW, w, h)
        right_eyebrow = get_points(landmarks, RIGHT_EYEBROW, w, h)
        left_eye = get_points(landmarks, LEFT_EYE, w, h)
        right_eye = get_points(landmarks, RIGHT_EYE, w, h)

        left_eye_height = np.max(left_eye[:,1]) - np.min(left_eye[:,1])
        right_eye_height = np.max(right_eye[:,1]) - np.min(right_eye[:,1])

        if left_eye_height == 0 or right_eye_height == 0:
            continue

        left_dist = (np.mean(left_eyebrow[:,1]) - np.mean(left_eye[:,1])) / left_eye_height
        right_dist = (np.mean(right_eyebrow[:,1]) - np.mean(right_eye[:,1])) / right_eye_height

        left_curve = np.polyfit(left_eyebrow[:,0], left_eyebrow[:,1], 2)[0]
        right_curve = np.polyfit(right_eyebrow[:,0], right_eyebrow[:,1], 2)[0]

        left_ear = left_eye_height / (np.max(left_eye[:,0]) - np.min(left_eye[:,0]) + 1e-6)
        right_ear = right_eye_height / (np.max(right_eye[:,0]) - np.min(right_eye[:,0]) + 1e-6)

        features = np.array([[left_dist, right_dist, left_curve, right_curve, left_ear, right_ear]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        cv2.putText(frame, prediction, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Final Attention Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
