import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time

MODEL_PATH = "landmark_model_v3.pkl"
SCALER_PATH = "scaler_v3.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Landmark IDs
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

# Stability control
prediction_buffer = deque(maxlen=35)
current_state = "Initializing"
last_change_time = time.time()
min_hold_time = 2.0

cap = cv2.VideoCapture(0)

def get_point(landmarks, idx, w, h):
    x = landmarks.landmark[idx].x * w
    y = landmarks.landmark[idx].y * h
    return np.array([x, y])

print("V3 Stable Real-Time Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

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

        # Normalized features
        features = np.array([[
            (inner_left[1] - left_eye_top[1]) / face_height,
            (inner_right[1] - right_eye_top[1]) / face_height,
            (outer_left[1] - inner_left[1]) / face_height,
            (outer_right[1] - inner_right[1]) / face_height,
            abs(left_eye_top[1] - left_eye_bottom[1]) / face_height,
            abs(right_eye_top[1] - right_eye_bottom[1]) / face_height
        ]])

        features_scaled = scaler.transform(features)

        probabilities = model.predict_proba(features_scaled)[0]
        prediction = model.classes_[np.argmax(probabilities)]
        confidence = np.max(probabilities)

        # Confidence threshold (ignore weak predictions)
        if confidence > 0.60:
            prediction_buffer.append(prediction)

        # Temporal smoothing
        if len(prediction_buffer) == prediction_buffer.maxlen:
            majority = max(set(prediction_buffer),
                           key=prediction_buffer.count)

            if majority != current_state:
                if time.time() - last_change_time > min_hold_time:
                    current_state = majority
                    last_change_time = time.time()

        cv2.putText(frame,
                    f"{current_state} ({confidence:.2f})",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Attention Detection V3 Stable", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
