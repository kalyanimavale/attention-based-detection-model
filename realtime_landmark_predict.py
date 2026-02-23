import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# ===============================
# LOAD TRAINED MODEL + SCALER
# ===============================

model = joblib.load("landmark_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# MEDIAPIPE FACE MESH SETUP
# ===============================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eyebrow and eye landmark indices
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Smoothing buffer (last 5 predictions)
prediction_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        def get_points(indices):
            pts = []
            for idx in indices:
                x = landmarks.landmark[idx].x * w
                y = landmarks.landmark[idx].y * h
                pts.append((x, y))
            return np.array(pts)

        # Extract landmark groups
        left_eyebrow = get_points(LEFT_EYEBROW)
        right_eyebrow = get_points(RIGHT_EYEBROW)
        left_eye = get_points(LEFT_EYE)
        right_eye = get_points(RIGHT_EYE)

        # Compute geometric features
        left_dist = np.mean(left_eyebrow[:,1]) - np.mean(left_eye[:,1])
        right_dist = np.mean(right_eyebrow[:,1]) - np.mean(right_eye[:,1])

        left_slope = np.polyfit(left_eyebrow[:,0], left_eyebrow[:,1], 1)[0]
        right_slope = np.polyfit(right_eyebrow[:,0], right_eyebrow[:,1], 1)[0]

        features = np.array([[left_dist, right_dist, left_slope, right_slope]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        prediction_buffer.append(prediction)

        # Majority vote smoothing
        final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

        cv2.putText(frame,
                    f"{final_prediction}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    else:
        cv2.putText(frame,
                    "No Face Detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    cv2.imshow("Eyebrow Attention Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
