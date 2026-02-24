# Landmark-Based Attention Detection

This project uses MediaPipe FaceMesh to extract eyebrow geometry and head position and classify:

1. Attentive
2. Attentive with No Understanding
3. Disattentive

Pipeline:
- Landmark extraction
- Geometric feature computation
- MLP classifier training
- Real-time OpenCV testing

Run order:
1. python3 extract_features.py
2. python3 train_landmark_model.py
3. python3 realtime_landmark_predict.py
