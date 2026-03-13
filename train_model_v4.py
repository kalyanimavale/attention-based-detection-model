import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Load features
X = np.load("X.npy")
y = np.load("y.npy")

print("Total samples:", len(X))


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Pipeline: Normalization + SVM
model = Pipeline([
    ('scaler', StandardScaler()),

    ('svm', SVC(
        kernel='rbf',
        C=20,
        gamma='scale',
        class_weight='balanced',
        probability=True
    ))
])


# Train model
print("Training model...")
model.fit(X_train, y_train)


# Predictions
pred = model.predict(X_test)


# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# Save model
joblib.dump(model, "engagement_model.pkl")

print("\nModel saved as engagement_model.pkl")
