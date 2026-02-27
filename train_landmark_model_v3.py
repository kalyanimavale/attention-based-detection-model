import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA_PATH = "landmark_features_v3.csv"
MODEL_PATH = "landmark_model_v3.pkl"
SCALER_PATH = "scaler_v3.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1)
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Stronger RandomForest
model = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== V3 Classification Report =====")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nV3 Model saved successfully.")
