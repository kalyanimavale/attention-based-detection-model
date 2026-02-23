import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load extracted features
df = pd.read_csv("landmark_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train MLP classifier
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Validation Accuracy:", acc)

# Save model
joblib.dump(model, "landmark_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Landmark model saved successfully.")
