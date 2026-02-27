import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("landmark_features_v2.csv")

X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Final Validation Accuracy:", acc)

joblib.dump(model, "landmark_model_v2.pkl")
joblib.dump(scaler, "scaler_v2.pkl")

print("Final model saved.")
