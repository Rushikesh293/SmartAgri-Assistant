# train_crop.py
import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommend.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "crop_recommend_log.txt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)   # replace with actual crop column
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

joblib.dump(model, MODEL_PATH)

with open(LOG_PATH, "w") as f:
    f.write(f"Accuracy: {acc:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"✅ Crop model trained with Accuracy: {acc:.2f}")
print(f"📂 Logs saved at {LOG_PATH}")


