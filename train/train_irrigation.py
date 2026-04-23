# train_irrigation.py
import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "TARP.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "irrigation_model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "irrigation_model_log.txt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
df["Status"] = df["Status"].astype(str).str.strip().str.upper().map({"ON": 1, "OFF": 0})
df = df.fillna(df.mean())

X = df[["Soil_Moisture", "Temperature", "Soil_Humidity", "rainfall"]]  # adjust column names
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

joblib.dump(model, MODEL_PATH)

with open(LOG_PATH, "w") as f:
    f.write(f"Accuracy: {acc:.2f}\n")
    f.write(report)

print(f"✅ Irrigation model trained with Accuracy: {acc:.2f}")
print(f"📂 Logs saved at {LOG_PATH}")




