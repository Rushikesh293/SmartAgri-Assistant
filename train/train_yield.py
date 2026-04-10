import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "yield_df.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yield_model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "yield_model.log")

df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# Encode categorical columns
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Save encoders for inference later
joblib.dump(label_encoders, os.path.join(BASE_DIR, "models", "yield_encoders.pkl"))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Save model
joblib.dump(model, MODEL_PATH)

# Save logs
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "w") as f:
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")

# Save feature columns
joblib.dump(X.columns.tolist(), os.path.join(BASE_DIR, "models", "yield_features.pkl"))


print("✅ Yield model trained and saved successfully!")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")





