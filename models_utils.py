# models_utils.py
import os, pickle, joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Crop recommendation
def load_crop_model():
    p = os.path.join(MODEL_DIR, "crop_recommend.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError("models/crop_recommend.pkl not found. Train the model first.")
    with open(p, "rb") as f:
        return joblib.load(f)

def load_yield_model():
    model_path = os.path.join(MODEL_DIR, "yield_model.pkl")
    enc_path = os.path.join(MODEL_DIR, "yield_encoders.pkl")
    feat_path = os.path.join(MODEL_DIR, "yield_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("yield_model.pkl not found. Train the model first.")

    model = joblib.load(model_path)
    encoders = joblib.load(enc_path)
    features = joblib.load(feat_path)

    return {"model": model, "encoders": encoders, "features": features}


# Irrigation
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "irrigation_model.pkl")

# -------------------------------
# Load Model
# -------------------------------
def load_irrigation_model():
    return joblib.load(MODEL_PATH)

# -------------------------------
# Irrigation Advice Function
# -------------------------------
def irrigation_advice(features):
    """
    features = [Soil Moisture, Temperature, Soil_Humidity, rainfall]
    """
    model = load_irrigation_model()
    prediction = model.predict([features])[0]

    soil_moist, temp, soil_humic, rainfall = features

    if prediction == 1:   # Irrigation needed
        # Basic recommendation logic (you can tweak as needed)
        if soil_moist < 30:
            water_amount = "25-30 mm of water"
            frequency = "Daily for 3-4 days"
        elif 30 <= soil_moist < 50:
            water_amount = "15-20 mm of water"
            frequency = "Every 2-3 days"
        else:
            water_amount = "10-15 mm of water"
            frequency = "Every 3-4 days"

        advice = (
            "💧 Irrigation Needed\n"
            f"- Apply {water_amount}\n"
            f"- Frequency: {frequency}\n"
            "- Prefer early morning or late evening irrigation.\n"
            "- Avoid over-irrigation to prevent root damage."
        )

    else:   # No irrigation needed
        advice = (
            "✅ No Irrigation Required\n"
            "- Current soil and weather conditions are sufficient.\n"
            "- Monitor soil moisture regularly.\n"
            "- Recheck after 2-3 days or after next rainfall."
        )

    return advice


# Disease model
def load_disease_model():
    p = os.path.join(MODEL_DIR, "disease_model.h5")
    if not os.path.exists(p):
        raise FileNotFoundError("models/disease_model.h5 not found. Train the model first.")
    return load_model(p)

# Predict functions
def recommend_crop(features_list):
    """features_list = [N,P,K,temperature,humidity,ph,rainfall]"""
    model = load_crop_model()
    arr = np.array([features_list])
    pred = model.predict(arr)
    return pred[0]

def predict_yield(features_dict):
    data = load_yield_model()
    model = data["model"]
    encoders = data["encoders"]
    features = data["features"]

    # Apply label encoders for categorical features
    for col, le in encoders.items():
        if col in features_dict:
            val = features_dict[col]
            if val in le.classes_:
                features_dict[col] = le.transform([val])[0]
            else:
                # unseen label → map to -1
                features_dict[col] = -1  

    # Create input in correct order
    arr = np.array([[features_dict.get(f, 0) for f in features]])
    pred = model.predict(arr)
    return float(pred[0])


def detect_disease_from_image(img_path, top_k=1):
    model = load_disease_model()
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    classes = sorted(os.listdir(os.path.join("datasets","PlantVillage")))
    return classes[idx], float(preds[idx])
