# app.py
import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

import joblib
import pandas as pd
import numpy as np

from models_utils import recommend_crop, irrigation_advice, detect_disease_from_image, predict_yield

# ---------- Paths & App ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "imgs", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------- DB Models ----------
class Disease(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(255), unique=True, nullable=False)
    description = db.Column(db.Text)
    prevention = db.Column(db.Text)
    cure = db.Column(db.Text)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    module = db.Column(db.String(50))
    input_data = db.Column(db.Text)
    result = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    if Disease.query.count() == 0:
        d1 = Disease(
            disease_name="Tomato___Late_blight",
            description="Fungal disease causing spots on leaves.",
            prevention="Avoid overhead irrigation; rotate crops.",
            cure="Use recommended fungicide; remove infected leaves."
        )
        db.session.add(d1)
        db.session.commit()

# ---------- Yield model & columns ----------
# We’ll align inputs to the exact training columns used by your model
YIELD_MODEL_PATH = os.path.join(BASE_DIR, "models", "yield_model.pkl")
yield_model = joblib.load(YIELD_MODEL_PATH)

YIELD_DATA_PATH = os.path.join(BASE_DIR, "datasets", "yield_df.csv")
yield_df = pd.read_csv(YIELD_DATA_PATH)
yield_feature_columns = yield_df.drop(columns=["hg/ha_yield"]).columns


# Helpers to find the right continuous column names regardless of exact spelling
def _find_col_by_token(columns, token):
    token = token.lower()
    for c in columns:
        if token in c.lower():
            return c
    raise KeyError(f"Could not find a column containing '{token}' in feature columns.")

def _set_crop_one_hot(input_dict, feature_cols, crop_name):
    # Find a one-hot column that matches the crop (case/space-insensitive)
    crop_norm = "".join(crop_name.lower().split())
    for c in feature_cols:
        if c.startswith("Item_"):
            suffix = c[5:]  # part after "Item_"
            suffix_norm = "".join(suffix.lower().split())
            if suffix_norm == crop_norm:
                input_dict[c] = 1
                return True
    # If the exact crop one-hot isn’t present (because of drop_first), model can still predict with all zeros
    return False

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

# Crop Recommendation
@app.route("/crop", methods=["GET", "POST"])
def crop_page():
    prediction = None
    if request.method == "POST":
        try:
            N = float(request.form.get("N", 0))
            P = float(request.form.get("P", 0))
            K = float(request.form.get("K", 0))
            temperature = float(request.form.get("temperature", 25))
            humidity = float(request.form.get("humidity", 70))
            ph = float(request.form.get("ph", 6.5))
            rainfall = float(request.form.get("rainfall", 100))

            features = [N, P, K, temperature, humidity, ph, rainfall]
            result = recommend_crop(features)
            prediction = result

            rec = Prediction(module="crop", input_data=str(features), result=str(result))
            db.session.add(rec)
            db.session.commit()
        except Exception as e:
            flash("Error: " + str(e), "danger")
    return render_template("crop.html", prediction=prediction)

# Yield Prediction
# ---------- Yield Prediction ----------
@app.route("/yield", methods=["GET", "POST"])
def yield_page():
    try:
        if request.method == "POST":
            crop = request.form.get("crop")
            pesticides_val = float(request.form.get("pesticides", 0))
            rainfall_val = float(request.form.get("rainfall", 0))
            temperature_val = float(request.form.get("temperature", 0))

            # ✅ Match dataset column names
            features = {
                "Item": crop,   # crop name (will be label encoded)
                "pesticides_tonnes": pesticides_val,
                "average_rain_fall_mm_per_year": rainfall_val,
                "avg_temp": temperature_val,
            }

            # Call prediction utility
            predicted = predict_yield(features)

            # Save prediction to DB
            rec = Prediction(
                module="yield",
                input_data=str(features),
                result=str(predicted),
            )
            db.session.add(rec)
            db.session.commit()

            return render_template("yield.html", predicted=round(predicted, 2))

        # If GET request → just render page
        return render_template("yield.html", predicted=None)

    except Exception as e:
       import traceback
       error_msg = f"Error: {str(e)} | Trace: {traceback.format_exc()}"
       flash(error_msg, "danger")
       return render_template("yield.html", predicted=None)




# Disease Detection
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import joblib
import os

# Load model
disease_model = load_model("models/disease_model.h5")

# Load labels (reverse the dict so index → class_name)
label_dict = joblib.load("models/disease_labels.pkl")
disease_labels = {v: k for k, v in label_dict.items()}

@app.route("/disease", methods=["GET", "POST"])
def disease_page():
    prediction = None
    if request.method == "POST":
        try:
            file = request.files["file"]
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = disease_model.predict(img_array)
            class_idx = np.argmax(preds)
            result = disease_labels[class_idx]

            prediction = f"Detected Disease: {result}"
        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template("disease.html", prediction=prediction)


# Irrigation
from models_utils import irrigation_advice

@app.route("/irrigation", methods=["GET", "POST"])
def irrigation_page():
    if request.method == "POST":
        soil_moist = float(request.form["Soil_Moist"])
        temperat = float(request.form["Temperat"])
        soil_humic = float(request.form["Soil_Humic"])
        rainfall = float(request.form["rainfall"])

        advice = irrigation_advice([soil_moist, temperat, soil_humic, rainfall])
        return render_template("irrigation.html", result=advice)
    return render_template("irrigation.html")




# Knowledge Base
@app.route("/knowledge")
def knowledge_page():
    diseases = Disease.query.all()
    return render_template("knowledge.html", diseases=diseases)

# Serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True)
