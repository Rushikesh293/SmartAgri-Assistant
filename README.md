# 🌱 SmartAgri Assistant

An AI-powered web application that helps farmers make better decisions using Machine Learning and Deep Learning.

---

## 🚀 Overview

SmartAgri Assistant is a smart agriculture system that integrates multiple AI models to provide:

* 🌾 Crop Recommendation
* 📈 Yield Prediction
* 💧 Irrigation Suggestion
* 🦠 Disease Detection (Image-based)
* 📄 Full Report Generation (PDF Download)

This system helps farmers improve productivity, reduce risks, and make data-driven decisions.

---

## 🎯 Features

* ✅ AI-based crop recommendation using soil & weather data
* ✅ Yield prediction using Machine Learning
* ✅ Irrigation advice for efficient water usage
* ✅ Disease detection using CNN (image classification)
* ✅ Detailed knowledge base for crops
* ✅ Interactive dashboard with charts
* ✅ Full report generation with PDF download
* ✅ User-friendly web interface

---

## 🧠 Technologies Used

### 🔹 Backend

* Python
* Flask

### 🔹 Frontend

* HTML
* CSS
* Bootstrap

### 🔹 Machine Learning

* Scikit-learn (Random Forest)
* TensorFlow / Keras (CNN)
* Pandas, NumPy

### 🔹 Other Tools

* OpenCV (Image Processing)
* Joblib / Pickle (Model Saving)
* ReportLab (PDF Generation)

---

## 🏗️ Project Structure

```
SmartAgri-Assistant/
│
├── app.py
├── models/
├── templates/
├── static/
├── crop_knowledge.py
├── crop_advisory.py
├── yield_tips.py
├── uploads/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/Rushikesh293/SmartAgri-Assistant.git
cd SmartAgri-Assistant
```

### 2️⃣ Create virtual environment (optional but recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the application

```
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Modules

### 🌾 Crop Recommendation

Recommends best crop based on:

* N, P, K values
* Temperature
* Humidity
* pH
* Rainfall

---

### 📈 Yield Prediction

Predicts crop yield using:

* Crop type
* Weather conditions
* Pesticide usage

---

### 💧 Irrigation Suggestion

Provides:

* Water requirement
* Irrigation timing
* Best practices

---

### 🦠 Disease Detection

* Upload plant leaf image
* Detect disease using CNN
* Get causes, symptoms, prevention, cure

---

### 📄 Report Generation

* Combines all outputs
* Includes farming insights
* Download as PDF

---

## 🔮 Future Scope

* 🌦️ Real-time weather API integration
* 📱 Mobile application
* 🌍 Multi-language support
* 🎤 Voice assistant for farmers
* ☁️ Cloud deployment

---

## 👨‍💻 Author

**Rushikesh Gughane**
B.Tech – Computer Science & Design

---

## 📜 License

This project is for academic and educational purposes.

---
