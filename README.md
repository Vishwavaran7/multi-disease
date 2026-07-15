# MediSense.AI — Multi-Disease Prediction & AI Advisory Platform

[![MediSense Live](https://img.shields.io/badge/MediSense.AI-Live-00e5ff?style=for-the-badge&logo=render)](https://multi-disease-1multi-disease.onrender.com/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-7c3aed?style=for-the-badge&logo=github)](https://github.com/Vishwavaran7/multi-disease)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

> **MediSense.AI** is an advanced, production-grade healthcare risk prediction platform. It integrates highly optimized Machine Learning models (trained using Scikit-Learn and XGBoost) with a conversational AI medical advisor powered by the Google Gemini API, allowing users to assess health risks, locate nearby medical facilities, and receive instant interactive feedback.

---

## 🌟 Key Features

*   🧠 **Multi-Disease ML Predictors:** Highly accurate classification models utilizing state-of-the-art algorithms to predict:
    *   **Asthma Risk** (`asthma_model.pkl` + scaler)
    *   **Diabetes Risk** (`diabetes_model.pkl` + scaler)
    *   **Heart Disease Risk** (`heart_model.pkl` + scaler)
    *   **Stroke Risk** (`stroke_model.pkl` + scaler)
*   💬 **AI Conversational Medical Assistant:** Integrated with the **Google Gemini API** to provide context-aware, empathetic health coaching, answering queries and explaining prediction results in detail.
*   📍 **Emergency Hospital Finder:** Uses the **Overpass API** and **Geopy** to locate the top 5 closest hospitals based on the user's geographic coordinates, calculating distances and providing direct Google Maps routing directions.
*   📧 **Automated Email Alerts:** Integrated email notification system (`EmailHelper`) that sends detailed prediction summaries, risk scores, Gemini recommendations, and nearby hospital contacts to users and designated parent/guardian emails.
*   🔐 **User Authentication & Dashboard:** Secure account registration, login session tracking, and a comprehensive user dashboard visualizing previous prediction histories.
*   🗄️ **Secure SQLite Database:** Keeps structured logs of all predictions, input features, and AI recommendations for history tracking.

---

## 🛠️ Tech Stack

| Component | Technologies Used |
| :--- | :--- |
| **Backend Framework** | Flask, Flask-SQLAlchemy |
| **Machine Learning** | XGBoost, Scikit-Learn, Pandas, NumPy, SciPy |
| **Generative AI** | Google Gemini API (via `google-generativeai` SDK) |
| **Mapping & Geolocation**| Geopy, Overpass API (OpenStreetMap) |
| **Database** | SQLite (SQLAlchemy ORM) |
| **Frontend/UI** | HTML5, CSS3, Vanilla JavaScript |
| **Deployment** | Render Web Services (configured with `gunicorn` & `wsgi.py`) |

---

## 📂 Repository Structure

```text
multi-disease/
│
├── multi_disease/           # Main application package
│   ├── models/              # Serialized trained ML models & scalers
│   │   ├── asthma_model.pkl
│   │   ├── asthma_scaler.pkl
│   │   ├── diabetes_model.pkl
│   │   ├── diabetes_scaler.pkl
│   │   ├── heart_model.pkl
│   │   ├── heart_scaler.pkl
│   │   ├── stroke_model.pkl
│   │   └── stroke_scaler.pkl
│   │
│   ├── static/              # Static frontend assets
│   ├── templates/           # HTML templates (Dashboard, Disease predict pages, etc.)
│   ├── app.py               # Main Flask application controllers & API routes
│   ├── config.py            # Global application configuration settings
│   ├── database.py          # SQLAlchemy Models (User, Prediction)
│   ├── email_helper.py      # Automated email dispatch service
│   ├── gemini_helper.py     # Gemini AI prompt configuration & API handler
│   └── multi-disease.ipynb  # ML Model training notebook
│
├── .venv/                   # Python virtual environment (ignored by Git)
├── app.py                   # Flask entry point file
├── wsgi.py                  # WSGI entry point for production servers
├── Procfile                 # Process file for Render/Heroku deployments
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started (Local Setup)

Follow these steps to run the application locally on your machine.

### 1. Prerequisites
Make sure you have python installed (version 3.9 or higher is recommended).
*   Download Python: [python.org](https://www.python.org/downloads/)

### 2. Clone the Repository
```bash
git clone https://github.com/Vishwavaran7/multi-disease.git
cd multi-disease
```

### 3. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up Environment Variables
Create a `.env` file (or set these inside your environment config) and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
FLASK_ENV=development
```

### 6. Run the Flask App
```bash
python wsgi.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the platform.

---

## 📊 Machine Learning Model Details

The disease risk classification is backed by model binaries trained on standardized datasets:

1.  **Diabetes Classifier:** Predicts risk levels using features like Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
2.  **Heart Disease Classifier:** Predicts cardiovascular risk using clinical markers like Chest Pain Type, RestBP, Chol, Fbs, RestECG, MaxHR, ExAng, Oldpeak, Slope, Ca, and Thal.
3.  **Asthma Classifier:** Assesses asthma onset probabilities based on demographic, environmental, and respiratory clinical variables.
4.  **Stroke Classifier:** Assesses stroke risk based on factors such as Age, Hypertension, Heart Disease, Marital Status, Work Type, Residence Type, Glucose Level, and BMI.

---

## 🔒 Security & Privacy Notice
*   **Disclaimer:** MediSense.AI is an AI-powered educational reference and screening assistance tool. It does not replace professional clinical diagnosis, advice, or therapy.
*   **Data Isolation:** SQLite is utilized locally. All input health credentials and conversational sessions are securely logged and are not sold or sent to third-party endpoints, except the anonymized queries passed to the Gemini API for medical chatbot assistance.

---

## ✉️ Contact & Support

Created by **Vishwavaran V**

*   **Email:** [vishwavaran7@gmail.com](mailto:vishwavaran7@gmail.com)
*   **LinkedIn:** [linkedin.com/in/vishwavaran](https://www.linkedin.com/in/vishwavaran)
*   **Portfolio:** [Vishwavaran's Portfolio](https://Vishwavaran7.github.io/portfolio/)
