# Soil Health Analytics & Crop Recommendation System

A comprehensive web-based soil health analytics platform that leverages machine learning to provide data-driven crop recommendations and farm management insights. This Flask application combines soil nutrient analysis, weather integration, and intelligent crop prediction to support farmers in optimizing their agricultural productivity.

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌾 Overview

**SoilSense** is an intelligent agricultural decision support system designed to help farmers understand their soil health and receive personalized crop recommendations. By analyzing soil nutrient levels (N, P, K), pH, and local climate conditions, the system predicts the most suitable crops and provides actionable recommendations for fertilization, irrigation, and sustainable farming practices.

### Key Capabilities

- **ML-Based Crop Prediction**: Random Forest model with 99.32% accuracy
- **Soil Health Scoring**: Comprehensive nutrient and pH analysis
- **Weather Integration**: Real-time weather data from OpenWeatherMap API
- **User Analytics**: Track prediction history and soil health trends
- **Multi-Language Support**: English and regional language translations
- **User Authentication**: Secure local authentication with password reset
- **Detailed Reporting**: Comprehensive reports with nutrient analysis and recommendations

---

## ✨ Features

### 🔐 Authentication & User Management
- User registration with email validation
- Secure login with password hashing (Werkzeug)
- Password reset via email verification
- Session management with CSRF protection
- User profile management

### 🌾 Soil Analysis & Prediction
- **Nutrient Analysis**: Nitrogen (N), Phosphorus (P), Potassium (K) level assessment
- **pH Analysis**: Soil pH range validation and recommendations
- **Soil Health Scoring**: 0-100 score based on nutrient composition and pH
- **Crop Recommendations**: Top 3 crops with confidence probabilities
- **Fertilizer Guidance**: Targeted fertilizer recommendations based on deficiencies

### 📊 Advanced Analytics
- **Prediction History**: Track all soil analyses and predictions
- **Trend Analysis**: Monthly and crop-wise statistics
- **Dashboard Metrics**: Real-time analytics and KPIs
- **Detailed Reports**: Downloadable prediction reports

### 💧 Smart Recommendations
- **Water Management**: Daily irrigation estimates based on climate conditions
- **Fertilizer Planning**: Conservative nutrient guidance with organic alternatives
- **Yield Potential**: Estimated yield index combining soil health and model confidence
- **Sustainable Farming**: Actionable suggestions for soil improvement

### 🗣️ Chatbot & Support
- **Agriculture Assistant**: Knowledge-based Q&A on soil, fertilizers, irrigation, and crop selection
- **Local Guidance**: Region-appropriate farming advice
- **Chat History**: Conversation history maintained per session

### 🌍 Multi-Language Support
- English (en)
- Hindi (hi)
- Kannada (ta)
- And more regional languages
- Language-aware translations for crops, fertilizers, and messages

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask (Python web framework) |
| **ML Model** | scikit-learn (Random Forest Classifier) |
| **Database** | SQLite |
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript |
| **Security** | Flask-WTF (CSRF protection), Werkzeug (password hashing) |
| **APIs** | OpenWeatherMap (Weather data) |
| **Serialization** | joblib (model persistence) |
| **Environment** | python-dotenv (configuration management) |

---

## 📋 Requirements
 How to Run
1. Clone the Repository
bash
git clone https://github.com/Shalinihr1402/soil_health_project.git
cd soil_health_project
2. Create a Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Set Up Environment Variables
Create a .env file in the root directory with:

Code
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
OPENWEATHER_API_KEY=your_openweather_api_key
DATABASE_URL=sqlite:///soil_health.db
5. Initialize Database
bash
python
>>> from app import db
>>> db.create_all()
>>> exit()
6. Run the Application
bash
flask run
The application will be available at http://localhost:5000

### System Requirements
- Python 3.8+
- pip (Python package manager)
- SQLite3 (usually included with Python)

### Python Dependencies
