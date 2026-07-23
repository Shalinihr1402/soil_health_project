"""Soil health analytics Flask application with local user authentication."""

import logging
import json
import os
import secrets
import smtplib
import sqlite3
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, session, url_for
from flask_wtf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash

from translations import SUPPORTED_LANGUAGES, translate, translate_crop, translate_fertilizer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATABASE_PATH = BASE_DIR / "database.db"
MODEL_PATH = BASE_DIR / "crop_model.pkl"
MODEL_METADATA_PATH = BASE_DIR / "model_metadata.json"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
REQUEST_TIMEOUT_SECONDS = 5

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USERNAME)
PASSWORD_RESET_EXPIRY_MINUTES = 30

app = Flask(__name__)
app.config.update(
    DATABASE=str(DATABASE_PATH),
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(32)),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
csrf = CSRFProtect(app)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Model Management
# ============================================================================

def load_model() -> Any:
    """Load the trained crop model from the project folder."""
    try:
        model = joblib.load(MODEL_PATH)
        app.logger.info(f"Model loaded successfully from {MODEL_PATH.name}")
        app.logger.info(f"Model type: {type(model).__name__}")
        
        # Check model capabilities
        if hasattr(model, 'predict_proba'):
            app.logger.info("Model supports probability predictions")
        else:
            app.logger.warning("Model does not support predict_proba")
            
        if hasattr(model, 'classes_'):
            app.logger.info(f"Model has {len(model.classes_)} classes")
        else:
            app.logger.warning("Model does not have classes_ attribute")
            
        return model
    except FileNotFoundError as error:
        raise RuntimeError(f"Trained model not found: {MODEL_PATH.name}") from error
    except Exception as error:
        raise RuntimeError(f"Unable to load model: {str(error)}") from error


# Load model with error handling
try:
    model = load_model()
except RuntimeError as e:
    app.logger.error(f"Failed to load model: {e}")
    model = None


def load_model_metadata() -> dict[str, Any]:
    """Load optional training metrics without preventing prediction on older models."""
    try:
        if MODEL_METADATA_PATH.exists():
            metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
            app.logger.info("Model metadata loaded successfully")
            return metadata
        else:
            app.logger.info("No model metadata file found")
            return {"feature_importance": [], "model_comparison": {}}
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as e:
        app.logger.warning(f"Could not load model metadata: {e}")
        return {"feature_importance": [], "model_comparison": {}}


model_metadata = load_model_metadata()


# ============================================================================
# Database Management
# ============================================================================

def get_database_connection() -> sqlite3.Connection:
    """Create a SQLite connection with named-row access."""
    connection = sqlite3.connect(app.config["DATABASE"])
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    """Create application tables required for authentication and history."""
    with get_database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                crop TEXT NOT NULL,
                soil_score INTEGER NOT NULL CHECK (soil_score BETWEEN 0 AND 100),
                fertilizer TEXT NOT NULL,
                N REAL,
                P REAL,
                K REAL,
                ph REAL,
                temperature REAL,
                humidity REAL,
                rainfall REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
        )
        
        # Check and update schema for existing databases
        history_columns = {row["name"] for row in connection.execute("PRAGMA table_info(history)")}
        
        if "created_at" not in history_columns:
            app.logger.info("Adding created_at column to history table")
            connection.execute("ALTER TABLE history ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
        
        if "user_id" not in history_columns:
            app.logger.info("Adding user_id column to history table")
            connection.execute("ALTER TABLE history ADD COLUMN user_id INTEGER REFERENCES users(id)")
        
        if "N" not in history_columns:
            app.logger.info("Adding nutrient columns to history table")
            connection.execute("ALTER TABLE history ADD COLUMN N REAL")
            connection.execute("ALTER TABLE history ADD COLUMN P REAL")
            connection.execute("ALTER TABLE history ADD COLUMN K REAL")
            connection.execute("ALTER TABLE history ADD COLUMN ph REAL")
            connection.execute("ALTER TABLE history ADD COLUMN temperature REAL")
            connection.execute("ALTER TABLE history ADD COLUMN humidity REAL")
            connection.execute("ALTER TABLE history ADD COLUMN rainfall REAL")
        
        connection.commit()


# ============================================================================
# Authentication Decorator
# ============================================================================

def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    """Redirect anonymous users to login before accessing protected routes."""
    @wraps(view)
    def wrapped_view(*args: Any, **kwargs: Any) -> Any:
        if "user_id" not in session:
            flash("Please log in to access the soil health system.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


# ============================================================================
# Form Validation
# ============================================================================

def validate_registration_form(form: Any) -> dict[str, str]:
    """Validate and normalize registration fields before database insertion."""
    values = {
        "full_name": form.get("full_name", "").strip(),
        "email": form.get("email", "").strip().lower(),
        "username": form.get("username", "").strip(),
        "password": form.get("password", ""),
        "confirm_password": form.get("confirm_password", ""),
    }
    if not all(values.values()):
        raise ValueError("All registration fields are required.")
    if len(values["full_name"]) < 2 or len(values["full_name"]) > 100:
        raise ValueError("Full name must contain 2 to 100 characters.")
    if "@" not in values["email"] or len(values["email"]) > 254:
        raise ValueError("Enter a valid email address.")
    if not values["username"].replace("_", "").isalnum() or not 3 <= len(values["username"]) <= 30:
        raise ValueError("Username must be 3 to 30 letters, numbers, or underscores.")
    if len(values["password"]) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    if values["password"] != values["confirm_password"]:
        raise ValueError("Password and confirmation password do not match.")
    return values


def parse_numeric_inputs(form: Any) -> dict[str, float]:
    """Validate all fields consumed by the crop model."""
    labels = {
        "N": "Nitrogen", 
        "P": "Phosphorus", 
        "K": "Potassium", 
        "temperature": "Temperature", 
        "humidity": "Humidity", 
        "ph": "pH", 
        "rainfall": "Rainfall"
    }
    values: dict[str, float] = {}
    
    for field, label in labels.items():
        raw_value = form.get(field, "").strip()
        if not raw_value:
            raise ValueError(f"{label} is required.")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"{label} must be a valid number.") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be a non-negative finite number.")
        values[field] = value
    
    if values["humidity"] > 100:
        raise ValueError("Humidity must be between 0 and 100.")
    if values["ph"] > 14:
        raise ValueError("pH must be between 0 and 14.")
    
    return values


# ============================================================================
# Weather Service
# ============================================================================

def get_weather(city: str) -> tuple[float | None, float | None]:
    """Return city weather or no values when the optional service is unavailable."""
    if not WEATHER_API_KEY:
        app.logger.info("No weather API key configured")
        return None, None
    
    try:
        response = requests.get(
            WEATHER_API_URL, 
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"}, 
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        weather = response.json()["main"]
        temp = float(weather["temp"])
        humidity = float(weather["humidity"])
        app.logger.info(f"Weather for {city}: {temp}°C, {humidity}% humidity")
        return temp, humidity
    except requests.Timeout:
        app.logger.warning(f"Weather API timeout for '{city}'")
    except requests.RequestException as error:
        app.logger.warning(f"Weather API request failed for '{city}': {error}")
    except (KeyError, TypeError, ValueError) as error:
        app.logger.warning(f"Weather data parsing failed for '{city}': {error}")
    
    return None, None


# ============================================================================
# Email Service
# ============================================================================

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send a plain-text email over SMTP. Returns False when unavailable or on failure."""
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD):
        app.logger.info("SMTP is not configured; cannot send email")
        return False

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=REQUEST_TIMEOUT_SECONDS * 2) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, [to_email], message.as_string())
        app.logger.info(f"Sent email to {to_email}")
        return True
    except smtplib.SMTPException as error:
        app.logger.warning(f"Failed to send email to {to_email}: {error}")
        return False


def send_password_reset_email(to_email: str, reset_url: str) -> bool:
    """Send the password reset link to the user."""
    body = (
        "You requested a password reset for your SoilSense account.\n\n"
        f"Reset your password using this link: {reset_url}\n\n"
        f"This link expires in {PASSWORD_RESET_EXPIRY_MINUTES} minutes. "
        "If you did not request this, you can safely ignore this email."
    )
    return send_email(to_email, "Reset your SoilSense password", body)


# ============================================================================
# Soil Analysis Functions
# ============================================================================

def calculate_soil_score(nitrogen: float, phosphorus: float, potassium: float, ph: float) -> int:
    """Calculate soil health score based on nutrient levels and pH."""
    try:
        n_score = max(0, 100 - abs(50 - nitrogen) * 0.3)
        p_score = max(0, 100 - abs(40 - phosphorus) * 0.3)
        k_score = max(0, 100 - abs(40 - potassium) * 0.3)
        score = (n_score + p_score + k_score) / 3
        
        if not 5.5 <= ph <= 7.5:
            score -= 15
        
        return max(0, min(100, round(score)))
    except Exception as e:
        app.logger.error(f"Error calculating soil score: {e}")
        return 50


def get_soil_status(score: int) -> str:
    """Get soil health status based on score."""
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Average"
    return "Poor"


def get_fertilizer_recommendation(nitrogen: float, phosphorus: float, potassium: float) -> str:
    """Get fertilizer recommendations based on nutrient levels."""
    fertilizers = []
    if nitrogen < 50:
        fertilizers.append("Nitrogen Fertilizer")
    if phosphorus < 40:
        fertilizers.append("Phosphorus Fertilizer")
    if potassium < 40:
        fertilizers.append("Potassium Fertilizer")
    return ", ".join(fertilizers) if fertilizers else "No fertilizer required"


def get_top_crop_recommendations(features: np.ndarray, prediction: str) -> list[dict[str, str | float]]:
    """Get top crop recommendations with probabilities."""
    try:
        if model is None:
            app.logger.warning("Model not loaded, returning single prediction")
            return [{"crop": str(prediction), "probability": 100.0}]
        
        if not hasattr(model, 'predict_proba'):
            app.logger.warning("Model doesn't have predict_proba method")
            return [{"crop": str(prediction), "probability": 100.0}]
        
        probabilities = model.predict_proba(features)[0]
        
        if not hasattr(model, 'classes_'):
            app.logger.warning("Model doesn't have classes_ attribute")
            return [{"crop": str(prediction), "probability": 100.0}]
        
        if len(probabilities) != len(model.classes_):
            app.logger.warning(f"Length mismatch: {len(probabilities)} probs vs {len(model.classes_)} classes")
            return [{"crop": str(prediction), "probability": 100.0}]
        
        indices = np.argsort(probabilities)[::-1][:3]
        
        results = []
        for index in indices:
            crop_name = str(model.classes_[index])
            prob = round(float(probabilities[index]) * 100, 2)
            results.append({"crop": crop_name, "probability": prob})
        
        app.logger.info(f"Top crops: {results}")
        return results
        
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        app.logger.warning(f"Unable to get crop probabilities: {error}")
        return [{"crop": str(prediction), "probability": 100.0}]
    except Exception as error:
        app.logger.error(f"Unexpected error in get_top_crop_recommendations: {error}")
        return [{"crop": str(prediction), "probability": 100.0}]


def get_nutrient_analysis(values: dict[str, float]) -> list[dict[str, str]]:
    """Describe nutrient and pH status using the system's recommendation thresholds."""
    checks = {
        "Nitrogen": (values["N"], 50, "Nitrogen Fertilizer"),
        "Phosphorus": (values["P"], 40, "Phosphorus Fertilizer"),
        "Potassium": (values["K"], 40, "Potassium Fertilizer"),
    }
    analysis = []
    for nutrient, (value, target, fertilizer) in checks.items():
        if value < target:
            analysis.append({
                "nutrient": nutrient, 
                "status": "Deficient", 
                "detail": f"{target - value:.1f} units below the target; consider {fertilizer}."
            })
        else:
            analysis.append({
                "nutrient": nutrient, 
                "status": "Adequate", 
                "detail": "Within the current recommendation threshold."
            })
    
    ph_status = "Adequate" if 5.5 <= values["ph"] <= 7.5 else "Needs adjustment"
    ph_detail = (
        "Within the preferred range for most crops." 
        if ph_status == "Adequate" 
        else "Outside the preferred 5.5–7.5 range; test soil before applying amendments."
    )
    analysis.append({"nutrient": "pH", "status": ph_status, "detail": ph_detail})
    return analysis


def get_fertilizer_plan(values: dict[str, float]) -> dict[str, Any]:
    """Produce conservative nutrient guidance; field testing determines final application rates."""
    deficits = {
        "Nitrogen": max(0.0, 50 - values["N"]), 
        "Phosphorus": max(0.0, 40 - values["P"]), 
        "Potassium": max(0.0, 40 - values["K"])
    }
    recommendations = []
    organic_options = {
        "Nitrogen": "well-composted manure or neem cake", 
        "Phosphorus": "bone meal or rock phosphate", 
        "Potassium": "wood ash or compost"
    }
    for nutrient, deficit in deficits.items():
        if deficit:
            recommendations.append({
                "nutrient": nutrient, 
                "deficit": round(deficit, 1), 
                "quantity": f"Correct a {deficit:.1f}-unit soil-test deficit", 
                "organic_alternative": organic_options[nutrient]
            })
    return {
        "items": recommendations, 
        "explanation": "Apply quantities only after confirming local soil-test guidance; crop type, area, and fertilizer grade determine field rates."
    }


def get_water_recommendation(values: dict[str, float]) -> dict[str, Any]:
    """Estimate an irrigation need index from heat, humidity, and recent rainfall."""
    heat_factor = max(0.0, values["temperature"] - 20) * 0.18
    dry_air_factor = max(0.0, 70 - values["humidity"]) * 0.04
    rainfall_offset = min(values["rainfall"] / 25, 2.5)
    daily_litres_per_sq_m = round(max(1.5, 3.5 + heat_factor + dry_air_factor - rainfall_offset), 1)
    schedule = (
        "Irrigate lightly every 2–3 days and reassess soil moisture." 
        if rainfall_offset >= 1 
        else "Check soil moisture daily; irrigate early morning when the root zone begins to dry."
    )
    return {
        "daily_litres_per_sq_m": daily_litres_per_sq_m, 
        "schedule": schedule, 
        "explanation": "This estimate uses submitted temperature, humidity, and rainfall as a planning aid, not a replacement for field moisture measurement."
    }


def get_yield_potential(score: int, top_probability: float) -> dict[str, Any]:
    """Return a yield-potential index because the current dataset has no yield target column."""
    potential = round(score * 0.7 + top_probability * 0.3, 1)
    confidence = "High" if top_probability >= 75 else "Moderate" if top_probability >= 50 else "Low"
    return {
        "potential_index": potential, 
        "confidence": confidence, 
        "explanation": "Potential combines soil-health score and model crop confidence. Actual yield requires local yield records, area, cultivar, and management data."
    }


def build_crop_explanations(values: dict[str, float], top_crops: list[dict[str, str | float]]) -> list[dict[str, Any]]:
    """Convert model probabilities and field inputs into plain-language guidance."""
    explanations = []
    
    try:
        nutrient_summary = (
            "balanced NPK levels" 
            if values["N"] >= 50 and values["P"] >= 40 and values["K"] >= 40 
            else "nutrient levels that need targeted improvement"
        )
        ph_summary = (
            "a suitable pH range" 
            if 5.5 <= values["ph"] <= 7.5 
            else "a pH level that may need correction"
        )
        climate_summary = (
            f"{values['temperature']:.1f}°C temperature, "
            f"{values['humidity']:.0f}% humidity, "
            f"and {values['rainfall']:.1f} mm rainfall"
        )
        
        for crop in top_crops:
            crop_name = str(crop.get("crop", "Unknown"))
            confidence = float(crop.get("probability", 0))
            
            risks = []
            if values["ph"] < 5.5 or values["ph"] > 7.5:
                risks.append("soil pH is outside the preferred general range")
            if values["rainfall"] < 50:
                risks.append("low rainfall may increase irrigation needs")
            if values["humidity"] > 85:
                risks.append("high humidity can increase fungal disease pressure")
            
            explanations.append({
                "crop": crop_name,
                "confidence": confidence,
                "reason": (
                    f"{crop_name.title()} matches the model's learned patterns for "
                    f"{nutrient_summary}, {ph_summary}, and {climate_summary}."
                ),
                "benefits": "Use soil testing, mulching, and timely irrigation to support healthy crop establishment.",
                "risks": risks or ["Continue monitoring soil moisture and nutrient levels through the growing season."],
            })
    except Exception as e:
        app.logger.error(f"Error building crop explanations: {e}")
        explanations = [{
            "crop": str(top_crops[0].get("crop", "Unknown")) if top_crops else "Unknown",
            "confidence": 0,
            "reason": "Unable to generate detailed explanation due to an error.",
            "benefits": "General soil care recommended.",
            "risks": ["Consult local agricultural expert for specific guidance."],
        }]
    
    return explanations


def get_smart_suggestions(values: dict[str, float]) -> list[str]:
    """Return practical, low-risk sustainable farming suggestions."""
    suggestions = [
        "Use mulch or crop residue to reduce evaporation and protect soil structure.",
        "Recheck soil nutrients before each major fertilizer application."
    ]
    if values["N"] < 50:
        suggestions.append("Add compost, green manure, or neem cake gradually to improve nitrogen availability.")
    if values["rainfall"] < 50:
        suggestions.append("Prefer drip irrigation or early-morning watering to reduce water loss.")
    if values["ph"] < 5.5:
        suggestions.append("Discuss agricultural lime application with a local soil laboratory before correcting acidic soil.")
    elif values["ph"] > 7.5:
        suggestions.append("Use organic matter and seek local agronomy advice before correcting alkaline soil.")
    return suggestions


# ============================================================================
# Chatbot Knowledge Base
# ============================================================================

def answer_agriculture_question(message: str) -> str:
    """Provide concise local guidance without depending on an external AI service."""
    question = message.lower()
    knowledge = [
        (
            ("soil", "ph", "nutrient"), 
            "For soil health, begin with a soil test. Keep pH in a crop-appropriate range, add organic matter, and correct nutrient shortages gradually."
        ),
        (
            ("fertilizer", "manure", "compost"), 
            "Apply fertilizer according to a soil test and crop stage. Compost, farmyard manure, neem cake, and green manure can improve organic matter alongside targeted nutrients."
        ),
        (
            ("water", "irrigation", "rain"), 
            "Irrigate based on root-zone moisture, not only the calendar. Morning drip irrigation, mulch, and rainwater harvesting help save water."
        ),
        (
            ("disease", "pest", "fungus"), 
            "For crop diseases, inspect plants early, remove badly affected material safely, improve airflow, and use local agricultural-extension guidance before applying treatments."
        ),
        (
            ("weather", "temperature", "humidity"), 
            "Weather affects irrigation and disease risk. High humidity can encourage fungal disease, while hot dry conditions increase water demand."
        ),
        (
            ("crop", "select", "recommend"), 
            "Choose crops using soil nutrients, pH, local climate, water availability, season, and market access. Use the New Prediction page for a data-driven starting point."
        ),
    ]
    for keywords, response in knowledge:
        if any(keyword in question for keyword in keywords):
            return response
    return "I can help with soil health, fertilizers, irrigation, crop diseases, weather, and crop selection. Please ask a specific farming question."


# ============================================================================
# Database Helpers
# ============================================================================

def save_prediction(crop: str, score: int, fertilizer: str, user_id: int = None, values: dict = None) -> None:
    """Save prediction to history database."""
    with get_database_connection() as connection:
        if values:
            connection.execute(
                """INSERT INTO history 
                   (user_id, crop, soil_score, fertilizer, N, P, K, ph, temperature, humidity, rainfall, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (user_id, crop, score, fertilizer, 
                 values.get('N'), values.get('P'), values.get('K'),
                 values.get('ph'), values.get('temperature'), 
                 values.get('humidity'), values.get('rainfall'))
            )
        else:
            connection.execute(
                """INSERT INTO history (user_id, crop, soil_score, fertilizer, created_at) 
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (user_id, crop, score, fertilizer)
            )
        connection.commit()


def get_dashboard_metrics(user_id: int = None) -> dict[str, Any]:
    """Return summary values for the authenticated dashboard."""
    with get_database_connection() as connection:
        prediction_summary = connection.execute(
            """SELECT COUNT(*) AS total_predictions,
                      COALESCE(ROUND(AVG(soil_score)), 0) AS average_soil_score,
                      SUM(CASE WHEN date(created_at) = date('now', 'localtime') THEN 1 ELSE 0 END) AS today_predictions
               FROM history
               WHERE (user_id = ? OR ? IS NULL)""",
            (user_id, user_id)
        ).fetchone()
        
        latest_prediction = connection.execute(
            """SELECT crop, soil_score, created_at 
               FROM history 
               WHERE (user_id = ? OR ? IS NULL)
               ORDER BY id DESC LIMIT 1""",
            (user_id, user_id)
        ).fetchone()
        
        user_total = connection.execute("SELECT COUNT(*) AS total_users FROM users").fetchone()

    return {
        "total_predictions": prediction_summary["total_predictions"] or 0,
        "average_soil_score": prediction_summary["average_soil_score"] or 0,
        "today_predictions": prediction_summary["today_predictions"] or 0,
        "recommended_crop": latest_prediction["crop"] if latest_prediction else "No data yet",
        "latest_prediction": latest_prediction,
        "registered_users": user_total["total_users"],
        "weather_status": "Ready for city lookup" if WEATHER_API_KEY else "Manual weather input",
    }


def format_relative_time(timestamp_str: str) -> str:
    """Convert a stored UTC timestamp into a short, human-friendly relative label."""
    try:
        then = datetime.fromisoformat(timestamp_str)
    except (TypeError, ValueError):
        return timestamp_str or ""

    seconds = (datetime.utcnow() - then).total_seconds()

    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if seconds < 172800:
        return "Yesterday"
    if seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} days ago"
    return then.strftime("%b %d, %Y")


def get_recent_notifications(user_id: int, lang: str) -> list[dict[str, Any]]:
    """Return the user's most recent predictions for the notification dropdown."""
    with get_database_connection() as connection:
        rows = connection.execute(
            """SELECT id, crop, soil_score, created_at
               FROM history
               WHERE user_id = ?
               ORDER BY id DESC LIMIT 5""",
            (user_id,)
        ).fetchall()

    notifications = []
    for row in rows:
        crop_name = translate_crop(row["crop"], lang)
        if row["soil_score"] >= 60:
            message = translate("notification_good_score", lang).format(crop=crop_name, score=row["soil_score"])
            icon, tone = "bi-check-circle-fill", "success"
        else:
            message = translate("notification_low_score", lang).format(crop=crop_name, score=row["soil_score"])
            icon, tone = "bi-exclamation-triangle-fill", "warning"
        notifications.append({
            "id": row["id"],
            "message": message,
            "icon": icon,
            "tone": tone,
            "created_at": format_relative_time(row["created_at"]),
        })
    return notifications


@app.context_processor
def inject_notifications() -> dict[str, Any]:
    """Make recent-activity notifications available to every template via base.html."""
    user_id = session.get("user_id")
    if not user_id:
        return {"notifications": []}
    try:
        return {"notifications": get_recent_notifications(user_id, session.get("language", "en"))}
    except sqlite3.Error:
        app.logger.exception("Failed to load notifications")
        return {"notifications": []}


@app.context_processor
def inject_translations() -> dict[str, Any]:
    """Expose the t() translation helper and language list to every template."""
    lang = session.get("language", "en")
    return {
        "t": lambda key: translate(key, lang),
        "tf": lambda text: translate_fertilizer(text, lang),
        "tc": lambda crop: translate_crop(crop, lang),
        "current_language": lang,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.route("/set-language/<lang_code>")
def set_language(lang_code: str) -> Any:
    """Store the farmer's chosen language and return to the referring page."""
    if lang_code in SUPPORTED_LANGUAGES:
        session["language"] = lang_code
    return redirect(request.referrer or url_for("dashboard"))


def safe_next_url() -> str:
    """Allow login redirects only to local application paths."""
    next_url = request.form.get("next") or request.args.get("next")
    return next_url if next_url and next_url.startswith("/") and not next_url.startswith("//") else url_for("dashboard")


# ============================================================================
# Initialize Database
# ============================================================================

initialize_database()


# ============================================================================
# Authentication Routes
# ============================================================================

@app.route("/register", methods=["GET", "POST"])
def register() -> Any:
    """Handle user registration."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        try:
            values = validate_registration_form(request.form)
            with get_database_connection() as connection:
                connection.execute(
                    "INSERT INTO users (full_name, email, username, password_hash) VALUES (?, ?, ?, ?)",
                    (values["full_name"], values["email"], values["username"], 
                     generate_password_hash(values["password"]))
                )
                connection.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except ValueError as error:
            flash(str(error), "danger")
        except sqlite3.IntegrityError:
            flash("That username or email address is already registered.", "danger")
        except sqlite3.Error:
            app.logger.exception("Registration failed")
            flash("Unable to create your account. Please try again.", "danger")
    
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    """Handle user login."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")
        
        if not identifier or not password:
            flash("Username or email and password are required.", "danger")
        else:
            try:
                with get_database_connection() as connection:
                    user = connection.execute(
                        "SELECT id, full_name, username, password_hash FROM users WHERE username = ? OR email = ?",
                        (identifier, identifier)
                    ).fetchone()
                
                if user and check_password_hash(user["password_hash"], password):
                    session.clear()
                    session.update(
                        user_id=user["id"], 
                        username=user["username"], 
                        full_name=user["full_name"]
                    )
                    session.permanent = request.form.get("remember_me") == "on"
                    flash("Welcome back!", "success")
                    return redirect(safe_next_url())
                
                flash("Invalid username/email or password.", "danger")
            except sqlite3.Error:
                app.logger.exception("Login failed")
                flash("Unable to log in right now. Please try again.", "danger")
    
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout() -> Any:
    """Handle user logout."""
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password() -> Any:
    """Request a password reset email."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if not email:
            flash("Enter your account email address.", "danger")
            return render_template("forgot_password.html")

        try:
            with get_database_connection() as connection:
                user = connection.execute(
                    "SELECT id, email FROM users WHERE email = ?", (email,)
                ).fetchone()

                if user:
                    token = secrets.token_urlsafe(32)
                    expires_at = (
                        datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_EXPIRY_MINUTES)
                    ).isoformat()
                    connection.execute(
                        "INSERT INTO password_resets (user_id, token, expires_at) VALUES (?, ?, ?)",
                        (user["id"], token, expires_at)
                    )
                    connection.commit()
                    reset_url = url_for("reset_password", token=token, _external=True)
                    if not send_password_reset_email(user["email"], reset_url):
                        flash(
                            "Email delivery is not configured on this server. "
                            f"Use this link to reset your password: {reset_url}",
                            "warning"
                        )
        except sqlite3.Error:
            app.logger.exception("Password reset request failed")

        flash("If that email is registered, a password reset link has been sent.", "success")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str) -> Any:
    """Reset a password using a valid, unexpired token."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    with get_database_connection() as connection:
        reset_row = connection.execute(
            "SELECT id, user_id, expires_at, used FROM password_resets WHERE token = ?",
            (token,)
        ).fetchone()

    token_valid = (
        reset_row is not None
        and not reset_row["used"]
        and datetime.fromisoformat(reset_row["expires_at"]) > datetime.utcnow()
    )

    if not token_valid:
        flash("This password reset link is invalid or has expired.", "danger")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(new_password) < 8:
            flash("Password must contain at least 8 characters.", "danger")
        elif new_password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            try:
                with get_database_connection() as connection:
                    connection.execute(
                        "UPDATE users SET password_hash = ? WHERE id = ?",
                        (generate_password_hash(new_password), reset_row["user_id"])
                    )
                    connection.execute(
                        "UPDATE password_resets SET used = 1 WHERE id = ?",
                        (reset_row["id"],)
                    )
                    connection.commit()
                flash("Your password has been reset. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.Error:
                app.logger.exception("Password reset failed")
                flash("Unable to reset your password. Please try again.", "danger")

    return render_template("reset_password.html", token=token)


@app.route("/terms")
def terms() -> str:
    """Render the Terms of Service page."""
    return render_template("terms.html")


@app.route("/privacy")
def privacy() -> str:
    """Render the Privacy Policy page."""
    return render_template("privacy.html")


# ============================================================================
# Main Application Routes
# ============================================================================

@app.route("/")
@login_required
def home() -> str:
    """Render the home/prediction form page."""
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard() -> str:
    """Render the signed-in user's analytics overview."""
    try:
        user_id = session.get("user_id")
        metrics = get_dashboard_metrics(user_id)
        return render_template("dashboard.html", metrics=metrics)
    except sqlite3.Error:
        app.logger.exception("Dashboard metrics could not be loaded")
        flash("Dashboard data is temporarily unavailable.", "warning")
        return render_template("dashboard.html", metrics={
            "total_predictions": 0,
            "average_soil_score": 0,
            "today_predictions": 0,
            "recommended_crop": "No data yet",
            "latest_prediction": None,
            "registered_users": 0,
            "weather_status": "Unavailable",
        })


@app.route("/history")
@login_required
def history() -> str:
    """Display prediction history for the current user."""
    try:
        user_id = session.get("user_id")
        today_str = date.today().isoformat()
        
        with get_database_connection() as connection:
            predictions = connection.execute(
                """SELECT id, crop, soil_score, fertilizer, N, P, K, ph, 
                   temperature, humidity, rainfall, created_at
                   FROM history 
                   WHERE user_id = ?
                   ORDER BY created_at DESC""",
                (user_id,)
            ).fetchall()
        
        history_list = []
        for pred in predictions:
            history_list.append({
                "id": pred["id"],
                "crop": pred["crop"],
                "soil_score": pred["soil_score"],
                "fertilizer": pred["fertilizer"],
                "N": pred["N"],
                "P": pred["P"],
                "K": pred["K"],
                "ph": pred["ph"],
                "temperature": pred["temperature"],
                "humidity": pred["humidity"],
                "rainfall": pred["rainfall"],
                "created_at": pred["created_at"],
                "status": get_soil_status(pred["soil_score"])
            })
        
        if history_list:
            stats = {
                "total": len(history_list),
                "average_score": round(sum(p["soil_score"] for p in history_list) / len(history_list)),
                "best_score": max(p["soil_score"] for p in history_list),
                "recent_crop": history_list[0]["crop"],
                "today_count": sum(1 for p in history_list if p["created_at"] and p["created_at"][:10] == today_str)
            }
        else:
            stats = {
                "total": 0, "average_score": 0, "best_score": 0,
                "recent_crop": "None", "today_count": 0
            }
        
        return render_template("history.html", predictions=history_list, stats=stats, today_date=today_str)
        
    except sqlite3.Error:
        app.logger.exception("Failed to load history")
        flash("Unable to load prediction history.", "danger")
        return render_template("history.html", predictions=[], stats={
            "total": 0, "average_score": 0, "best_score": 0, "recent_crop": "None", "today_count": 0
        }, today_date=date.today().isoformat())


@app.route("/report/<int:prediction_id>")
@login_required
def report(prediction_id: int) -> str:
    """Display detailed report for a specific prediction."""
    try:
        user_id = session.get("user_id")
        with get_database_connection() as connection:
            prediction = connection.execute(
                """SELECT id, crop, soil_score, fertilizer, N, P, K, ph, 
                   temperature, humidity, rainfall, created_at
                   FROM history 
                   WHERE id = ? AND user_id = ?""",
                (prediction_id, user_id)
            ).fetchone()
        
        if not prediction:
            flash("Prediction not found or access denied.", "warning")
            return redirect(url_for("history"))
        
        values = {
            "N": prediction["N"] or 50,
            "P": prediction["P"] or 40,
            "K": prediction["K"] or 40,
            "ph": prediction["ph"] or 7.0,
            "temperature": prediction["temperature"] or 25.0,
            "humidity": prediction["humidity"] or 60.0,
            "rainfall": prediction["rainfall"] or 100.0
        }
        
        features = np.array([[
            values["N"], values["P"], values["K"],
            values["temperature"], values["humidity"],
            values["ph"], values["rainfall"]
        ]], dtype=float)
        
        prediction_text = prediction["crop"]
        score = prediction["soil_score"]
        fertilizer = prediction["fertilizer"]
        soil_status = get_soil_status(score)
        
        top_crops = get_top_crop_recommendations(features, prediction_text)
        top_probability = float(top_crops[0]["probability"]) if top_crops else 50.0
        
        nutrient_analysis = get_nutrient_analysis(values)
        fertilizer_plan = get_fertilizer_plan(values)
        water_recommendation = get_water_recommendation(values)
        yield_potential = get_yield_potential(score, top_probability)
        crop_explanations = build_crop_explanations(values, top_crops)
        smart_suggestions = get_smart_suggestions(values)
        
        return render_template(
            "report.html",
            prediction=prediction,
            prediction_text=prediction_text,
            score=score,
            soil_status=soil_status,
            fertilizer=fertilizer,
            top_crops=top_crops,
            nutrient_analysis=nutrient_analysis,
            fertilizer_plan=fertilizer_plan,
            water_recommendation=water_recommendation,
            yield_potential=yield_potential,
            crop_explanations=crop_explanations,
            smart_suggestions=smart_suggestions,
            values=values,
            model_metadata=model_metadata
        )
        
    except sqlite3.Error:
        app.logger.exception("Failed to load report")
        flash("Unable to load the prediction report.", "danger")
        return redirect(url_for("history"))
    except Exception as e:
        app.logger.exception("Error generating report")
        flash(f"Error generating report: {str(e)}", "danger")
        return redirect(url_for("history"))


@app.route("/analytics")
@login_required
def analytics() -> str:
    """Render the analytics page."""
    try:
        user_id = session.get("user_id")
        with get_database_connection() as connection:
            # Get all predictions for charts
            predictions = connection.execute(
                """SELECT crop, soil_score, created_at 
                   FROM history 
                   WHERE user_id = ?
                   ORDER BY created_at ASC""",
                (user_id,)
            ).fetchall()
            
            # Get crop distribution
            crop_stats = connection.execute(
                """SELECT crop, COUNT(*) as count, ROUND(AVG(soil_score), 1) as avg_score
                   FROM history 
                   WHERE user_id = ?
                   GROUP BY crop
                   ORDER BY count DESC""",
                (user_id,)
            ).fetchall()
            
            # Get monthly stats
            monthly_stats = connection.execute(
                """SELECT strftime('%Y-%m', created_at) as month,
                          COUNT(*) as count,
                          ROUND(AVG(soil_score), 1) as avg_score
                   FROM history 
                   WHERE user_id = ?
                   GROUP BY month
                   ORDER BY month ASC
                   LIMIT 12""",
                (user_id,)
            ).fetchall()
        
        analytics_data = {
            "total_predictions": len(predictions),
            "crop_stats": [dict(row) for row in crop_stats],
            "monthly_stats": [dict(row) for row in monthly_stats],
            "predictions": [dict(row) for row in predictions]
        }
        
        return render_template("analytics.html", data=analytics_data)
        
    except sqlite3.Error:
        app.logger.exception("Failed to load analytics")
        flash("Unable to load analytics data.", "warning")
        return render_template("analytics.html", data={
            "total_predictions": 0, "crop_stats": [], "monthly_stats": [], "predictions": []
        })


@app.route("/reports")
@login_required
def reports() -> str:
    """Render the reports overview page."""
    try:
        user_id = session.get("user_id")
        with get_database_connection() as connection:
            recent_predictions = connection.execute(
                """SELECT id, crop, soil_score, fertilizer, created_at
                   FROM history 
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT 20""",
                (user_id,)
            ).fetchall()
        
        return render_template("reports.html", predictions=recent_predictions)
        
    except sqlite3.Error:
        app.logger.exception("Failed to load reports")
        flash("Unable to load reports.", "warning")
        return render_template("reports.html", predictions=[])


@app.route("/profile")
@login_required
def profile() -> str:
    """Render the user profile page."""
    try:
        user_id = session.get("user_id")
        with get_database_connection() as connection:
            user = connection.execute(
                "SELECT id, full_name, email, username, created_at FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            
            stats = connection.execute(
                """SELECT COUNT(*) as total_predictions,
                          COALESCE(ROUND(AVG(soil_score)), 0) as avg_score,
                          MAX(soil_score) as best_score,
                          COUNT(DISTINCT crop) as unique_crops
                   FROM history WHERE user_id = ?""",
                (user_id,)
            ).fetchone()
        
        return render_template("profile.html", user=user, stats=stats)
        
    except sqlite3.Error:
        app.logger.exception("Failed to load profile")
        flash("Unable to load profile.", "warning")
        return redirect(url_for("dashboard"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings() -> str:
    """Render the settings page and handle profile/password updates."""
    user_id = session.get("user_id")

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_profile":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()

            if len(full_name) < 2 or len(full_name) > 100:
                flash("Full name must contain 2 to 100 characters.", "danger")
            elif "@" not in email or len(email) > 254:
                flash("Enter a valid email address.", "danger")
            else:
                try:
                    with get_database_connection() as connection:
                        connection.execute(
                            "UPDATE users SET full_name = ?, email = ? WHERE id = ?",
                            (full_name, email, user_id)
                        )
                        connection.commit()
                    session["full_name"] = full_name
                    flash("Profile updated successfully.", "success")
                except sqlite3.IntegrityError:
                    flash("That email address is already registered to another account.", "danger")
                except sqlite3.Error:
                    app.logger.exception("Profile update failed")
                    flash("Unable to update profile. Please try again.", "danger")

        elif action == "change_password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")

            with get_database_connection() as connection:
                user_row = connection.execute(
                    "SELECT password_hash FROM users WHERE id = ?", (user_id,)
                ).fetchone()

            if not user_row or not check_password_hash(user_row["password_hash"], current_password):
                flash("Current password is incorrect.", "danger")
            elif len(new_password) < 8:
                flash("New password must contain at least 8 characters.", "danger")
            elif new_password != confirm_password:
                flash("New password and confirmation do not match.", "danger")
            else:
                try:
                    with get_database_connection() as connection:
                        connection.execute(
                            "UPDATE users SET password_hash = ? WHERE id = ?",
                            (generate_password_hash(new_password), user_id)
                        )
                        connection.commit()
                    flash("Password changed successfully.", "success")
                except sqlite3.Error:
                    app.logger.exception("Password change failed")
                    flash("Unable to change password. Please try again.", "danger")

    with get_database_connection() as connection:
        user = connection.execute(
            "SELECT full_name, email, username FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    return render_template("settings.html", user=user)


@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather() -> str:
    """Look up current temperature and humidity for a city."""
    city = ""
    temperature = humidity = None
    searched = False

    if request.method == "POST":
        city = request.form.get("city", "").strip()
        searched = True
        if not city:
            flash("Enter a city name to check the weather.", "danger")
        elif not WEATHER_API_KEY:
            flash("Weather lookups require an OPENWEATHER_API_KEY to be configured on the server.", "warning")
        else:
            temperature, humidity = get_weather(city)
            if temperature is None:
                flash(f"Could not find weather data for '{city}'.", "danger")

    return render_template(
        "weather.html",
        city=city,
        temperature=temperature,
        humidity=humidity,
        searched=searched,
        weather_configured=bool(WEATHER_API_KEY),
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict() -> str:
    """Process soil data and return crop predictions."""
    try:
        app.logger.info("Starting prediction process")
        values = parse_numeric_inputs(request.form)
        app.logger.info(f"Parsed values: {values}")
        
        if model is None:
            raise RuntimeError("Prediction model is not loaded. Please contact support.")
        
        city = request.form.get("city", "").strip()
        if city:
            app.logger.info(f"Fetching weather for city: {city}")
            temperature, humidity = get_weather(city)
            if temperature is not None and humidity is not None:
                values["temperature"], values["humidity"] = temperature, humidity
                app.logger.info(f"Using weather data: {temperature}°C, {humidity}%")
        
        features = np.array([[
            values["N"], values["P"], values["K"], 
            values["temperature"], values["humidity"], 
            values["ph"], values["rainfall"]
        ]], dtype=float)
        
        app.logger.info("Making prediction...")
        prediction = str(model.predict(features)[0])
        app.logger.info(f"Predicted crop: {prediction}")
        
        score = calculate_soil_score(values["N"], values["P"], values["K"], values["ph"])
        fertilizer = get_fertilizer_recommendation(values["N"], values["P"], values["K"])
        app.logger.info(f"Soil score: {score}, Fertilizer: {fertilizer}")
        
        top_crops = get_top_crop_recommendations(features, prediction)
        app.logger.info(f"Top crops calculated: {len(top_crops)} results")
        
        top_probability = float(top_crops[0].get("probability", 50)) if top_crops else 50.0
        
        nutrient_analysis = get_nutrient_analysis(values)
        fertilizer_plan = get_fertilizer_plan(values)
        water_recommendation = get_water_recommendation(values)
        yield_potential = get_yield_potential(score, top_probability)
        crop_explanations = build_crop_explanations(values, top_crops)
        smart_suggestions = get_smart_suggestions(values)
        
        save_prediction(
            crop=prediction, 
            score=score, 
            fertilizer=fertilizer,
            user_id=session.get("user_id"),
            values=values
        )
        
        return render_template(
            "result.html",
            prediction=prediction,
            score=score,
            soil_status=get_soil_status(score),
            fertilizer=fertilizer,
            top_crops=top_crops,
            nutrient_analysis=nutrient_analysis,
            fertilizer_plan=fertilizer_plan,
            water_recommendation=water_recommendation,
            yield_potential=yield_potential,
            crop_explanations=crop_explanations,
            smart_suggestions=smart_suggestions,
            model_metadata=model_metadata,
            N=values["N"],
            P=values["P"],
            K=values["K"],
        )
        
    except ValueError as error:
        app.logger.warning(f"Validation error: {error}")
        abort(400, description=str(error))
    except RuntimeError as error:
        app.logger.error(f"Runtime error: {error}")
        abort(500, description=str(error))
    except Exception as error:
        app.logger.exception(f"Unexpected error in predict: {error}")
        abort(500, description="Unable to process the prediction. Please try again.")


@app.route("/chatbot", methods=["GET", "POST"])
@login_required
def chatbot() -> Any:
    """Render the local agriculture assistant and serve its conversation API."""
    if request.method == "GET":
        return render_template("chatbot.html", chat_history=session.get("chat_history", []))

    payload = request.get_json(silent=True) or request.form
    message = str(payload.get("message", "")).strip()
    
    if not message:
        return jsonify({"error": "Please enter a question."}), 400
    if len(message) > 500:
        return jsonify({"error": "Please limit questions to 500 characters."}), 400

    response = answer_agriculture_question(message)
    history = session.get("chat_history", [])
    history.extend([
        {"role": "user", "message": message}, 
        {"role": "assistant", "message": response}
    ])
    session["chat_history"] = history[-20:]
    session.modified = True
    
    return jsonify({"response": response, "history": session["chat_history"]})


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    app.logger.warning(f"400 error: {error}")
    try:
        return render_template("error.html", 
                              error_code=400,
                              error_message=str(error.description)), 400
    except:
        return f"<h1>400 Bad Request</h1><p>{error.description}</p>", 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    try:
        return render_template("error.html", 
                              error_code=404,
                              error_message="The page you're looking for doesn't exist."), 404
    except:
        return "<h1>404 Not Found</h1><p>The page you're looking for doesn't exist.</p>", 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    app.logger.error(f"500 error: {error}")
    try:
        return render_template("error.html", 
                              error_code=500,
                              error_message="Something went wrong. Please try again later."), 500
    except:
        return "<h1>500 Internal Server Error</h1><p>Something went wrong. Please try again later.</p>", 500


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True)
    