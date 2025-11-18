# fastapi_app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import time
from typing import Optional

# ---------------------------
# HARD-CODED API KEYS (backend uses these silently)
# ---------------------------
from dotenv import load_dotenv
import os

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")
# (Nominatim requires only a User-Agent header, no API key)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Demand Surge Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load model & artifacts
# Put your artifacts under ./model/
# - model/lgbm_model.txt  OR  model/lgbm_model.pkl  (if saved via joblib)
# - model/scaler.pkl
# - model/label_encoders.pkl  (dict: col -> LabelEncoder)
# - model/feature_columns.pkl  (list)
# - model/categorical_columns.pkl (list)
# ---------------------------
model = None
scaler = None
label_encoders = {}
feature_columns = []
categorical_cols = []

try:
    # Try joblib-loaded model first (sklearn wrapper or pickled model object)
    model = joblib.load("model/lgbm_model.pkl")
except Exception:
    try:
        # Try LightGBM text model (Booster)
        import lightgbm as lgb
        model = lgb.Booster(model_file="model/lgbm_model.txt")
    except Exception:
        model = None

try:
    scaler = joblib.load("model/scaler.pkl")
except Exception:
    scaler = None

try:
    label_encoders = joblib.load("model/label_encoders.pkl")
except Exception:
    label_encoders = {}

try:
    feature_columns = joblib.load("model/feature_columns.pkl")
except Exception:
    feature_columns = []

try:
    categorical_cols = joblib.load("model/categorical_columns.pkl")
except Exception:
    categorical_cols = []

# If feature_columns contains the target 'demand_surge', remove it for prediction
if "demand_surge" in feature_columns:
    feature_columns = [c for c in feature_columns if c != "demand_surge"]

# ---------------------------
# Request/Response models
# ---------------------------
class PredictRequest(BaseModel):
    address: str
    use_current_time: bool = True
    prediction_datetime: Optional[str] = None  # "YYYY-MM-DD HH:MM:SS"
    manual_lat: Optional[float] = None
    manual_lon: Optional[float] = None
    manual_temp: Optional[float] = None
    manual_humidity: Optional[float] = None
    manual_weather: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: float
    surge_level: str
    location: dict
    weather: dict
    traffic: dict
    events: dict
    timestamp: str

# ---------------------------
# Helpers
# ---------------------------
def geocode_nominatim(address: str):
    """Return (lat, lon, display_name) or raise HTTPException"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "DemandSurgeApp/1.0 (contact@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Nominatim geocoding failed")
    data = r.json()
    if not data:
        raise HTTPException(status_code=400, detail="Could not geocode address")
    # Respect rate-limit: 1 request per second recommended
    time.sleep(1)
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    display = data[0].get("display_name", "")
    return lat, lon, display

def get_openweather(lat: float, lon: float):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        # return defaults if API fails
        return {"temperature": 25.0, "humidity": 60.0, "precipitation": 0.0, "weather_condition": "Clear"}
    j = r.json()
    temp = j.get("main", {}).get("temp", 25.0)
    humidity = j.get("main", {}).get("humidity", 60.0)
    precipitation = 0.0
    if "rain" in j:
        precipitation = j["rain"].get("1h", precipitation)
    if "snow" in j:
        precipitation += j["snow"].get("1h", 0.0)
    weather_condition = j.get("weather", [{}])[0].get("main", "Clear")
    return {"temperature": temp, "humidity": humidity, "precipitation": precipitation, "weather_condition": weather_condition}

def get_tomtom_traffic(lat: float, lon: float):
    # TomTom Flow Segment endpoint
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"point": f"{lat},{lon}", "key": TOMTOM_API_KEY}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return {"avg_speed": 40.0, "traffic_ratio": 1.25}
    j = r.json()
    fs = j.get("flowSegmentData", {})
    current_speed = fs.get("currentSpeed", 40.0)  # km/h
    free_flow = fs.get("freeFlowSpeed", 50.0)
    if current_speed <= 0:
        current_speed = 40.0
    traffic_ratio = free_flow / current_speed if current_speed > 0 else 1.0
    return {"avg_speed": current_speed, "traffic_ratio": traffic_ratio}

def get_ticketmaster_events(lat: float, lon: float, radius_km: int = 10):
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {"apikey": TICKETMASTER_API_KEY, "latlong": f"{lat},{lon}", "radius": radius_km, "unit": "km", "size": 20}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return {"event_nearby": 0, "event_type": "No_Event"}
    j = r.json()
    if "_embedded" in j and "events" in j["_embedded"]:
        events = j["_embedded"]["events"]
        count = len(events)
        ev_type = "No_Event"
        try:
            if count > 0:
                ev_type = events[0]["classifications"][0]["segment"]["name"]
        except Exception:
            ev_type = "No_Event"
        return {"event_nearby": count, "event_type": ev_type}
    return {"event_nearby": 0, "event_type": "No_Event"}

def safe_label_transform(col_name: str, value):
    """Use saved LabelEncoder; unseen-> -1"""
    if not label_encoders or col_name not in label_encoders:
        # if encoder missing, return original value
        return value
    le = label_encoders[col_name]
    try:
        return int(le.transform([str(value)])[0])
    except Exception:
        return -1

def prepare_features_for_model(lat, lon, weather, traffic, events, pred_dt: datetime):
    day_of_week = pred_dt.weekday()
    hour_of_day = pred_dt.hour
    is_peak_hour = 1 if ((7 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20)) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    if 0 <= hour_of_day < 6:
        time_period = "Night"
    elif 6 <= hour_of_day < 12:
        time_period = "Morning"
    elif 12 <= hour_of_day < 18:
        time_period = "Afternoon"
    else:
        time_period = "Evening"
    weather_sev_map = {"Clear": 1, "Clouds": 2, "Rain": 3, "Storm": 4, "Snow": 4, "Thunderstorm": 4}
    weather_sev = weather_sev_map.get(weather.get("weather_condition", "Clear"), 2)
    temp = float(weather.get("temperature", 25.0))
    humidity = float(weather.get("humidity", 60.0))
    temp_humidity_interaction = (temp * humidity) / 100.0
    traffic_weather_interaction = (traffic.get("traffic_ratio", 1.0) * humidity) / 100.0
    precipitation_log = np.log1p(max(0.0, float(weather.get("precipitation", 0.0))))
    event_nearby = int(events.get("event_nearby", 0))
    event_type = events.get("event_type", "No_Event")
    has_event = 1 if event_nearby > 0 else 0
    event_peak_combo = event_nearby * is_peak_hour

    feat = {
        "latitude": lat,
        "longitude": lon,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "temperature": temp,
        "humidity": humidity,
        "weather_condition": weather.get("weather_condition", "Clear"),
        "avg_speed": float(traffic.get("avg_speed", 40.0)),
        "traffic_ratio": float(traffic.get("traffic_ratio", 1.25)),
        "event_nearby": event_nearby,
        "event_type": event_type,
        "precipitation_log": precipitation_log,
        "is_peak_hour": is_peak_hour,
        "is_weekend": is_weekend,
        "time_period": time_period,
        "weather_severity_score": weather_sev,
        "temp_humidity_interaction": temp_humidity_interaction,
        "traffic_weather_interaction": traffic_weather_interaction,
        "has_event": has_event,
        "event_peak_combo": event_peak_combo,
    }
    return feat

def encode_and_scale(feat_dict: dict):
    """Return scaled numpy array or DataFrame ready for model.predict"""
    df = pd.DataFrame([feat_dict])

    # Ensure all categorical columns exist in DF with string type
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "None"
        df[c] = df[c].astype(str)

    # Apply label encoders
    for c in categorical_cols:
        if c in df.columns and c in label_encoders:
            df[c] = df[c].map(lambda x: safe_label_transform(c, x))

    # Ensure all expected feature columns exist
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns required by model: {missing}")

    df = df[feature_columns]

    if scaler is None:
        # If no scaler, return raw df
        return df.values

    try:
        scaled = scaler.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler.transform failed: {e}")

    return scaled

def predict_with_model(X):
    """Return scalar prediction (float)"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")
    # If sklearn-like object with predict
    try:
        preds = model.predict(X)
        # preds could be array-like; take first
        if hasattr(preds, "__len__"):
            return float(preds[0])
        return float(preds)
    except Exception:
        # Try LightGBM Booster (predict expects np.array or dataset)
        try:
            preds = model.predict(X)
            if isinstance(preds, (list, np.ndarray)):
                return float(preds[0])
            return float(preds)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def root():
    return {"message": "Demand Surge Prediction API", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "time": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) geocode
    if req.manual_lat is not None and req.manual_lon is not None:
        lat = float(req.manual_lat)
        lon = float(req.manual_lon)
        formatted = req.address
    else:
        lat, lon, formatted = geocode_nominatim(req.address)

    # 2) weather
    if (req.manual_temp is not None) and (req.manual_humidity is not None) and req.manual_weather:
        weather = {
            "temperature": req.manual_temp,
            "humidity": req.manual_humidity,
            "precipitation": 0.0,
            "weather_condition": req.manual_weather
        }
    else:
        weather = get_openweather(lat, lon)

    # 3) traffic (TomTom)
    traffic = get_tomtom_traffic(lat, lon)

    # 4) events (Ticketmaster)
    events = get_ticketmaster_events(lat, lon)

    # 5) prediction datetime
    if req.use_current_time or not req.prediction_datetime:
        pred_dt = datetime.now()
    else:
        try:
            pred_dt = datetime.strptime(req.prediction_datetime, "%Y-%m-%d %H:%M:%S")
        except Exception:
            raise HTTPException(status_code=400, detail="prediction_datetime must be 'YYYY-MM-DD HH:MM:SS'")

    # 6) prepare features
    feat = prepare_features_for_model(lat, lon, weather, traffic, events, pred_dt)

    # 7) encode & scale
    X = encode_and_scale(feat)

    # 8) predict
    pred_val = predict_with_model(X)

    # 9) surge level
    if pred_val > 100:
        level = "VERY HIGH"
    elif pred_val > 70:
        level = "HIGH"
    elif pred_val > 40:
        level = "MODERATE"
    else:
        level = "LOW"

    return PredictResponse(
        prediction=pred_val,
        surge_level=level,
        location={"lat": lat, "lng": lon, "formatted_address": formatted},
        weather=weather,
        traffic=traffic,
        events=events,
        timestamp=datetime.now().isoformat(),
    )
