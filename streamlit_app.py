# streamlit_app.py
import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from datetime import datetime

# ---------------------------
# Minimal UI (Option A)
# ---------------------------
st.set_page_config(page_title="Demand Surge Predictor", layout="wide")
st.markdown("<h1 style='text-align:center'>🚗 Demand Surge Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Backend URL (local default)
FASTAPI_URL = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000")

# Connection check
try:
    h = requests.get(f"{FASTAPI_URL}/health", timeout=3)
    if h.status_code == 200:
        st.sidebar.success("API: connected")
    else:
        st.sidebar.error("API: unhealthy")
except Exception:
    st.sidebar.error("API not reachable")

st.markdown("### Enter location and time")
address = st.text_input("Address / Place / Landmark", value="Times Square, New York")
col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Date", value=datetime.now().date())
with col2:
    time_val = st.time_input("Time", value=datetime.now().time())
use_now = st.checkbox("Use current time (ignore date/time)", value=True)

if st.button("🚀 Predict Demand Surge"):
    if not address:
        st.error("Please type an address")
    else:
        # prepare request
        payload = {
            "address": address,
            "use_current_time": use_now,
            "prediction_datetime": None if use_now else datetime.combine(date, time_val).strftime("%Y-%m-%d %H:%M:%S"),
            "manual_lat": None,
            "manual_lon": None,
            "manual_temp": None,
            "manual_humidity": None,
            "manual_weather": None
        }
        try:
            r = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=30)
            if r.status_code != 200:
                st.error(f"API Error: {r.status_code} - {r.text}")
            else:
                res = r.json()
                pred = res["prediction"]
                level = res["surge_level"]
                loc = res["location"]
                weather = res["weather"]
                traffic = res["traffic"]
                events = res["events"]

                # Prediction box
                st.markdown("---")
                color = "#28a745"
                if level == "VERY HIGH":
                    color = "#dc3545"
                elif level == "HIGH":
                    color = "#fd7e14"
                elif level == "MODERATE":
                    color = "#ffc107"
                st.markdown(f"""<div style='background:{color};padding:20px;border-radius:10px;text-align:center;color:#fff;font-size:22px;'>
                                <b>Predicted Demand Surge:</b> {pred:.2f} &nbsp;&nbsp; <b>Level:</b> {level}
                              </div>""", unsafe_allow_html=True)

                # Cards
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📍 Location")
                    st.write(res["location"]["formatted_address"])
                    st.write(f"Lat: {res['location']['lat']:.6f}  Lon: {res['location']['lng']:.6f}")
                    st.subheader("🌤️ Weather")
                    st.write(f"Condition: {weather.get('weather_condition')}")
                    st.write(f"Temp: {weather.get('temperature'):.1f} °C")
                    st.write(f"Humidity: {weather.get('humidity'):.0f}%")
                with c2:
                    st.subheader("🚗 Traffic")
                    st.write(f"Avg speed: {traffic.get('avg_speed'):.1f} km/h")
                    st.write(f"Traffic ratio: {traffic.get('traffic_ratio'):.2f}")
                    st.subheader("🎫 Events")
                    st.write(f"Nearby events: {events.get('event_nearby')}")
                    st.write(f"Type: {events.get('event_type')}")

                # Map
                st.subheader("🗺️ Map")
                m = folium.Map(location=[res["location"]["lat"], res["location"]["lng"]], zoom_start=13)
                folium.Marker(
                    [res["location"]["lat"], res["location"]["lng"]],
                    popup=f"Surge: {pred:.2f}",
                    tooltip=res["location"]["formatted_address"],
                    icon=folium.Icon(color='red' if pred > 100 else 'orange' if pred > 70 else 'green')
                ).add_to(m)
                folium_static(m, width=700, height=450)
        except requests.exceptions.Timeout:
            st.error("Request timed out")
        except Exception as e:
            st.error(f"Error: {e}")
