import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD TRAINED COMPONENTS
# =========================
model = joblib.load("model/demand_surge_model_final.pkl")
scaler = joblib.load("model/scaler.pkl")

# =========================
# FINAL FEATURE SET
# =========================
final_features = [
    'hour', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'demand_lag_1', 'demand_lag_2', 'demand_lag_48', 'demand_lag_336',
    'demand_rolling_mean_3', 'demand_rolling_mean_6',
    'demand_rolling_std_3', 'demand_rolling_max_6',
    'event_nearby', 'event_attendance',
    'minutes_to_event_start', 'minutes_to_event_end',
    'event_type_Conference', 'event_type_Festival',
    'event_type_Sports', 'event_type_none'
]

numerical_cols = [
    'hour', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'demand_lag_1', 'demand_lag_2', 'demand_lag_48', 'demand_lag_336',
    'demand_rolling_mean_3', 'demand_rolling_mean_6',
    'demand_rolling_std_3', 'demand_rolling_max_6',
    'event_nearby', 'event_attendance',
    'minutes_to_event_start', 'minutes_to_event_end'
]

# =========================
# STREAMLIT UI
# =========================
st.title("🚖 Real-Time Demand Surge Prediction App")
st.write("Provide input details to predict surge demand for the next 30 minutes.")

col1, col2 = st.columns(2)

# -------------------------
# USER INPUTS
# -------------------------
hour = col1.slider("Hour of the Day", 0, 23, 12)
is_weekend = col2.selectbox("Is Weekend?", [0, 1])
is_rush_hour = col1.selectbox("Is Rush Hour?", [0, 1])

event_nearby = col2.selectbox("Event Nearby?", [0, 1])
event_attendance = col1.number_input("Event Attendance", min_value=0, value=500)
minutes_to_event_start = col2.number_input("Minutes to Event Start", value=30)
minutes_to_event_end = col1.number_input("Minutes to Event End", value=60)

event_type = st.selectbox(
    "Event Type",
    ["Conference", "Festival", "Sports", "none"]
)

# -------------------------
# COMPUTE CYCLIC FEATURES
# -------------------------
day_of_week = 2  # Dummy fixed weekday (for cyclic transformation)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)

# -------------------------
# PLACEHOLDER HISTORICAL FEATURES
# -------------------------
demand_lag_1 = 10.0
demand_lag_2 = 9.5
demand_lag_48 = 8.8
demand_lag_336 = 7.5
demand_rolling_mean_3 = 9.4
demand_rolling_mean_6 = 9.1
demand_rolling_std_3 = 1.1
demand_rolling_max_6 = 11.0

# -------------------------
# CREATE INPUT DATAFRAME
# -------------------------
data = {
    'hour': hour,
    'is_weekend': is_weekend,
    'is_rush_hour': is_rush_hour,
    'hour_sin': hour_sin,
    'hour_cos': hour_cos,
    'day_sin': day_sin,
    'day_cos': day_cos,
    'demand_lag_1': demand_lag_1,
    'demand_lag_2': demand_lag_2,
    'demand_lag_48': demand_lag_48,
    'demand_lag_336': demand_lag_336,
    'demand_rolling_mean_3': demand_rolling_mean_3,
    'demand_rolling_mean_6': demand_rolling_mean_6,
    'demand_rolling_std_3': demand_rolling_std_3,
    'demand_rolling_max_6': demand_rolling_max_6,
    'event_nearby': event_nearby,
    'event_attendance': event_attendance,
    'minutes_to_event_start': minutes_to_event_start,
    'minutes_to_event_end': minutes_to_event_end,
    'event_type_Conference': 1 if event_type == "Conference" else 0,
    'event_type_Festival': 1 if event_type == "Festival" else 0,
    'event_type_Sports': 1 if event_type == "Sports" else 0,
    'event_type_none': 1 if event_type == "none" else 0
}

input_df = pd.DataFrame([data])
input_df = input_df.reindex(columns=final_features, fill_value=0)

# -------------------------
# SCALE + PREDICT
# -------------------------
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

prediction = model.predict(input_df)

# -------------------------
# DISPLAY OUTPUT
# -------------------------
st.subheader("Predicted Surge Demand (Next 30 min):")
st.metric(label="Predicted Demand", value=round(float(prediction[0]), 2))
