"""
Demand Surge Prediction - Data Preprocessing Pipeline
=====================================================
This script combines trips.csv, weather.csv, and events.csv into a 
unified dataset for demand surge prediction with engineered features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEMAND SURGE PREDICTION - DATA PREPROCESSING PIPELINE")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading CSV files...")

# Load the three CSV files
trips = pd.read_csv('trips.csv')
weather = pd.read_csv('weather.csv')
events = pd.read_csv('events.csv')

print(f"✓ Trips loaded: {len(trips):,} records")
print(f"✓ Weather loaded: {len(weather):,} records")
print(f"✓ Events loaded: {len(events):,} records")

# ============================================================================
# STEP 2: PARSE TIMESTAMPS
# ============================================================================
print("\n[STEP 2] Parsing timestamps...")

trips['timestamp'] = pd.to_datetime(trips['timestamp'])
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
events['start_time'] = pd.to_datetime(events['start_time'])
events['end_time'] = pd.to_datetime(events['end_time'])

print("All timestamps parsed successfully")

# ============================================================================
# STEP 3: CREATE 30-MINUTE TIME BINS FOR TRIPS
# ============================================================================
print("\n[STEP 3] Aggregating trips into 30-minute windows...")

# Create 30-minute bins
trips['time_window'] = trips['timestamp'].dt.floor('30min')

# Aggregate demand by zone and time window
demand_df = trips.groupby(['zone_id', 'time_window']).size().reset_index(name='demand')

print(f"Created {len(demand_df):,} zone-time window combinations")
print(f"Date range: {demand_df['time_window'].min()} to {demand_df['time_window'].max()}")

# ============================================================================
# STEP 4: CREATE COMPLETE TIME GRID (Fill missing intervals with 0 demand)
# ============================================================================
print("\n[STEP 4] Creating complete time grid...")

# Get all unique zones and time windows
all_zones = trips['zone_id'].unique()
min_time = demand_df['time_window'].min()
max_time = demand_df['time_window'].max()

# Create complete time range (30-min intervals)
time_range = pd.date_range(start=min_time, end=max_time, freq='30min')

# Create all combinations of zones and time windows
from itertools import product
complete_grid = pd.DataFrame(
    list(product(all_zones, time_range)),
    columns=['zone_id', 'time_window']
)

# Merge with actual demand (fill missing with 0)
demand_df = complete_grid.merge(demand_df, on=['zone_id', 'time_window'], how='left')
demand_df['demand'] = demand_df['demand'].fillna(0).astype(int)

print(f"Complete grid created: {len(demand_df):,} records")
print(f"  Zones: {len(all_zones)}, Time windows: {len(time_range)}")

# ============================================================================
# STEP 5: EXTRACT TEMPORAL FEATURES
# ============================================================================
print("\n[STEP 5] Extracting temporal features...")

demand_df['hour'] = demand_df['time_window'].dt.hour
demand_df['day_of_week'] = demand_df['time_window'].dt.dayofweek
demand_df['day'] = demand_df['time_window'].dt.day
demand_df['month'] = demand_df['time_window'].dt.month
demand_df['is_weekend'] = (demand_df['day_of_week'] >= 5).astype(int)
demand_df['is_rush_hour'] = ((demand_df['hour'] >= 7) & (demand_df['hour'] <= 9) | 
                               (demand_df['hour'] >= 17) & (demand_df['hour'] <= 19)).astype(int)

# Cyclical encoding for hour
demand_df['hour_sin'] = np.sin(2 * np.pi * demand_df['hour'] / 24)
demand_df['hour_cos'] = np.cos(2 * np.pi * demand_df['hour'] / 24)

# Cyclical encoding for day of week
demand_df['day_sin'] = np.sin(2 * np.pi * demand_df['day_of_week'] / 7)
demand_df['day_cos'] = np.cos(2 * np.pi * demand_df['day_of_week'] / 7)

print("Temporal features extracted:")
print("  - hour, day_of_week, month, day")
print("  - is_weekend, is_rush_hour")
print("  - Cyclical encodings (hour_sin, hour_cos, day_sin, day_cos)")

# ============================================================================
# STEP 6: CREATE LAG FEATURES (Historical Demand)
# ============================================================================
print("\n[STEP 6] Creating lag features...")

# Sort by zone and time
demand_df = demand_df.sort_values(['zone_id', 'time_window'])

# Create lag features for each zone
lag_features = []
for zone in all_zones:
    zone_data = demand_df[demand_df['zone_id'] == zone].copy()
    
    # Lag features (previous demand)
    zone_data['demand_lag_1'] = zone_data['demand'].shift(1)  # 30 min ago
    zone_data['demand_lag_2'] = zone_data['demand'].shift(2)  # 1 hour ago
    zone_data['demand_lag_48'] = zone_data['demand'].shift(48)  # 24 hours ago (same time yesterday)
    zone_data['demand_lag_336'] = zone_data['demand'].shift(336)  # 1 week ago (same time last week)
    
    # Rolling statistics (moving averages and std dev)
    zone_data['demand_rolling_mean_3'] = zone_data['demand'].rolling(window=3, min_periods=1).mean()
    zone_data['demand_rolling_mean_6'] = zone_data['demand'].rolling(window=6, min_periods=1).mean()
    zone_data['demand_rolling_std_3'] = zone_data['demand'].rolling(window=3, min_periods=1).std()
    zone_data['demand_rolling_max_6'] = zone_data['demand'].rolling(window=6, min_periods=1).max()
    
    lag_features.append(zone_data)

demand_df = pd.concat(lag_features, ignore_index=True)

print("Lag features created:")
print("  - demand_lag_1 (30 min ago)")
print("  - demand_lag_2 (1 hour ago)")
print("  - demand_lag_48 (24 hours ago)")
print("  - demand_lag_336 (1 week ago)")
print("  - Rolling mean (3, 6 windows)")
print("  - Rolling std (3 windows)")
print("  - Rolling max (6 windows)")

# ============================================================================
# STEP 7: MERGE WEATHER DATA
# ============================================================================
print("\n[STEP 7] Merging weather data...")

# Round weather timestamps to 30-min to match time windows
weather['time_window'] = weather['timestamp'].dt.floor('30min')

# Select relevant weather columns
weather_cols = ['time_window', 'temperature', 'precipitation', 
                'weather_condition', 'wind_speed', 'humidity', 'feels_like_temp']
weather_subset = weather[weather_cols].drop_duplicates(subset=['time_window'])

# Check weather data coverage
print(f"  Weather date range: {weather_subset['time_window'].min()} to {weather_subset['time_window'].max()}")
print(f"  Demand date range: {demand_df['time_window'].min()} to {demand_df['time_window'].max()}")

# Merge weather with demand data
demand_df = demand_df.merge(weather_subset, on='time_window', how='left')

# Count missing weather values before filling
missing_weather = demand_df['temperature'].isna().sum()
print(f"  Missing weather records: {missing_weather:,} ({missing_weather/len(demand_df)*100:.2f}%)")

# Fill missing weather data using multiple strategies
if missing_weather > 0:
    print("  Applying missing data interpolation...")
    
    # Sort by time to ensure proper forward/backward fill
    demand_df = demand_df.sort_values('time_window')
    
    # Strategy 1: Forward fill (use last known value) - up to 2 hours
    weather_features = ['temperature', 'precipitation', 'humidity', 'wind_speed', 'feels_like_temp']
    for feature in weather_features:
        demand_df[feature] = demand_df[feature].fillna(method='ffill', limit=4)  # 4 windows = 2 hours
    
    # Strategy 2: Backward fill for remaining gaps at the beginning
    for feature in weather_features:
        demand_df[feature] = demand_df[feature].fillna(method='bfill', limit=4)
    
    # Strategy 3: Fill weather_condition with most common value
    most_common_weather = weather['weather_condition'].mode()[0]
    demand_df['weather_condition'] = demand_df['weather_condition'].fillna(most_common_weather)
    
    # Strategy 4: Fill any remaining with reasonable defaults
    demand_df['temperature'] = demand_df['temperature'].fillna(15)  # Mild temperature
    demand_df['precipitation'] = demand_df['precipitation'].fillna(0)
    demand_df['humidity'] = demand_df['humidity'].fillna(60)
    demand_df['wind_speed'] = demand_df['wind_speed'].fillna(10)
    demand_df['feels_like_temp'] = demand_df['feels_like_temp'].fillna(demand_df['temperature'])
    demand_df['weather_condition'] = demand_df['weather_condition'].fillna('Clear')
    
    final_missing = demand_df['temperature'].isna().sum()
    print(f"  ✓ Missing values after interpolation: {final_missing}")

# Create derived weather features
demand_df['is_bad_weather'] = demand_df['weather_condition'].isin(['Rain', 'Snow', 'Fog']).astype(int)
demand_df['temperature_comfort'] = ((demand_df['temperature'] >= 15) & 
                                     (demand_df['temperature'] <= 25)).astype(int)

print("Weather features merged and cleaned:")
print("  - temperature, precipitation, humidity, wind_speed")
print("  - weather_condition, feels_like_temp")
print("  - is_bad_weather, temperature_comfort")

# ============================================================================
# STEP 8: MERGE EVENT DATA
# ============================================================================
print("\n[STEP 8] Merging event data...")

# Get zone locations from trips
zone_locations = trips.groupby('zone_id').agg({
    'pickup_lat': 'mean',
    'pickup_lon': 'mean'
}).reset_index()
zone_locations.columns = ['zone_id', 'zone_lat', 'zone_lon']

demand_df = demand_df.merge(zone_locations, on='zone_id', how='left')

# Function to calculate distance (Haversine formula)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Initialize event features
demand_df['event_nearby'] = 0
demand_df['event_type'] = 'none'
demand_df['event_attendance'] = 0
demand_df['minutes_to_event_start'] = -999
demand_df['minutes_to_event_end'] = -999

# For each time window and zone, check for nearby events
print("  Processing event proximity (this may take a moment)...")
for idx, row in demand_df.iterrows():
    # Find events that overlap with this time window (±2 hours)
    time_start = row['time_window'] - timedelta(hours=2)
    time_end = row['time_window'] + timedelta(hours=2)
    
    nearby_events = events[
        (events['start_time'] >= time_start) & 
        (events['start_time'] <= time_end)
    ]
    
    # Check distance to events
    for _, event in nearby_events.iterrows():
        distance = haversine_distance(
            row['zone_lat'], row['zone_lon'],
            event['venue_lat'], event['venue_lon']
        )
        
        # If event is within 2km
        if distance <= 2.0:
            demand_df.at[idx, 'event_nearby'] = 1
            demand_df.at[idx, 'event_type'] = event['event_type']
            demand_df.at[idx, 'event_attendance'] = event['expected_attendance']
            
            # Calculate time to event
            time_to_start = (event['start_time'] - row['time_window']).total_seconds() / 60
            time_to_end = (event['end_time'] - row['time_window']).total_seconds() / 60
            
            demand_df.at[idx, 'minutes_to_event_start'] = time_to_start
            demand_df.at[idx, 'minutes_to_event_end'] = time_to_end
            break  # Only consider the closest/first event

print("Event features merged:")
print("  - event_nearby (within 2km)")
print("  - event_type, event_attendance")
print("  - minutes_to_event_start, minutes_to_event_end")

# ============================================================================
# STEP 9: CREATE TARGET VARIABLE (Next 30-min demand)
# ============================================================================
print("\n[STEP 9] Creating target variable...")

# Sort by zone and time
demand_df = demand_df.sort_values(['zone_id', 'time_window'])

# Create target: demand in next 30-min window
target_features = []
for zone in all_zones:
    zone_data = demand_df[demand_df['zone_id'] == zone].copy()
    zone_data['demand_next_30min'] = zone_data['demand'].shift(-1)
    target_features.append(zone_data)

demand_df = pd.concat(target_features, ignore_index=True)

# Remove last time window for each zone (no target available)
demand_df = demand_df.dropna(subset=['demand_next_30min'])

print(f"Target variable created: demand_next_30min")
print(f"Final dataset size: {len(demand_df):,} records")

# ============================================================================
# STEP 10: CLEAN AND PREPARE FINAL DATASET
# ============================================================================
print("\n[STEP 10] Final data preparation...")

# Fill remaining NaN values in lag features (early time windows)
lag_cols = ['demand_lag_1', 'demand_lag_2', 'demand_lag_48', 'demand_lag_336',
            'demand_rolling_mean_3', 'demand_rolling_mean_6', 
            'demand_rolling_std_3', 'demand_rolling_max_6']

for col in lag_cols:
    demand_df[col] = demand_df[col].fillna(0)

# Encode categorical variables
demand_df['weather_condition'] = demand_df['weather_condition'].fillna('Clear')

# One-hot encode event_type
event_dummies = pd.get_dummies(demand_df['event_type'], prefix='event')
demand_df = pd.concat([demand_df, event_dummies], axis=1)

print("Data cleaning completed")
print(f"NaN values filled, categorical variables encoded")

# ============================================================================
# STEP 11: SAVE PROCESSED DATASET
# ============================================================================
print("\n[STEP 11] Saving processed dataset...")

# Select final feature columns
feature_columns = [
    'zone_id', 'time_window', 'demand',
    # Temporal features
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    # Lag features
    'demand_lag_1', 'demand_lag_2', 'demand_lag_48', 'demand_lag_336',
    'demand_rolling_mean_3', 'demand_rolling_mean_6', 
    'demand_rolling_std_3', 'demand_rolling_max_6',
    # Weather features
    'temperature', 'precipitation', 'wind_speed', 'humidity',
    'feels_like_temp', 'is_bad_weather', 'temperature_comfort',
    # Event features
    'event_nearby', 'event_attendance', 
    'minutes_to_event_start', 'minutes_to_event_end',
    # Target
    'demand_next_30min'
]

# Add event type dummies
feature_columns.extend([col for col in demand_df.columns if col.startswith('event_')])

# Create final dataset
final_df = demand_df[feature_columns].copy()

# Save to CSV
final_df.to_csv('demand_prediction_dataset.csv', index=False)

print(f"Dataset saved: demand_prediction_dataset.csv")
print(f"Total features: {len(feature_columns)}")
print(f"Total records: {len(final_df):,}")

# ============================================================================
# STEP 12: DISPLAY SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nDataset Shape: {final_df.shape}")
print(f"Date Range: {final_df['time_window'].min()} to {final_df['time_window'].max()}")
print(f"Number of Zones: {final_df['zone_id'].nunique()}")

print("\nTarget Variable Statistics (demand_next_30min):")
print(final_df['demand_next_30min'].describe())

print("\nFeature Categories:")
print(f"  - Temporal features: 9")
print(f"  - Lag features: 8")
print(f"  - Weather features: 7")
print(f"  - Event features: {len([c for c in feature_columns if 'event' in c])}")
print(f"  - Total features: {len(feature_columns) - 3}")  # Exclude zone_id, time_window, target

print("\nMissing Values:")
print(final_df.isnull().sum().sum(), "total missing values")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Load 'demand_prediction_dataset.csv'")
print("2. Split into train/validation/test sets (temporal split)")
print("3. Train LightGBM regression model")
print("4. Evaluate and visualize results")
print("\nReady for model training!")