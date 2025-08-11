import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

# Load the trained model
model_path = r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\models\trained_model.pkl"
try:
    model = joblib.load(model_path)
    st.success(f"✅ Model loaded from: {model_path}")
except FileNotFoundError:
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

# Define feature columns expected by the model
FEATURE_COLS = [
    'Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h',
    'Visibility_km', 'Press_kPa', 'Hour', 'Month', 'DayOfWeek',
    'Is_Night', 'Temp_Diff', 'Hum_Vis_Interaction'
]

# Scaling parameters
SCALING_PARAMS = {
    'Temp_C': {'mean': 8.0, 'std': 11.0},
    'Dew Point Temp_C': {'mean': 2.0, 'std': 10.0},
    'Rel Hum_%': {'mean': 70.0, 'std': 15.0},
    'Wind Speed_km/h': {'mean': 15.0, 'std': 10.0},
    'Visibility_km': {'mean': 25.0, 'std': 10.0},
    'Press_kPa': {'mean': 101.0, 'std': 1.0}
}

def scale_features(df):
    for col in ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']:
        df[col] = (df[col] - SCALING_PARAMS[col]['mean']) / SCALING_PARAMS[col]['std']
    return df

def engineer_features(df):
    current_time = datetime.now()
    df['Month'] = current_time.month
    df['DayOfWeek'] = current_time.weekday()
    df['Is_Night'] = ((df['Hour'] >= 18) | (df['Hour'] < 6)).astype(int)
    df['Temp_Diff'] = df['Temp_C'] - df['Dew Point Temp_C']
    df['Hum_Vis_Interaction'] = df['Rel Hum_%'] * df['Visibility_km']
    return df

# ------------------ Streamlit UI ------------------ #
st.title("🌦 Weather Prediction App ")

st.markdown("Enter the weather conditions below to predict the weather category.")

col1, col2 = st.columns(2)

with col1:
    temp_c = st.number_input("Temperature (°C)", value=20.0)
    dew_point = st.number_input("Dew Point Temperature (°C)", value=10.0)
    rel_hum = st.number_input("Relative Humidity (%)", value=70.0)
with col2:
    wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
    visibility = st.number_input("Visibility (km)", value=25.0)
    pressure = st.number_input("Pressure (kPa)", value=101.0)

hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)

if st.button("🔮 Predict Weather"):
    try:
        input_data = {
            'Temp_C': temp_c,
            'Dew Point Temp_C': dew_point,
            'Rel Hum_%': rel_hum,
            'Wind Speed_km/h': wind_speed,
            'Visibility_km': visibility,
            'Press_kPa': pressure,
            'Hour': hour
        }

        df = pd.DataFrame([input_data])
        df = engineer_features(df)
        df = scale_features(df)

        for col in FEATURE_COLS:
            if col not in df.columns:
                st.error(f"Missing feature: {col}")
                st.stop()

        X = df[FEATURE_COLS]
        probabilities = model.predict_proba(X)[0]
        class_names = model.classes_

        # Remove "Fog" and "Snow"
        remove_labels = ["Fog", "Snow"]
        mask = ~np.isin(class_names, remove_labels)
        class_names = class_names[mask]
        probabilities = probabilities[mask]

        # Recalculate prediction without Fog & Snow
        best_idx = np.argmax(probabilities)
        prediction = class_names[best_idx]
        confidence = probabilities[best_idx] * 100

        prob_dict = {class_names[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}

        st.subheader("Prediction Results")
        st.write(f"**Predicted Weather:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.write("### Class Probabilities")
        st.bar_chart(pd.DataFrame(prob_dict.values(), index=prob_dict.keys(), columns=["Probability"]))

    except Exception as e:
        st.error(f"Error: {str(e)}. Please ensure all inputs are valid numbers.")
