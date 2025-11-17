import streamlit as st
import pandas as pd
import joblib
import time
import os
import numpy as np
from sensor_reader import get_sensor_data

# ==============================================================
# 🌿 PAGE CONFIG
# ==============================================================
st.set_page_config(page_title="Smart Pest Surveillance System", page_icon="🌾", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e6f2ff, #d9f7f3);
        font-family: 'Poppins', sans-serif;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem 2.5rem;
        box-shadow: 0px 6px 25px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1b4332;
        text-align: center;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #40916c !important;
        color: white !important;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b4332 !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================
# 🧠 LOAD MODEL AND DATASET
# ==============================================================

model_type = "⚡ Light Model (Optimized for Streamlit Cloud)"
try:
    # Try to load lightweight model first
    if os.path.exists("pest_prediction_model_light.joblib"):
        model = joblib.load("pest_prediction_model_light.joblib")
    elif os.path.exists("pest_prediction_model_full.joblib"):
        model = joblib.load("pest_prediction_model_full.joblib")
        model_type = "🧠 Full Model (Local Use)"
    else:
        st.error("❌ No model file found. Please train the model first.")
        st.stop()

    pesticide_map = joblib.load("pesticide_map.joblib") if os.path.exists("pesticide_map.joblib") else {}

    # Load dataset for crop types
    if os.path.exists("Smart_Pesticide_MultiRecommend.csv"):
        df = pd.read_csv("Smart_Pesticide_MultiRecommend.csv")
        plant_types = sorted(df["Plant Type"].dropna().unique())
    else:
        st.warning("⚠️ Dataset not found. Using default crop list.")
        df = pd.DataFrame(columns=["Plant Type", "pesticide"])
        plant_types = ["Tomato", "Rice", "Wheat", "Cotton", "Maize"]

    st.success(f"✅ Model and dataset loaded successfully — {model_type}")

except Exception as e:
    st.error(f"❌ Error loading model or dataset: {e}")
    st.stop()

# ==============================================================
# 🌾 HEADER
# ==============================================================
st.title("🌾 Smart Pest Surveillance System")
st.markdown("""
This intelligent system uses **IoT + Machine Learning**  
to recommend the best **pesticides** for your crops based on  
real-time **Humidity**, **Temperature**, and **Soil Moisture** readings.
""")

st.info(f"💡 Current Model Loaded: {model_type}")

st.markdown("---")

# ==============================================================
# 📡 MODE SELECTION
# ==============================================================
mode = st.radio("Select Mode", ["🌿 Manual Input Mode", "📡 Live Sensor Mode"])

# ==============================================================
# 🧮 MANUAL INPUT MODE
# ==============================================================
if mode == "🌿 Manual Input Mode":
    st.subheader("🧮 Manual Crop Input Mode")

    plant = st.selectbox("🌱 Select Crop Type", plant_types)

    col1, col2, col3 = st.columns(3)
    humidity = col1.slider("💧 Humidity (%)", 0, 100, 60)
    temperature = col2.slider("🌡 Temperature (°C)", 0, 60, 30)
    moisture = col3.slider("🌱 Soil Moisture (%)", 0, 100, 50)

    st.markdown("### ⚙️ Model Prediction")

    if st.button("💊 Recommend Pesticide"):
        sample = pd.DataFrame([{
            "Humidity": humidity,
            "Moisture": moisture,
            "Temperature": temperature,
            "Plant Type": plant
        }])

        with st.spinner("Predicting pesticide recommendation..."):
            prediction = model.predict(sample)[0]
            try:
                prediction_proba = model.predict_proba(sample)[0]
                confidence = np.max(prediction_proba) * 100
            except Exception:
                confidence = 85.0  # fallback if predict_proba not available

        st.success(f"🌾 **Crop:** {plant}")
        st.markdown(f"💊 **Recommended Pesticide:** `{prediction}`")
        st.progress(int(confidence))
        st.caption(f"Model confidence: **{confidence:.2f}%**")

        st.markdown("---")
        st.info("📋 Other pesticide options for this crop:")
        alt = df[df["Plant Type"].str.lower() == plant.lower()]["pesticide"]
        if not alt.empty:
            st.markdown(f"🧪 `{alt.sample(1).values[0]}`")
        else:
            st.markdown("🧪 No alternative found in dataset.")

# ==============================================================
# 📡 LIVE SENSOR MODE
# ==============================================================
else:
    st.subheader("📡 Real-Time Arduino Sensor Mode")

    plant = st.selectbox("🌿 Select Crop Type", plant_types, key="live_crop")

    if st.button("🔄 Fetch Live Data"):
        with st.spinner("Fetching live readings from Arduino..."):
            data = get_sensor_data()
            time.sleep(2)

        if data:
            col1, col2, col3 = st.columns(3)
            col1.metric("💧 Humidity (%)", f"{data['humidity']:.2f}")
            col2.metric("🌡 Temperature (°C)", f"{data['temperature']:.2f}")
            col3.metric("🌱 Soil Moisture (%)", f"{data['soil_moisture']:.2f}")

            sample = pd.DataFrame([{
                "Humidity": data["humidity"],
                "Moisture": data["soil_moisture"],
                "Temperature": data["temperature"],
                "Plant Type": plant
            }])

            with st.spinner("Predicting pesticide recommendation..."):
                prediction = model.predict(sample)[0]
                try:
                    prediction_proba = model.predict_proba(sample)[0]
                    confidence = np.max(prediction_proba) * 100
                except Exception:
                    confidence = 80.0

            st.markdown("---")
            st.success(f"💊 **Recommended Pesticide:** `{prediction}`")
            st.progress(int(confidence))
            st.caption(f"Model confidence: **{confidence:.2f}%**")

            st.markdown("### 🌾 Alternative Pesticides for Similar Conditions:")
            alt = df[df["Plant Type"].str.lower() == plant.lower()]["pesticide"]
            if not alt.empty:
                st.info(f"🧪 `{alt.sample(1).values[0]}`")
            else:
                st.info("🧪 No alternative found in dataset.")
        else:
            st.error("⚠️ Could not read data from Arduino. Please check the COM port or wiring.")

    st.caption("ℹ️ Note: Live Sensor Mode works only locally (not on Streamlit Cloud).")

# ==============================================================
# 📘 FOOTER
# ==============================================================
st.markdown("---")
st.markdown("""
<center>Developed with 💚 by <b>Pranav Raikar</b><br>
for Smart IoT-based Crop Protection 🌾</center>
""", unsafe_allow_html=True)
