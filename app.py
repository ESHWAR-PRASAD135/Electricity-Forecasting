import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Electricity Consumption Forecasting",
                   page_icon="⚡", layout="wide")

st.title("⚡ Electricity Consumption Forecasting")
st.caption("Deep Learning Case Study using LSTM")

required = ["electricity_lstm.keras",
            scaler.pkl", recent_sequence.npy"]

if not all(os.path.exists(x) for x in required):
    st.error("Model files are missing. Run `python train_model.py` first.")
    st.stop()

model = load_model(required[0])
with open(required[1], "rb") as f:
    scaler = pickle.load(f)
sequence = np.load(required[2])

st.sidebar.header("Model Information")
st.sidebar.write("Model: LSTM")
st.sidebar.write("Input window: 24 hours")
st.sidebar.write("Forecast: Next hour")
st.sidebar.write("Optimizer: Adam")
st.sidebar.write("Loss: MSE")

st.subheader("🔮 Next-Hour Forecast")

if st.button("Predict Next Hour", use_container_width=True):
    x = sequence.reshape(1, 24, 1)
    scaled_prediction = model.predict(x, verbose=0)
    prediction = scaler.inverse_transform(scaled_prediction)[0, 0]
    st.success(f"Predicted Global Active Power: {prediction:.3f} kW")

st.divider()
st.subheader("📈 Recent 24-Hour Consumption")
recent = scaler.inverse_transform(sequence.reshape(-1, 1)).flatten()
st.line_chart(pd.DataFrame({"Global Active Power (kW)": recent}))

st.divider()
st.subheader("How the System Works")
st.markdown("""
1. Historical electricity data is collected.
2. Missing values are removed.
3. Minute readings are aggregated into hourly readings.
4. Data is normalized using Min-Max Scaling.
5. Previous 24 hours are given to the LSTM.
6. LSTM learns temporal patterns.
7. The next-hour consumption is predicted.
""")

st.info("Dataset: UCI Individual Household Electric Power Consumption. Target: Global Active Power.")
