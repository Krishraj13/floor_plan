# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("material_predictor_model.pkl")
scaler = joblib.load("xgb_feature_scaler.pkl")

# App UI
st.set_page_config(page_title="Construction Estimator", layout="centered")
st.title("🏗️ Construction Material & Labor Estimator")
st.markdown("Enter flat details below to estimate required materials and labor.")

# Input form
with st.form("input_form"):
    floor_number = st.number_input("Floor Number", min_value=0, max_value=20, value=0)
    no_of_flats = st.number_input("Number of Flats on This Floor", min_value=1, max_value=10, value=1)
    flat_area = st.number_input("Total Flat Area (sq. ft)", min_value=200, max_value=3000, value=1000)
    no_of_rooms = st.number_input("Number of Rooms", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict Requirements")

# On submit
if submitted:
    # Inform user if no rooms
    if no_of_rooms == 0:
        st.info("You are building a single flat with no separate rooms — treated as one large open unit.")

    # Derived features
    area_per_room = flat_area / no_of_rooms if no_of_rooms > 0 else flat_area
    area_per_flat = flat_area / no_of_flats
    room_density = no_of_rooms / flat_area

    # Full feature input
    input_features = np.array([[floor_number, no_of_flats, flat_area, no_of_rooms, area_per_room, area_per_flat, room_density]])

    # Scale features
    input_scaled = scaler.transform(input_features)

    # Predict
    prediction = model.predict(input_scaled)[0]
    sand, cement, bricks, labour, hours = prediction

    # Output results
    st.subheader("📦 Estimated Construction Requirements")
    st.success(f"**Sand:** {sand:.2f} tons")
    st.success(f"**Cement:** {cement:.2f} tons")
    st.success(f"**Bricks:** {int(bricks)} units")
    st.success(f"**Labor Required:** {int(labour)} workers")
    st.success(f"**Total Labor Hours:** {int(hours)} hours")

