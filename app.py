import streamlit as st
import numpy as np
import joblib

# Load model & preprocessors
import joblib

model = ("models/best_model.joblib")
scaler = ("models/scaler.pkl")
encoder = ("models/encoder.pkl")


st.title("🧠 Mental Health Prediction App")

col1, col2, col3 = st.columns(3)

with col1:
    time_spent = st.slider("Hours on Social Media Daily", 0, 12, 3)

with col2:
    posts = st.slider("Posts Per Week", 0, 40, 5)

with col3:
    likes = st.slider("Average Likes", 0, 500, 100)

if st.button("Predict"):
    data = np.array([[time_spent, posts, likes]])
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)[0]
    result = encoder.inverse_transform([pred])[0]

    st.success(f"🧠 Predicted Mental Health Status: **{result}**")
