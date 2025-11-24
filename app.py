import joblib
import streamlit as st
import numpy as np

# Load trained model & preprocessors
model = ("best_model.joblib")
scaler = ("scaler.pkl")
encoder = ("encoder.pkl")

st.title("🧠 Mental Health Prediction App")

time_spent = st.slider("Daily Social Media Time (minutes)", 0, 600, 60)
posts = st.slider("Posts Per Week", 0, 50, 5)
likes = st.slider("Average Likes", 0, 500, 100)

if st.button("Predict"):

    data = np.array([[time_spent, posts, likes]])
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)[0]
    result = encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Mental Health Condition: {result}")
