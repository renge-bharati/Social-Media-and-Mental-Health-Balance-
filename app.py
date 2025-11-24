import streamlit as st
import numpy as np
import joblib

# Load the trained scaler (make sure the file exists in models folder)
scaler = joblib.load("models/scaler.pkl")

# Streamlit input sliders
time_spent = st.slider("Daily Social Media Time (minutes)", 0, 600, 60)
posts = st.slider("Posts Per Week", 0, 50, 5)
likes = st.slider("Average Likes", 0, 500, 100)

# Button to predict
if st.button("Transform Data"):
    # Prepare data as 2D array
    data = np.array([[time_spent, posts, likes]])

    # Transform using loaded scaler
    data_scaled = scaler.transform(data)

    st.write("Scaled Data:", data_scaled)
