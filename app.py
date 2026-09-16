import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AirPulse",
    page_icon="🌍",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌍 AirPulse")
st.subheader("Machine Learning-Based Air Quality Prediction")

st.write(
    "Enter the air quality parameters below to predict the Air Quality Index (AQI)."
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/knn_regression.pkl")
    return model


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Trained model not found. Please train the model and save it "
        "inside the models folder."
    )
    st.stop()


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.header("Enter Air Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=50.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=80.0
    )

    no2 = st.number_input(
        "NO₂",
        min_value=0.0,
        value=30.0
    )

with col2:
    so2 = st.number_input(
        "SO₂",
        min_value=0.0,
        value=10.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.0
    )

    o3 = st.number_input(
        "O₃",
        min_value=0.0,
        value=30.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Air Quality"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "PM2.5": [pm25],
        "PM10": [pm10],
        "NO2": [no2],
        "SO2": [so2],
        "CO": [co],
        "O3": [o3]
    })

    try:

        # Make prediction
        prediction = model.predict(input_data)

        predicted_aqi = prediction[0]

        # Display result
        st.success(
            f"Predicted AQI: {predicted_aqi:.2f}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AirPulse | Machine Learning Lab Project"
)