import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

# Custom CSS (for better UI)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stSlider label, .stSelectbox label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("titanic_rf_model.pkl")

# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Enter Passenger Details")

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 25)

with col2:
    sibsp = st.slider("Siblings/Spouses", 0, 10, 0)
    parch = st.slider("Parents/Children", 0, 10, 0)
    fare = st.slider("Fare", 0.0, 500.0, 50.0)

embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs
sex = 0 if sex == "male" else 1
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Input array
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    survival_prob = prob[0][1] * 100
    not_survival_prob = prob[0][0] * 100

    st.markdown("---")

    # Result
    if prediction[0] == 1:
        st.success(f"🎉 Survived (Probability: {survival_prob:.2f}%)")
    else:
        st.error(f"❌ Not Survived (Probability: {not_survival_prob:.2f}%)")

    # Progress bars
    st.subheader("📊 Prediction Probability")
    st.write("Survival Chance")
    st.progress(int(survival_prob))

    st.write("Not Survival Chance")
    st.progress(int(not_survival_prob))