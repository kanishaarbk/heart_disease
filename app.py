import streamlit as st
import pickle
import pandas as pd

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Fill in the patient's details below.")

# Numeric Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, step=0.1)

# Categorical Inputs
sex_option = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_option == "Female" else 1

cp_option = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)
cp = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}[cp_option]

fbs_option = st.selectbox(
    "Fasting Blood Sugar",
    [
        "≤ 120 mg/dl",
        "> 120 mg/dl"
    ]
)
fbs = 0 if fbs_option == "≤ 120 mg/dl" else 1

restecg_option = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)
restecg = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}[restecg_option]

exang_option = st.selectbox(
    "Exercise Induced Angina",
    [
        "No",
        "Yes"
    ]
)
exang = 0 if exang_option == "No" else 1

slope_option = st.selectbox(
    "ST Segment Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)
slope = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}[slope_option]

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal_option = st.selectbox(
    "Thalassemia",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)
thal = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}[thal_option]

if st.button("🔍 Predict"):

    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    prediction = model.predict(input_data)[0]

if prediction == 1:
    st.success("✅ Low Risk of Heart Disease")
else:
    st.error("⚠️ High Risk of Heart Disease")

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(input_data)[0]

    st.write(f"**Probability of Heart Disease:** {probability[0]*100:.2f}%")
    st.write(f"**Probability of No Heart Disease:** {probability[1]*100:.2f}%")
