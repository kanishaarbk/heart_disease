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
st.write("Enter the patient's details below.")

user_input = {}

for col in columns:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict"):

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"**Probability of No Disease:** {probability[0]*100:.2f}%")
    st.write(f"**Probability of Heart Disease:** {probability[1]*100:.2f}%")
