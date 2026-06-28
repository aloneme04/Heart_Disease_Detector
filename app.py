import streamlit as st
import pandas as pd
import joblib
model = joblib.load("KNN_heart.pkl")  # Load your pre-trained model
scaler = joblib.load("scaler.pkl")  # Load your pre-trained scaler
expected_columns=joblib.load("columns.pkl") 
 # Load expected columns
st.title("heart stroke prediction❤️ (❁´◡`❁)")
st.markdown("Provide the  following details")
age=st.slider("age",18,100,40)
sex=st.selectbox("SEX",["Male","Female"])
chest_pain=st.selectbox("chest pain type",["ATA","NAP","ASY","TA"])
resting_bp=st.number_input("resting blood pressure(mg Hg)",80,200,120)
cholesterol=st.number_input("cholesterol(mg/dl)",100,400,200)
fasting_bs=st.selectbox("fasting blood sugar > 120 mg/dl",[0,1])
resting_ecg=st.selectbox("resting ECG",["Normal","ST","LVH"])
max_heart_rate=st.slider("maximum heart rate",60,220,150)
exercise_angina=st.selectbox("exercise induced angina",["Y","N"])
oldpeak=st.slider("oldpeak",0.0,6.0,1.0)
st_slope=st.selectbox("ST slope",["Up","Flat","Down"])

if st.button("Predict"):
    raw_input={
        "age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,
        "Sex_"+ sex: 1,
        "chestPainType_"+ chest_pain: 1,
        "RestingECG_"+ resting_ecg: 1,
        "ExerciseAngina_"+ exercise_angina: 1,  
        "ST_Slope_"+ st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]  # Reorder columns to match expected        
    scaler_input = scaler.transform(input_df)
    prediction = model.predict(scaler_input),[0]

    if prediction == 1:
        st.error("⚠️ high risk of heart disease")
    else:
        st.success("✅ low risk of heart disease")