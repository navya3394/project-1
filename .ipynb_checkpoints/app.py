import streamlit as st
import pandas as pd
import joblib as jb

model=jb.load('RF.pkl')
scaler=jb.load('Scaler.pkl')
expected_columns=jb.load('columns.pkl')

st.title("Chrun Prediction")
gender= st.selectbox("GENDER",['M','F'])
SeniorCitizen=st.selectbox("SeniorCitizen",[0,1])
Partner=st.selectbox("Partner",['Yes','No'])  
Dependents=st.selectbox("Dependents",['Yes','No'])
tenure=st.number_input("tenure",min_value=0,max_value=72,value=0,step=1)
PhoneService=st.selectbox("PhoneService",['Yes','No'])
MultipleLines=st.selectbox("MultipleLines",['Yes','No','No phone service'])
InternetService=st.selectbox("InternetService",['DSL','Fiber optic','No'])
OnlineSecurity=st.selectbox("OnlineSecurity",['Yes','No','No internet service'])
OnlineBackup=st.selectbox("OnlineBackup",['Yes','No','No internet service'])
DeviceProtection=st.selectbox("DeviceProtection",['Yes','No','No internet service'])
TechSupport=st.selectbox("TechSupport",['Yes','No','No internet service'])
StreamingTV=st.selectbox("StreamingTV",['Yes','No','No internet service'])
StreamingMovies=st.selectbox("StreamingMovies",['Yes','No','No internet service'])
Contract=st.selectbox("Contract",['Month-to-month','One year','Two year'])
PaperlessBilling=st.selectbox("PaperlessBilling",['Yes','No'])
PaymentMethod=st.selectbox("PaymentMethod",['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
MonthlyCharges=st.number_input("MonthlyCharges",min_value=0.0,max_value=150.0,value=0.0,step=0.1)
TotalCharges=st.number_input("TotalCharges",min_value=0.0,max_value=10000.0,value=0.0,step=0.1)

if st.button("Predict"):
    input={
        'Gender':gender,
        'SeniorCitizen':SeniorCitizen,
        'Partner':Partner,
        'Dependents':Dependents,
        'tenure':tenure,
        'PhoneService':PhoneService,
        'MultipleLines':MultipleLines,
        'InternetService':InternetService,
        'OnlineSecurity':OnlineSecurity,
        'OnlineBackup':OnlineBackup,
        'DeviceProtection':DeviceProtection,
        'TechSupport':TechSupport,
        'StreamingTV':StreamingTV,
        'StreamingMovies':StreamingMovies,
        'Contract':Contract,
        'PaperlessBilling':PaperlessBilling,
        'PaymentMethod':PaymentMethod,
        'MonthlyCharges':MonthlyCharges,
        'TotalCharges':TotalCharges
    }

    input_df=pd.DataFrame([input])

    scaled_data=scaler.transform(input_df)
    prediction=model.predict(scaled_data)[0]
    if prediction==0:
        st.success("The customer is not likely to churn")
    else:
        st.error("The customer is likely to churn")
