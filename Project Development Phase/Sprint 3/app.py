import streamlit as st
import pandas as pd
import joblib

#st.title("University Admition Eligibility Predictor")
st.header("Car Resale Value Predictor")



year = st.number_input("Enter The Year")
Kms_driven = st.number_input("Enter The Kilometer")
owner = st.number_input("Enter The Owner Value")
engine = st.number_input("Enter The Engine Value")
FuelType = st.text_input("Enter The Fuel Type Value")
SellerType = st.text_input("Enter The Seller Type Value")
Transmissions = st.text_input("Enter The Transmissions Value")

#Research = st.selectbox("Select Research or Not", ("Research", "Not"))


if st.button("Submit"):
    clf = joblib.load("model.pkl")
    X = pd.DataFrame([[year,Kms_driven,owner,engine,FuelType,SellerType,Transmissions]], 
                        columns = ["year","Kms_driven","owner","engine","FuelType","SellerType","Transmissions"])
    #X = X.replace(["Research", "Not"], [1, 0],)
    prediction = clf.predict(X)[0]
    st.subheader(f"The possibility is {prediction}")

#st.subheader("Created By")
#st.write("*Batrick Swaistan - 963319104011*")
#st.write("*Berdin Jasper - 963319104011*")
#st.write("*Shane Ratheesh - 963319104011*")
#st.write("*Benishan- 963319104011*")