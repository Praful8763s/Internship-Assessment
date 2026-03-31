
import streamlit as st
import pandas as pd
import pickle

st.title("Supply Chain Capstone Dashboard 📦")
st.write("Analyze and predict supply chain risks.")

df = pd.read_csv("../data/supply_chain_data.csv")
st.subheader("Data Overview")
st.dataframe(df.head())

st.subheader("Delay Risk Prediction")
qty = st.number_input("Quantity", min_value=1, value=50)
price = st.number_input("Price", min_value=1.0, value=100.0)
shipping = st.number_input("Shipping Cost", min_value=1.0, value=15.0)
lead_time = st.number_input("Lead Time (Days)", min_value=1, value=10)

if st.button("Predict Delay Risk"):
    with open("../models/final_model.pkl", "rb") as f:
        model = pickle.load(f)
    pred = model.predict([[qty, price, shipping, lead_time]])
    risk = "High Risk of Delay" if pred[0] == 1 else "On Time"
    st.write(f"**Prediction:** {risk}")
