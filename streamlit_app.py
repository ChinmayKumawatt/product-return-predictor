import streamlit as st
import pandas as pd

from src.pipeline.prediction_pipeline import PredictPipeline


st.set_page_config(page_title="Customer Intelligence Platform")

st.title("Customer Intelligence Platform")

st.sidebar.header("Select Service")

service = st.sidebar.selectbox(
    "Choose a Service",
    (
        "Return Prediction",
        "High Value Customer Detection",
        "Customer Segmentation"
    )
)

pipeline = PredictPipeline()


if service == "Return Prediction":

    st.header("Return Prediction")

    Quantity = st.number_input("Quantity")
    Price = st.number_input("Price")
    Order_Value = st.number_input("Order Value")
    Invoice_Month = st.number_input("Invoice Month")
    Invoice_Day = st.number_input("Invoice Day")
    Country = st.text_input("Country")

    if st.button("Predict Return"):

        data = pd.DataFrame({
            "Quantity":[Quantity],
            "Price":[Price],
            "Order_Value":[Order_Value],
            "Invoice_Month":[Invoice_Month],
            "Invoice_Day":[Invoice_Day],
            "Country":[Country]
        })

        prediction = pipeline.predict_return(data)

        if prediction[0] == 1:
            st.error("Order Likely to be Returned")
        else:
            st.success("Order Not Likely to be Returned")


elif service == "High Value Customer Detection":

    st.header("High Value Customer Detection")

    Recency = st.number_input("Recency")
    Frequency = st.number_input("Frequency")
    Monetary = st.number_input("Monetary")

    if st.button("Predict Customer Value"):

        data = pd.DataFrame({
            "Recency":[Recency],
            "Frequency":[Frequency],
            "Monetary":[Monetary]
        })

        prediction = pipeline.predict_high_value(data)

        if prediction[0] == 1:
            st.success("High Value Customer")
        else:
            st.warning("Regular Customer")


elif service == "Customer Segmentation":

    st.header("Customer Segmentation")

    Recency = st.number_input("Recency")
    Frequency = st.number_input("Frequency")
    Monetary = st.number_input("Monetary")

    if st.button("Predict Segment"):

        data = pd.DataFrame({
            "Recency":[Recency],
            "Frequency":[Frequency],
            "Monetary":[Monetary]
        })

        cluster = pipeline.predict_customer_segment(data)

        st.info(f"Customer belongs to Segment {cluster[0]}")