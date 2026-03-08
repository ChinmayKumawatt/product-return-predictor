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

# Country dropdown list (replace with your full list)
countries = [
    "United Kingdom",
    "USA",
    "Japan",
    "Czech Republic",
    "Korea",
    "Saudi Arabia",
    "Channel Islands",
    "Malta",
    "Australia",
    "Germany",
    "Poland"
]


# Segment labels (edit according to your clustering meaning)
segment_labels = {
    0: "Low Value Customers",
    1: "Regular Customers",
    2: "High Value Customers",
    3: "Premium Customers"
}


# ---------------- RETURN PREDICTION ---------------- #

if service == "Return Prediction":

    st.header("Return Prediction")

    Quantity = st.number_input("Quantity", min_value=0, step=1)
    Price = st.number_input("Price", min_value=0.0, step=0.1)
    Order_Value = st.number_input("Order Value", min_value=0.0, step=0.1)

    Invoice_Month = st.number_input(
        "Invoice Month", min_value=1, max_value=12, step=1
    )

    Invoice_Day = st.number_input(
        "Invoice Day", min_value=1, max_value=31, step=1
    )

    Country = st.selectbox("Country", countries)

    if st.button("Predict Return"):

        try:

            data = pd.DataFrame({
                "Quantity": [Quantity],
                "Price": [Price],
                "Order_Value": [Order_Value],
                "Invoice_Month": [Invoice_Month],
                "Invoice_Day": [Invoice_Day],
                "Country": [Country]
            })

            prediction = pipeline.predict_return(data)

            if prediction[0] == 1:
                st.error("Order Likely to be Returned")
            else:
                st.success("Order Not Likely to be Returned")

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ---------------- HIGH VALUE CUSTOMER ---------------- #

elif service == "High Value Customer Detection":

    st.header("High Value Customer Detection")

    Recency = st.number_input("Recency", min_value=0, step=1)
    Frequency = st.number_input("Frequency", min_value=0, step=1)
    Monetary = st.number_input("Monetary", min_value=0.0, step=0.1)

    if st.button("Predict Customer Value"):

        try:

            data = pd.DataFrame({
                "Recency": [Recency],
                "Frequency": [Frequency],
                "Monetary": [Monetary]
            })

            prediction = pipeline.predict_high_value(data)

            if prediction[0] == 1:
                st.success("High Value Customer")
            else:
                st.warning("Regular Customer")

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ---------------- CUSTOMER SEGMENTATION ---------------- #

elif service == "Customer Segmentation":

    st.header("Customer Segmentation")

    Recency = st.number_input("Recency", min_value=0, step=1)
    Frequency = st.number_input("Frequency", min_value=0, step=1)
    Monetary = st.number_input("Monetary", min_value=0.0, step=0.1)

    if st.button("Predict Segment"):

        try:

            data = pd.DataFrame({
                "Recency": [Recency],
                "Frequency": [Frequency],
                "Monetary": [Monetary]
            })

            cluster = pipeline.predict_customer_segment(data)

            label = segment_labels.get(cluster[0], f"Segment {cluster[0]}")

            st.success(f"Customer Segment: {label}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")