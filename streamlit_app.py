import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline

st.set_page_config(
    page_title="Customer Intelligence Platform",
    layout="wide"
)

st.title("Customer Intelligence Platform")
st.markdown("Predict returns, detect high-value customers, and segment customers using ML.")

st.sidebar.title("Services")

service = st.sidebar.radio(
    "Choose Service",
    [
        "Return Prediction",
        "High Value Customer Detection",
        "Customer Segmentation"
    ]
)

pipeline = PredictPipeline()

# Country dropdown list
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

# Segment labels (correct mapping)
segment_labels = {
    0: "VIP Customers",
    1: "Mid Value Customers",
    2: "At Risk Customers"
}


# ---------------- RETURN PREDICTION ---------------- #

if service == "Return Prediction":

    st.header("Product Return Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Quantity = st.number_input("Quantity", min_value=0, step=1)
        Price = st.number_input("Price", min_value=0.0)

    with col2:
        Order_Value = st.number_input("Order Value", min_value=0.0)
        Country = st.selectbox("Country", countries)

    col3, col4 = st.columns(2)

    with col3:
        Invoice_Month = st.number_input("Invoice Month", min_value=1, max_value=12)

    with col4:
        Invoice_Day = st.number_input("Invoice Day", min_value=1, max_value=31)

    if st.button("Predict Return"):

        try:

            data = pd.DataFrame({
                "Quantity":[Quantity],
                "Price":[Price],
                "Order_Value":[Order_Value],
                "Invoice_Month":[Invoice_Month],
                "Invoice_Day":[Invoice_Day],
                "Country":[Country]
            })

            prediction = pipeline.predict_return(data)

            st.subheader("Prediction Result")

            if prediction[0] == 1:
                st.error("⚠️ Order Likely to be Returned")
            else:
                st.success("✅ Order Not Likely to be Returned")

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ---------------- HIGH VALUE CUSTOMER ---------------- #

elif service == "High Value Customer Detection":

    st.header("High Value Customer Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        Recency = st.number_input("Recency", min_value=0, step=1)

    with col2:
        Frequency = st.number_input("Frequency", min_value=0, step=1)

    with col3:
        Monetary = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Customer Value"):

        try:

            data = pd.DataFrame({
                "Recency":[Recency],
                "Frequency":[Frequency],
                "Monetary":[Monetary]
            })

            prediction = pipeline.predict_high_value(data)

            st.subheader("Prediction Result")

            if prediction[0] == 1:
                st.success("🌟 High Value Customer")
            else:
                st.warning("Regular Customer")

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ---------------- CUSTOMER SEGMENTATION ---------------- #

elif service == "Customer Segmentation":

    st.header("Customer Segmentation")

    col1, col2, col3 = st.columns(3)

    with col1:
        Recency = st.number_input("Recency", min_value=0, step=1)

    with col2:
        Frequency = st.number_input("Frequency", min_value=0, step=1)

    with col3:
        Monetary = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Segment"):

        try:

            data = pd.DataFrame({
                "Recency":[Recency],
                "Frequency":[Frequency],
                "Monetary":[Monetary]
            })

            cluster = pipeline.predict_customer_segment(data)

            label = segment_labels.get(cluster[0], f"Segment {cluster[0]}")

            st.subheader("Customer Segment")

            if cluster[0] == 0:
                st.success(f"👑 {label}")

            elif cluster[0] == 1:
                st.info(f"📊 {label}")

            else:
                st.warning(f"⚠️ {label}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")