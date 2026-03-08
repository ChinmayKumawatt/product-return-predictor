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


# ---------------- RETURN PREDICTION ---------------- #

if service == "Return Prediction":

    st.header("Return Prediction")

    Quantity = int(st.number_input("Quantity", min_value=0, step=1))
    Price = float(st.number_input("Price", min_value=0.0))
    Order_Value = float(st.number_input("Order Value", min_value=0.0))

    Invoice_Month = int(
        st.number_input("Invoice Month", min_value=1, max_value=12, step=1)
    )

    Invoice_Day = int(
        st.number_input("Invoice Day", min_value=1, max_value=31, step=1)
    )

    Country = st.text_input("Country")

    if st.button("Predict Return"):

        if Country.strip() == "":
            st.warning("Please enter a country")
        else:

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

    Recency = float(st.number_input("Recency", min_value=0.0))
    Frequency = float(st.number_input("Frequency", min_value=0.0))
    Monetary = float(st.number_input("Monetary", min_value=0.0))

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

    Recency = float(st.number_input("Recency", min_value=0.0))
    Frequency = float(st.number_input("Frequency", min_value=0.0))
    Monetary = float(st.number_input("Monetary", min_value=0.0))

    if st.button("Predict Segment"):

        try:

            data = pd.DataFrame({
                "Recency": [Recency],
                "Frequency": [Frequency],
                "Monetary": [Monetary]
            })

            cluster = pipeline.predict_customer_segment(data)

            st.info(f"Customer belongs to Segment {cluster[0]}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")