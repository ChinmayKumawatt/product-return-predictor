import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):

        try:

            self.return_model_path = os.path.join(
                "artifacts",
                "return_model.pkl"
            )

            self.high_value_model_path = os.path.join(
                "artifacts",
                "high_value_model.pkl"
            )

            self.segmentation_model_path = os.path.join(
                "artifacts",
                "segmentation_model.pkl"
            )

            self.preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            self.rfm_scaler_path = os.path.join(
                "artifacts",
                "rfm_scaler.pkl"
            )

        except Exception as e:
            raise CustomException(e, sys)




    def predict_return(self, features):

        try:

            model = load_object(self.return_model_path)

            preprocessor = load_object(self.preprocessor_path)

            transformed_data = preprocessor.transform(features)

            prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)




    def predict_high_value(self, rfm_features):

        try:

            high_value_model = load_object(self.high_value_model_path)

            scaler = load_object(self.rfm_scaler_path)

            segmentation_model = load_object(self.segmentation_model_path)

            scaled_features = scaler.transform(rfm_features)

            cluster = segmentation_model.predict(scaled_features)

            rfm_features["Cluster"] = cluster

            prediction = high_value_model.predict(rfm_features)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)



    def predict_customer_segment(self, rfm_features):

        try:

            model = load_object(self.segmentation_model_path)

            scaler = load_object(self.rfm_scaler_path)

            scaled_features = scaler.transform(rfm_features)

            cluster = model.predict(scaled_features)

            return cluster

        except Exception as e:
            raise CustomException(e, sys)




class CustomData:

    def __init__(

        self,

        Quantity,
        Price,
        Order_Value,
        Invoice_Month,
        Invoice_Day,
        Country,

        Recency=None,
        Frequency=None,
        Monetary=None

    ):

        self.Quantity = Quantity
        self.Price = Price
        self.Order_Value = Order_Value
        self.Invoice_Month = Invoice_Month
        self.Invoice_Day = Invoice_Day
        self.Country = Country

        self.Recency = Recency
        self.Frequency = Frequency
        self.Monetary = Monetary




    def get_return_dataframe(self):

        try:

            input_dict = {

                "Quantity": [self.Quantity],
                "Price": [self.Price],
                "Order_Value": [self.Order_Value],
                "Invoice_Month": [self.Invoice_Month],
                "Invoice_Day": [self.Invoice_Day],
                "Country": [self.Country]

            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)




    def get_rfm_dataframe(self):

        try:

            input_dict = {

                "Recency": [self.Recency],
                "Frequency": [self.Frequency],
                "Monetary": [self.Monetary]

            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)