import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        'artifacts',
        'preprocessor.pkl'
    )
    rfm_scaler_path = os.path.join("artifacts","rfm_scaler.pkl")
    kmeans_model_path = os.path.join("artifacts","kmeans_model.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_rfm_features(self, df):

        logging.info("Creating RFM features")

        try:

            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            df["TotalPrice"] = df["Quantity"] * df["Price"]

            reference_date = df["InvoiceDate"].max()

            rfm = df.groupby("Customer ID").agg({
                "InvoiceDate": lambda x: (reference_date - x.max()).days,
                "Invoice": "nunique",
                "TotalPrice": "sum"
            })

            rfm.columns = ["Recency", "Frequency", "Monetary"]

            rfm.reset_index(inplace  = True)
            logging.info("RFM features created")

            return rfm

        except Exception as e:
            raise CustomException(e, sys)
        

    def get_return_preprocessor(self):

        try:

            numerical_columns = [

                "Quantity",
                "Price",
                "Order_Value",
                "Invoice_Month",
                "Invoice_Day"

            ]

            categorical_columns = [
                "Country"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(

                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df = train_df.dropna(subset=["Customer ID"])
            test_df = test_df.dropna(subset=["Customer ID"])

            logging.info("Train and test datasets loaded")

            

            for df in [train_df,test_df]:

                df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

                df["Order_Value"] = abs(df["Quantity"]) * df["Price"]

                df["Invoice_Month"] = df["InvoiceDate"].dt.month

                df["Invoice_Day"] = df["InvoiceDate"].dt.dayofweek

                df["Returned"] = (df["Quantity"] < 0).astype(int)

            logging.info("Return features created")

            return_features = [

                "Quantity",
                "Price",
                "Order_Value",
                "Invoice_Month",
                "Invoice_Day",
                "Country"

            ]

            X_train_return = train_df[return_features]
            y_train_return = train_df["Returned"]

            X_test_return = test_df[return_features]
            y_test_return = test_df["Returned"]

            return_preprocessor = self.get_return_preprocessor()

            X_train_return_processed = return_preprocessor.fit_transform(X_train_return)

            X_test_return_processed = return_preprocessor.transform(X_test_return)

            save_object(

                self.data_transformation_config.preprocessor_obj_file_path,
                return_preprocessor

            )

            logging.info("Return preprocessing completed")


            combined_df = pd.concat([train_df,test_df],axis=0)
            rfm_table = self.create_rfm_features(combined_df)

            train_customers = train_df["Customer ID"].unique()
            test_customers = test_df["Customer ID"].unique()

            rfm_train = rfm_table[rfm_table["Customer ID"].isin(train_customers)].copy()
            rfm_test = rfm_table[rfm_table["Customer ID"].isin(test_customers)].copy()

            rfm_train_features = rfm_train[["Recency","Frequency","Monetary"]]
            rfm_test_features = rfm_test[["Recency","Frequency","Monetary"]]

            scaler = StandardScaler()

            rfm_train_scaled = scaler.fit_transform(rfm_train_features)

            rfm_test_scaled = scaler.transform(rfm_test_features)

            save_object(

                self.data_transformation_config.rfm_scaler_path,
                scaler

            )

            logging.info("RFM scaling completed")



            kmeans = KMeans(

                n_clusters=3,
                random_state=42

            )

            train_clusters = kmeans.fit_predict(rfm_train_scaled)

            test_clusters = kmeans.predict(rfm_test_scaled)

            rfm_train["Cluster"] = train_clusters
            rfm_test["Cluster"] = test_clusters

            save_object(

                self.data_transformation_config.kmeans_model_path,
                kmeans

            )

            logging.info("Customer segmentation completed")


            threshold = rfm_train["Monetary"].quantile(0.75)

            rfm_train["High_Value"] = (

                rfm_train["Monetary"] > threshold

            ).astype(int)

            rfm_test["High_Value"] = (

                rfm_test["Monetary"] > threshold

            ).astype(int)

            logging.info("High value customer label created")

            X_train_high_value = rfm_train[

                ["Recency","Frequency","Monetary","Cluster"]

            ]

            y_train_high_value = rfm_train["High_Value"]

            X_test_high_value = rfm_test[

                ["Recency","Frequency","Monetary","Cluster"]

            ]

            y_test_high_value = rfm_test["High_Value"]
            y_train_return = y_train_return.values
            y_test_return = y_test_return.values

            X_train_high_value = X_train_high_value.values
            X_test_high_value = X_test_high_value.values

            y_train_high_value = y_train_high_value.values
            y_test_high_value = y_test_high_value.values

            return(

                X_train_return_processed,
                X_test_return_processed,
                y_train_return,
                y_test_return,

                X_train_high_value,
                X_test_high_value,
                y_train_high_value,
                y_test_high_value,

                rfm_train_scaled,
                rfm_test_scaled

            )

        except Exception as e:
            raise CustomException(e,sys)