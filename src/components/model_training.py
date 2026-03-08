import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:

    return_model_path = os.path.join(
        "artifacts",
        "return_model.pkl"
    )

    high_value_model_path = os.path.join(
        "artifacts",
        "high_value_model.pkl"
    )

    segmentation_model_path = os.path.join(
        "artifacts",
        "segmentation_model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(

        self,

        X_train_return,
        X_test_return,
        y_train_return,
        y_test_return,

        X_train_high_value,
        X_test_high_value,
        y_train_high_value,
        y_test_high_value,

        rfm_train_scaled,
        rfm_test_scaled

    ):

        try:

            logging.info("Starting Model Training Pipeline")




            logging.info("Training models for Return Prediction")

            return_models = {

                "LogisticRegression": LogisticRegression(max_iter=1000),

                "RandomForest": RandomForestClassifier(),

                "XGBoost": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss'
                )

            }

            return_params = {

                "LogisticRegression": {

                    "C": [0.01, 0.1, 1, 10]

                },

                "RandomForest": {

                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]

                },

                "XGBoost": {

                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6, 10]

                }

            }


            return_report, return_best_models = evaluate_models(

                X_train_return,
                y_train_return,
                X_test_return,
                y_test_return,
                return_models,
                return_params,
                report_name="return_model_report"

            )


            best_return_model_name = max(
                return_report,
                key=lambda x: return_report[x]["f1_score"]
            )

            best_return_model = return_best_models[best_return_model_name]


            logging.info(f"Best Return Model Selected: {best_return_model_name}")
            logging.info(f"Return Model Metrics: {return_report[best_return_model_name]}")


            save_object(

                self.model_trainer_config.return_model_path,
                best_return_model

            )




            logging.info("Training models for High Value Customer Prediction")


            high_value_models = {

                "LogisticRegression": LogisticRegression(max_iter=1000),

                "RandomForest": RandomForestClassifier(),

                "XGBoost": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss'
                )

            }


            high_value_params = {

                "LogisticRegression": {

                    "C": [0.01, 0.1, 1, 10]

                },

                "RandomForest": {

                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]

                },

                "XGBoost": {

                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6, 10]

                }

            }


            high_value_report, high_value_best_models = evaluate_models(

                X_train_high_value,
                y_train_high_value,
                X_test_high_value,
                y_test_high_value,
                high_value_models,
                high_value_params,
                report_name="high_value_model_report"

            )


            best_high_value_model_name = max(
                high_value_report,
                key=lambda x: high_value_report[x]["f1_score"]
            )

            best_high_value_model = high_value_best_models[best_high_value_model_name]


            logging.info(f"Best High Value Model Selected: {best_high_value_model_name}")
            logging.info(f"High Value Model Metrics: {high_value_report[best_high_value_model_name]}")


            save_object(

                self.model_trainer_config.high_value_model_path,
                best_high_value_model

            )




            logging.info("Training Customer Segmentation Model (KMeans)")


            best_k = None
            best_score = -1
            best_kmeans = None


            for k in range(2, 7):

                kmeans = KMeans(n_clusters=k, random_state=42)

                labels = kmeans.fit_predict(rfm_train_scaled)

                score = silhouette_score(rfm_train_scaled, labels)

                logging.info(f"KMeans clusters={k} silhouette_score={score}")

                if score > best_score:

                    best_score = score
                    best_k = k
                    best_kmeans = kmeans


            logging.info(f"Best KMeans clusters selected: {best_k}")
            logging.info(f"Best silhouette score: {best_score}")


            save_object(

                self.model_trainer_config.segmentation_model_path,
                best_kmeans

            )




            logging.info("Model Training Completed Successfully")

            return {

                "return_model": best_return_model_name,
                "high_value_model": best_high_value_model_name,
                "segmentation_clusters": best_k

            }


        except Exception as e:
            raise CustomException(e, sys)