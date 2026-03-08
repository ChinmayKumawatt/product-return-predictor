import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer


class TrainingPipeline:

    def __init__(self):
        pass


    def start_training_pipeline(self):

        try:

            logging.info("Training Pipeline Started")


            logging.info("Starting Data Ingestion")

            data_ingestion = DataIngestion()

            train_path, test_path = data_ingestion.initiate_data_ingestion()

            logging.info("Data Ingestion Completed")



            logging.info("Starting Data Transformation")

            data_transformation = DataTransformation()

            (

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

            ) = data_transformation.initiate_data_transformation(
                train_path,
                test_path
            )

            logging.info("Data Transformation Completed")



            logging.info("Starting Model Training")

            model_trainer = ModelTrainer()

            model_trainer.initiate_model_trainer(

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

            )

            logging.info("Model Training Completed")

            logging.info("Training Pipeline Finished Successfully")


        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.start_training_pipeline()