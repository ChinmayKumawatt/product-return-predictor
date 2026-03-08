import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# from src.components.data_transformation import DataTransformation

# from src.components.model_trainer import model_trainer_config
# from src.components.model_trainer import modelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts','train.csv')
    test_data_path :str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv(os.path.join("notebooks","data","raw","online.csv"))
            logging.info(f"Dataset shape: {df.shape}")
            logging.info("Database read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train and Test split completed")
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

# if __name__ =="__main__":
#     obj = DataIngestion()
#     train_data,test_data = obj.initiate_data_ingestion()


#     data_transformation = DataTransformation()

#     (
#         X_train_return,
#         X_test_return,
#         y_train_return,
#         y_test_return,

#         X_train_high_value,
#         X_test_high_value,
#         y_train_high_value,
#         y_test_high_value,

#         rfm_train_scaled,
#         rfm_test_scaled

#     ) = data_transformation.initiate_data_transformation(train_data, test_data)


#     print("Data Transformation Completed")

#     print("Return Train Shape:", X_train_return.shape)
#     print("High Value Train Shape:", X_train_high_value.shape)
#     print("Segmentation Train Shape:", rfm_train_scaled.shape)
#         # modelTrainer = modelTrainer()
#         # print( modelTrainer.initiate_model_trainer(train_arr,test_arr))