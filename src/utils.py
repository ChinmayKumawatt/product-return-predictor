import os
import sys
import dill
import numpy as np
import pandas as pd
import json
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
    logging.info("pickle file created")


def evaluate_models(X_train, y_train, X_test, y_test, models, params, report_name):

    try:

        report = {}
        best_models = {}

        os.makedirs("model_reports", exist_ok=True)

        for model_name, model in models.items():

            param_grid = params[model_name]

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            report[model_name] = {

                "accuracy": round(accuracy,4),
                "precision": round(precision,4),
                "recall": round(recall,4),
                "f1_score": round(f1,4)

            }

            best_models[model_name] = best_model

        report_path = os.path.join(
            "model_reports",
            f"{report_name}.json"
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):

    try:

        if not os.path.exists(file_path):
            raise Exception(f"File not found at path: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)