import sys
import os
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test_pred, y_test)
            recall = recall_score(y_test_pred, y_test)
            precision = precision_score(y_test_pred, y_test)
            report[list(models.keys())[i]] = precision
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)