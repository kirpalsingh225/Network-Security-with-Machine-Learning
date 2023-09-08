import os
import sys
import seaborn as sns
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils import save_object
from src.utils import evaluate_models
from sklearn.metrics import precision_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(y_train)
            models = {
                "Random Forest" : RandomForestClassifier(),
                "Logistic Regression" : LogisticRegression()
            }
            model_report : dict = evaluate_models(x_train, y_train, x_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info("model report generated")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            sns.heatmap(confusion_matrix(predicted, y_test), fmt='g', annot=True)
            precision = precision_score(predicted, y_test, average='weighted')
            return precision

        except Exception as e:
            raise CustomException(e, sys)