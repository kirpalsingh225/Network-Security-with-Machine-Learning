import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            num_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'level']

            cat_cols = ['protocol_type', 'service', 'flag']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("numerical columns scaling completed")
            logging.info("categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test dataset completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.data_transformer_object()
            num_cols = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'level']

            cat_cols = ['protocol_type', 'service', 'flag']

            target_feature = "attack"

            input_feature_train_df = train_df.drop([target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop([target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info("applying preprocessing pipeline of train and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)
            
            # columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            # 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            # 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            # 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            # 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            # 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            # 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            # 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            # 'dst_host_srv_count', 'dst_host_same_srv_rate',
            # 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            # 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            # 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            # 'dst_host_srv_rerror_rate', 'level', 'attack']
            # train_dataf = pd.DataFrame(data=input_feature_train_arr, columns=columns)
            # test_dataf = pd.DataFrame(data=input_feature_test_arr, columns=columns)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)