import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            pre_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = pre_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self,duration, protocol_type, service, flag, src_bytes,
       dst_bytes, land, wrong_fragment, urgent, hot,
       num_failed_logins, logged_in, num_compromised, root_shell,
       su_attempted, num_root, num_file_creations, num_shells,
       num_access_files, num_outbound_cmds, is_host_login,
       is_guest_login, count, srv_count, serror_rate,
       srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate,
       diff_srv_rate, srv_diff_host_rate, dst_host_count,
       dst_host_srv_count, dst_host_same_srv_rate,
       dst_host_diff_srv_rate, dst_host_same_src_port_rate,
       dst_host_srv_diff_host_rate, dst_host_serror_rate,
       dst_host_srv_serror_rate, dst_host_rerror_rate,
       dst_host_srv_rerror_rate, level):
        self.duration = duration
        self.protocol_type = protocol_type
        self.service = service
        self.flag = flag
        self.src_bytes = src_bytes
        self.dst_bytes = dst_bytes
        self.land = land
        self.wrong_fragment = wrong_fragment
        self.hot = hot
        self.num_failed_logins = num_failed_logins
        self.logged_in = logged_in
        self.num_compromised = num_compromised
        self.root_shell = root_shell
        self.su_attempted = su_attempted
        self.num_file_creations = num_file_creations
        self.num_root = num_root
        self.num_shells = num_shells
        self.num_access_files = num_access_files
        self.num_outbound_cmds = num_outbound_cmds
        self.is_host_login = is_host_login
        self.is_guest_login = is_guest_login
        self.count = count
        self.srv_count = srv_count
        self.serror_rate = serror_rate
        self.srv_serror_rate = srv_serror_rate
        self.rerror_rate = rerror_rate
        self.srv_rerror_rate = srv_rerror_rate
        self.same_srv_rate = same_srv_rate
        self.diff_srv_rate = diff_srv_rate
        self.srv_diff_host_rate = srv_diff_host_rate
        self.dst_host_count = dst_host_count
        self.dst_host_srv_count = dst_host_srv_count
        self.dst_host_same_srv_rate = dst_host_same_srv_rate
        self.dst_host_diff_srv_rate = dst_host_diff_srv_rate
        self.dst_host_same_src_port_rate = dst_host_same_src_port_rate
        self.dst_host_srv_diff_host_rate = dst_host_srv_diff_host_rate
        self.dst_host_serror_rate = dst_host_serror_rate
        self.dst_host_srv_serror_rate = dst_host_srv_serror_rate
        self.dst_host_rerror_rate = dst_host_rerror_rate
        self.dst_host_srv_rerror_rate = dst_host_srv_rerror_rate
        self.urgent = urgent
        self.level = level

    def get_data_as_dataframe(self):
            try:
                custom_data_dict = {
                    'duration':[self.duration]
                    , 'protocol_type':[self.protocol_type]
                    , 'service':[self.service], 
                    'flag':[self.flag], 
                    'src_bytes':[self.src_bytes],
                    'dst_bytes':[self.dst_bytes],
                    'land':[self.land], 
                    'wrong_fragment':[self.wrong_fragment], 
                    'urgent':[self.urgent], 
                    'hot':[self.hot],
                    'num_failed_logins':[self.num_failed_logins], 
                    'logged_in':[self.logged_in], 
                    'num_compromised':[self.num_compromised], 
                    'root_shell':[self.root_shell],
                    'su_attempted':[self.su_attempted], 
                    'num_root':[self.num_root], 
                    'num_file_creations':[self.num_file_creations], 
                    'num_shells':[self.num_shells],
                    'num_access_files':[self.num_access_files], 
                    'num_outbound_cmds':[self.num_outbound_cmds], 
                    'is_host_login':[self.is_host_login],
                    'is_guest_login':[self.is_guest_login], 
                    'count':[self.count], 
                    'srv_count':[self.srv_count], 
                    'serror_rate':[self.serror_rate],
                    'srv_serror_rate':[self.srv_serror_rate], 
                    'rerror_rate':[self.rerror_rate], 
                    'srv_rerror_rate':[self.srv_rerror_rate], 
                    'same_srv_rate':[self.same_srv_rate],
                    'diff_srv_rate':[self.diff_srv_rate], 
                    'srv_diff_host_rate':[self.srv_diff_host_rate], 
                    'dst_host_count':[self.dst_host_count],
                    'dst_host_srv_count':[self.dst_host_srv_count], 'dst_host_same_srv_rate':[self.dst_host_same_srv_rate],
                    'dst_host_diff_srv_rate':[self.dst_host_diff_srv_rate], 'dst_host_same_src_port_rate':[self.dst_host_same_src_port_rate],
                    'dst_host_srv_diff_host_rate':[self.dst_host_srv_diff_host_rate], 'dst_host_serror_rate':[self.dst_host_serror_rate],
                    'dst_host_srv_serror_rate':[self.dst_host_srv_serror_rate], 'dst_host_rerror_rate':[self.dst_host_rerror_rate],
                    'dst_host_srv_rerror_rate':[self.dst_host_srv_rerror_rate], 
                    'level':[self.level]
                }
                return pd.DataFrame(custom_data_dict)
            except Exception as e:
                raise CustomException(e, sys)


        