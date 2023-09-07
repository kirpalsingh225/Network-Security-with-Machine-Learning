from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

## Route for home page
@app.route("/")
def home():
    return render_template("home.html")

# @app.route("/precision/<float:pre>")
# def show(pre):
#     return "the precision was" + str(pre)

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST", "GET"])
def submit():
   if request.method=="POST":
        data = CustomData(
            duration = float(request.form["duration"]),
            protocol_type=request.form["protocol-type"],
            service = request.form["service"],
            flag= request.form["flag"],
            dst_host_srv_rerror_rate = float(request.form["dst_host_srv_rerror_rate"]),
            land=request.form["land"],
            wrong_fragment=request.form["wrong-fragment"],
            urgent=request.form["urgent"],
            num_failed_logins=request.form["failed_logins"],
            logged_in=  request.form["logged_in"],
            root_shell=request.form["root_shell"],
            su_attempted=request.form["su_attempt"],
            num_shells=request.form["num_shells"],
            num_access_files=request.form["num_access_files"],
            num_outbound_cmds=request.form["num_outbound_cmds"],
            is_host_login=request.form["is_host_login"],
            is_guest_login=request.form["is_guest_login"],
            level=request.form["level"],
            src_bytes=float(request.form["src_bytes"]),
            dst_bytes=float(request.form["dst_bytes"]),
            hot=float(request.form["hot"]),
            num_compromised=float(request.form["num_compromised"]),
            num_root=float(request.form["num_root"]),
            num_file_creations=float(request.form["file_creation"]),
            count=float(request.form["count"]),
            srv_count=float(request.form["srv_count"]),
            serror_rate=float(request.form["serror_rate"]),
            srv_serror_rate=float(request.form["srv_serror_rate"]),
            rerror_rate=float(request.form["rerror_rate"]),
            srv_rerror_rate=float(request.form["srv_rerror_rate"]),
            same_srv_rate=float(request.form["same_srv_rate"]),
            diff_srv_rate=float(request.form["diff_srv_rate"]),
            srv_diff_host_rate=float(request.form["srv_diff_host_rate"]),
            dst_host_count = float(request.form["dst_host_count"]),
            dst_host_srv_count = float(request.form["dst_host_srv_count"]),
            dst_host_same_srv_rate = float(request.form["dst_host_same_srv_rate"]),
            dst_host_diff_srv_rate = float(request.form["dst_host_diff_srv_rate"]),
            dst_host_same_src_port_rate = float(request.form["dst_host_same_src_port_rate"]),
            dst_host_srv_diff_host_rate = float(request.form["dst_host_srv_diff_host_rate"]),
            dst_host_serror_rate = float(request.form["dst_host_serror_rate"]),
            dst_host_srv_serror_rate = float(request.form["dst_host_srv_serror_rate"]),
            dst_host_rerror_rate = float(request.form["dst_host_rerror_rate"])

        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        if result[0]==1:
            return render_template("predict.html", final_result="attackable")
        else:
            return render_template("predict.html", final_result="normal")
            
        
        

if __name__=="__main__":
    app.run()

