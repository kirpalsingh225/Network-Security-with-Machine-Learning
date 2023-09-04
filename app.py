from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## Route for home page
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/precision/<int:pre>")
def show(pre):
    return "the precision was" + str(pre)

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST", "GET"])
def submit():
    val = 0
    if request.method=="POST":
        dur = float(request.form["duration"])
        val = dur
    return redirect(url_for("precision", score = val))    

if __name__=="__main__":
    app.run(debug=True)

