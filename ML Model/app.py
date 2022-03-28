from contextlib import nullcontext
from distutils.log import debug
from flask import Flask, render_template, request, jsonify
import pickle
import warnings
import numpy as np
import pandas as pd
from processor import pipeline

deja = list()

app = Flask(__name__)

warnings.filterwarnings('ignore')

@app.route("/",methods=["POST"])
def predict():
    disease_descr = None
    disease_prec = None


    # Reroute moderations to global variable
    global deja

    text = request.form.get('message')

    with open("data.pkl","rb") as file:
        data = pickle.load(file)

    model = data["model"]


    processed = pipeline(text,deja)

    if (type(processed) == str) and (text.lower() != "no"):
        output_text = processed
    elif (len(deja) == 0) and (text.lower() != "no"):
        output_text = "Do you have any other symptoms? If yes, enter remaining symptoms separated by commas."
        deja = processed
    else:
        disease = model.predict([processed])[0]
        output_text = "It seems like you may have " + disease
        descr = pd.read_csv("Database/symptom_Description.csv")
        prec = pd.read_csv("Database/symptom_precaution.csv")
        disease_descr = str(descr[descr['Disease'] == disease]['Description'].values[0])
        disease_prec = "Until you see your doctor, it is advisable to " + str(prec[prec['Disease'] == disease]['Precaution_1'].values[0]) + " " + str(prec[prec['Disease'] == disease]['Precaution_2'].values[0]) + " " + str(prec[prec['Disease'] == disease]['Precaution_3'].values[0]) + " " + str(prec[prec['Disease'] == disease]['Precaution_4'].values[0])
        deja.clear()

    output = {"disease":output_text,"disease_descr":disease_descr,"disease_prec":disease_prec}
    
    return jsonify(output)

@app.route("/test")
def test():
    return "API WORKS"

if __name__ == "__main__":
    app.run(debug=True)