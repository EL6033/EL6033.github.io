import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from flask import Flask, render_template, request
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import pickle

loaded_model = pickle.load(open('/Users/qifeng/Documents/polygence app/finalized_model.sav', 'rb'))
imputer = pickle.load(open('/Users/qifeng/Documents/polygence app/imputer.sav', 'rb'))
logistic_loaded_model = pickle.load(open('/Users/qifeng/Documents/polygence app/finalized_logistic_model.sav', 'rb'))
scaler = pickle.load(open('/Users/qifeng/Documents/polygence app/scaler.sav','rb'))
print("T_T")



# Flask web app
app = Flask(__name__)

@app.route('/')
def index():
    # return str(os.listdir('mysite/'))
    return render_template('diabetes.html')


# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input values from the HTML form
        event_58 = float(request.form['event_58'])
        event_33 = float(request.form['event_33'])
        event_34 = float(request.form['event_34'])
        event_62 = float(request.form['event_62'])
        event_60 = float(request.form['event_60'])

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[event_58, event_33, event_34, event_62, event_60]], columns=[58, 33, 34, 62, 60])

        # Fill any missing values in the input data (optional, depending on your data preprocessing)
        input_data_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
        input_data_imputed = scaler.transform(input_data_imputed)
        # Make the prediction
        prediction = loaded_model.predict(input_data_imputed)[0]
        prediction_two = logistic_loaded_model.predict(input_data_imputed)[0]
        message = "Higher"
        range = "[" + str(round(prediction)) + "," + str(round(prediction+50)) + "]"
        if prediction_two != True:
            message = "Lower"
            range = "[" + str(round(prediction-50)) + "," + str(round([prediction])) + "]"
        return render_template('result.html', prediction=prediction, message = message, range = range)
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug = True)  # Use a different available port

