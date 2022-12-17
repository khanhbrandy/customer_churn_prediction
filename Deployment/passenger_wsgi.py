#    Created by: Ngoc Khanh Trinh
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib


app = Flask(__name__, static_url_path='')
CORS(app)

@app.route('/')
def index():
    return "Customer Churn Prediction Model"

@app.route('/churn')
def churn():
    return render_template("index.html")

@app.route("/churn/predict", methods=['POST', 'GET'])
def predict():
    # Dummy features
    features = ['account_length', 'number_vmail_messages', 'total_day_minutes',
       'total_day_calls', 'total_eve_minutes', 'total_eve_calls',
       'total_night_minutes', 'total_night_calls', 'total_intl_minutes',
       'total_intl_calls', 'number_customer_service_calls', 'state_AL',
       'state_AR', 'state_AZ', 'state_CA', 'state_CO', 'state_CT', 'state_DC',
       'state_DE', 'state_FL', 'state_GA', 'state_HI', 'state_IA', 'state_ID',
       'state_IL', 'state_IN', 'state_KS', 'state_KY', 'state_LA', 'state_MA',
       'state_MD', 'state_ME', 'state_MI', 'state_MN', 'state_MO', 'state_MS',
       'state_MT', 'state_NC', 'state_ND', 'state_NE', 'state_NH', 'state_NJ',
       'state_NM', 'state_NV', 'state_NY', 'state_OH', 'state_OK', 'state_OR',
       'state_PA', 'state_RI', 'state_SC', 'state_SD', 'state_TN', 'state_TX',
       'state_UT', 'state_VA', 'state_VT', 'state_WA', 'state_WI', 'state_WV',
       'state_WY', 'area_code_415', 'area_code_510', 'international_plan_yes',
       'voice_mail_plan_yes']
    # Load trained model
    clf = joblib.load('trained_model.pkl')
    # Get validation data
    record = json.loads(request.data)
    df = pd.DataFrame(record.items()).T
    new_header = df.iloc[0] 
    df = df[1:] 
    df.columns = new_header 
    # Transform data
    mising_dummies = [f for f in features if f not in df.columns]
    for f in mising_dummies:
        df[f] = 0
    # Re-arrange the order
    df = df[features]
    df.fillna(0, inplace=True)
    # Predict
    prob = clf.predict_proba(df)[:,1][0]
    cls = clf.predict(df)[0]
    if cls ==0:
        label = "Not churn"
    else:
        label = "Churn"
    result = {"prob":prob, "label":label}
    response = jsonify(result)
    
    return response

application = app
# app.run()