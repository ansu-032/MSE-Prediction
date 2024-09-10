from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(_name_)

# Load encoders and models
country_encoder = joblib.load('models/country_encoder.pkl')
balance_encoder = joblib.load('models/balance_encoder.pkl')
product_encoder = joblib.load('models/product_encoder.pkl')
month_encoder = joblib.load('models/month_encoder.pkl')
model_1 = joblib.load('models/model_1.pkl')
model_2 = joblib.load('models/model_2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract input data
    country = data['Country']
    balance = data['Balance']
    product = data['Product']
    month = data['Month']
    value = data['Value']

    # Encode the input data
    encoded_country = country_encoder.transform([country])[0]
    encoded_balance = balance_encoder.transform([balance])[0]
    encoded_product = product_encoder.transform([product])[0]
    encoded_month = month_encoder.transform([month])[0]

    # Create the feature array
    input_features = [[encoded_country, encoded_balance, encoded_product, encoded_month, value]]

    # Select the model based on 'Balance'
    if balance == 'Net Electricity Production':
        model = model_1
    else:
        model = model_2

    # Make prediction
    predicted_value = model.predict(input_features)[0]

    return jsonify({'predicted_value': predicted_value})

if _name_ == '_main_':
    app.run(debug=True)