# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model_path = '../models/random_forest_fraud_model.pkl'
model = joblib.load(model_path)
logging.info("Model loaded successfully.")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert JSON data to DataFrame
    input_data = pd.DataFrame([data])
    
    # Ensure input format matches training data
    input_data = pd.get_dummies(input_data, columns=['source', 'browser', 'sex'], drop_first=True)
    
    # Fill any missing columns with zeros
    missing_cols = set(model.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    
    input_data = input_data[model.feature_names_in_]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    fraud_prob = model.predict_proba(input_data)[0][1]
    
    # Log the request and response
    logging.info(f"Prediction request: {data} - Result: {prediction}, Probability of fraud: {fraud_prob}")
    
    # Send back the prediction response
    return jsonify({'fraud_prediction': int(prediction), 'fraud_probability': fraud_prob})

# Define a status endpoint
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'API is up and running'})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
