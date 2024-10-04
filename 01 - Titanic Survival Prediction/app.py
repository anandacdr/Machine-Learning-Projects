from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
dt_model = pickle.load(open('dt_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    lr_prediction = lr_model.predict(features_scaled)
    dt_prediction = dt_model.predict(features_scaled)
    
    # Get prediction probabilities
    lr_proba = lr_model.predict_proba(features_scaled)[0][1]
    dt_proba = dt_model.predict_proba(features_scaled)[0][1]
    
    return render_template('index.html',
                         lr_prediction=f'Logistic Regression Prediction: {"Survived" if lr_prediction[0] == 1 else "Did Not Survive"} (Probability: {lr_proba:.2f})',
                         dt_prediction=f'Decision Tree Prediction: {"Survived" if dt_prediction[0] == 1 else "Did Not Survive"} (Probability: {dt_proba:.2f})')

if __name__ == '__main__':
    app.run(debug=True)