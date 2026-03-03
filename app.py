from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and scaler
# We do this once when the server starts, not every time a user makes a request
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Route 1: The Home Page
@app.route('/')
def home():
    # This serves the HTML file to the user's browser
    return render_template('index.html')

# Route 2: The Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the data typed into the HTML form
    attendance = float(request.form['attendance'])
    marks = float(request.form['marks'])
    assignments = float(request.form['assignments'])
    tests = float(request.form['tests'])
    participation = float(request.form['participation'])
    
    # 2. Arrange the data exactly how the model expects it (a 2D numpy array)
    input_data = np.array([[attendance, marks, assignments, tests, participation]])
    
    # 3. Scale the data using the EXACT SAME scaler we used in train.py
    scaled_data = scaler.transform(input_data)
    
    # 4. Make the prediction (0 or 1)
    prediction = model.predict(scaled_data)[0]
    
    # 5. Get the probability (e.g., 0.85 means 85% sure they will drop out)
    probability = model.predict_proba(scaled_data)[0][1]
    
    # 6. Format the result to send back to the web page
    if prediction == 1:
        risk_level = "HIGH RISK"
        recommendation = "Immediate academic counseling required. Send early warning alert."
        color_class = "text-red-600 bg-red-100 border-red-400"
    else:
        risk_level = "LOW RISK"
        recommendation = "Student is performing well. No immediate action needed."
        color_class = "text-green-700 bg-green-100 border-green-400"
        
    return render_template('index.html', 
                           prediction_text=f"Risk Level: {risk_level}",
                           probability_text=f"Probability: {probability:.0%}",
                           recommendation=recommendation,
                           color_class=color_class)

if __name__ == "__main__":
    # Start the local web server
    app.run(debug=True, port=5001)