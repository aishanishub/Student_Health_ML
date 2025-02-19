from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load ML Model
with open('health_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Homepage Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        exercise = int(request.form['exercise'])

        # Prepare input for ML model
        features = np.array([[age, bmi, exercise]])
        prediction = model.predict(features)[0]

        # Interpret the result
        health_status = "Healthy ✅" if prediction == 1 else "At Risk ⚠️"
        return render_template('result.html', prediction=health_status)

    except Exception as e:
        return f"Error: {e}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
