from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("loan_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/result', methods=['POST'])
def result():

    applicant_income = float(request.form['income'])
    loan_amount = float(request.form['loan_amount'])
    credit_history = float(request.form['credit'])

    features = np.array([[applicant_income, loan_amount, credit_history]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Not Approved ❌"

    return render_template("predict.html", prediction_text=result)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True,port=5004)