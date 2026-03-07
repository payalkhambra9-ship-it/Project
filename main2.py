from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("insurance_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route('/result', methods=['POST'])
def result():

    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])

    features = np.array([[age, bmi, children]])

    prediction = model.predict(features)

    return render_template("predict.html",
                           prediction_text=f"Estimated Insurance Cost: ${round(prediction[0],2)}")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True,port=5003)