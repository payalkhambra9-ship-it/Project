from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("house_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route('/result', methods=['POST'])
def result():

    rm = float(request.form['rm'])
    lstat = float(request.form['lstat'])
    ptratio = float(request.form['ptratio'])

    features = np.array([[rm, lstat, ptratio]])

    prediction = model.predict(features)

    return render_template("predict.html",
                           prediction_text=f"Estimated House Price: ${round(prediction[0],2)}")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True,port=5005)