from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("cust_model.lb")

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route('/predict_result', methods=['POST'])
def predict_result():

    age = int(request.form['age'])
    flight_distance = int(request.form['flight_distance'])
    entertainment = int(request.form['entertainment'])
    baggage = int(request.form['baggage'])
    cleanliness = int(request.form['cleanliness'])
    dep_delay = int(request.form['dep_delay'])
    arr_delay = int(request.form['arr_delay'])

    features = np.array([[age, flight_distance, entertainment, baggage, cleanliness, dep_delay, arr_delay]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Customer is Satisfied 😊"
    else:
        result = "Customer is Not Satisfied 😞"

    return render_template("predict.html", prediction_text=result)


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)