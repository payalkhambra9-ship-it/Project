from flask import Flask, render_template
import joblib

app = Flask(__name__)

model = joblib.load("laptop_model.lb")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True,port=5002)