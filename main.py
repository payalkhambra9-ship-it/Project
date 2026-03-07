from flask import Flask ,render_template,request
import numpy as np
import joblib
app=Flask(__name__)
model=joblib.load('linear_model.lb')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict')
def project():
    return render_template('project.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        brand_name= request.form['brand_name']
        owner= int(request.form['owner'])
        age= int(request.form['age'])
        power= int(request.form['power'])
        kms_driven= int(request.form['kms_driven'])
        # print("upload data")
        brand_dict={
                'TVS':1,   'Royal Enfield':2,         'Triumph':3,          'Yamaha':4,
           'Honda':5,            'Hero':6,           'Bajaj':7,          'Suzuki':8,
         'Benelli':9,             'KTM':10,        'Mahindra':11,        'Kawasaki':12,
          'Ducati':13,         'Hyosung':14, 'Harley-Davidson':15,            'Jawa':16,
             'BMW':17,          'Indian':18,         'Rajdoot':19,             'LML':20,
           'Yezdi':21,              'MV':22,           'Ideal':23
                }
        brand_name=brand_dict[brand_name]
        print("labels: ",brand_name, owner, age, power, kms_driven)
        labels=[[brand_name, owner, age, power, kms_driven]]
        pred=model.predict(labels)
        # print(type(pred))
        # pred=np.ravel(pred)
        print("output: ",pred)
    return render_template('project.html', prediction=pred)







if __name__=='__main__':
    app.run(debug=True,port=5001)