from flask import Flask, render_template, flash, redirect, url_for, session, request
#from data import Articles

import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



#Articles = Articles()

# Index
@app.route('/')
def index():
    return render_template('about.html')



@app.route('/mlcode')
def mlcode():
    return render_template('MLCODE.html')

@app.route('/waterpollution')
def waterpollution():
    return render_template('waterpollution.html')

@app.route('/details')
def details():
    return render_template('details.html')







@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('about.html', prediction_text='Water Quality Index value should be "wqi" {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


















if __name__ == '__main__':
    app.secret_key='\xaf\xce\xa0\xe7\x97\x05\xb1o\x9f\xcf(\x15\xd3\xa0\xd5\xf7k\xd2\xc3,b\x8aI4'
    app.run(debug=True)
