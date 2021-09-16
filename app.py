import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, request, redirect, session
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
app.secret_key = 'ItShouldBeAnythingButSecret'

user = {"username": "admin", "password": "admin"}


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == user['username'] and password == user['password']:
            session['user'] = username
            return redirect('/')

        return "<h1>Wrong username or password</h1>"  # if the username or password does not matches

    return render_template("login.html")


@app.route('/')
def home():
    if ('user' in session and session['user'] == user['username']):
        return render_template('index.html')
    else:
        return redirect('/login')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    int_features[0] = np.log(int(int_features[0]))
    int_features[1] = int(int_features[1])
    int_features[2] = int(int_features[2])
    int_features[3] = np.log(int(int_features[3]))
    int_features[4] = np.log(int(int_features[4]))
    int_features[5] = int(int_features[5])
    int_features[6] = int(int_features[6])
    int_features[7] = np.log(int(int_features[7]))
    int_features[8] = int(int_features[8])
    int_features[9] = float(int_features[9])
    int_features[10] = int(int_features[10])
    int_features[11] = int(int_features[11])
    int_features[12] = int(int_features[12])
    final_features = (np.array(int_features))
    final_features = final_features.reshape(1, -1)
    # print(final_features)
    prediction = model.predict(final_features)
    # output = prediction[0]
    if prediction[0] == 0:
        output = 'patient is not at risk of a Heart Attack'
    else:
        output = 'patient is at risk of a Heart Attack!'

    return render_template("index.html", prediction_text='Currently, {}'.format(output))


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)

