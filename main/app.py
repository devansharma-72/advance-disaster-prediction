from flask import Flask, render_template, redirect
import subprocess


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guidelines')
def guidelines():
    return render_template('guidelines.html')

@app.route('/casestudy')
def casestudy():
    return render_template('casestudy.html')

@app.route('/forest_fire')
def forest_fire():
    # Redirect to the forest fire prediction page of the external Flask app
    return redirect('http://127.0.0.1:5001')
    
@app.route('/earthquake-prediction')
def earthquake_prediction():
    return redirect('http://localhost:8501/')

    
if __name__ == '__main__':
    app.run(port=5000)
